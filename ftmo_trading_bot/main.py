"""
===============================================================================
FTMO Trading Bot — จุดเริ่มต้นของโปรแกรม (Main Entry Point)
===============================================================================
ไฟล์หลักที่ใช้เริ่มต้น Bot รวมถึง:
- เริ่มต้นทุกโมดูล (MT5, Risk Manager, Position Sizer)
- Main Loop สำหรับตรวจสอบความเสี่ยงและรันกลยุทธ์
- Graceful Shutdown เมื่อหยุด Bot

การใช้งาน:
    python main.py                # รันปกติ (live trading)
    python main.py --status       # แสดงสถานะปัจจุบัน
===============================================================================
"""

import json
import os
import sys
import signal
import time as time_module
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import argparse

from config.settings import bot_config, get_symbol_config
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager, BotState
from core.position_sizer import PositionSizer
# v8.0 (2026-05-06): live path uses Mean Reversion strategy by default.
# `LiveMRScanner` is a drop-in replacement for `SMCStrategy` (same
# `scan_all_symbols()` + `_ltf_data`/`_mtf_data` accessors). SMC code is kept
# in `strategy/smc_strategy.py` as a deprecated reference module — the live
# entry no longer imports it.
from strategy.mean_reversion_strategy import (
    LiveMRScanner as SMCStrategy,   # alias for backward-compat in this file
    MRSignalType as SignalType,
)
from execution.trade_executor import TradeExecutor
from execution.trade_manager import TradeManager
from analytics.trade_logger import TradeLogger
from analytics.performance import PerformanceAnalyzer
from core.time_manager import TimeManager
from core.notifier import DiscordNotifier
from core.news_scheduler import NewsCalendarScheduler

try:
    from ml.rl_agent import SelfLearningAgent
except ImportError:
    pass


class FTMOTradingBot:
    """
    คลาสหลักของ FTMO Trading Bot
    
    ทำหน้าที่:
    - เริ่มต้นและจัดการทุกโมดูล
    - รัน Main Loop (ตรวจสอบความเสี่ยง → วิเคราะห์กลยุทธ์ → ส่งคำสั่ง)
    - จัดการ Shutdown อย่างปลอดภัย
    
    ลำดับความสำคัญ (เช็คทุก Loop):
    1. Risk Check (ความเสี่ยง) — สำคัญที่สุด
    2. Session Check (ช่วงเวลา)
    3. Strategy Signal (สัญญาณเทรด)
    4. Trade Execution (ส่งคำสั่ง)
    5. Position Management (จัดการ SL/TP)
    """

    def __init__(self):
        """เริ่มต้น Bot — สร้างทุกโมดูลแต่ยังไม่เชื่อมต่อ"""
        self._running = False
        self._notifier = DiscordNotifier()
        
        # === สร้างโมดูลหลัก ===
        self._connector = MT5Connector()
        self._risk_manager = RiskManager(self._connector)
        self._position_sizer = PositionSizer(self._connector)
        self._strategy = SMCStrategy(self._connector)
        # Project root — ใช้ทั้ง TradeLogger + NewsCalendarScheduler
        _project_root = os.path.dirname(os.path.abspath(__file__))
        # v6.9: TradeLogger เปิดอีกครั้ง — เก็บ live demo data สำหรับวิเคราะห์
        # Schema v3 = 63 cols (core + ML v2 + E1/E2 enhanced + Obs27 JSON) + Signals sheet (per-scan)
        try:
            self._logger = TradeLogger(
                log_dir=os.path.join(_project_root, "logs")
            )
        except Exception as e:
            print(f"⚠️ [Bot] TradeLogger init failed: {e} — running without trade log")
            self._logger = None
        self._analyzer = PerformanceAnalyzer(initial_balance=100000.0)
        self._executor = TradeExecutor(
            self._connector, 
            self._risk_manager, 
            self._position_sizer,
            logger=self._logger,
            analyzer=self._analyzer
        )
        self._trade_manager = TradeManager(self._connector, self._risk_manager, self._executor)
        
        # === AI Signal Filter Agent (Phase 5, v8.0: MR model first, SMC fallback) ===
        # v8.0 — live now expects MR model under models/mr/. Load order:
        #   1. models/mr/ppo_mr_filter.zip (rename to ppo_signal_filter.zip
        #      because SelfLearningAgent.model_path is hardcoded). The
        #      auto_train_pipeline copies the best MR model to this name.
        #   2. fallback to legacy models/ppo_signal_filter.zip
        self._rl_agent = None
        _mr_model_dir = os.path.join(bot_config.paths.model_dir, "mr")
        _candidate_dirs = (
            [_mr_model_dir] if os.path.isdir(_mr_model_dir) else []
        ) + [bot_config.paths.model_dir]
        for _md in _candidate_dirs:
            try:
                _agent = SelfLearningAgent(model_dir=_md, verbose=1)
                # MR pipeline names model `ppo_mr_filter.zip`. SelfLearningAgent
                # expects `ppo_signal_filter.zip` — accept either via override.
                _mr_zip = os.path.join(_md, "ppo_mr_filter.zip")
                _mr_vec = os.path.join(_md, "vec_normalize_mr.pkl")
                if os.path.exists(_mr_zip):
                    _agent.model_path = _mr_zip
                if os.path.exists(_mr_vec):
                    _agent.vec_normalize_path = _mr_vec
                _agent.initialize_model(strict=True)
                self._rl_agent = _agent
                print(f"   ✅ RL agent loaded from {_md}")
                break
            except Exception as e:
                print(f"   ⚠️ RL agent load from {_md} failed: {e}")
        if self._rl_agent is None:
            print(f"⚠️ [Bot] AI Signal Filter ไม่พร้อม — ใช้ MR strategy โดยตรง")

        # === ML Signal Quality Model (GBM, AUC ~0.59) ===
        # ให้ probability ว่า signal จะ win → feed เป็น obs feature ให้ RL agent
        # v8.0: prefer MR-trained GBM (mr_signal_quality_model.pkl) over legacy SMC GBM
        self._quality_model = None
        try:
            import os as _os
            from ml.signal_quality import SignalQualityModel
            _mr_gbm = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "data", "mr_signal_quality_model.pkl"
            )
            _mpath = _mr_gbm if _os.path.exists(_mr_gbm) else _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "data", "signal_quality_model.pkl"
            )
            if _os.path.exists(_mpath):
                self._quality_model = SignalQualityModel(_mpath)
                print(f"✅ [Bot] โหลด ML Quality Model สำเร็จ")
            else:
                print(f"⚠️ [Bot] ML Quality Model ไม่พบที่ {_mpath} — obs[ml_score]=0.5 (neutral)")
        except Exception as e:
            print(f"⚠️ [Bot] ML Quality Model load fail: {e}")

        # === Chronos 2 Forecaster (v7, 2026-05-01) — feeds obs[27,28] ===
        # Zero-shot M15 forecast (median + q10 + q90 ใน 8 bars ahead)
        # Disabled via bot_config.ml.CHRONOS_ENABLED = False หรือ env BOT_DISABLE_CHRONOS=1
        self._chronos = None
        try:
            if getattr(bot_config.ml, "CHRONOS_ENABLED", True):
                from ml.chronos_forecaster import ChronosForecaster
                self._chronos = ChronosForecaster(
                    model_name=bot_config.ml.CHRONOS_MODEL_NAME,
                    device=bot_config.ml.CHRONOS_DEVICE,
                    prediction_length=bot_config.ml.CHRONOS_PREDICTION_LENGTH,
                    context_length=bot_config.ml.CHRONOS_CONTEXT_LENGTH,
                    verbose=1,
                )
                if not self._chronos.is_available:
                    print(f"⚠️ [Bot] Chronos ยังไม่พร้อม — obs[27,28] จะเป็น 0 (neutral)")
            else:
                print(f"ℹ️ [Bot] Chronos disabled via config — obs[27,28] = 0")
        except Exception as e:
            print(f"⚠️ [Bot] Chronos init fail: {e} — obs[27,28] = 0")
            self._chronos = None

        # === News Calendar Auto-Scheduler (อัพเดททุกอาทิตย์ 23:30 EET) ===
        # _project_root กำหนดข้างบนแล้ว
        self._news_scheduler = NewsCalendarScheduler(
            inbox_dir=os.path.join(_project_root, "config", "news_inbox"),
            output_json=os.path.join(_project_root, "config", "news_calendar.json"),
            state_file=os.path.join(_project_root, "logs", "news_scheduler_state.json"),
        )

        # === ตัวแปรสถิติ ===
        self._loop_count = 0
        self._start_time = None

        # v8.0.63: Drift warning throttle state — กัน spam "GBM Drift" warning ที่ fire ทุกชม.
        # Re-print เฉพาะเมื่อ drift count เปลี่ยน ≥ 3 features หรือผ่านไป ≥ 6 ชม.
        self._last_drift_count: int = -1
        self._last_drift_announce_ts: Optional[float] = None  # epoch seconds

        # v8.0.70: Signal scan throttle — wall-clock gate (เดิม % 12 loops = 60s @ 5s loop)
        # ต้องแยกจาก loop_count เพราะ v8.0.70 adaptive ทำให้ loop เป็น 1s ทุกครั้งที่มี
        # position open → ถ้าใช้ loop_count จะ scan ทุก 12s (เร็วเกิน). ใช้ wall-clock
        # 60s gate แทน → scan cadence คงที่ไม่ว่า loop จะเร็วแค่ไหน
        self._last_signal_scan_ts: Optional[float] = None  # epoch seconds

        # v6.9: track trade open history for overtrading detection
        # List of tuples (datetime, symbol) — capped at last 200 entries
        self._trade_open_history: list = []

        # === Announce-once flags (idle states — print on entry, silence on repeat) ===
        self._daily_halt_announced = False
        self._weekend_announced = False
        self._friday_announced = False
        self._daily_close_announced = False
        self._rollover_announced = False

        # === v6.10: Daily summary + Stats sheet trigger ===
        # _last_logged_day = วันสุดท้ายที่ log_daily_summary ถูกเรียก
        # None = ยังไม่เคย log → ตั้งครั้งแรกใน loop (ไม่ flush ของวันก่อน)
        self._last_logged_day = None

        # === จัดการ Signal สำหรับ Graceful Shutdown ===
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._print_banner()

    def _print_banner(self):
        """แสดง Banner ตอนเริ่มต้น Bot"""
        print("\n" + "=" * 70)
        print("""
    ███████╗████████╗███╗   ███╗ ██████╗     ██████╗  ██████╗ ████████╗
    ██╔════╝╚══██╔══╝████╗ ████║██╔═══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
    █████╗     ██║   ██╔████╔██║██║   ██║    ██████╔╝██║   ██║   ██║   
    ██╔══╝     ██║   ██║╚██╔╝██║██║   ██║    ██╔══██╗██║   ██║   ██║   
    ██║        ██║   ██║ ╚═╝ ██║╚██████╔╝    ██████╔╝╚██████╔╝   ██║   
    ╚═╝        ╚═╝   ╚═╝     ╚═╝ ╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   
        """)
        print("    🤖 Algorithmic Forex Trading Bot — FTMO Compliant")
        print("    📋 Version: 3.0.0 (Phase 3 — Trade Execution & Management)")
        print("    🛡️ Risk Management: ACTIVE")
        print("=" * 70 + "\n")

    # =========================================================================
    # 🚀 การเริ่มต้นและเชื่อมต่อ
    # =========================================================================

    def initialize(self) -> bool:
        """
        เริ่มต้นทุกโมดูลและเชื่อมต่อ MT5
        
        ลำดับ:
        1. เชื่อมต่อ MT5
        2. เริ่มต้น Risk Manager (โหลดสถานะเดิม)
        3. ตรวจสอบ Position ที่เปิดค้าง
        
        Returns:
            bool: True ถ้าเริ่มต้นสำเร็จ
        """
        print("🔄 [Bot] กำลังเริ่มต้นระบบ...\n")

        if self._rl_agent:
            print("━" * 40)
            print("🧠 AI Signal Filter Agent — พร้อมกรอง MR signals (PPO + AuxTask)")
            print("━" * 40)

        # ขั้นตอนที่ 1: เชื่อมต่อ MT5
        print("━" * 40)
        print("📡 ขั้นตอนที่ 1: เชื่อมต่อ MetaTrader 5")
        print("━" * 40)
        if not self._connector.connect():
            print("❌ [Bot] เชื่อมต่อ MT5 ล้มเหลว — หยุดการทำงาน")
            return False

        # ขั้นตอนที่ 2: เริ่มต้น Risk Manager
        print("\n" + "━" * 40)
        print("🛡️ ขั้นตอนที่ 2: เริ่มต้น Risk Manager")
        print("━" * 40)
        if not self._risk_manager.initialize():
            print("❌ [Bot] เริ่มต้น Risk Manager ล้มเหลว")
            return False

        # ขั้นตอนที่ 3: ตรวจสอบ Position ที่เปิดค้าง + Orphan Recovery (v8.0.13)
        print("\n" + "━" * 40)
        print("📋 ขั้นตอนที่ 3: ตรวจสอบ Position ค้าง")
        print("━" * 40)
        open_positions = self._connector.get_open_positions()
        if open_positions:
            print(f"⚠️ มี {len(open_positions)} Position เปิดค้างอยู่:")
            for pos in open_positions:
                print(f"   📌 {pos['symbol']} {pos['type']} Vol={pos['volume']} P/L=${pos['profit']:,.2f}")
            # v8.0.13: Re-attach orphan positions to executor._active_trades
            # กัน duplicate-open + ให้ TradeManager จัดการต่อ (BE/Partial/Trail/News)
            print("♻️ [Bot] Orphan recovery — re-attaching positions to active_trades…")
            self._executor.sync_with_mt5()
            print(f"✅ [Bot] Active trades หลัง sync: {len(self._executor.active_trades)} ticket")
        else:
            print("✅ ไม่มี Position ค้าง")

        # ขั้นตอนที่ 3.5: Seed Analyzer ด้วย balance จริงจาก broker + peak จาก state
        # → Max DD / Sharpe / equity curve จะสอดคล้องกับบัญชีจริง ไม่ใช่ hardcoded 100k
        try:
            real_initial = self._risk_manager.initial_balance
            peak_balance = getattr(self._risk_manager, "_highest_balance", None)
            if real_initial > 0:
                self._analyzer.set_initial_balance(real_initial, peak_balance)
        except Exception as e:
            print(f"⚠️ [Bot] seed analyzer balance ล้มเหลว: {e}")

        # ขั้นตอนที่ 3.6: v6.14 — re-enable ftmo_trades.xlsx replay เข้า Analyzer
        #  → Stats sheet จะสะท้อนสถานะ challenge ทั้งหมด ไม่ใช่เฉพาะ session ปัจจุบัน
        #  → equity curve / Max DD / Sharpe ต่อเนื่องข้าม restart
        #  → ถ้าต้องการ reset ให้ลบ logs/ftmo_trades.xlsx ก่อน run
        try:
            excel_path = os.path.join(self._logger._log_dir, "ftmo_trades.xlsx") \
                if self._logger is not None else None
            if excel_path and os.path.exists(excel_path):
                loaded = self._analyzer.load_from_excel(excel_path)
                print(f"📊 [Bot] Analyzer replay: {loaded} closed trades from {excel_path}")
        except Exception as e:
            print(f"⚠️ [Bot] Analyzer load_from_excel ล้มเหลว: {e}")

        # ขั้นตอนที่ 4: เตรียมกลยุทธ์ Mean Reversion (v8.0+)
        from config.settings import bot_config as _bc
        _mr = _bc.mr
        print("\n" + "━" * 40)
        print("🎯 ขั้นตอนที่ 4: เริ่มต้น Mean Reversion Strategy Engine")
        print("━" * 40)
        print(f"   📊 Strategy: Mean Reversion + Trend Filter (v8.0)")
        print(f"   📦 Entry: BB%B {_mr.bb_oversold:.2f}/{_mr.bb_overbought:.2f} + RSI {_mr.rsi_oversold:.0f}/{_mr.rsi_overbought:.0f} + ADX H1 ≤ {_mr.adx_trend_block:.0f}")
        print(f"   🎯 SL = {_mr.sl_atr_mult:.1f}×ATR  |  TP = {_mr.rr_ratio:.1f}×SL  (RR 1:{_mr.rr_ratio:.1f}, quick TP)")
        # `SMCStrategy` is aliased to `LiveMRScanner` in v8.0+ — class attr `MIN_CONFLUENCE_SCORE`
        # is the MR confluence floor (default 30). Kept import name for backward-compat.
        print(f"   🧭 ML threshold: {_bc.ftmo.ML_FILTER_THRESHOLD:.2f}  |  MR setup score floor: {SMCStrategy.MIN_CONFLUENCE_SCORE:.0f}/100")

        # ขั้นตอนที่ 5: แสดงสรุปการตั้งค่า
        self._print_config_summary()

        print("\n✅ [Bot] เริ่มต้นระบบสำเร็จ — พร้อมทำงาน!\n")
        self._notifier.send_startup()
        return True

    # =========================================================================
    # 🧠 RL Agent — Live Observation & Daily Re-tune
    # =========================================================================

    def _compute_symbol_regime(self, symbol: str) -> dict:
        """
        คำนวณ regime signal ของ 1 symbol จาก H1 100 แท่งล่าสุด

        Returns:
            dict หรือ None ถ้าดึงข้อมูลไม่ได้
        """
        try:
            df = self._connector.get_ohlcv(symbol, "H1", count=100)
            if df is None or len(df) < 50:
                return None

            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            n = len(close)

            # True Range → ATR
            hl = high[1:] - low[1:]
            hc = np.abs(high[1:] - close[:-1])
            lc = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(hl, hc), lc)

            atr_recent = float(np.mean(tr[-14:]))
            atr_baseline = float(np.mean(tr))
            atr_std = float(np.std(tr)) or 1e-8
            atr_zscore = (atr_recent - atr_baseline) / atr_std

            # ATR pips — JPY pairs ใช้ pip_size 0.01, คู่อื่น 0.0001
            pip_size = 0.01 if float(np.mean(close)) > 50 else 0.0001
            atr_pips = atr_recent / pip_size

            # Trend slope (normalized to ATR/bar — scale-invariant ข้าม symbols)
            window = min(50, n)
            y = close[-window:]
            x = np.arange(window, dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0])
            slope_per_atr = slope / max(atr_recent, 1e-8)

            # จำแนก regime
            if abs(slope_per_atr) > 0.15:
                regime = "trending"
            elif atr_zscore > 1.0:
                regime = "volatile"
            elif atr_zscore < -0.8:
                regime = "quiet"
            else:
                regime = "ranging"

            # Multi-window slope consistency
            consistent = 0
            total = 0
            for w in (20, 40, 60, 80, min(100, n)):
                if w > n:
                    continue
                yy = close[-w:]
                xx = np.arange(w, dtype=np.float64)
                s = float(np.polyfit(xx, yy, 1)[0])
                total += 1
                if np.sign(s) == np.sign(slope) and abs(s) > 0:
                    consistent += 1
            consistency = (consistent / total) if total > 0 else 0.0

            return {
                "regime": regime,
                "atr_zscore": atr_zscore,
                "atr_pips": atr_pips,
                "slope_sign": int(np.sign(slope_per_atr)),
                "consistency": consistency,
            }
        except Exception:
            return None

    def _compute_live_regime(self) -> dict:
        """
        คำนวณ regime aggregate จาก **ทุก symbols ที่บอทเทรด** (bot_config.symbols.symbols)
        สะท้อนสถานะ portfolio จริง ไม่ bias ไปทาง EURUSD อย่างเดียว

        Aggregation:
        - regime_trend_norm: majority vote (trending=+1, ranging=-1, อื่น=0)
        - atr_zscore: mean ข้าม symbols (normalize แล้ว)
        - volatility_norm: mean ของ atr_pips/pair_baseline — ใช้ baseline 12 สำหรับ non-JPY, 120 สำหรับ JPY
        - regime_consistency: mean ของ per-symbol consistency

        Returns:
            dict: {regime_trend_norm, atr_zscore, volatility_norm, regime_consistency}
        """
        default = {
            "regime_trend_norm": 0.0,
            "atr_zscore": 0.0,
            "volatility_norm": 0.0,
            "regime_consistency": 0.0,
        }
        try:
            symbols = bot_config.symbols.symbols
            per_symbol = []
            for sym in symbols:
                info = self._compute_symbol_regime(sym)
                if info is not None:
                    per_symbol.append((sym, info))

            if not per_symbol:
                return default

            # Regime vote: trending=+1, ranging=-1, อื่น=0
            regime_votes = []
            for _, info in per_symbol:
                if info["regime"] == "trending":
                    regime_votes.append(1.0)
                elif info["regime"] == "ranging":
                    regime_votes.append(-1.0)
                else:
                    regime_votes.append(0.0)
            regime_trend_norm = float(np.mean(regime_votes))

            # ATR z-score: mean (แต่ละ symbol z-score ของตัวเอง → scale เปรียบเทียบได้)
            atr_zscore = float(np.mean([info["atr_zscore"] for _, info in per_symbol]))

            # Volatility norm: normalize atr_pips ต่อ baseline ของ pair type
            # Non-JPY baseline ~12 pips, JPY baseline ~80 pips (H1 ATR ปกติ)
            vol_norms = []
            for sym, info in per_symbol:
                is_jpy = sym.endswith("JPY")
                baseline = 80.0 if is_jpy else 12.0
                scale = 40.0 if is_jpy else 6.0
                vol_norms.append((info["atr_pips"] - baseline) / scale)
            volatility_norm = float(np.mean(vol_norms))

            consistency = float(np.mean([info["consistency"] for _, info in per_symbol]))

            return {
                "regime_trend_norm": float(np.clip(regime_trend_norm, -1.0, 1.0)),
                "atr_zscore": float(np.clip(atr_zscore, -3.0, 3.0)),
                "volatility_norm": float(np.clip(volatility_norm, -2.0, 2.0)),
                "regime_consistency": float(np.clip(consistency, 0.0, 1.0)),
            }
        except Exception as e:
            print(f"⚠️ [RL] compute_live_regime ล้มเหลว: {e}")
            return default

    def _get_challenge_day(self, today: date) -> int:
        """
        นับวันที่ของ FTMO challenge (business days ตั้งแต่เริ่ม)
        เก็บ start_date ในไฟล์ persistent — สร้างอัตโนมัติรอบแรก

        Returns:
            int: วันที่ของ challenge (1..30+) — 1 = วันแรก
        """
        state_path = os.path.join(bot_config.paths.model_dir, "challenge_state.json")
        start_date = None
        try:
            if os.path.exists(state_path):
                import json
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                start_date = date.fromisoformat(data.get("start_date", ""))
        except Exception:
            start_date = None

        if start_date is None:
            start_date = today
            try:
                import json
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({"start_date": start_date.isoformat()}, f)
                print(f"📅 [RL] เริ่ม FTMO Challenge วันที่ {start_date.isoformat()} (Day 1)")
            except Exception:
                pass

        # นับ business days (Mon-Fri) ระหว่าง start_date → today, inclusive
        delta = (today - start_date).days
        if delta < 0:
            return 1
        business = 0
        for i in range(delta + 1):
            d = start_date.fromordinal(start_date.toordinal() + i)
            if d.weekday() < 5:
                business += 1
        return max(1, business)

    def _build_signal_observation(self, sig) -> np.ndarray:
        """
        สร้าง 27-dim observation จาก TradeSignal + portfolio state สำหรับ Signal Filter Agent
        ต้อง match กับ FTMOSignalFilterEnv._get_obs() ลำดับและ scale
        """
        try:
            risk = self._risk_manager.get_risk_status()
            total_dd = float(risk.get("overall_drawdown_pct", 0.0))
            daily_dd = float(risk.get("daily_loss_pct", 0.0))
            balance = float(risk.get("current_balance", self._risk_manager.initial_balance))
            initial = float(self._risk_manager.initial_balance) or 100_000.0
        except Exception:
            total_dd, daily_dd, balance, initial = 0.0, 0.0, 100_000.0, 100_000.0

        profit = balance - initial
        target_amount = initial * 0.10
        progress = (profit / target_amount) * 100.0 if target_amount > 0 else 0.0

        # Signal features (7 original)
        confluence_norm = (sig.confluence_score - 50.0) / 50.0
        rr_norm = (sig.rr_ratio - 1.0) / 4.0
        direction = 1.0 if sig.signal_type.value == "BUY" else -1.0
        atr_val = max(sig.atr_value, 1e-8)
        atr_pips = atr_val / (0.01 if sig.entry_price > 50 else 0.0001)
        atr_norm = (atr_pips - 15.0) / 10.0
        # v8.0 — obs[4] reinterprets ob_norm as bb_extreme (MR signal strength)
        # MR signals carry `bb_extreme` in [0, 1]; SMC fallback uses ob_score/100
        ob_norm = float(getattr(sig, "bb_extreme", None) or sig.ob_score / 100.0)
        bias_align = direction * sig.market_bias
        sl_atr = sig.sl_distance / atr_val

        # Signal features — momentum & context (4)
        rsi_norm = (sig.rsi_value - 50.0) / 50.0
        macd_norm = sig.macd_histogram / atr_val
        trend_str = sig.trend_strength / 100.0
        # v8.0 — obs[10] reinterprets ob_size_atr as bb_band_width_atr/3
        # MR signals carry `bb_band_width_atr`; SMC fallback uses OB body / ATR
        bb_bw_atr = float(getattr(sig, "bb_band_width_atr", 0.0) or 0.0)
        if bb_bw_atr > 0.0:
            ob_size_atr = min(bb_bw_atr / 3.0, 1.0)
        else:
            ob_range = (
                abs(sig.ob_high - sig.ob_low)
                if sig.ob_high is not None and sig.ob_low is not None
                else 0.0
            )
            ob_size_atr = ob_range / atr_val

        # Signal features — market regime (5 ใหม่)
        adx_norm = sig.adx / 100.0
        stoch_norm = (sig.stoch_k - 50.0) / 50.0
        bb_pctb = sig.bb_pctb
        atr_chg = sig.atr_change_ratio
        price_roc = sig.price_roc

        # ML quality score — GBM P(win), AUC 0.59 ⭐
        if self._quality_model is not None:
            try:
                # ต้องแปลง TradeSignal → dict หรือให้ model อ่าน attrs ตรง ๆ
                ml_score = float(self._quality_model.score(sig))
            except Exception:
                ml_score = 0.5
        else:
            ml_score = 0.5
        ml_score_norm = (ml_score - 0.5) * 2.0  # map [0,1] → [-1,+1]

        # Portfolio features
        # v6.10b: ใช้ broker date (EEST) — กัน timezone offset จาก VPS local time
        try:
            challenge_day = self._get_challenge_day(TimeManager.get_server_time().date())
        except Exception:
            challenge_day = 0
        day_progress = float(challenge_day) / 45.0

        open_positions = len(self._connector.get_open_positions() or [])
        trades_today_n = min(open_positions, 3) / 3.0

        recent_wr_norm = 0.0
        consec_losses = 0
        try:
            trades = getattr(self._analyzer, "_trades", None) or []
            if trades:
                recent = trades[-10:]
                wins = sum(1 for t in recent if float(getattr(t, "profit", 0.0) or 0.0) > 0)
                recent_wr_norm = (wins / len(recent)) * 2.0 - 1.0
                for t in reversed(trades[-5:]):
                    if float(getattr(t, "profit", 0.0) or 0.0) < 0:
                        consec_losses += 1
                    else:
                        break
        except Exception:
            pass

        # === v7 (2026-05-01): Chronos 2 forecast features [27-28] ===
        # ใช้ M15 cache ของ strategy (อัปเดตทุก scan รอบ ~60s)
        # cache key (symbol, last_bar_ts) ใน ChronosForecaster → hit ทุก ~15min/symbol
        chronos_align = 0.0
        chronos_unc = 0.0
        if self._chronos is not None and self._chronos.is_available:
            try:
                ltf_df = getattr(self._strategy, "_ltf_data", None)
                if ltf_df is not None and len(ltf_df) >= 32:
                    chronos_align, chronos_unc = self._chronos.forecast_features(
                        sig.symbol, ltf_df, float(direction), float(atr_val)
                    )
            except Exception as e:
                if not getattr(self, "_chronos_warned", False):
                    print(f"⚠️ [main] Chronos forecast failed: {e} — fallback 0")
                    self._chronos_warned = True

        obs = np.array([
            # Signal core [0-11]
            float(np.clip(confluence_norm, -1.0, 1.0)),
            float(np.clip(rr_norm, 0.0, 1.0)),
            float(direction),
            float(np.clip(atr_norm, -2.0, 2.0)),
            float(np.clip(ob_norm, 0.0, 1.0)),
            float(np.clip(bias_align, -1.0, 1.0)),
            float(np.clip(sl_atr, 0.0, 2.0)),
            float(np.clip(rsi_norm, -1.0, 1.0)),
            float(np.clip(macd_norm, -2.0, 2.0)),
            float(np.clip(trend_str, 0.0, 1.0)),
            float(np.clip(ob_size_atr, 0.0, 3.0)),
            float(np.clip(adx_norm, 0.0, 1.0)),
            # Market regime [12-15]
            float(np.clip(stoch_norm, -1.0, 1.0)),
            float(np.clip(bb_pctb, -0.5, 1.5)),
            float(np.clip(atr_chg, -1.0, 1.0)),
            float(np.clip(price_roc, -3.0, 3.0)),
            # ML quality [16] ⭐
            float(np.clip(ml_score_norm, -1.0, 1.0)),
            # Portfolio [17-23]
            float(np.clip(-total_dd / 0.10, -5.0, 0.0)),
            float(np.clip(-daily_dd / 0.05, -5.0, 0.0)),
            float(np.clip(progress / 100.0, -1.0, 2.0)),
            float(np.clip(day_progress, 0.0, 1.0)),
            float(np.clip(trades_today_n, 0.0, 1.0)),
            float(np.clip(recent_wr_norm, -1.0, 1.0)),
            float(np.clip(consec_losses / 5.0, 0.0, 1.0)),
            # v6 Cost/Flip/HTF [24-26] — match FTMOSignalFilterEnv (v8.0 MR semantics)
            float(np.clip(self._build_spread_pct_of_atr(sig, atr_pips), 0.0, 3.0)),
            float(self._has_opposite_recently_closed(sig)),
            # obs[26] — v8.0 MR: adx_inverse_norm (1.0 when ADX low = ranging =
            # favorable for MR). Falls back to bias_align when ADX missing.
            float(np.clip(1.0 - sig.adx / 50.0, 0.0, 1.0))
            if hasattr(sig, "adx") and sig.adx is not None
            else float(np.clip(bias_align, -1.0, 1.0)),
            # v7 Chronos forecast [27-28]
            float(np.clip(chronos_align, -1.0, 1.0)),
            float(np.clip(chronos_unc, 0.0, 3.0)),
            # v7.1 Portfolio realtime + session timing [29-31]
            float(np.clip(self._compute_floating_pnl_norm(), -3.0, 1.0)),
            float(np.clip(self._compute_open_losing_count_norm(), 0.0, 1.0)),
            float(np.clip(self._compute_mins_since_session_norm(), 0.0, 1.0)),
        ], dtype=np.float32)
        return obs

    def _compute_floating_pnl_norm(self) -> float:
        """v7.2.1 (Option B) — return 0 to match training env (leak fix).

        Training env ตั้ง obs[29] = 0 ตลอดเพื่อตัด future leak (ดู
        signal_filter_env.py v7.2.1 comment). Live ก็ต้อง 0 ตามเพื่อ
        distribution match — ป้องกัน VecNormalize blow-up จาก var≈0 ตอน train.
        Concurrent risk awareness ทำผ่าน RiskManager.check_unrealized_circuit_breaker
        (execution path, ไม่ใช่ obs).
        """
        return 0.0

    def _compute_open_losing_count_norm(self) -> float:
        """v7.2.1 (Option B) — return 0 to match training env (leak fix). ดู
        _compute_floating_pnl_norm docstring."""
        return 0.0

    def _compute_current_session(self) -> str:
        """Mirror TradeExecutor session classifier (server time = EET).

        Used by `_log_signal_scan` to populate Signals sheet `Session` column,
        which was previously hardcoded empty (Trades sheet got it via different path).
        """
        try:
            h = TimeManager.get_server_time().hour
            if 7 <= h < 12:
                return "LONDON"
            if 12 <= h < 13:
                return "LONDON_NY_OVERLAP"
            if 13 <= h < 17:
                return "NEW_YORK"
            if 17 <= h < 21:
                return "NY_AFTERNOON"
            if 0 <= h < 7:
                return "ASIAN"
            return "OFF_HOURS"
        except Exception:
            return ""

    def _compute_mins_since_session_norm(self) -> float:
        """v7.1 — minutes since London/NY session open / 480 (8 hours)"""
        try:
            from datetime import datetime as _dt
            import pytz as _pytz
            now_utc = TimeManager.get_server_time().astimezone(_pytz.UTC)
            hour, minute = now_utc.hour, now_utc.minute
            if 8 <= hour < 13:
                mins = (hour - 8) * 60 + minute  # London-relative
            elif 13 <= hour < 16:
                mins = (hour - 13) * 60 + minute  # NY-relative
            else:
                return 0.0
            return min(mins / 480.0, 1.0)
        except Exception:
            return 0.0

    def _build_spread_pct_of_atr(self, sig, atr_pips: float) -> float:
        """spread_pips / atr_pips — normalize ต้นทุน spread (v6)"""
        try:
            price_info = self._connector.get_current_price(sig.symbol)
            if price_info is None:
                return 0.0
            symbol_info = self._connector.get_symbol_info(sig.symbol)
            if symbol_info is None:
                return 0.0
            pip_size = 0.01 if symbol_info["digits"] <= 3 else 0.0001
            spread_pips = price_info["spread"] / (pip_size / symbol_info["point"])
            return spread_pips / max(atr_pips, 1e-6) if atr_pips > 0 else 0.0
        except Exception:
            return 0.0

    def _has_opposite_recently_closed(self, sig) -> float:
        """1.0 ถ้ามี trade ตรงข้ามปิดภายใน 30 นาที (flip-lock context, v6)"""
        try:
            flip_lock = getattr(self._risk_manager, '_flip_lock', None)
            if flip_lock is None:
                # RiskManager ยังไม่ init _flip_lock → log เพื่อ debug
                if not getattr(self, '_flip_lock_warned', False):
                    print("⚠️ [main] RiskManager._flip_lock missing — obs[25] จะเป็น 0 ตลอด")
                    self._flip_lock_warned = True
                return 0.0
            lock = flip_lock.get(sig.symbol)
            if not lock:
                return 0.0
            closed_dir = lock.get("closed_direction", "")
            signal_dir = sig.signal_type.value
            if closed_dir and closed_dir != signal_dir:
                return 1.0
            return 0.0
        except Exception as e:
            if not getattr(self, '_flip_lock_warned', False):
                print(f"⚠️ [main] flip_lock read failed: {e}")
                self._flip_lock_warned = True
            return 0.0

    def _build_live_context(self, sig) -> Dict:
        """v6.9 — สร้าง dict ของ live context สำหรับ logging + executor.

        เก็บข้อมูล ณ moment ของ signal scan ที่ executor ไม่รู้เอง:
          - ml_score (calibrated + raw)
          - market context: ADX H1/H4, MTF/D1 bias
          - bid/ask snapshot
          - account balance
        """
        # v6.11 (Tier 2.4) อ่าน per-component pts + htf_bias จาก signal โดยตรง
        # — เก่า: hardcode 0 ทุกตัว → Trades sheet HTF/MTF/OB/FVG/Sweep pts ว่างเปล่าหมด
        ctx = {
            "ml_score": 0.5,
            "ml_score_raw": 0.5,
            "agent_action_value": 0.0,
            "agent_decision": "",
            "ml_threshold_used": float(bot_config.ftmo.ML_FILTER_THRESHOLD),
            "htf_score": int(getattr(sig, "htf_score", 0)),
            "mtf_score": int(getattr(sig, "mtf_score", 0)),
            "ob_pts": int(getattr(sig, "ob_pts", 0)),
            "fvg_pts": int(getattr(sig, "fvg_pts", 0)),
            "sweep_pts": int(getattr(sig, "sweep_pts", 0)),
            "htf_bias": getattr(sig, "htf_bias", "") or "",
            "bid_at_entry": 0.0,
            "ask_at_entry": 0.0,
            "spread_pips_actual": 0.0,
            "adx_h1": 0.0,
            "adx_h4": 0.0,
            "mtf_bias": int(getattr(sig, "market_bias", 0)),
            "d1_bias": int(getattr(sig, "d1_bias", 0)),
            "balance_at_entry": 0.0,
            # v7 Chronos forecast features (mirror obs[27,28]) — for Excel logging
            "chronos_align": 0.0,
            "chronos_unc": 0.0,
        }

        # v7: Chronos forecast — same compute path as _build_signal_observation
        if self._chronos is not None and self._chronos.is_available:
            try:
                ltf_df = getattr(self._strategy, "_ltf_data", None)
                atr_val = max(float(getattr(sig, "atr_value", 0.0) or 0.0), 1e-8)
                direction = 1.0 if sig.signal_type.value == "BUY" else -1.0
                if ltf_df is not None and len(ltf_df) >= 32:
                    a, u = self._chronos.forecast_features(
                        sig.symbol, ltf_df, direction, atr_val
                    )
                    ctx["chronos_align"] = float(a)
                    ctx["chronos_unc"] = float(u)
            except Exception:
                pass

        # v7.1 — compute temporal/regime features (สำหรับ GBM input + RL obs)
        try:
            from ml.signal_quality import compute_temporal_features
            ltf_df_for_temp = getattr(self._strategy, "_ltf_data", None)
            symbol_info = self._connector.get_symbol_info(sig.symbol)
            pip = 0.0001 if symbol_info and symbol_info.get("digits", 5) >= 4 else 0.01
            is_metal = "XAU" in sig.symbol.upper() or "XAG" in sig.symbol.upper()
            atr_floor = float(get_symbol_config(
                sig.symbol, "atr_floor_pips", 100.0 if is_metal else 8.0
            ))
            temporal_feats = compute_temporal_features(
                timestamp=getattr(sig, "timestamp", None) or datetime.now(timezone.utc),
                ltf_df=ltf_df_for_temp,
                atr_floor_pips=atr_floor,
                pip_size=pip,
            )
            # Keep ใน ctx เพื่อให้ _build_signal_observation อ่านได้ (RL obs v7.1)
            ctx["_temporal_feats"] = temporal_feats
            for k, v in temporal_feats.items():
                ctx[k] = v
        except Exception:
            ctx["_temporal_feats"] = {}

        # ML scores (calibrated via SignalQualityModel + raw via base GBM)
        # v7.1 — augment sig ด้วย temporal_feats ก่อน score (GBM ต้องการ 24 features)
        try:
            if self._quality_model is not None:
                # สร้าง dict รวม sig attrs + temporal_feats สำหรับ GBM
                score_input = {
                    k: self._quality_model._extract(sig, k)
                    for k in self._quality_model.keys
                }
                # Override ด้วย temporal feats ที่เพิ่ง compute (กัน 0.0 fallback)
                for k, v in ctx.get("_temporal_feats", {}).items():
                    if k in self._quality_model.keys:
                        score_input[k] = float(v)
                ctx["ml_score"] = float(self._quality_model.score(score_input))
                # raw probability (skip calibrator)
                if self._quality_model.calibrator is not None:
                    raw = self._quality_model.model.predict_proba(
                        np.array([[score_input[k] for k in self._quality_model.keys]],
                                 dtype=np.float64)
                    )[0, 1]
                    ctx["ml_score_raw"] = float(raw)
                else:
                    ctx["ml_score_raw"] = ctx["ml_score"]

                # v7.1 — record live signal สำหรับ drift monitor
                # (record_input = score_input ที่เพิ่งคำนวณ — re-use เพื่อหลีก redundancy)
                try:
                    self._quality_model.record_live_signal(score_input)
                except Exception:
                    pass
        except Exception:
            pass

        # Confluence breakdown (parse จาก signal.reasons text)
        try:
            for r in sig.reasons:
                if "HTF" in r and "(+" in r:
                    pass  # could extract numeric — skip for now
        except Exception:
            pass

        # Market context — read from strategy state
        try:
            mtf = self._strategy._mtf_data
            htf = self._strategy._htf_data
            if mtf is not None and "adx" in mtf.columns:
                ctx["adx_h1"] = float(mtf["adx"].iloc[-1])
            if htf is not None and "adx" in htf.columns:
                ctx["adx_h4"] = float(htf["adx"].iloc[-1])
            ctx["mtf_bias"] = int(self._strategy._structure_mtf.get_current_bias())
            ctx["d1_bias"] = int(self._strategy._get_d1_bias(sig.symbol))
        except Exception:
            pass

        # Bid/ask snapshot
        try:
            tick = self._connector.get_current_price(sig.symbol)
            if tick:
                ctx["bid_at_entry"] = float(tick.get("bid", 0.0))
                ctx["ask_at_entry"] = float(tick.get("ask", 0.0))
                spread = ctx["ask_at_entry"] - ctx["bid_at_entry"]
                # pip size = 0.01 for JPY/Gold (digits<=3), else 0.0001
                symbol_info = self._connector.get_symbol_info(sig.symbol)
                pip = 0.0001 if symbol_info and symbol_info.get("digits", 5) >= 4 else 0.01
                ctx["spread_pips_actual"] = float(spread / pip) if pip > 0 else 0.0
        except Exception:
            pass

        # Account balance
        try:
            risk_status = self._risk_manager.get_risk_status()
            ctx["balance_at_entry"] = float(risk_status.get("current_balance", 0.0))
        except Exception:
            pass

        # ML threshold (v6.12 — อ่านจาก bot_config.ftmo เพื่อ sync live ↔ training)
        # เดิม: try getattr(self._rl_agent, "ml_filter_threshold") — แต่ attribute นี้
        # อยู่บน FTMOSignalFilterEnv ไม่ใช่ SelfLearningAgent → ตกค่า 0.0 ตลอด
        ctx["ml_threshold_used"] = float(bot_config.ftmo.ML_FILTER_THRESHOLD)

        # === Overtrading metrics ===
        # นับ trade history vs current time → detect overtrading patterns
        # v6.10b: ใช้ broker time (EEST) → ตรงกับ trade open_time ที่บันทึกไว้
        try:
            now = TimeManager.get_server_time().replace(tzinfo=None)
            today = now.date()
            one_hour_ago = now - timedelta(hours=1)

            ctx["trades_today_at_open"] = sum(
                1 for (t, _s) in self._trade_open_history if t.date() == today
            )
            ctx["trades_last_hour_at_open"] = sum(
                1 for (t, _s) in self._trade_open_history if t >= one_hour_ago
            )

            # delta from last trade (any symbol)
            if self._trade_open_history:
                last_t, _ = self._trade_open_history[-1]
                ctx["secs_since_last_trade_open"] = (now - last_t).total_seconds()

            # delta from last trade on SAME symbol
            same_symbol_history = [
                t for (t, s) in self._trade_open_history if s == sig.symbol
            ]
            if same_symbol_history:
                ctx["secs_since_last_trade_same_symbol"] = (
                    now - same_symbol_history[-1]
                ).total_seconds()
        except Exception:
            pass

        # === Obs vector (JSON) — for offline RL retrain ===
        # ใช้ existing _build_signal_observation (ที่ feed เข้า agent อยู่แล้ว)
        # ⚠️ key ยังเก็บชื่อ "obs_27_json" เพื่อ backward-compat กับ Excel retrain reader
        # แต่ตั้งแต่ v7 (2026-05-01) เก็บ 29 dims จริง (เพิ่ม chronos_align, chronos_uncertainty)
        # round 4 decimals → file size ~270 chars/row
        try:
            if self._rl_agent is not None:
                obs = self._build_signal_observation(sig)
                ctx["obs_27_json"] = json.dumps(
                    [round(float(x), 4) for x in obs.tolist()]
                )
        except Exception:
            ctx["obs_27_json"] = ""

        # === v6.10: Account state audit (verify DD@Entry calculation) ===
        # เก็บ raw equity / balance / floating + daily_start_equity เพื่อ debug
        # กรณี DD@Entry % ดูแปลก (เช่น 11% ทั้งที่ net P/L บวก)
        try:
            acc = self._connector.get_account_info() or {}
            bal = float(acc.get("balance", 0) or 0)
            eq = float(acc.get("equity", 0) or 0)
            ctx["balance_at_entry"] = bal
            ctx["equity_at_entry"] = eq
            ctx["floating_pnl_at_entry"] = eq - bal
        except Exception:
            pass

        try:
            ctx["daily_start_equity"] = float(
                getattr(self._risk_manager, "_daily_start_equity", 0) or 0
            )
        except Exception:
            pass

        return ctx

    def _log_signal_scan(self, sig, live_context: Dict, result: str):
        """v6.9 — บันทึก signal scan ลง Signals sheet ของ TradeLogger."""
        if self._logger is None:
            return
        try:
            scan_data = {
                "time": sig.timestamp,
                "symbol": sig.symbol,
                "direction": sig.signal_type.value,
                "result": result,
                "confluence": sig.confluence_score,
                "atr": sig.atr_value,
                "rr_target": sig.rr_ratio,
                "ml_score": live_context.get("ml_score", 0),
                "ml_score_raw": live_context.get("ml_score_raw", 0),
                "agent_action_value": live_context.get("agent_action_value", 0),
                "agent_decision": live_context.get("agent_decision", ""),
                "ml_threshold": live_context.get("ml_threshold_used", 0),
                "adx_h1": live_context.get("adx_h1", 0),
                # v6.11 (Tier 2.4) ใช้ signal.htf_bias (string "BULLISH/BEARISH/RANGING") ตรงๆ
                # — เก่า: int จาก _strategy._htf_bias → Excel แสดงเลข 1/-1/0 ไม่ informative
                "htf_bias": getattr(sig, "htf_bias", "") or live_context.get("htf_bias", ""),
                "mtf_bias": live_context.get("mtf_bias", 0),
                "d1_bias": live_context.get("d1_bias", 0),
                "session": live_context.get("session", "") or self._compute_current_session(),
                "spread_pips": live_context.get("spread_pips_actual", 0),
                "reasons": sig.reasons[:5] if isinstance(sig.reasons, list) else str(sig.reasons),
                # v6.10d: propagate executor reject reason ลง Signals sheet col 20
                # main.py scan loop ตั้ง live_context["executor_reject_reason"] หลัง execute_signal คืน None
                "executor_reject_reason": live_context.get("executor_reject_reason", ""),
                # v6.10b: propagate obs_27_json ลง Signals sheet col 21 (สำหรับ retrain)
                "obs_27_json": live_context.get("obs_27_json", ""),
                # v7: Chronos forecast features ลง Signals sheet col 22-23
                "chronos_align": live_context.get("chronos_align", 0.0),
                "chronos_unc": live_context.get("chronos_unc", 0.0),
            }
            self._logger.log_signal_scan(scan_data)
        except Exception as e:
            print(f"⚠️ [Bot] log_signal_scan error: {e}")

    def _print_config_summary(self):
        """แสดงสรุปการตั้งค่าทั้งหมด"""
        print("\n" + "━" * 40)
        print("⚙️ สรุปการตั้งค่า")
        print("━" * 40)
        sym = bot_config.symbols
        ftmo = bot_config.ftmo
        print(f"   📊 คู่เงิน ({len(sym.symbols)}):     {', '.join(sym.symbols)}")
        print(f"   ⏱️ Timeframes (v8.0): {sym.primary_timeframe} (Entry) + "
              f"{sym.structure_timeframe} (ADX trend filter)")
        print(f"   🛡️ Daily Stop:       {ftmo.DAILY_LOSS_HARD_STOP_PCT:.0%} "
              f"(FTMO 5% — buffer 1%)")
        print(f"   🛡️ Max Drawdown:     {ftmo.MAX_DRAWDOWN_HARD_STOP_PCT:.0%} "
              f"(FTMO 10% — buffer 2%)")
        print(f"   💰 Risk/Trade:       {ftmo.MIN_RISK_PER_TRADE_PCT:.2%} - "
              f"{ftmo.MAX_RISK_PER_TRADE_PCT:.2%}  "
              f"(default {ftmo.DEFAULT_RISK_PER_TRADE_PCT:.2%})")
        print(f"   🎯 R:R:              Fixed 1:{ftmo.PREFERRED_RISK_REWARD_RATIO:.1f} "
              f"(MR quick-TP) — min 1:{ftmo.MIN_RISK_REWARD_RATIO:.1f}")
        print(f"   🧭 ML threshold:     {ftmo.ML_FILTER_THRESHOLD:.2f}  "
              f"(signals < threshold filtered before agent)")
        print(f"   📋 Max Positions:    {ftmo.MAX_OPEN_POSITIONS}")
        print(f"   🎯 Profit Target:    {ftmo.PROFIT_TARGET_PCT:.0%}  "
              f"(FTMO Challenge)")
        print(f"   🔁 Loop Interval:    {bot_config.main_loop_interval}s")

    # =========================================================================
    # 🔄 Main Loop
    # =========================================================================

    def run(self):
        """
        รัน Main Loop ของ Bot
        
        ทุก Loop จะ:
        1. ตรวจสอบ Risk (Daily Loss, Max Drawdown)
        2. ตรวจสอบ Trading Session
        3. วิเคราะห์สัญญาณ (Phase 2)
        4. ดำเนินการเทรด (Phase 3)
        5. จัดการ Position (Phase 3)
        6. บันทึก Log (Phase 4)
        """
        self._running = True
        self._start_time = datetime.now()
        self._loop_count = 0

        print("\n" + "=" * 70)
        print("🚀 [Bot] เริ่มต้น Main Loop — กด Ctrl+C เพื่อหยุด")
        print("=" * 70)

        # แสดง market status ตอน start เพื่อให้ผู้ใช้รู้ว่าบอทจะ sleep หรือ active
        try:
            _srv_time = TimeManager.get_server_time()
            _weekday = _srv_time.weekday()
            _weekday_th = ['จันทร์','อังคาร','พุธ','พฤหัส','ศุกร์','เสาร์','อาทิตย์'][_weekday]
            print(f"🕐 [Bot] เวลาโบรกเกอร์: วัน{_weekday_th} {_srv_time.strftime('%Y-%m-%d %H:%M:%S')} EET")
            if _weekday in (5, 6):
                print(f"🌙 [Bot] ตลาด Forex ปิด (เสาร์-อาทิตย์) → เข้าโหมด Weekend Sleep")
                print(f"          จะตื่นอัตโนมัติเมื่อ Monday 00:00 EET (Sydney session เปิด)")
            elif _weekday == 4 and _srv_time.time() >= bot_config.sessions.friday_force_close:
                print(f"🛑 [Bot] หลัง Friday 20:45 EET → Weekend Halt (ปิด position + รอ Monday)")
            else:
                print(f"✅ [Bot] ตลาดเปิด — พร้อมสแกนสัญญาณ")
        except Exception as _e:
            print(f"⚠️ [Bot] ไม่สามารถตรวจสถานะตลาด: {_e}")
        print("=" * 70 + "\n")

        while self._running:
            try:
                self._loop_count += 1

                # === ขั้นตอนที่ 0: News Calendar Auto-Import (Sunday 23:30 EET) ===
                # Exception-safe: ไม่ block main loop แม้ scheduler พัง
                self._news_scheduler.check_and_run()

                # === v6.10: Daily summary on day rollover ===
                # Detect day change ก่อน check_risk() (เพราะ check_risk จะ reset _daily_closed_pnl)
                # → log ของวันก่อนต้อง snapshot ก่อน reset
                try:
                    broker_today = TimeManager.get_server_time().date()
                    if self._last_logged_day != broker_today:
                        if self._last_logged_day is not None and self._logger is not None:
                            risk_status = self._risk_manager.get_risk_status() or {}
                            self._logger.log_daily_summary(
                                balance=risk_status.get("current_balance", 0),
                                daily_dd_pct=risk_status.get("daily_loss_pct", 0),
                                max_dd_pct=risk_status.get("overall_drawdown_pct", 0),
                            )
                            try:
                                stats = self._analyzer.get_full_report()
                                self._logger.update_stats_sheet(stats)
                            except Exception as e:
                                print(f"⚠️ [Bot] Update Stats sheet failed: {e}")
                        self._last_logged_day = broker_today
                except Exception as e:
                    print(f"⚠️ [Bot] Daily summary check failed: {e}")

                # === v6.10: Periodic Stats update (every 720 loops = 1h @ 5s) ===
                # ทำให้ Stats sheet update realtime — user เปิด Excel ดูสถานะได้
                if (self._logger is not None
                        and self._loop_count % 720 == 0
                        and self._loop_count > 0):
                    try:
                        stats = self._analyzer.get_full_report()
                        self._logger.update_stats_sheet(stats)
                    except Exception as e:
                        print(f"⚠️ [Bot] Hourly Stats update failed: {e}")

                # === ขั้นตอนที่ 1: ตรวจสอบความเสี่ยง (สำคัญที่สุด) ===
                bot_state = self._risk_manager.check_risk()
                
                if bot_state == BotState.MAX_DRAWDOWN_HALT:
                    print("🛑 [Bot] Max Drawdown เกินขีดจำกัด — Bot หยุดถาวร")
                    self._notifier.send_risk_alert("🛑 MAX DRAWDOWN LIMIT", "พอร์ตสูญเสียเกิน Max Drawdown ของ FTMO เรียบร้อยแล้ว (Bot จะหยุดถาวร)")
                    self._running = False
                    break
                    
                if bot_state == BotState.DAILY_HALT:
                    if not self._daily_halt_announced:
                        print(f"🔒 [Bot] Daily Halt — รอวันถัดไป (เวลาปัจจุบัน: {TimeManager.get_server_time().strftime('%H:%M:%S')} EEST)")
                        self._notifier.send_risk_alert("🔒 DAILY LOSS LIMIT", "พอร์ตชนค่าจำกัดขาดทุนรายวัน (Daily Drawdown) บอทเข้าโหมดระงับการเทรดชั่วคราวจนกว่าจะขึ้นวันใหม่รอยัลโอเวอร์")
                        self._daily_halt_announced = True
                    time_module.sleep(bot_config.main_loop_interval)
                    continue
                else:
                    # ออกจาก Daily Halt → reset flag เพื่อพร้อม print ครั้งหน้า
                    self._daily_halt_announced = False
                    
                if bot_state == BotState.DISCONNECTED:
                    print("⚠️ [Bot] MT5 ไม่ได้เชื่อมต่อ — พยายามเชื่อมต่อใหม่...")
                    if not self._connector.reconnect():
                        print("❌ [Bot] เชื่อมต่อใหม่ไม่สำเร็จ — รอ 30 วินาที")
                        time_module.sleep(30)
                    continue

                # === ขั้นตอนที่ 1.5: ตรวจสอบระดับการจัดการเวลาเซิร์ฟเวอร์แบบเข้มงวด (FTMO Compliance) ===
                current_server_time = TimeManager.get_server_time()

                # หากเลยเวลา Friday 20:45 EET ระบบจะบังคับปิดทุก position + หยุดเทรดจนสัปดาห์หน้า
                if TimeManager.is_friday_close_time(current_server_time):
                    if not self._friday_announced:
                        print(f"🛑 [Bot] Friday Force Close (20:45 EET) — ปิดทุก position + หยุดสุดสัปดาห์")
                        self._friday_announced = True
                    # 🚨 FORCE CLOSE: check_session_close() มี logic ปิดทุก position ตอน Friday
                    closed = self._trade_manager.check_session_close()
                    if closed > 0:
                        print(f"🚨 [Bot] Friday Force Close — ปิด {closed} positions สำเร็จ")
                    # manage_all_positions() สำหรับ position ที่ปิดไม่สำเร็จ (retry ผ่าน trailing)
                    self._trade_manager.manage_all_positions()
                    time_module.sleep(bot_config.main_loop_interval)
                    continue
                else:
                    self._friday_announced = False

                # === Weekend Sleep (Saturday/Sunday) — market closed ===
                # ปิดตลาด Forex วันเสาร์อาทิตย์ → sleep นาน + ไม่ต้อง scan
                if TimeManager.is_weekend(current_server_time):
                    if not self._weekend_announced:
                        weekday_th = ['จันทร์','อังคาร','พุธ','พฤหัส','ศุกร์','เสาร์','อาทิตย์'][current_server_time.weekday()]
                        print(f"🌙 [Bot] Weekend Sleep — ตลาดปิดวัน{weekday_th} "
                              f"({current_server_time.strftime('%H:%M')} EET) — รอจันทร์ 00:00 EET...")
                        self._weekend_announced = True
                    # Sleep นานกว่าปกติ (60s แทน 5s) เพื่อไม่ spam CPU
                    time_module.sleep(60)
                    continue
                else:
                    self._weekend_announced = False

                # Zero-Overnight Policy (Mon-Thu 23:30 EET) — ปิดทุก position ก่อนข้ามวัน
                if TimeManager.is_daily_close_time(current_server_time):
                    if not self._daily_close_announced:
                        print(f"🌙 [Bot] Daily Close (23:30 EET) — ปิดทุก position ก่อนข้ามวัน")
                        self._daily_close_announced = True
                    # 🚨 FORCE CLOSE: check_session_close() มี logic Daily Overnight Close
                    closed = self._trade_manager.check_session_close()
                    if closed > 0:
                        print(f"🌙 [Bot] Daily Overnight Close — ปิด {closed} positions สำเร็จ")
                    self._trade_manager.manage_all_positions()  # retry สำหรับที่ปิดไม่สำเร็จ
                    time_module.sleep(bot_config.main_loop_interval)
                    continue
                else:
                    self._daily_close_announced = False

                # หากช่วงเวลา 23:55 น. ถึง 01:05 น. (Rollover / Spread Expansion)
                if TimeManager.is_rollover_period(current_server_time):
                    if not self._rollover_announced:
                        print(f"💤 [Bot] Rollover Pause — เข้าสู่โหมดหลับพักหนีสเปรดถ่างประจำวัน ({current_server_time.strftime('%H:%M:%S')} EET)")
                        self._rollover_announced = True
                    time_module.sleep(bot_config.main_loop_interval)
                    continue
                else:
                    self._rollover_announced = False

                # === ขั้นตอนที่ 2: สแกนสัญญาณ + กรองด้วย AI + ส่งคำสั่งเทรด ===
                # v8.0.70: wall-clock 60s gate (เดิม % 12 loops). decoupled จาก loop_count
                # เพราะ adaptive ทำให้ loop เป็น 1s ตอนมี position open — % 12 จะกลายเป็น
                # 12s scan (เร็วเกิน). ใช้ wall-clock 60s ให้ scan cadence คงที่
                _now_ts = time_module.time()
                _scan_due = (
                    self._last_signal_scan_ts is None
                    or (_now_ts - self._last_signal_scan_ts) >= 60.0
                )
                if _scan_due:
                    self._last_signal_scan_ts = _now_ts
                    try:
                        signals = self._strategy.scan_all_symbols()
                        for sig in signals:
                            # === v6.9: Build live_context สำหรับ logging + executor ===
                            live_context = self._build_live_context(sig)

                            # === ML quality gate (live ↔ training distribution sync, v8.0.3) ===
                            # FTMOSignalFilterEnv กรอง signals ที่ ml_score < 0.30 ตอน train
                            # → live ก็ต้องกรองเดียวกัน ไม่งั้น agent เห็น distribution กว้างกว่าที่ฝึก
                            ml_threshold = float(bot_config.ftmo.ML_FILTER_THRESHOLD)
                            if ml_threshold > 0.0 and live_context.get("ml_score", 0.0) < ml_threshold:
                                live_context["agent_decision"] = "ML_FILTERED"
                                self._log_signal_scan(sig, live_context, result="ML_FILTERED")
                                continue

                            agent_decision = "NO_AGENT"
                            agent_action_value = 0.0

                            if self._rl_agent:
                                signal_obs = self._build_signal_observation(sig)
                                take = self._rl_agent.should_take_signal(signal_obs)
                                confidence = self._rl_agent.get_action_confidence(signal_obs)
                                agent_action_value = float(confidence)
                                agent_decision = "TAKE" if take else "SKIP"
                                live_context["agent_action_value"] = agent_action_value
                                live_context["agent_decision"] = agent_decision

                                if not take:
                                    # SKIP: log into Signals sheet only (no console — silenced to reduce loop noise)
                                    self._log_signal_scan(sig, live_context, result="AGENT_SKIP")
                                    continue
                                # TAKE: keep — สัญญาณกำลังจะเปิดเทรด, event สำคัญ
                                print(f"📡 [Agent] TAKE {sig.signal_type.value} {sig.symbol} "
                                      f"Conf={sig.confluence_score:.0f} RR=1:{sig.rr_ratio:.1f} "
                                      f"(confidence={confidence:.2f})")
                            else:
                                live_context["agent_decision"] = "NO_AGENT"

                            executed = self._executor.execute_signal(sig, live_context=live_context)
                            if executed:
                                print(f"✅ [Bot] เปิดเทรดสำเร็จ: Ticket {executed.ticket}")
                                self._log_signal_scan(sig, live_context, result="AGENT_TAKE")
                                # v8.0.26: record last open time → bulk-trading guard
                                # v8.0.55: pass symbol+direction → cluster theme cooldown
                                self._risk_manager.record_trade_open(
                                    symbol=sig.symbol,
                                    direction=sig.signal_type.value,
                                )
                                # v6.9: บันทึก trade open history สำหรับ overtrading detection
                                # v6.10b: ใช้ broker time (EEST) — ตรงกับ executed.open_time
                                self._trade_open_history.append(
                                    (TimeManager.get_server_time().replace(tzinfo=None), sig.symbol)
                                )
                                # cap ที่ 200 entries (กัน memory leak run นาน ๆ)
                                if len(self._trade_open_history) > 200:
                                    self._trade_open_history = self._trade_open_history[-200:]
                            else:
                                # v6.10: capture executor reject reason for log
                                reject_reason = getattr(self._executor, "_last_reject_reason", None) or "unknown"
                                live_context["executor_reject_reason"] = reject_reason
                                self._log_signal_scan(sig, live_context, result="AGENT_TAKE_FAIL")
                    except Exception as e:
                        print(f"⚠️ [Bot] Strategy/Execution error: {e}")
                
                # === ขั้นตอนที่ 4: จัดการ Position ที่เปิดอยู่ (Trailing, BE, Partial TP) ===
                try:
                    self._trade_manager.manage_all_positions()

                    # v8.0.17 + v8.0.22: Daily Profit/Loss Cap (Option D + mirror)
                    # ใช้ closed + floating P/L เทียบกับ initial × cap_pct
                    try:
                        account = self._connector.get_account_info()
                        if account:
                            equity = account.get("equity", 0.0)
                            floating = self._connector.get_total_floating_pnl()
                            # v8.0.17: Profit cap (+1.6%)
                            if self._risk_manager.check_daily_profit_cap(equity, floating):
                                cap_closed = self._trade_manager.close_all_positions(
                                    reason="Daily Profit Cap (Option D)"
                                )
                                print(f"🎯 [Bot] Daily profit cap ถึง — ปิด {cap_closed} positions, "
                                      f"หยุดเทรดวันนี้")
                            # v8.0.22: Loss cap (-3.0%) — mirror, ตรวจคู่ขนาน
                            if self._risk_manager.check_daily_loss_cap(equity, floating):
                                cap_closed = self._trade_manager.close_all_positions(
                                    reason="Daily Loss Cap (Option D mirror)"
                                )
                                print(f"🛑 [Bot] Daily loss cap ถึง — ปิด {cap_closed} positions, "
                                      f"หยุดเทรดวันนี้")
                    except Exception as e:
                        print(f"⚠️ [Bot] Daily profit/loss cap check error: {e}")

                    # v7.1.10: Pre-news close — ปิด position ก่อนข่าวแรง sync กับ block สัญญาณใหม่
                    news_closed = self._trade_manager.check_news_close()
                    if news_closed > 0:
                        print(f"📰 [Bot] ปิด {news_closed} Position ก่อนชนข่าว")

                    # ตรวจ Session Close
                    closed = self._trade_manager.check_session_close()
                    if closed > 0:
                        print(f"⏰ [Bot] ปิด {closed} Position ก่อนหมด Session")
                except Exception as e:
                    print(f"⚠️ [Bot] Trade Manager error: {e}")
                
                # แสดงสถานะตาม verbose level (v8.0.30)
                # 0=silent: ไม่ print เลย
                # 1=normal: ทุก 720 loops (~1 ชม.) print สั้น
                # 2=debug: ทุก 60 loops (~5 นาที) print เต็ม
                _vl = getattr(bot_config, 'verbose_level', 1)
                if _vl >= 2 and self._loop_count % 60 == 0:
                    self._print_periodic_status()
                elif _vl == 1 and self._loop_count % 720 == 0 and self._loop_count > 0:
                    self._print_periodic_status()

                # v7.1 — GBM drift monitor ทุก 720 loops (~1 ชม.)
                if self._loop_count % 720 == 0 and self._loop_count > 0:
                    self._check_gbm_drift()

                # v8.0.70: Adaptive loop — มี position open → 1s ทุกครั้ง (เดิม v8.0.46 เฉพาะ profit ≥ 0.5R)
                # เหตุผล: BE trigger ที่ 0.3R (v8.0.56) ต้อง modify ภายใน 1 tick มิฉะนั้น price spike
                # ผ่าน 0.3R + revert ก่อน loop หน้า → BE ไม่ถูกตั้ง. การ poll 1s ตลอด lifecycle ของ
                # position ทำให้ Partial 0.8R + Stage 2/3 trail แม่นยำสุด. Signal scan แยก wall-clock
                # 60s gate (ด้านบน) ไม่กระทบ cadence ของการเปิด order ใหม่
                sleep_interval = bot_config.main_loop_interval
                try:
                    open_positions = self._connector.get_open_positions()
                    if open_positions:
                        sleep_interval = 1  # any open position → 1s for accurate BE/SL/TP
                except Exception:
                    pass  # fail-safe: use default interval

                # รอก่อน Loop ถัดไป
                time_module.sleep(sleep_interval)

            except Exception as e:
                print(f"❌ [Bot] เกิดข้อผิดพลาดใน Main Loop: {e}")
                import traceback
                traceback.print_exc()
                time_module.sleep(10)  # รอ 10 วินาทีก่อนลองใหม่

    def _print_periodic_status(self):
        """แสดงสถานะ Bot เป็นระยะ"""
        risk_status = self._risk_manager.get_risk_status()
        uptime = datetime.now() - self._start_time if self._start_time else None
        
        print(f"\n{'─' * 50}")
        print(f"📊 สถานะ Bot — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─' * 50}")
        print(f"   🔵 State:     {risk_status['state']}")
        print(f"   💰 Balance:   ${risk_status['current_balance']:,.2f}")
        print(f"   💎 Equity:    ${risk_status['current_equity']:,.2f}")
        print(f"   📉 Daily DD:  {risk_status['daily_loss_pct']:.2%}")
        print(f"   📉 Max DD:    {risk_status['overall_drawdown_pct']:.2%}")
        print(f"   📋 Positions: {risk_status['open_positions']}/{risk_status['max_positions']}")
        print(f"   🔁 Loop:      #{self._loop_count}")
        if uptime:
            hours = int(uptime.total_seconds() // 3600)
            minutes = int((uptime.total_seconds() % 3600) // 60)
            print(f"   ⏱️ Uptime:    {hours}h {minutes}m")
        print(f"{'─' * 50}\n")
        # v8.0.31: ลบ Discord periodic status notification (เคยส่งทุก 1 ชม.)
        # User feedback: ไม่ต้องการ status alert ใน Discord
        # Trade open/close + breach warnings ยังส่งปกติผ่าน notifier event hooks

    def _check_gbm_drift(self) -> None:
        """
        v7.1 — KS test เทียบ live signal distribution กับ training distribution.

        เรียกทุก 720 loops (~1 ชม.). ถ้ามี ≥ 3 features drift → log + Discord ping.
        ไม่ block trading — แค่ alert ให้ user รู้ว่าควร retrain.

        v8.0.14 — Append ลงไฟล์ logs/gbm_drift.log ด้วย กัน console scroll-out.
        v8.0.63 — Throttle console print: re-announce เฉพาะถ้า drift count เปลี่ยน ≥ 3
                  features หรือผ่านไป ≥ 6 ชม. (file log ยัง append ทุกครั้ง — full audit trail)
        """
        if self._quality_model is None:
            return
        try:
            drifts = self._quality_model.detect_drift(ks_threshold=0.15)
        except Exception as e:
            print(f"⚠️ [Drift] check failed: {e}")
            return

        if len(drifts) >= 3:
            top = sorted(drifts.items(), key=lambda x: -x[1])[:5]
            top_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
            count = len(drifts)

            # v8.0.63: Throttle console — print เฉพาะถ้า count เปลี่ยน ≥3 หรือ 6h elapsed
            now_ts = time_module.time()
            count_changed = abs(count - self._last_drift_count) >= 3
            time_elapsed = (
                self._last_drift_announce_ts is None
                or (now_ts - self._last_drift_announce_ts) >= 21600  # 6h
            )
            should_announce = count_changed or time_elapsed

            msg = (
                f"⚠️ [GBM Drift] {count} features ห่างจาก training "
                f"(KS > 0.15). Top: {top_str}"
            )
            if should_announce:
                print(msg)
                self._last_drift_count = count
                self._last_drift_announce_ts = now_ts
                # Discord notify เฉพาะตอน announce — กัน spam
                try:
                    self._notifier.send_alert(msg, level="warning")
                except Exception:
                    pass

            # v8.0.14: persistent drift log — append ทุกครั้งสำหรับ audit
            # v8.0.63: timezone-aware UTC (fix DeprecationWarning datetime.utcnow)
            try:
                import os
                from datetime import datetime, timezone
                root = os.path.dirname(os.path.abspath(__file__))
                log_dir = os.path.join(root, "logs")
                os.makedirs(log_dir, exist_ok=True)
                drift_log = os.path.join(log_dir, "gbm_drift.log")
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                with open(drift_log, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] count={count} top={top_str}\n")
            except Exception as e:
                print(f"⚠️ [Drift] log file write failed: {e}")

    # =========================================================================
    # 🛑 Shutdown
    # =========================================================================

    def shutdown(self):
        """หยุด Bot อย่างปลอดภัย (Graceful Shutdown)"""
        print("\n" + "=" * 70)
        print("🛑 [Bot] กำลังหยุดการทำงาน...")
        print("=" * 70)
        
        self._notifier.send_shutdown()
        self._running = False

        # === v6.10: Final Daily/Stats flush ก่อน shutdown ===
        # User กด Ctrl+C → flush latest data ลง Excel
        if self._logger is not None and self._analyzer is not None:
            try:
                stats = self._analyzer.get_full_report()
                self._logger.update_stats_sheet(stats)
                risk_status = self._risk_manager.get_risk_status() or {}
                self._logger.log_daily_summary(
                    balance=risk_status.get("current_balance", 0),
                    daily_dd_pct=risk_status.get("daily_loss_pct", 0),
                    max_dd_pct=risk_status.get("overall_drawdown_pct", 0),
                )
                print("✅ [Bot] บันทึก Daily + Stats sheets ก่อน shutdown")
            except Exception as e:
                print(f"⚠️ [Bot] Final logging ก่อน shutdown ล้มเหลว: {e}")

        # บันทึก state ล่าสุดก่อนปิด (atomic write)
        try:
            self._risk_manager.save()
            print("💾 [Bot] บันทึก state ก่อนปิดเรียบร้อย")
        except Exception as e:
            print(f"⚠️ [Bot] บันทึก state ก่อนปิดไม่สำเร็จ: {e}")

        # ตัดการเชื่อมต่อ MT5
        self._connector.disconnect()
        
        # แสดงสรุป
        if self._start_time:
            uptime = datetime.now() - self._start_time
            print(f"\n📊 สรุปการทำงาน:")
            print(f"   ⏱️ ระยะเวลาทำงาน: {uptime}")
            print(f"   🔁 จำนวน Loop: {self._loop_count}")
        
        print("\n✅ [Bot] หยุดการทำงานเรียบร้อย — สวัสดีครับ 👋\n")

    def _signal_handler(self, signum, frame):
        """จัดการ Signal (Ctrl+C) เพื่อหยุด Bot อย่างปลอดภัย"""
        print(f"\n⚠️ [Bot] ได้รับ Signal {signum} — กำลังหยุดอย่างปลอดภัย...")
        self.shutdown()
        sys.exit(0)


# =============================================================================
# 🏁 Entry Point
# =============================================================================

if __name__ == "__main__":
    # จัดการ Arguments
    parser = argparse.ArgumentParser(description="FTMO Trading Bot")
    parser.add_argument("--status", action="store_true", help="แสดงสถานะปัจจุบัน")
    args = parser.parse_args()

    if args.status:
        # โหมดแสดงสถานะ
        connector = MT5Connector()
        connector.connect()
        risk_mgr = RiskManager(connector)
        risk_mgr.initialize()
        connector.disconnect()
    else:
        # โหมดปกติ — รัน Bot
        bot = FTMOTradingBot()

        if bot.initialize():
            try:
                bot.run()
            except KeyboardInterrupt:
                pass
            finally:
                bot.shutdown()
        else:
            print("❌ เริ่มต้น Bot ล้มเหลว — ตรวจสอบการตั้งค่าและลองใหม่")
            sys.exit(1)
