"""
===============================================================================
FTMO Trading Bot — Trade Executor (ระบบส่งคำสั่งซื้อขาย)
===============================================================================
จัดการการส่งคำสั่งเทรดจากสัญญาณ SMC ไปยัง MT5

ขั้นตอนการทำงาน:
1. รับ TradeSignal จาก SMC Strategy
2. ตรวจสอบ Risk final time (Double-check)
3. คำนวณ Lot Size จาก Position Sizer
4. ส่ง Market Order พร้อม SL/TP
5. ตรวจสอบผลลัพธ์ + Retry ถ้าล้มเหลว
6. บันทึกข้อมูลเทรดสำหรับ Logger (Phase 4)

⚠️ ทุกคำสั่งต้องมี Stop Loss — ห้ามส่งคำสั่งโดยไม่มี SL เด็ดขาด
===============================================================================
"""

import time as time_module
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np

from config.settings import bot_config
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from core.position_sizer import PositionSizer
from core.time_manager import TimeManager
# v8.0.6: TradeSignal/SignalType moved to strategy.mean_reversion_strategy
# as part of SMC cleanup. MRSignal mimics the old TradeSignal field set
# (signal_type, entry_price, sl_price, confluence_score, atr_value, ...);
# SignalType is aliased to MRSignalType. trade_executor uses these as a
# duck-type interface — no behavior change.
from strategy.mean_reversion_strategy import TradeSignal

try:
    from analytics.trade_logger import TradeLogger
    from analytics.performance import PerformanceAnalyzer
except ImportError:
    pass

from core.notifier import DiscordNotifier


@dataclass
class ExecutedTrade:
    """
    ข้อมูลเทรดที่ส่งคำสั่งสำเร็จแล้ว
    ใช้ส่งต่อไปยัง Trade Logger (Phase 4) และ Trade Manager
    """
    ticket: int                     # หมายเลข Ticket จาก MT5
    symbol: str                     # คู่เงิน
    trade_type: str                 # "BUY" หรือ "SELL"
    entry_price: float              # ราคาที่เปิดจริง (อาจต่างจาก Signal เล็กน้อย)
    sl_price: float                 # Stop Loss
    tp_price: float                 # Take Profit
    lot_size: float                 # ขนาด Lot ปัจจุบัน (ลดลงหลัง partial close)
    risk_amount: float              # จำนวนเงินที่เสี่ยง (USD)
    risk_pct: float                 # เปอร์เซ็นต์ความเสี่ยง
    rr_ratio: float                 # Risk:Reward Ratio
    confluence_score: float         # คะแนน Confluence จาก Strategy
    atr_value: float                # ATR ตอนเปิดเทรด (ใช้สำหรับ Trailing)
    open_time: datetime             # เวลาเปิดเทรด
    signal_reasons: List[str]       # เหตุผลจาก Strategy
    magic_number: int = 123456      # Magic Number ของ Bot
    
    # สถานะ
    is_open: bool = True            # ยังเปิดอยู่หรือไม่
    close_price: float = 0.0        # ราคาปิด (เมื่อปิดแล้ว)
    close_time: Optional[datetime] = None  # เวลาปิด
    profit: float = 0.0             # กำไร/ขาดทุน (USD)
    close_reason: str = ""          # เหตุผลที่ปิด

    # === ML features v2 (Schema v2) ===
    session: str = ""                       # Asian / London / NY / Overlap
    day_of_week: int = 0                    # 0=Mon..6=Sun
    hour_of_day: int = 0
    spread_at_entry: float = 0.0            # points
    slippage: float = 0.0                   # price diff (signal vs fill)
    htf_bias: str = ""                      # BULLISH / BEARISH / RANGING
    volatility_regime: str = ""             # quiet / normal / high
    consec_loss_before: int = 0             # แพ้ติดกันก่อนเปิดเทรดนี้
    dd_at_entry_pct: float = 0.0            # daily DD % ณ เวลาเปิด
    mae: float = 0.0                        # Max Adverse Excursion (price)
    mfe: float = 0.0                        # Max Favorable Excursion (price)
    time_in_trade: int = 0                  # seconds
    exit_path: str = ""                     # SL / TP / Trail / BE / Manual / Friday / SessionEnd

    # === Partial close tracking ===
    original_lot_size: float = 0.0          # Lot ตอนเปิด (immutable — ใช้คำนวณ partial pct)
    partial_close_count: int = 0            # จำนวนครั้งที่ปิดบางส่วนไปแล้ว

    # === v6.9 enhanced live logging (E1+E2 deploy) ===
    # ML / Agent decision context
    ml_score: float = 0.5                   # GBM calibrated P(win), 0.5 = neutral
    ml_score_raw: float = 0.5               # uncalibrated raw probability
    agent_action_value: float = 0.0         # raw policy output [-1, 1]
    agent_decision: str = ""                # TAKE / SKIP / FALLBACK / NO_AGENT
    ml_threshold_used: float = 0.0          # ml_filter_threshold at decision time

    # Confluence scoring breakdown (parsed from strategy)
    htf_score: int = 0                      # HTF (H4) trend points
    mtf_score: int = 0                      # MTF (H1) bias points
    ob_pts: int = 0                         # Order Block points
    fvg_pts: int = 0                        # Fair Value Gap points
    sweep_pts: int = 0                      # Liquidity sweep points

    # Trade management state (TradeManager interactions, final at close)
    be_moved: bool = False                  # SL moved to entry (BE)
    partial_closed_flag: bool = False       # 50% partial executed (fired)
    partial_close_skipped: bool = False     # v6.10: partial trigger but skipped (lot too small)
    trailing_active: bool = False           # trail SL activated
    final_sl_at_close: float = 0.0          # SL price at exit time

    # Live execution details
    bid_at_entry: float = 0.0
    ask_at_entry: float = 0.0
    spread_pips_actual: float = 0.0         # real spread at fill (pips)
    bid_at_exit: float = 0.0
    ask_at_exit: float = 0.0

    # Market context (HTF analysis at signal time)
    adx_h1: float = 0.0                     # H1 ADX value
    adx_h4: float = 0.0                     # H4 ADX value
    mtf_bias: int = 0                       # H1 market bias 1/-1/0
    d1_bias: int = 0                        # D1 bias 1/-1/0

    # Account state
    balance_at_entry: float = 0.0
    balance_at_close: float = 0.0
    equity_peak_during_trade: float = 0.0

    # === Overtrading detection (v6.9 — added 2026-04-25) ===
    trades_today_at_open: int = 0           # # trades opened TODAY before this one
    trades_last_hour_at_open: int = 0       # # trades opened in trailing 60 min
    secs_since_last_trade_open: float = 0.0  # delta from last trade open (any symbol)
    secs_since_last_trade_same_symbol: float = 0.0  # delta from last trade open (same symbol)

    # === Retrain capability (v6.9 — added 2026-04-25) ===
    # JSON-encoded 27-dim obs vector at decision time → unlock RL retrain from live data
    obs_27_json: str = ""

    # === Chronos forecast features (v7 — added 2026-05-01) ===
    # mirrors obs[27,28] at entry decision time
    chronos_align: float = 0.0
    chronos_unc: float = 0.0

    def to_dict(self) -> Dict:
        """แปลงเป็น Dictionary สำหรับ Excel Logging"""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "type": self.trade_type,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "lot_size": self.lot_size,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "rr_ratio": self.rr_ratio,
            "confluence": self.confluence_score,
            "atr": self.atr_value,
            "open_time": str(self.open_time),
            "close_price": self.close_price,
            "close_time": str(self.close_time) if self.close_time else "",
            "profit": self.profit,
            "close_reason": self.close_reason,
            "reasons": "; ".join(self.signal_reasons),
            # --- ML features v2 ---
            "session": self.session,
            "day_of_week": self.day_of_week,
            "hour_of_day": self.hour_of_day,
            "spread_at_entry": self.spread_at_entry,
            "slippage": self.slippage,
            "htf_bias": self.htf_bias,
            "volatility_regime": self.volatility_regime,
            "consec_loss_before": self.consec_loss_before,
            "dd_at_entry_pct": self.dd_at_entry_pct,
            "mae": self.mae,
            "mfe": self.mfe,
            "time_in_trade": self.time_in_trade,
            "exit_path": self.exit_path,
            # --- v6.9 enhanced (E1+E2 live deploy) ---
            "ml_score": self.ml_score,
            "ml_score_raw": self.ml_score_raw,
            "agent_action_value": self.agent_action_value,
            "agent_decision": self.agent_decision,
            "ml_threshold_used": self.ml_threshold_used,
            "htf_score": self.htf_score,
            "mtf_score": self.mtf_score,
            "ob_pts": self.ob_pts,
            "fvg_pts": self.fvg_pts,
            "sweep_pts": self.sweep_pts,
            "be_moved": self.be_moved,
            "partial_closed_flag": self.partial_closed_flag,
            "partial_close_skipped": self.partial_close_skipped,
            "trailing_active": self.trailing_active,
            "final_sl_at_close": self.final_sl_at_close,
            "bid_at_entry": self.bid_at_entry,
            "ask_at_entry": self.ask_at_entry,
            "spread_pips_actual": self.spread_pips_actual,
            "bid_at_exit": self.bid_at_exit,
            "ask_at_exit": self.ask_at_exit,
            "adx_h1": self.adx_h1,
            "adx_h4": self.adx_h4,
            "mtf_bias": self.mtf_bias,
            "d1_bias": self.d1_bias,
            "balance_at_entry": self.balance_at_entry,
            "balance_at_close": self.balance_at_close,
            "equity_peak_during_trade": self.equity_peak_during_trade,
            # --- Overtrading metrics ---
            "trades_today_at_open": self.trades_today_at_open,
            "trades_last_hour_at_open": self.trades_last_hour_at_open,
            "secs_since_last_trade_open": self.secs_since_last_trade_open,
            "secs_since_last_trade_same_symbol": self.secs_since_last_trade_same_symbol,
            # --- Retrain capability ---
            "obs_27_json": self.obs_27_json,
            # --- v7 Chronos forecast @ entry ---
            "chronos_align": self.chronos_align,
            "chronos_unc": self.chronos_unc,
        }


class TradeExecutor:
    """
    ระบบส่งคำสั่งเทรดหลัก

    ทำหน้าที่:
    - รับสัญญาณจาก SMC Strategy → แปลงเป็นคำสั่ง MT5
    - ตรวจสอบ Risk ซ้ำอีกครั้งก่อนส่ง (Double Safety)
    - คำนวณ Lot Size ที่เหมาะสม
    - ส่ง Market Order พร้อม SL/TP
    - จัดการ Retry เมื่อคำสั่งล้มเหลว
    - เก็บประวัติเทรดที่เปิดได้สำเร็จ

    ⚠️ ทุกคำสั่งต้องผ่าน 3 ด่านตรวจสอบ:
    1. Strategy Confluence >= 60
    2. Risk Manager approve
    3. Position Sizer validate
    """

    # จำนวนครั้ง Retry สูงสุดเมื่อคำสั่งล้มเหลว
    MAX_RETRY = 3
    RETRY_DELAY_SEC = 2

    # Magic Number สำหรับระบุว่าเป็นคำสั่งจาก Bot
    MAGIC_NUMBER = 123456

    # === MT5 DEAL_REASON → human-readable string ===
    # ใช้ตัดสินใจ cooldown policy ใน RiskManager ด้วย (อย่าเปลี่ยน code โดยไม่อัพเดททั้งสองฝั่ง)
    _DEAL_REASON_MAP = {
        0: "Manual Close",
        1: "Manual Close (Mobile)",
        2: "Manual Close (Web)",
        3: "Bot Close",
        4: "🔴 SL Hit",
        5: "🟢 TP Hit",
        6: "🆘 Stop-out",
        7: "Rollover",
        8: "Variation Margin",
        9: "Split",
    }

    # === Correlation Groups ===
    # กลุ่มเดียวกัน = exposure ด้านเดียวกัน (ทั้ง BUY/SELL ต้องดูทิศ USD/JPY ไม่ใช่ order direction)
    # EURUSD BUY = USD อ่อน, USDCHF SELL = USD อ่อน → exposure เดียวกัน
    CORRELATION_GROUPS = {
        "USD_WEAK": {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
        "USD_STRONG": {"USDJPY", "USDCAD", "USDCHF"},
        "JPY_CROSS": {"USDJPY", "EURJPY", "GBPJPY"},
        "EUR_PAIRS": {"EURUSD", "EURJPY"},
        "GBP_PAIRS": {"GBPUSD", "GBPJPY"},
        # Safe-haven: Gold มักเคลื่อนไหวตรงข้ามกับ USD strength
        # BUY XAUUSD = USD weak / risk-off sentiment
        "SAFE_HAVEN": {"XAUUSD"},
    }

    # ทิศทางที่ "positive" สำหรับแต่ละ symbol ในกลุ่ม
    # BUY symbol ที่ base=X → X strength. SELL → X weakness.
    # กลุ่ม USD_WEAK: BUY EURUSD = USD weak, กลุ่ม USD_STRONG: BUY USDJPY = USD strong
    # ถ้า symbol อยู่ในกลุ่ม "ด้านเดียว" → BUY = positive exposure
    # ถ้า symbol อยู่ในกลุ่ม "ด้านตรงข้าม" → SELL = positive exposure
    # ตาราง: symbol → direction ที่ให้ "group-positive exposure"
    _GROUP_POSITIVE_DIR = {
        # USD_WEAK group: BUY = USD weak (positive exposure ต่อ theme "USD อ่อน")
        "USD_WEAK": {"EURUSD": "BUY", "GBPUSD": "BUY", "AUDUSD": "BUY", "NZDUSD": "BUY"},
        # USD_STRONG group: BUY = USD strong
        "USD_STRONG": {"USDJPY": "BUY", "USDCAD": "BUY", "USDCHF": "BUY"},
        # JPY_CROSS group: BUY = JPY weak
        "JPY_CROSS": {"USDJPY": "BUY", "EURJPY": "BUY", "GBPJPY": "BUY"},
        "EUR_PAIRS": {"EURUSD": "BUY", "EURJPY": "BUY"},
        "GBP_PAIRS": {"GBPUSD": "BUY", "GBPJPY": "BUY"},
        # SAFE_HAVEN: BUY XAUUSD = risk-off / USD weak (positive safe-haven exposure)
        "SAFE_HAVEN": {"XAUUSD": "BUY"},
    }

    # จำนวน Position ต่อกลุ่ม correlation ที่ยอมรับได้
    # (ใช้จาก config FTMOConfig.MAX_CORRELATED_POSITIONS ถ้ามี)
    MAX_CORRELATED_POSITIONS: int = getattr(bot_config.ftmo, "MAX_CORRELATED_POSITIONS", 1)

    # === Cross-group USD Theme Guard (v7.1) ===
    # ปัญหา 2026-05-04: USDCHF SELL (USD_STRONG group, effective −1) + NZDUSD BUY (USD_WEAK,
    # effective +1) ผ่านได้เพราะอยู่คนละ group แม้ทั้งคู่คือ "shorting USD"
    # USD_THEME_DIR map: (symbol, direction) → "SHORT" (เก็งกำไร USD อ่อน) หรือ "LONG"
    USD_THEME_DIR = {
        # USD weak — short USD theme (BUY ของ xxxUSD = USD ขายออก)
        ("EURUSD", "BUY"): "SHORT", ("EURUSD", "SELL"): "LONG",
        ("GBPUSD", "BUY"): "SHORT", ("GBPUSD", "SELL"): "LONG",
        ("AUDUSD", "BUY"): "SHORT", ("AUDUSD", "SELL"): "LONG",
        ("NZDUSD", "BUY"): "SHORT", ("NZDUSD", "SELL"): "LONG",
        # USD strong base — SELL ของ USDxxx = USD ขายออก
        ("USDJPY", "BUY"): "LONG",  ("USDJPY", "SELL"): "SHORT",
        ("USDCAD", "BUY"): "LONG",  ("USDCAD", "SELL"): "SHORT",
        ("USDCHF", "BUY"): "LONG",  ("USDCHF", "SELL"): "SHORT",
        # Gold inverted — BUY XAU = USD weak/risk-off
        ("XAUUSD", "BUY"): "SHORT", ("XAUUSD", "SELL"): "LONG",
    }
    MAX_USD_THEME_POSITIONS: int = getattr(bot_config.ftmo, "MAX_USD_THEME_POSITIONS", 2)

    def __init__(
        self,
        connector: MT5Connector,
        risk_manager: RiskManager,
        position_sizer: PositionSizer,
        logger: Optional['TradeLogger'] = None,
        analyzer: Optional['PerformanceAnalyzer'] = None
    ):
        """
        เริ่มต้น Trade Executor

        Args:
            connector: ตัวเชื่อมต่อ MT5
            risk_manager: ระบบจัดการความเสี่ยง
            position_sizer: ระบบคำนวณขนาด Position
        """
        self._connector = connector
        self._risk_manager = risk_manager
        self._position_sizer = position_sizer
        self._logger = logger
        self._analyzer = analyzer

        # เก็บประวัติเทรดที่เปิดอยู่ (key = ticket)
        self._active_trades: Dict[int, ExecutedTrade] = {}

        # เก็บประวัติเทรดที่ปิดแล้ว
        self._closed_trades: List[ExecutedTrade] = []

        # สถิติ
        self._total_executed = 0
        self._total_rejected = 0
        self._notifier = DiscordNotifier()

        # v8.0.55: Spread spike filter — rolling history per symbol
        # บอทเรียน "spread ปกติของ broker นี้" เอง จากการสังเกตเอง
        # ตอน entry ใหม่: เทียบ spread ตอนนี้ vs median 30 ค่าล่าสุด
        # เกิน 2x = SKIP (ตลาดประหม่า, อย่าเข้า)
        self._spread_history: Dict[str, deque] = {}

        print("⚡ [Trade Executor] เริ่มต้นระบบส่งคำสั่งเทรด")

    # =========================================================================
    # ♻️ Orphan Recovery (v8.0.13)
    # =========================================================================

    def _rebuild_executed_trade_from_mt5(self, pos: Dict) -> ExecutedTrade:
        """
        สร้าง ExecutedTrade จาก MT5 position dict — ใช้ตอน orphan recovery
        (bot restart แล้ว position ยังค้างที่ broker)

        ค่าที่ recover ไม่ได้จาก MT5 (ATR, ML score, signal reasons) จะ:
        - ATR: ใช้ค่าปัจจุบัน (acceptable approximation — ใช้ใน trail distance เท่านั้น)
        - ML/agent fields: ใส่ default 0.0 / "" (ไม่ส่งผลต่อ trade management)

        Args:
            pos: dict จาก connector.get_open_positions() — มี ticket, symbol, type,
                 volume, price_open, sl, tp, time, magic, ...

        Returns:
            ExecutedTrade ที่ถือว่า is_open=True, พร้อมให้ TradeManager จัดการต่อ
        """
        # ประมาณ ATR ปัจจุบันสำหรับ trail (ของเดิมไม่มีใน MT5)
        # ATR ใช้แค่กับ trail distance — recover ค่าปัจจุบันก็เพียงพอ
        atr_value = 0.0
        try:
            from strategy.indicators import TechnicalIndicators
            rates = self._connector.get_ohlcv(pos["symbol"], "M15", 100)
            if rates is not None and len(rates) >= 50:
                atr_df = TechnicalIndicators().calculate_atr(rates, period=14)
                atr_value = float(atr_df["atr"].iloc[-1])
        except Exception:
            pass

        # คำนวณ SL distance สำหรับ rr_ratio
        entry = float(pos["price_open"])
        sl = float(pos["sl"]) if pos.get("sl") else 0.0
        tp = float(pos["tp"]) if pos.get("tp") else 0.0
        sl_dist = abs(entry - sl) if sl > 0 else 0.0
        tp_dist = abs(tp - entry) if tp > 0 else 0.0
        rr = (tp_dist / sl_dist) if sl_dist > 0 else 1.0

        return ExecutedTrade(
            ticket=int(pos["ticket"]),
            symbol=str(pos["symbol"]),
            trade_type=str(pos["type"]),
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            lot_size=float(pos["volume"]),
            risk_amount=0.0,            # unknown — would need historical balance
            risk_pct=0.0,
            rr_ratio=rr,
            confluence_score=0.0,
            atr_value=atr_value,
            open_time=pos.get("time", datetime.now(timezone.utc)),
            signal_reasons=["recovered from MT5 (orphan)"],
            magic_number=int(pos.get("magic", 123456)),
            is_open=True,
            original_lot_size=float(pos["volume"]),
            agent_decision="RECOVERED",
        )

    # =========================================================================
    # 🚀 ส่งคำสั่งเทรดจากสัญญาณ
    # =========================================================================

    def execute_signal(
        self,
        signal: TradeSignal,
        live_context: Optional[Dict] = None,
    ) -> Optional[ExecutedTrade]:
        """
        ดำเนินการส่งคำสั่งเทรดจาก TradeSignal

        ขั้นตอน:
        1. ตรวจสอบสัญญาณ Valid
        2. คำนวณ Lot Size
        3. ตรวจสอบ Risk ซ้ำ (Double-check)
        4. ตรวจสอบ Spread ปัจจุบัน
        5. ส่ง Market Order
        6. บันทึกผลลัพธ์

        Args:
            signal: สัญญาณเทรดจาก SMC Strategy

        Returns:
            ExecutedTrade หรือ None: ข้อมูลเทรดที่เปิดสำเร็จ
        """
        # v6.10: reset reject reason for every call → main.py reads after None return
        self._last_reject_reason: Optional[str] = None

        # === ด่านที่ 1: ตรวจสอบสัญญาณ ===
        if not signal.is_valid:
            print("❌ [Executor] สัญญาณไม่ Valid — ยกเลิก")
            self._last_reject_reason = "signal_invalid"
            return None

        symbol = signal.symbol
        print(f"\n{'━' * 60}")
        print(f"⚡ [Executor] ดำเนินการ {signal.signal_type.value} {symbol}")
        print(f"   Confluence: {signal.confluence_score:.0f}/100 | RR: 1:{signal.rr_ratio:.1f}")
        print(f"{'━' * 60}")

        # === ด่านที่ 1.5: ตรวจ Duplicate + Correlation ===
        # กัน Over-exposure ต่อ Symbol เดียวกัน หรือกลุ่มที่เคลื่อนไหวไปทางเดียวกัน
        allowed_corr, corr_reason = self._check_correlation_risk(
            symbol, signal.signal_type.value
        )
        if not allowed_corr:
            print(f"🚫 [Executor] Correlation Risk: {corr_reason}")
            self._last_reject_reason = f"correlation:{corr_reason}"
            self._total_rejected += 1
            return None

        # === ด่านที่ 2: คำนวณ Lot Size ===
        lot_result = self._position_sizer.calculate_lot_size(
            symbol=symbol,
            sl_distance_price=signal.sl_distance,
        )

        if lot_result is None:
            print("❌ [Executor] คำนวณ Lot Size ล้มเหลว — ยกเลิก")
            self._last_reject_reason = "lot_calc_failed"
            self._total_rejected += 1
            return None

        lot_size = lot_result["lot_size"]
        risk_amount = lot_result["risk_amount"]
        risk_pct = lot_result["risk_pct"]

        # === ด่านที่ 3: ตรวจสอบ Risk ซ้ำ (Double Safety) ===
        allowed, reason = self._risk_manager.can_open_trade(
            symbol=symbol,
            risk_amount=risk_amount,
            sl_distance_pips=lot_result["sl_pips"],
            rr_ratio=signal.rr_ratio,
            direction=signal.signal_type.value,
            atr=signal.atr_value,
            confluence_score=signal.confluence_score,  # v8.0.44: Asian conf exception
        )

        if not allowed:
            print(f"🚫 [Executor] Risk Manager ปฏิเสธ: {reason}")
            self._last_reject_reason = f"risk_manager:{reason}"
            self._total_rejected += 1
            return None

        # === ด่านที่ 4: ตรวจสอบ Spread ===
        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            print("❌ [Executor] ดึงราคาปัจจุบันล้มเหลว — ยกเลิก")
            self._last_reject_reason = "price_fetch_failed"
            self._total_rejected += 1
            return None

        symbol_info = self._connector.get_symbol_info(symbol)
        max_spread = bot_config.symbols.max_spread_points.get(symbol, 20)

        current_spread_pts = 0.0
        if symbol_info:
            current_spread_pts = price_info["spread"] / symbol_info["point"]
            if current_spread_pts > max_spread:
                print(f"🚫 [Executor] Spread สูงเกินไป ({current_spread_pts:.0f} > {max_spread}) — รอ Spread ลด")
                self._last_reject_reason = f"spread_high:{current_spread_pts:.0f}>{max_spread}"
                self._total_rejected += 1
                # บันทึก observation ก่อน return (สำหรับ baseline)
                self._record_spread_observation(symbol, current_spread_pts)
                return None

        # === ด่านที่ 4.5: Spread Spike (v8.0.55 — dynamic, broker-agnostic) ===
        # เทียบ spread ปัจจุบัน vs median rolling 30 ค่าล่าสุด → กรองตลาดประหม่า
        if current_spread_pts > 0:
            spike_ok, spike_reason = self._check_spread_spike(symbol, current_spread_pts)
            if not spike_ok:
                print(f"🚫 [Executor] Spread Spike: {spike_reason}")
                self._last_reject_reason = f"spread_spike:{spike_reason}"
                self._total_rejected += 1
                # ยังบันทึก observation เพื่อ track baseline ต่อ
                self._record_spread_observation(symbol, current_spread_pts)
                return None
            # บันทึก observation ปัจจุบันก่อน gate ต่อ (เผื่อ entry confirm reject)
            self._record_spread_observation(symbol, current_spread_pts)

        # === ด่านที่ 4.6: Entry Confirmation (v8.0.55 — pre-execution chart re-check) ===
        # เช็ค 3 อย่างก่อนยิง: slip / M1 direction / BB %B still extreme
        # ปัญหาที่แก้: signal stale ระหว่าง scan → execute (~30-60 วินาที)
        ec_ok, ec_reason = self._check_entry_confirmation(signal)
        if not ec_ok:
            print(f"🚫 [Executor] Entry Confirmation Failed: {ec_reason}")
            self._last_reject_reason = f"entry_confirm:{ec_reason}"
            self._total_rejected += 1
            return None

        # === ด่านที่ 5: Final Validation ===
        remaining_budget = self._risk_manager.get_remaining_daily_budget()
        valid, val_reason = self._position_sizer.validate_trade_risk(
            symbol=symbol,
            lot_size=lot_size,
            sl_distance_price=signal.sl_distance,
            remaining_daily_budget=remaining_budget,
        )

        if not valid:
            print(f"🚫 [Executor] Final Validation ล้มเหลว: {val_reason}")
            self._last_reject_reason = f"final_validation:{val_reason}"
            self._total_rejected += 1
            return None

        # === ด่านที่ 6: ส่ง Market Order (พร้อม Retry) ===
        order_result = self._send_order_with_retry(
            symbol=symbol,
            order_type=signal.signal_type.value,
            volume=lot_size,
            sl=signal.sl_price,
            tp=signal.tp_price,
        )

        if order_result is None:
            print("❌ [Executor] ส่งคำสั่งล้มเหลวหลังจาก Retry ทุกครั้ง")
            self._last_reject_reason = "order_send_failed"
            self._total_rejected += 1
            return None

        # === Entry price fallback (3-tier + retry) ===
        # บาง broker/filling-mode คืน result.price = 0 ทำให้ Discord/Log โชว์ 0
        # → Tier 1: order_result.price | Tier 2: MT5 position (retry 3x, 150ms) | Tier 3: market price
        resolved_entry = order_result.get("price") or 0
        fallback_tier = 1
        if not resolved_entry:
            ticket_no = order_result.get("ticket")
            # Tier 2: position อาจยังไม่ sync ทันที — retry สูงสุด 3 ครั้ง × 150ms
            for attempt in range(3):
                for pos in self._connector.get_open_positions():
                    if pos.get("ticket") == ticket_no:
                        resolved_entry = pos.get("price_open") or pos.get("price") or 0
                        break
                if resolved_entry:
                    fallback_tier = 2
                    break
                time_module.sleep(0.15)
            # Tier 3: ตกมาที่ market price (ยอมรับ slippage เล็กน้อย)
            if not resolved_entry:
                px = self._connector.get_current_price(symbol)
                if px:
                    resolved_entry = px["ask"] if signal.signal_type.value == "BUY" else px["bid"]
                    fallback_tier = 3
            print(f"⚠️ [Executor] MT5 คืน price=0 → ใช้ fallback tier-{fallback_tier} entry={resolved_entry}")

            # Log slippage ให้ชัดเจน (tier-3 = เราไม่รู้ราคา fill จริง → ถือเป็น untracked slippage)
            if fallback_tier == 3:
                print(f"🚨 [Executor] TIER-3 FALLBACK: ราคา entry ไม่ confirmed จาก MT5 → "
                      f"actual slippage ไม่สามารถวัดได้. ใช้ market price เป็น proxy เท่านั้น")
            else:
                expected_entry = order_result.get("requested_price") or 0
                if expected_entry and resolved_entry:
                    slip = abs(resolved_entry - expected_entry)
                    print(f"📏 [Executor] Tier-{fallback_tier} slippage: {slip:.5f}")

        # === Capture ML features ณ เวลาเปิดเทรด ===
        # v6.10b: ใช้ broker time (EEST) — match กับ MT5 close time + Friday/Daily close logic
        # strip tzinfo เพื่อ openpyxl เขียน Excel ได้สะอาด (และเทียบกับ open_time/close_time ได้)
        now = TimeManager.get_server_time().replace(tzinfo=None)
        entry_ctx = self._capture_entry_context(signal, resolved_entry, symbol_info, price_info)

        # === ด่านที่ 7: บันทึกผลลัพธ์ ===
        executed = ExecutedTrade(
            ticket=order_result["ticket"],
            symbol=symbol,
            trade_type=signal.signal_type.value,
            entry_price=resolved_entry,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            lot_size=lot_size,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            rr_ratio=signal.rr_ratio,
            confluence_score=signal.confluence_score,
            atr_value=signal.atr_value,
            open_time=now,
            signal_reasons=signal.reasons,
            magic_number=self.MAGIC_NUMBER,
            # --- ML features v2 ---
            session=entry_ctx["session"],
            day_of_week=now.weekday(),
            hour_of_day=now.hour,
            spread_at_entry=entry_ctx["spread"],
            slippage=entry_ctx["slippage"],
            htf_bias=entry_ctx["htf_bias"],
            volatility_regime=entry_ctx["volatility_regime"],
            consec_loss_before=getattr(self._risk_manager, "_consecutive_losses", 0),
            dd_at_entry_pct=entry_ctx["dd_pct"],
            # --- Partial close tracking ---
            original_lot_size=lot_size,
        )

        # === v6.9: Apply live_context (ML score, agent decision, market context) ===
        # ส่งจาก main.py.run() — ข้อมูลที่ executor ไม่รู้เอง
        if live_context:
            executed.ml_score = float(live_context.get("ml_score", 0.5))
            executed.ml_score_raw = float(live_context.get("ml_score_raw", 0.5))
            executed.agent_action_value = float(live_context.get("agent_action_value", 0.0))
            executed.agent_decision = str(live_context.get("agent_decision", ""))
            executed.ml_threshold_used = float(live_context.get("ml_threshold_used", 0.0))
            # Confluence breakdown (ถ้ามี)
            executed.htf_score = int(live_context.get("htf_score", 0))
            executed.mtf_score = int(live_context.get("mtf_score", 0))
            executed.ob_pts = int(live_context.get("ob_pts", 0))
            executed.fvg_pts = int(live_context.get("fvg_pts", 0))
            executed.sweep_pts = int(live_context.get("sweep_pts", 0))
            # Live execution snapshot (bid/ask)
            executed.bid_at_entry = float(live_context.get("bid_at_entry", 0.0))
            executed.ask_at_entry = float(live_context.get("ask_at_entry", 0.0))
            executed.spread_pips_actual = float(live_context.get("spread_pips_actual", 0.0))
            # Market context (ADX, biases)
            executed.adx_h1 = float(live_context.get("adx_h1", 0.0))
            executed.adx_h4 = float(live_context.get("adx_h4", 0.0))
            executed.mtf_bias = int(live_context.get("mtf_bias", 0))
            executed.d1_bias = int(live_context.get("d1_bias", 0))
            # Account state
            executed.balance_at_entry = float(live_context.get("balance_at_entry", 0.0))
            # Overtrading metrics
            executed.trades_today_at_open = int(live_context.get("trades_today_at_open", 0))
            executed.trades_last_hour_at_open = int(live_context.get("trades_last_hour_at_open", 0))
            executed.secs_since_last_trade_open = float(live_context.get("secs_since_last_trade_open", 0.0))
            executed.secs_since_last_trade_same_symbol = float(
                live_context.get("secs_since_last_trade_same_symbol", 0.0)
            )
            # Obs vector (JSON) — for offline retrain reconstruction
            executed.obs_27_json = str(live_context.get("obs_27_json", ""))
            # v7: Chronos forecast features @ entry (mirrors obs[27,28])
            executed.chronos_align = float(live_context.get("chronos_align", 0.0) or 0.0)
            executed.chronos_unc = float(live_context.get("chronos_unc", 0.0) or 0.0)

        # เก็บใน Active Trades
        self._active_trades[executed.ticket] = executed
        self._total_executed += 1

        if self._logger:
            self._logger.log_trade_opened(executed.to_dict())
            
        self._notifier.send_trade_open(executed.to_dict())

        # อัพเดท Risk Manager
        print(f"\n✅ [Executor] เปิดเทรดสำเร็จ!")
        print(f"   🎫 Ticket:   {executed.ticket}")
        print(f"   💱 Symbol:   {executed.symbol} {executed.trade_type}")
        print(f"   📍 Entry:    {executed.entry_price}")
        print(f"   🔴 SL:       {executed.sl_price}")
        print(f"   🟢 TP:       {executed.tp_price}")
        print(f"   📦 Lot:      {executed.lot_size}")
        print(f"   🎯 Risk:     {executed.risk_pct:.2%} (${executed.risk_amount:,.2f})")
        print(f"   ⚖️ RR:       1:{executed.rr_ratio:.1f}")
        print(f"   🏆 Confluence: {executed.confluence_score:.0f}")

        return executed

    # =========================================================================
    # 🧠 ML Features — Capture entry context
    # =========================================================================

    def _capture_entry_context(self, signal, resolved_entry, symbol_info, price_info) -> Dict:
        """
        Capture ML features ณ เวลาเปิดเทรด (Schema v2)
        - session (Asian/London/NY/Overlap)
        - spread (points)
        - slippage (price diff vs signal.entry_price ถ้ามี)
        - htf_bias, volatility_regime (ถ้า signal มี)
        - dd_pct (daily DD % ณ เวลาเปิด)
        """
        # Session จากชั่วโมง UTC
        hour = datetime.now(timezone.utc).hour
        if 7 <= hour < 12:
            session = "LONDON"
        elif 12 <= hour < 17:
            session = "LONDON_NY_OVERLAP" if hour < 13 else "NEW_YORK"
        elif 0 <= hour < 7:
            session = "ASIAN"
        else:
            session = "OFF_HOURS"

        # Spread ณ เวลาเปิด (points)
        spread_pts = 0.0
        if symbol_info and price_info:
            point = symbol_info.get("point") or 0.0001
            if point > 0:
                spread_pts = float(price_info.get("spread", 0)) / point

        # Slippage = |fill - signal.entry_price|
        slippage = 0.0
        sig_entry = getattr(signal, "entry_price", 0) or 0
        if sig_entry and resolved_entry:
            slippage = abs(resolved_entry - sig_entry)

        # DD% ณ เวลาเปิด (จาก risk_manager)
        dd_pct = 0.0
        try:
            status = self._risk_manager.get_risk_status() if hasattr(self._risk_manager, "get_risk_status") else {}
            dd_pct = float(status.get("daily_loss_pct", 0) or 0)
        except Exception:
            dd_pct = 0.0

        # HTF bias / volatility regime — อ่านจาก signal ถ้ามี
        htf_bias = getattr(signal, "htf_bias", "") or ""
        vol_regime = getattr(signal, "volatility_regime", "") or ""
        # Fallback จาก ATR (ถ้ายังว่าง)
        if not vol_regime and getattr(signal, "atr_value", 0):
            atr = signal.atr_value
            if atr > 0.0020:
                vol_regime = "high"
            elif atr > 0.0010:
                vol_regime = "normal"
            else:
                vol_regime = "quiet"

        return {
            "session": session,
            "spread": round(spread_pts, 1),
            "slippage": round(slippage, 6),
            "htf_bias": str(htf_bias),
            "volatility_regime": vol_regime,
            "dd_pct": round(dd_pct * 100, 2) if dd_pct <= 1 else round(dd_pct, 2),
        }

    # =========================================================================
    # 🔗 ตรวจสอบ Correlation Risk (กัน Over-exposure)
    # =========================================================================

    def _check_correlation_risk(self, symbol: str, trade_type: str) -> tuple:
        """
        ตรวจสอบว่าการเปิดเทรดใหม่จะทำให้ exposure สูงเกินไปหรือไม่

        กฎ:
        1. ห้ามเปิด Position ใหม่ในคู่เงินเดียวกัน (Duplicate Symbol)
        2. ห้ามมี Position เกิน MAX_CORRELATED_POSITIONS ในกลุ่มเดียวกัน ทิศทางเดียวกัน

        Args:
            symbol: คู่เงินที่จะเปิด
            trade_type: "BUY" หรือ "SELL"

        Returns:
            Tuple[bool, str]: (ผ่าน/ไม่ผ่าน, เหตุผล)
        """
        # === 1. Duplicate Symbol Check ===
        for trade in self._active_trades.values():
            if trade.symbol == symbol and trade.is_open:
                return (False, f"มีเทรด {symbol} ({trade.trade_type}) เปิดอยู่แล้ว (Ticket {trade.ticket})")

        # === 2. Correlation Group Check ===
        # นับเทรด active ในแต่ละกลุ่ม correlation (แยกตาม direction)
        symbol_upper = symbol.upper()
        tt_upper = trade_type.upper()

        for group_name, group_symbols in self.CORRELATION_GROUPS.items():
            if symbol_upper not in group_symbols:
                continue

            same_direction_count = 0
            group_dir_map = self._GROUP_POSITIVE_DIR.get(group_name, {})
            new_pos_dir = group_dir_map.get(symbol_upper)
            new_effective = 1 if tt_upper == new_pos_dir else -1

            for trade in self._active_trades.values():
                if not trade.is_open:
                    continue
                existing_sym = trade.symbol.upper()
                if existing_sym not in group_symbols:
                    continue
                existing_pos_dir = group_dir_map.get(existing_sym)
                existing_effective = 1 if trade.trade_type.upper() == existing_pos_dir else -1
                if existing_effective == new_effective:
                    same_direction_count += 1

            if same_direction_count >= self.MAX_CORRELATED_POSITIONS:
                return (
                    False,
                    f"กลุ่ม {group_name} มี effective exposure เดียวกันเปิดแล้ว {same_direction_count} ตัว "
                    f"(จำกัด {self.MAX_CORRELATED_POSITIONS})"
                )

        # === 3. USD Theme Cross-group Check (v7.1) ===
        # กัน scenario "ทุก position ชอร์ต USD พร้อมกัน" ที่ legacy correlation guard มองไม่เห็น
        # เพราะ USD_STRONG/USD_WEAK เป็น 2 group แยก
        new_theme = self.USD_THEME_DIR.get((symbol_upper, tt_upper))
        if new_theme is not None:
            same_theme_count = 0
            for trade in self._active_trades.values():
                if not trade.is_open:
                    continue
                ex_theme = self.USD_THEME_DIR.get(
                    (trade.symbol.upper(), trade.trade_type.upper())
                )
                if ex_theme == new_theme:
                    same_theme_count += 1
            if same_theme_count >= self.MAX_USD_THEME_POSITIONS:
                return (
                    False,
                    f"USD theme {new_theme} เปิดแล้ว {same_theme_count} ตำแหน่ง "
                    f"(จำกัด {self.MAX_USD_THEME_POSITIONS}) — กัน double-bet on USD direction"
                )

        return (True, "ผ่านการตรวจ Correlation Risk")

    # =========================================================================
    # 🕯️ Entry Confirmation + Spread Spike (v8.0.55 — pre-execution gates)
    # =========================================================================

    def _record_spread_observation(self, symbol: str, spread_points: float) -> None:
        """เก็บ spread observation ปัจจุบันใน rolling history per symbol.

        ใช้ใน `_check_spread_spike` เพื่อเทียบกับ median ภายหลัง.
        Auto-learns baseline ของ broker — ไม่ต้อง config ต่อ broker.
        """
        if spread_points <= 0:
            return
        cfg = bot_config.ftmo
        lookback = int(getattr(cfg, "SPREAD_SPIKE_LOOKBACK_BARS", 30))
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=lookback)
        dq = self._spread_history[symbol]
        # Re-size ถ้า config เปลี่ยน (rare)
        if dq.maxlen != lookback:
            new_dq = deque(dq, maxlen=lookback)
            self._spread_history[symbol] = new_dq
            dq = new_dq
        dq.append(float(spread_points))

    def _check_spread_spike(
        self, symbol: str, current_spread_pts: float
    ) -> Tuple[bool, str]:
        """v8.0.55 — Spread Spike Filter (broker-agnostic, live-only).

        เทียบ spread ปัจจุบัน vs median ล่าสุด.
        เกิน 2x = ตลาดประหม่า/news/illiquidity → SKIP

        ตรรกะ: บอทเรียน "ค่าปกติของ broker นี้" จากการสังเกตเอง.
        ไม่ต้องตั้งค่า spread แต่ละ broker ล่วงหน้า.

        Returns:
            (passed, reason)
        """
        cfg = bot_config.ftmo
        if not getattr(cfg, "SPREAD_SPIKE_ENABLED", False):
            return (True, "disabled")
        if current_spread_pts <= 0:
            return (True, "no_spread_data")

        hist = self._spread_history.get(symbol)
        min_samples = int(getattr(cfg, "SPREAD_SPIKE_MIN_SAMPLES", 10))
        if not hist or len(hist) < min_samples:
            # Warmup — ยังไม่มี baseline พอ ใช้ fixed threshold (max_spread_points) เป็น fallback
            return (True, f"warmup ({len(hist) if hist else 0}/{min_samples})")

        median = float(np.median(list(hist)))
        if median <= 0:
            return (True, "no_baseline")

        ratio = current_spread_pts / median
        limit = float(getattr(cfg, "SPREAD_SPIKE_RATIO_LIMIT", 2.0))
        if ratio > limit:
            return (
                False,
                f"spread spike: {current_spread_pts:.1f} vs median {median:.1f} "
                f"= {ratio:.2f}x (limit {limit:.1f}x)"
            )
        return (True, f"spread ok ({ratio:.2f}x median)")

    def _check_entry_confirmation(self, signal: 'TradeSignal') -> Tuple[bool, str]:
        """v8.0.55 — Entry Confirmation Gate (pre-execution chart re-check).

        เช็ค 3 อย่างก่อนยิง:
        1. Slip: ราคาตอนนี้เคลื่อนสวนเรา > MAX_SLIP_R × SL distance → SKIP
        2. M1 last closed candle direction ต้องตรงทิศ signal
        3. BB %B ยังอยู่ใน extreme zone (ไม่หลุดเขตไปแล้ว)

        ปัญหาที่แก้: signal generated ตอน scan แต่ execute ห่างไป 30-60 วินาที.
        ภายในช่วงนั้นราคาอาจเคลื่อนผ่านจุด rejection ไปแล้ว.
        Mirrored ใน MeanReversionBacktester (first future bar direction match).

        Returns:
            (passed, reason)
        """
        cfg = bot_config.ftmo
        if not getattr(cfg, "ENTRY_CONFIRM_ENABLED", False):
            return (True, "disabled")

        symbol = signal.symbol
        direction = signal.signal_type.value.upper()  # "BUY" / "SELL"

        # ─── เช็ค 1: Slip จาก signal entry → current price ─────────────
        price_now = self._connector.get_current_price(symbol)
        if price_now is None:
            return (True, "no_current_price")

        sl_distance = abs(float(signal.entry_price) - float(signal.sl_price))
        if sl_distance <= 0:
            return (True, "no_sl_distance")

        if direction == "BUY":
            current_exec_price = float(price_now.get("ask", signal.entry_price))
            # adverse = price moved DOWN (away from BUY) ก่อนยิง
            adverse_move = max(0.0, float(signal.entry_price) - current_exec_price)
        else:
            current_exec_price = float(price_now.get("bid", signal.entry_price))
            # adverse = price moved UP (away from SELL) ก่อนยิง
            adverse_move = max(0.0, current_exec_price - float(signal.entry_price))

        slip_r = adverse_move / sl_distance
        max_slip_r = float(getattr(cfg, "ENTRY_CONFIRM_MAX_SLIP_R", 0.30))
        if slip_r > max_slip_r:
            return (
                False,
                f"slip {slip_r:.2f}R > {max_slip_r:.2f}R "
                f"(entry {signal.entry_price:.5f} → exec {current_exec_price:.5f}, "
                f"SL dist {sl_distance:.5f})"
            )

        # ─── เช็ค 2: M1 last closed candle — wick-aware (v8.0.77 RF-1 fix) ──
        # MR = "รับมีดที่กำลังตก": แท่ง M1 สุดท้ายที่ก้นจริงมักยังแดง (เพิ่งสวนลงก่อนเด้ง)
        # เดิมบังคับ body ตรงทิศ → ตัดไม้กลับตัว edge สูงทิ้ง (82/142 entry-confirm rejects).
        # ใหม่: ผ่านถ้า body ตรงทิศ "หรือ" มี rejection wick ฝั่งที่เด้งกลับ ≥ REJECT_WICK_MIN.
        # ตัดทิ้งเฉพาะแท่งวิ่งสวนแรง + ไม่มีหางเด้ง (clean opposite body) = โมเมนตัมสวนยังแรงจริง
        if getattr(cfg, "ENTRY_CONFIRM_M1_DIRECTION_MATCH", True):
            try:
                m1_df = self._connector.get_ohlcv(symbol, "M1", count=100)
                if m1_df is not None and len(m1_df) >= 2:
                    # iloc[-1] = แท่งกำลังก่อตัว (อาจ partial), iloc[-2] = แท่งล่าสุดที่ปิด
                    last_closed = m1_df.iloc[-2]
                    o = float(last_closed["open"])
                    c = float(last_closed["close"])
                    h = float(last_closed["high"])
                    lo = float(last_closed["low"])
                    bar_body = c - o
                    rng = h - lo
                    # อนุญาต doji (body ≈ 0) ผ่าน — ไม่ใช่ contra signal
                    body_threshold = max(abs(bar_body), rng) * 0.05
                    wick_min = float(getattr(cfg, "ENTRY_CONFIRM_M1_REJECT_WICK_MIN", 0.40))
                    if rng > 0 and abs(bar_body) > body_threshold:
                        # หางล่าง = ราคาสวนลงแล้วถูกดันกลับขึ้น (rejection ที่ดีต่อ BUY)
                        lower_wick_ratio = (min(o, c) - lo) / rng
                        # หางบน = ราคาสวนขึ้นแล้วถูกดันกลับลง (rejection ที่ดีต่อ SELL)
                        upper_wick_ratio = (h - max(o, c)) / rng
                        # BUY: ตัดเฉพาะแท่งแดงชัด + หางล่างเด้งน้อย (วิ่งลงแรง ไม่มีสัญญาณกลับ)
                        if (direction == "BUY" and bar_body < 0
                                and lower_wick_ratio < wick_min):
                            return (
                                False,
                                f"M1 last bar strong bearish, no reject-wick "
                                f"(O={o:.5f} C={c:.5f}, lower_wick {lower_wick_ratio:.2f} "
                                f"< {wick_min:.2f}) — BUY momentum still falling"
                            )
                        # SELL: ตัดเฉพาะแท่งเขียวชัด + หางบนเด้งน้อย
                        if (direction == "SELL" and bar_body > 0
                                and upper_wick_ratio < wick_min):
                            return (
                                False,
                                f"M1 last bar strong bullish, no reject-wick "
                                f"(O={o:.5f} C={c:.5f}, upper_wick {upper_wick_ratio:.2f} "
                                f"< {wick_min:.2f}) — SELL momentum still rising"
                            )
            except Exception:
                # fail-open: ดึง M1 ไม่ได้ก็ผ่าน (network glitch ฯลฯ)
                pass

        # ─── เช็ค 3-5: Keltner / ATR Band filters (v8.0.74 — anti-whipsaw) ─
        # BB %B recheck ลบแล้ว v8.0.74 — KC distance ทำหน้าที่เดียวกันแต่ดีกว่า
        # (ปรับตามความผันผวนจริง ไม่โดน outlier บิดเหมือน BB std dev)
        # 4. ราคาต้องอยู่นอก Keltner Band (overextended) ถึงจะเข้า MR
        # v8.0.76 — Dynamic threshold: ATR regime สูง = require deeper extreme,
        #          ATR regime ต่ำ = allow shallower extreme (catch sideway swings)
        if getattr(cfg, "KC_ENTRY_FILTER_ENABLED", False):
            kc_dist = getattr(signal, "kc_distance_norm", None)
            if kc_dist is not None:
                base_thresh = float(getattr(cfg, "KC_DIST_THRESHOLD_BASE", 0.60))
                if getattr(cfg, "KC_DIST_ATR_ADAPTIVE", False):
                    atr_z = float(getattr(signal, "atr_zscore_30bars", 0.0))
                    gain = float(getattr(cfg, "KC_DIST_ATR_GAIN", 0.20))
                    scale = float(np.clip(
                        1.0 + atr_z * gain,
                        float(getattr(cfg, "KC_DIST_ATR_SCALE_MIN", 0.6)),
                        float(getattr(cfg, "KC_DIST_ATR_SCALE_MAX", 1.5)),
                    ))
                    dist_thresh = base_thresh * scale
                    regime_tag = f" (base {base_thresh:.2f} × {scale:.2f}, atr_z={atr_z:+.2f})"
                else:
                    dist_thresh = base_thresh
                    regime_tag = ""

                kc_dist_f = float(kc_dist)
                if direction == "BUY" and kc_dist_f > -dist_thresh:
                    return (
                        False,
                        f"KC filter: dist {kc_dist_f:.2f} > -{dist_thresh:.2f}{regime_tag} "
                        f"— price not overextended down"
                    )
                if direction == "SELL" and kc_dist_f < dist_thresh:
                    return (
                        False,
                        f"KC filter: dist {kc_dist_f:.2f} < {dist_thresh:.2f}{regime_tag} "
                        f"— price not overextended up"
                    )

        # 5. EMA slope ต้องไม่ชันเกิน (trending = ไม่ใช่ MR environment)
        # v8.0.75 — Dynamic cap: scale baseline by ATR z-score regime
        #   volatile (atr_z > 0)  → ขยายเพดาน (อนุญาตชันได้มากขึ้น, ลด missed trades ช่วงข่าว)
        #   quiet    (atr_z < 0)  → หดเพดาน (slope เล็กๆ ก็ trend จริง, ลด whipsaw ช่วง Asian)
        if getattr(cfg, "KC_SLOPE_FILTER_ENABLED", False):
            slope = getattr(signal, "ema_slope_norm", None)
            base_thresh = float(getattr(cfg, "KC_SLOPE_THRESHOLD", 0.15))
            if slope is not None:
                slope_val = float(slope)
                if getattr(cfg, "KC_SLOPE_ATR_ADAPTIVE", False):
                    atr_z = float(getattr(signal, "atr_zscore_30bars", 0.0))
                    gain = float(getattr(cfg, "KC_SLOPE_ATR_GAIN", 0.15))
                    scale = float(np.clip(
                        1.0 + atr_z * gain,
                        float(getattr(cfg, "KC_SLOPE_ATR_SCALE_MIN", 0.7)),
                        float(getattr(cfg, "KC_SLOPE_ATR_SCALE_MAX", 1.6)),
                    ))
                    slope_thresh = base_thresh * scale
                    regime_tag = f" (base {base_thresh:.2f} × {scale:.2f}, atr_z={atr_z:+.2f})"
                else:
                    slope_thresh = base_thresh
                    regime_tag = ""

                if direction == "BUY" and slope_val < -slope_thresh:
                    return (
                        False,
                        f"KC slope: {slope_val:.2f} < -{slope_thresh:.2f}{regime_tag} "
                        f"— EMA falling too fast for BUY"
                    )
                if direction == "SELL" and slope_val > slope_thresh:
                    return (
                        False,
                        f"KC slope: {slope_val:.2f} > {slope_thresh:.2f}{regime_tag} "
                        f"— EMA rising too fast for SELL"
                    )

        # 6. ราคาอยู่นอกกรอบต่อเนื่อง >= N แท่ง = เทรนด์จริง ไม่ใช่ spike
        if getattr(cfg, "KC_CONSEC_OUTSIDE_FILTER_ENABLED", False):
            consec = getattr(signal, "consecutive_outside", None)
            max_consec = int(getattr(cfg, "KC_CONSEC_OUTSIDE_MAX", 3))
            if consec is not None and int(consec) >= max_consec:
                return (
                    False,
                    f"KC consec: {consec} bars outside >= {max_consec} — sustained trend, not spike"
                )

        return (True, "ok")

    # =========================================================================
    # 🔄 ส่งคำสั่งพร้อม Retry
    # =========================================================================

    def _send_order_with_retry(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float,
        tp: float,
    ) -> Optional[Dict]:
        """
        ส่ง Market Order พร้อม Retry Logic

        ถ้าล้มเหลวจะลองใหม่สูงสุด MAX_RETRY ครั้ง
        แต่ละครั้งจะรอ RETRY_DELAY_SEC วินาที

        Args:
            symbol: คู่เงิน
            order_type: "BUY" หรือ "SELL"
            volume: ขนาด Lot
            sl: ราคา Stop Loss
            tp: ราคา Take Profit

        Returns:
            Dict หรือ None: ผลลัพธ์การส่งคำสั่ง
        """
        for attempt in range(1, self.MAX_RETRY + 1):
            print(f"📤 [Executor] ส่งคำสั่ง {order_type} {symbol} (ครั้งที่ {attempt}/{self.MAX_RETRY})")

            result = self._connector.send_market_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                sl=sl,
                tp=tp,
                comment="",  # v8.0.23: empty — bot identified by magic number
                magic=self.MAGIC_NUMBER,
            )

            if result is not None:
                print(f"✅ [Executor] คำสั่งสำเร็จในครั้งที่ {attempt}")
                return result

            if attempt < self.MAX_RETRY:
                print(f"⚠️ [Executor] ล้มเหลว — รอ {self.RETRY_DELAY_SEC}s ก่อนลองใหม่...")
                time_module.sleep(self.RETRY_DELAY_SEC)

        return None

    # =========================================================================
    # 📋 จัดการเทรดที่เปิดอยู่
    # =========================================================================

    def close_trade(
        self,
        ticket: int,
        reason: str = "Manual Close"
    ) -> bool:
        """
        ปิดเทรดที่ระบุด้วย Ticket Number

        Args:
            ticket: หมายเลข Ticket
            reason: เหตุผลที่ปิด

        Returns:
            bool: True ถ้าปิดสำเร็จ
        """
        if ticket not in self._active_trades:
            print(f"⚠️ [Executor] ไม่พบ Ticket {ticket} ใน Active Trades")
            # ลองปิดจาก MT5 โดยตรง
            return self._connector.close_position(ticket)

        trade = self._active_trades[ticket]

        if self._connector.close_position(ticket):
            # อัพเดทข้อมูลเทรด
            price_info = self._connector.get_current_price(trade.symbol)
            if price_info:
                if trade.trade_type == "BUY":
                    trade.close_price = price_info["bid"]
                else:
                    trade.close_price = price_info["ask"]
                # v6.9: capture bid/ask snapshot at exit for analysis
                trade.bid_at_exit = float(price_info.get("bid", 0.0))
                trade.ask_at_exit = float(price_info.get("ask", 0.0))

            trade.is_open = False
            # v6.10b: ใช้ broker time (EEST) — same as open_time
            trade.close_time = TimeManager.get_server_time().replace(tzinfo=None)
            trade.close_reason = reason
            # v8.0.78: คำนวณ time_in_trade ที่นี่ด้วย (เดิมมีแค่ใน record_external_close
            # → ไม้ที่ปิดผ่าน close_trade เช่น Pre-news/Friday close โชว์ Time-in-Trade=0)
            if trade.open_time:
                trade.time_in_trade = int((trade.close_time - trade.open_time).total_seconds())
            if not trade.exit_path:
                trade.exit_path = reason
            # v6.9: capture balance + final SL at close
            try:
                rs = self._risk_manager.get_risk_status()
                trade.balance_at_close = float(rs.get("current_balance", 0.0))
                trade.equity_peak_during_trade = float(rs.get("current_equity", 0.0))
            except Exception:
                pass
            if not trade.final_sl_at_close:
                trade.final_sl_at_close = float(trade.sl_price)

            # คำนวณ P/L (ประมาณ) — ใช้ contract_size จริงของโบรกเกอร์ (ไม่ hardcode 100000)
            # และแปลงเป็นสกุลเงินของบัญชี (USD) สำหรับคู่ที่ Quote != USD
            sym_info = self._connector.get_symbol_info(trade.symbol)
            contract_size = sym_info.get("trade_contract_size", 100000) if sym_info else 100000

            if trade.trade_type == "BUY":
                price_diff = trade.close_price - trade.entry_price
            else:
                price_diff = trade.entry_price - trade.close_price

            # สำหรับคู่ xxxUSD (EURUSD, GBPUSD): profit = diff × lot × contract_size
            # สำหรับคู่ USDxxx หรือ cross (USDJPY, EURJPY): ต้องหารด้วยราคาปิดเพื่อแปลงเป็น USD
            quote_ccy = trade.symbol[3:6].upper() if len(trade.symbol) >= 6 else "USD"
            raw_profit = price_diff * trade.lot_size * contract_size

            if quote_ccy == "USD":
                trade.profit = raw_profit
            elif trade.close_price > 0:
                # แปลงจาก Quote Currency → USD
                trade.profit = raw_profit / trade.close_price
            else:
                trade.profit = raw_profit  # fallback

            # ย้ายจาก Active → Closed
            self._closed_trades.append(trade)
            del self._active_trades[ticket]

            # แจ้ง Risk Manager
            # reason_code=-1 → fallback ใช้ pnl sign; direction+close_price ช่วย post-TP lock ทำงานเมื่อ bot ปิดที่ TP level
            self._risk_manager.update_daily_pnl(
                trade.profit,
                symbol=trade.symbol,
                direction=trade.trade_type,
                close_price=trade.close_price,
            )

            if self._logger:
                self._logger.log_trade_closed(trade.to_dict())
            if self._analyzer:
                self._analyzer.add_trade(trade.to_dict())
                
            self._notifier.send_trade_close(trade.to_dict())

            print(f"✅ [Executor] ปิดเทรด Ticket {ticket}: P/L=${trade.profit:,.2f} ({reason})")
            return True
        else:
            print(f"❌ [Executor] ปิดเทรด Ticket {ticket} ล้มเหลว")
            return False

    def _populate_close_metadata(self, trade) -> None:
        """
        v7.1 — populate close-time metadata บน trade object.

        เดิม `record_external_close` (path SL/TP hit) ข้ามขั้นนี้ → trade_data dict ที่ส่งให้
        TradeLogger.log_trade_closed มี bid_at_exit / ask_at_exit / balance_at_close /
        equity_peak_during_trade = 0.0 (default จาก dataclass) → Excel เห็น 0/None.

        ตอนนี้เรียกจากทุก close path (self-close แล้วก็ทำ inline, external-close เรียก method นี้).
        """
        # Bid/Ask snapshot ณ ตอนปิด
        try:
            tick = self._connector.get_current_price(trade.symbol)
            if tick:
                trade.bid_at_exit = float(tick.get("bid", 0.0))
                trade.ask_at_exit = float(tick.get("ask", 0.0))
        except Exception:
            pass

        # Balance + Equity peak ณ ตอนปิด — ผ่าน RiskManager (single source of truth)
        try:
            rs = self._risk_manager.get_risk_status()
            trade.balance_at_close = float(rs.get("current_balance", 0.0))
            # Equity peak สูงสุดในวัน (ไม่ใช่ peak ของ trade อย่างเดียว — แต่ใช้แทนได้)
            trade.equity_peak_during_trade = float(rs.get("current_equity", 0.0))
        except Exception:
            pass

        # Final SL ใช้ค่าปัจจุบัน (อาจถูก trailing/BE updated ก่อนปิด)
        if not getattr(trade, "final_sl_at_close", 0):
            trade.final_sl_at_close = float(getattr(trade, "sl_price", 0.0) or 0.0)

    def record_external_close(
        self,
        ticket: int,
        close_price: float,
        profit: float,
        reason: str = "TP/SL Hit",
        reason_code: int = -1
    ):
        """
        บันทึกเทรดที่ปิดจากภายนอก (SL/TP Hit โดย MT5)

        เมื่อ SL หรือ TP ถูก Hit โดย MT5 โดยตรง Bot ต้องอัพเดทข้อมูล

        Args:
            ticket: หมายเลข Ticket
            close_price: ราคาปิด
            profit: กำไร/ขาดทุน
            reason: เหตุผล (human-readable string สำหรับ log/Excel)
            reason_code: MT5 DEAL_REASON code (4=SL, 5=TP, 6=SO, 3=EXPERT, 0-2=Manual).
                         -1 = ไม่ทราบ (จะ fallback ใช้ pnl sign ใน risk manager)
        """
        if ticket in self._active_trades:
            trade = self._active_trades[ticket]
            trade.is_open = False
            trade.close_price = close_price
            # v6.10b: ใช้ broker time (EEST)
            trade.close_time = TimeManager.get_server_time().replace(tzinfo=None)
            trade.profit = profit
            trade.close_reason = reason
            # Finalize ML features ตอนปิด
            if trade.open_time:
                trade.time_in_trade = int((trade.close_time - trade.open_time).total_seconds())
            trade.exit_path = reason

            # v7.1 — populate close-time metadata (Bid@Exit, Balance@Close, Equity Peak)
            # เดิม record_external_close ข้ามขั้นนี้ → Excel close rows มี field เป็น 0/None
            self._populate_close_metadata(trade)

            self._closed_trades.append(trade)
            del self._active_trades[ticket]

            self._risk_manager.update_daily_pnl(
                profit,
                symbol=trade.symbol,
                reason_code=reason_code,
                direction=trade.trade_type,
                close_price=close_price,
            )

            if self._logger:
                self._logger.log_trade_closed(trade.to_dict())
            if self._analyzer:
                self._analyzer.add_trade(trade.to_dict())

            self._notifier.send_trade_close(trade.to_dict())

            pnl_emoji = "🟢" if profit >= 0 else "🔴"
            print(f"{pnl_emoji} [Executor] เทรด Ticket {ticket} ปิดจากภายนอก: "
                  f"P/L=${profit:,.2f} ({reason})")

    # =========================================================================
    # 🔍 ตรวจสอบสถานะเทรด
    # =========================================================================

    def sync_with_mt5(self):
        """
        ซิงค์สถานะเทรดกับ MT5

        ตรวจสอบ:
        1. เทรดใน Active Trades ที่ปิดไปแล้ว (SL/TP Hit)
        2. เทรดที่เปิดใน MT5 แต่ไม่อยู่ใน Active Trades

        ⚠️ ข้ามในโหมดจำลอง (Mock) เพราะ get_open_positions คืน []
        ซึ่งจะทำให้ระบบเข้าใจผิดว่าทุกเทรดถูกปิดไปแล้ว
        """
        # ข้ามถ้าอยู่ในโหมดจำลอง — ไม่มีข้อมูล Position จริงจาก MT5
        try:
            from core.mt5_connector import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                return
        except ImportError:
            return

        mt5_positions = self._connector.get_open_positions()
        mt5_tickets = {pos["ticket"] for pos in mt5_positions}

        # === v8.0.13: Orphan Recovery (MT5 → Active Trades) ===
        # หา position ที่อยู่ใน MT5 แต่ไม่อยู่ใน _active_trades — เกิดได้เมื่อ:
        # 1. Bot restart กลางทาง (active_trades ใน RAM หาย แต่ broker ยังถือ position)
        # 2. Manual order ที่เปิดผ่าน MT5 client (magic ไม่ตรง — บอทไม่จัดการ ข้าม)
        # ก่อน v8.0.13 บอทมองไม่เห็น orphan → เปิดซ้ำ + ไม่จัดการของเก่า (เห็นจริงในรูป NZDUSD)
        for pos in mt5_positions:
            ticket = pos["ticket"]
            if ticket in self._active_trades:
                continue  # already tracked
            # ข้าม manual orders (magic ไม่ตรง bot magic)
            if pos.get("magic", 0) != 123456:
                continue
            try:
                self._active_trades[ticket] = self._rebuild_executed_trade_from_mt5(pos)
                print(f"♻️ [Executor] Orphan recovery: re-attached ticket {ticket} "
                      f"({pos['symbol']} {pos['type']} {pos['volume']} lot @ {pos['price_open']:.5f})")
            except Exception as e:
                print(f"⚠️ [Executor] ไม่สามารถ recover orphan ticket {ticket}: {e}")

        # ตรวจหาเทรดที่ปิดไปแล้ว (อยู่ใน Active แต่ไม่อยู่ใน MT5)
        closed_tickets = []
        for ticket, trade in self._active_trades.items():
            if ticket not in mt5_tickets:
                closed_tickets.append(ticket)

        for ticket in closed_tickets:
            trade = self._active_trades[ticket]
            # ดึง deals ของ position นี้โดยตรง (O(1) แทนสแกน 7 วัน)
            # รวม commission+swap+profit จาก **ทุก deals** ของ position (IN + OUT + partials)
            deals = self._connector.get_deals_by_position(ticket)
            actual_profit = 0.0
            actual_close_price = 0.0
            out_reason_code = -1  # -1 = ยังไม่ทราบ (จะ fallback ใช้ pnl sign ใน risk manager)
            matched = False

            for d in deals:
                actual_profit += (d.get("profit", 0) or 0) \
                              + (d.get("swap", 0) or 0) \
                              + (d.get("commission", 0) or 0)
                # close price + reason จาก deal entry = 1 (OUT) ไม้สุดท้าย
                if d.get("entry") == 1:
                    actual_close_price = d.get("price", 0) or 0
                    out_reason_code = d.get("reason", -1)
                    matched = True  # ต้องมี OUT deal เท่านั้นถึงจะถือว่าปิดจริง

            # ถ้า match ไม่เจอ → ข้าม record_external_close รอบนี้
            # profit=0.0 placeholder จะถูก Risk Manager ตีความเป็น loss (pnl <= 0)
            # ทำให้ consecutive_losses เพิ่มผิด → DAILY_HALT ผิด
            # Retry ใน cycle ถัดไปแทน (sync_with_mt5 ถูกเรียกทุก tick)
            if not matched:
                print(f"⚠️ [Executor] Ticket {ticket} ไม่พบใน deal history — รอ retry cycle ถัดไป")
                continue

            # ถ้าราคาไม่มี → ประมาณจากราคาปัจจุบัน (profit มาจาก history แล้ว ใช้ต่อได้)
            if actual_close_price == 0:
                price_info = self._connector.get_current_price(trade.symbol)
                if price_info:
                    actual_close_price = price_info["bid"] if trade.trade_type == "BUY" else price_info["ask"]

            reason_str = self._DEAL_REASON_MAP.get(
                out_reason_code,
                f"Close (reason={out_reason_code})"
            )
            self.record_external_close(
                ticket=ticket,
                close_price=actual_close_price,
                profit=actual_profit,
                reason=reason_str,
                reason_code=out_reason_code
            )

    # =========================================================================
    # 📊 ข้อมูลสรุป
    # =========================================================================

    @property
    def active_trades(self) -> Dict[int, ExecutedTrade]:
        """เทรดที่เปิดอยู่ทั้งหมด"""
        return self._active_trades

    @property
    def closed_trades(self) -> List[ExecutedTrade]:
        """เทรดที่ปิดแล้วทั้งหมด"""
        return self._closed_trades

    @property
    def active_count(self) -> int:
        """จำนวนเทรดที่เปิดอยู่"""
        return len(self._active_trades)

    def get_stats(self) -> Dict:
        """สรุปสถิติการทำงาน"""
        wins = sum(1 for t in self._closed_trades if t.profit > 0)
        losses = sum(1 for t in self._closed_trades if t.profit <= 0)
        total_pnl = sum(t.profit for t in self._closed_trades)
        win_rate = (wins / len(self._closed_trades) * 100) if self._closed_trades else 0

        return {
            "total_executed": self._total_executed,
            "total_rejected": self._total_rejected,
            "active_trades": self.active_count,
            "closed_trades": len(self._closed_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        }

    def __repr__(self) -> str:
        return (
            f"TradeExecutor(active={self.active_count}, "
            f"executed={self._total_executed}, rejected={self._total_rejected})"
        )
