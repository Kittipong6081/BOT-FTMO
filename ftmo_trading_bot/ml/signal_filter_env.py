"""
===============================================================================
FTMO Trading Bot — Signal Filter Environment (RL Agent กรอง Signal)
===============================================================================
Gymnasium Environment สำหรับ RL Agent ที่ตัดสินใจ TAKE/SKIP signal แต่ละตัว
จาก SMC Strategy โดยตรง

1 step  = 1 signal decision (take or skip)
1 episode = 1 FTMO Challenge (45 วัน, pre-generated signals)

Observation (29 dims, v7 2026-05-01) — ต้องตรงกับ SelfLearningAgent.OBS_DIM และ main._build_signal_observation:
  Signal core       [0-11]:  confluence, rr, direction, atr, ob_score, bias_align,
                             sl_atr, rsi, macd, trend_strength, ob_size_atr, adx
  Market regime     [12-15]: stoch_k, bb_pctb, atr_change_ratio, price_roc
  ML quality        [16]:    ml_score (P(win) จาก GBM model, AUC~0.59) ⭐
  Portfolio state   [17-23]: total_dd, daily_dd, progress, day_progress,
                             trades_today, recent_wr, consecutive_losses
  Cost/Flip/HTF     [24-26]: spread_pct_of_atr, has_opposite_recently_closed, htf_trend_alignment
  Chronos forecast  [27-28]: chronos_alignment, chronos_uncertainty_norm
                             v7: Amazon Chronos 2 zero-shot M15 forecast (8 bars ahead)

Action (1 dim, continuous [-1, 1]):
  > 0 → TAKE signal
  ≤ 0 → SKIP signal

Phase Curriculum (ควบคุมผ่าน enable_risk_penalty):
  Phase 1 (False): เรียนอ่านกราฟ + ทำกำไร — ไม่มี DD penalty, oracle SKIP reward
                   สอน agent ว่า signal ไหนควร TAKE/SKIP โดยใช้ future outcome เป็น label
  Phase 2 (True):  เพิ่ม daily DD + total DD penalty ตาม FTMO rules
                   Risk guard clamp + exponential penalty เมื่อเข้า danger zone

ML Pre-filter (Option F — Hybrid Architecture):
  ml_filter_threshold > 0 → signals with ml_score < threshold จะถูกตัดทิ้ง
  Agent เห็นเฉพาะ "quality signals" → เรียน "timing + DD" แทน filter
  Recommended: 0.36 (19% kept, WR 43% baseline) / 0.40 (5% kept, WR 48% baseline)
===============================================================================
"""

import os
import io
import pickle
import contextlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional

# v8.0.6: env now uses MeanReversionBacktester directly. The base
# StrategyBacktester is data-infrastructure only and raises NotImplementedError
# on generate_episode_signals — only subclasses produce signals. MR is the
# only strategy in the v8.0+ runtime.
from ml.mean_reversion_backtester import MeanReversionBacktester as StrategyBacktester


class FTMOSignalFilterEnv(gym.Env):
    """Agent ตัดสินใจ TAKE/SKIP signal แต่ละตัวจาก SMC Strategy"""

    metadata = {'render_modes': ['human', 'console']}

    INITIAL_BALANCE: float = 100_000.0
    CHALLENGE_DAYS: int = 45
    # v6.13: 0.003 → 0.007 (sync กับ live)
    # v7.1.9 (2026-05-05): 0.007 → 0.0099 — sync กับ live
    # v8.0.43 (2026-05-19): 0.0099 → 0.007 — Option X (trail) ชดเชย EV
    # v8.0.43c (2026-05-19): 0.007 → 0.0085 — สายกลาง balance
    RISK_PER_TRADE: float = 0.0085
    DAILY_DD_LIMIT: float = 0.05
    TOTAL_DD_LIMIT: float = 0.10
    TARGET_PCT: float = 0.10
    MAX_TRADES_PER_DAY: int = 3

    DAILY_DD_SAFE: float = 0.025
    TOTAL_DD_SAFE: float = 0.04
    # v8.0.4 (2026-05-06): tightened daily 4.0% → 3.0% so eval `daily_dd_max`
    # is no longer pinned at the guard ceiling (was always reading 4% even
    # when auto-tune lowered risk). Stays well under FTMO 5% live limit.
    DAILY_DD_GUARD: float = 0.030
    # v8.0.5 (2026-05-07): tightened total 8.5% → 5.8% (under our 6.0% gate).
    # Same logic as daily: previous guard 8.5% pinned eval `total_dd_max` to
    # 8.5% even when auto-tune lowered risk. Stays well under FTMO 10% limit.
    TOTAL_DD_GUARD: float = 0.058

    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_days: int = 45,
        verbose: bool = False,
        enable_risk_penalty: bool = True,
        signal_pool_path: Optional[str] = None,
        outcome_noise_std: float = 0.05,
        ml_filter_threshold: float = 0.0,
        risk_per_trade: Optional[float] = None,
    ):
        super().__init__()

        self.max_days = min(int(max_days), self.CHALLENGE_DAYS)
        self.verbose = verbose
        self.enable_risk_penalty = enable_risk_penalty
        self.outcome_noise_std = float(outcome_noise_std)
        # ML pre-filter: signals with ml_score < threshold จะถูกตัดทิ้ง
        # ไม่ส่งให้ agent เห็น (agent เห็นเฉพาะ "quality signals")
        # 0.0 = ปิด (ให้ agent filter เอง) / 0.36 = แนะนำ (WR baseline 43%)
        self.ml_filter_threshold = float(ml_filter_threshold)
        # Risk per trade (% of balance) — override class constant ได้
        # 0.003 = 0.3% (conservative) / 0.005 = 0.5% (balanced) / 0.007 = 0.7% (aggressive)
        self.risk_per_trade = float(risk_per_trade) if risk_per_trade is not None else self.RISK_PER_TRADE

        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "ohlcv"
            )
        self._data_dir = data_dir

        # ─── Signal Pool (optional) — ลด reset time จาก ~6 วิ → <1 ms ─────
        # ถ้ามี pool file: โหลดจาก disk (fast path)
        # ถ้าไม่มี: fallback → generate signals on-the-fly ตอน reset (slow path)
        self._signal_pool: List[List[Dict]] = []
        if signal_pool_path and os.path.exists(signal_pool_path):
            try:
                with open(signal_pool_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, list) and loaded:
                    self._signal_pool = loaded
                    if self.verbose:
                        print(f"✓ [Env] โหลด signal pool {len(loaded)} episodes จาก {signal_pool_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ [Env] โหลด pool ล้มเหลว ({e}) — ใช้ on-the-fly fallback")

        with self._suppress_stdout():
            try:
                self._backtester = StrategyBacktester(data_dir)
                if not self._backtester.is_available:
                    raise RuntimeError("No data available")
            except Exception as e:
                raise RuntimeError(
                    f"Signal Filter Env requires OHLCV data with 3 TFs: {e}"
                )

        self._seq_symbols = self._backtester.get_sequential_symbols(self.max_days)
        if not self._seq_symbols:
            raise RuntimeError("No symbols with enough data for sequential episode")

        # Observation: 12 signal core + 4 market regime + 1 ML quality + 7 portfolio + 3 cost/flip/htf + 2 chronos + 3 v7.1 + 3 keltner = 35
        # v6 (2026-04-22): เพิ่ม spread_pct_of_atr, has_opposite_recently_closed, htf_trend_alignment
        # v7 (2026-05-01): เพิ่ม chronos_alignment, chronos_uncertainty_norm จาก Amazon Chronos 2
        # v7.1 (2026-05-04): เพิ่ม floating_pnl_norm, losing_count_norm, mins_since_session_norm
        # v8.0.74 (2026-05-27): เพิ่ม kc_distance_norm, ema_slope_norm, band_squeeze_ratio (ATR Band anti-whipsaw)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(35,), dtype=np.float32
        )

        # Action: 1 dim continuous — >0 = TAKE, ≤0 = SKIP
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._reset_state()

    def _suppress_stdout(self):
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    # ─── v7.0.2: Correlation simulator (mirror TradeExecutor) ────────────
    # ทำให้ training distribution ตรงกับ live (live block 96.3% ของ TAKE
    # จาก correlation rules แต่ training เดิมไม่ check → silent regression).
    # ค่าทั้งหมด mirror TradeExecutor.CORRELATION_GROUPS / _GROUP_POSITIVE_DIR /
    # MAX_CORRELATED_POSITIONS — ห้ามแก้ฝั่งใดฝั่งนึงโดยไม่อัพเดทอีกฝั่ง.
    CORRELATION_GROUPS = {
        "USD_WEAK":   {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
        "USD_STRONG": {"USDJPY", "USDCAD", "USDCHF"},
        "JPY_CROSS":  {"USDJPY", "EURJPY", "GBPJPY"},
        "EUR_PAIRS":  {"EURUSD", "EURJPY"},
        "GBP_PAIRS":  {"GBPUSD", "GBPJPY"},
        "SAFE_HAVEN": {"XAUUSD"},
    }
    # symbol → direction ที่ให้ "group-positive exposure" (BUY มาตรฐาน)
    _GROUP_POSITIVE_DIR = {
        "USD_WEAK":   {"EURUSD": "BUY", "GBPUSD": "BUY", "AUDUSD": "BUY", "NZDUSD": "BUY"},
        "USD_STRONG": {"USDJPY": "BUY", "USDCAD": "BUY", "USDCHF": "BUY"},
        "JPY_CROSS":  {"USDJPY": "BUY", "EURJPY": "BUY", "GBPJPY": "BUY"},
        "EUR_PAIRS":  {"EURUSD": "BUY", "EURJPY": "BUY"},
        "GBP_PAIRS":  {"GBPUSD": "BUY", "GBPJPY": "BUY"},
        "SAFE_HAVEN": {"XAUUSD": "BUY"},
    }
    MAX_CORRELATED_POSITIONS = 1   # mirror config (FTMOConfig.MAX_CORRELATED_POSITIONS)
    # v7.0.3 (2026-05-02): Approximate hold time ใน pool. ก่อนหน้านี้ใช้ 1 → block 6h ใน live time
    # (1 pool slot = 6 ชม. live, แต่ live avg hold = 75 นาที = 1.25 ชม.) → over-block 4-5 เท่า
    # ผล eval v7.0.2: Pass Rate ตก 9.7% → 0.6%. ปรับ 0 = block correlation แค่ step ของ TAKE
    # แล้ว clear ทันที next step → effective ใกล้ live (live position ปิดเร็วกว่า next pool scan)
    # หมายเหตุ: ค่า 0 ทำให้ correlation simulator effectively off แต่ infrastructure คงไว้
    # เผื่อปรับเป็น 1 หากต้องการ block strict กลับ
    HOLD_SIGNALS_APPROX = 0

    def _is_correlation_blocked(self, sig: Dict) -> bool:
        """Mirror TradeExecutor._check_correlation_risk — ตรวจว่า signal นี้
        จะถูก correlation filter block ใน live หรือไม่.

        Returns True = block (forced SKIP), False = allow.
        """
        new_sym = (sig.get('symbol') or '').upper()
        if not new_sym:
            return False  # ไม่มี symbol = ไม่ block (defensive)
        new_dir_int = 1 if sig.get('direction', 0) > 0 else -1
        new_dir_str = "BUY" if new_dir_int > 0 else "SELL"

        # 1. Duplicate symbol
        for op in self._open_positions:
            if op['symbol'].upper() == new_sym:
                return True

        # 2. Group exposure — same effective direction within same correlation group
        for group_name, group_syms in self.CORRELATION_GROUPS.items():
            if new_sym not in group_syms:
                continue
            pos_dir_map = self._GROUP_POSITIVE_DIR.get(group_name, {})
            new_pos_dir = pos_dir_map.get(new_sym, "BUY")
            new_effective = 1 if new_dir_str == new_pos_dir else -1

            same_dir_count = 0
            for op in self._open_positions:
                existing_sym = op['symbol'].upper()
                if existing_sym not in group_syms:
                    continue
                existing_pos_dir = pos_dir_map.get(existing_sym, "BUY")
                existing_dir_str = "BUY" if op['direction'] > 0 else "SELL"
                existing_effective = 1 if existing_dir_str == existing_pos_dir else -1
                if existing_effective == new_effective:
                    same_dir_count += 1

            if same_dir_count >= self.MAX_CORRELATED_POSITIONS:
                return True

        return False

    def _reset_state(self):
        self.balance = self.INITIAL_BALANCE
        self.peak_balance = self.INITIAL_BALANCE
        self.daily_start_balance = self.INITIAL_BALANCE
        self.total_dd_pct = 0.0
        self.daily_dd_pct = 0.0
        self.min_balance = self.INITIAL_BALANCE
        self.max_daily_dd_pct = 0.0
        self.target_progress_pct = 0.0
        self._last_progress = 0.0
        # Track if episode ever hit target (สำหรับ sticky "passed" metric)
        # เมื่อ Phase 1 ไม่ terminate ที่ target, agent อาจ drop below 10% กลับ
        # → ต้อง track peak เพื่อบอกว่า "เคยผ่าน"
        self._target_bonus_given = False
        self._peak_passed = False
        # v6.3 B1: milestone bonuses — sticky flags ปลด bonus ทีละ 30/60/90 %
        self._milestone_30_given = False
        self._milestone_60_given = False
        self._milestone_90_given = False
        # v6.3 B1v2: mid-episode undertrading checks — sticky (fire once)
        # v6.13: เพิ่ม day 10 check (early signal) — push agent take ตั้งแต่ต้น episode
        self._mid_check_day10_fired = False
        self._mid_check_day20_fired = False
        self._mid_check_day35_fired = False

        self._signals: List[Dict] = []
        self._signal_idx = 0
        self._current_day = -1
        self._trades_today = 0
        self._trade_results: List[float] = []
        self._consecutive_losses = 0
        self._total_takes = 0
        self._total_skips = 0
        # v6: track last closed trade direction for flip-lock feature
        self._last_closed_direction: float = 0.0  # +1 BUY / -1 SELL / 0 none
        self._last_closed_signal_step: int = -999
        # v7.0.2: open positions for correlation simulator (mirror live TradeExecutor)
        # list of {'symbol': str, 'direction': int (+1/-1), 'opened_at_idx': int}
        self._open_positions: List[Dict] = []
        # counter สำหรับ telemetry: trades ที่ถูก correlation block (forced SKIP)
        self._correlation_forced_skips: int = 0
        # v8.0.55: counter trades ที่ถูก entry-confirm block (mirror live gate)
        self._entry_confirm_forced_skips: int = 0
        # v7.1: floating portfolio state for obs[29-30]
        # อัพเดทใน step() ตอน open/close position (simulator)
        self._floating_pnl_norm: float = 0.0
        self._open_losing_count_norm: float = 0.0

    def _pick_episode(self, rng: np.random.Generator):
        symbol = self._seq_symbols[int(rng.integers(0, len(self._seq_symbols)))]
        m15_len = len(self._backtester._m15_cache[symbol])
        needed = self._backtester.get_min_bars_for_episode(self.max_days)
        max_start = m15_len - needed
        if max_start <= self._backtester.MIN_M15_BARS:
            start_bar = self._backtester.MIN_M15_BARS
        else:
            start_bar = int(rng.integers(self._backtester.MIN_M15_BARS, max_start))
        return symbol, start_bar

    def _get_obs(self) -> np.ndarray:
        if self._signal_idx >= len(self._signals):
            return np.zeros(35, dtype=np.float32)

        sig = self._signals[self._signal_idx]

        # ─── Signal core (12) ─────────────────────────────────
        confluence_norm = (sig['confluence_score'] - 50.0) / 50.0
        rr_norm = (sig['rr_ratio'] - 1.0) / 4.0
        direction = sig['direction']
        atr_pips = sig['atr_pips']
        atr_norm = (atr_pips - 15.0) / 10.0
        ob_norm = sig['ob_score'] / 100.0
        bias_align = sig['bias_alignment']
        sl_atr = sig['sl_distance_atr']
        rsi_norm = (sig.get('rsi_value', 50.0) - 50.0) / 50.0
        atr_val = sig.get('atr_value', 1e-8)
        macd_norm = sig.get('macd_histogram', 0.0) / max(atr_val, 1e-8)
        trend_str = sig.get('trend_strength', 0.0) / 100.0
        ob_size_atr = sig.get('ob_size_atr', 0.0)
        adx_norm = sig.get('adx', 0.0) / 100.0

        # ─── Market regime (4) — match main._build_signal_observation ──
        stoch_norm = (sig.get('stoch_k', 50.0) - 50.0) / 50.0
        bb_pctb = sig.get('bb_pctb', 0.5)
        atr_chg = sig.get('atr_change_ratio', 0.0)
        price_roc = sig.get('price_roc', 0.0)

        # ─── ML quality score (1) — GBM P(win), AUC ~0.59 ⭐ ──
        # 0.5 = neutral (no info); >0.4 → edge profitable; center at 0.5
        ml_score = sig.get('ml_score', 0.5)
        ml_score_norm = (ml_score - 0.5) * 2.0  # map [0,1] → [-1,+1]

        # ─── Portfolio state (7) ──────────────────────────────
        total_dd_n = -self.total_dd_pct / 0.10
        daily_dd_n = -self.daily_dd_pct / 0.05
        progress_n = self.target_progress_pct / 100.0
        day_progress = sig['day'] / max(self.max_days, 1)
        trades_today_n = min(self._trades_today, self.MAX_TRADES_PER_DAY) / self.MAX_TRADES_PER_DAY
        recent = self._trade_results[-10:] if self._trade_results else []
        if recent:
            wr = sum(1 for t in recent if t > 0) / len(recent)
            recent_wr_norm = wr * 2.0 - 1.0
        else:
            recent_wr_norm = 0.0
        consec_norm = min(self._consecutive_losses, 5) / 5.0

        # ─── v6: Cost / Flip / HTF (3) ────────────────────────
        # [24] spread_pct_of_atr — normalize ต้นทุน spread เทียบ volatility
        #      GBPJPY จะเห็นค่าสูง (~1.0-2.0) → agent เรียนหลีกเลี่ยง setup RR ต่ำ
        spread_pips = sig.get('spread_pips', 0.0)
        spread_pct_of_atr = spread_pips / max(atr_pips, 1e-6) if atr_pips > 0 else 0.0

        # [25] has_opposite_recently_closed — flip-lock context
        #      1.0 ถ้ามี trade ตรงข้ามปิดภายใน last signal (sync กับ flip-lock P0)
        # ใช้ _last_closed_direction_step ที่เพิ่มใน __init__ (track step # at close)
        last_closed_dir = getattr(self, '_last_closed_direction', 0)
        last_closed_step = getattr(self, '_last_closed_signal_step', -999)
        # ถือว่า "recent" = ภายใน 3 signals
        recent_opposite = 0.0
        if last_closed_dir != 0 and last_closed_dir != direction:
            gap = self._signal_idx - last_closed_step
            if gap >= 0 and gap <= 3:
                recent_opposite = 1.0

        # [26] htf_trend_alignment — สรุป H1 EMA200 + D1 bias vs signal direction
        # ใช้ bias_alignment ของ signal (มีอยู่แล้ว) ผสมกับ adx
        # sig['htf_trend_alignment'] ถ้ามีใน pool (backtester v2) ไม่งั้น fallback = bias_align * sign(adx_norm-0.2)
        htf_align = sig.get('htf_trend_alignment', bias_align)

        # ─── v7: Chronos forecast (2) — Amazon Chronos 2 zero-shot ──────
        # [27] chronos_alignment — direction × sign(median_h+8 − close), ±1
        # [28] chronos_uncertainty_norm — log1p((q90 − q10) / (atr×√8)) / 2, [0, 3]
        chronos_align = float(sig.get('chronos_alignment', 0.0))
        chronos_unc = float(sig.get('chronos_uncertainty_norm', 0.0))

        # ─── v7.1 Portfolio risk realtime + session timing (3) ──────
        # [29] floating_pnl_norm — sum ของ open positions' floating PnL / daily_start_balance
        # [30] open_losing_count_norm — จำนวน open positions ที่กำลังขาดทุน / 3
        # [31] mins_since_session_start_norm — minutes since London/NY open / 480
        # ใน training env เลียนแบบผ่าน virtual portfolio simulator (ดู step())
        # Live env: main._build_signal_observation จะอ่าน RiskManager.get_unrealized_drawdown_pct
        floating_pnl_norm = getattr(self, '_floating_pnl_norm', 0.0)
        open_losing_count_norm = getattr(self, '_open_losing_count_norm', 0.0)
        # mins_since_session — pull from sig dict (compute_temporal_features เก็บไว้)
        mins_since = float(sig.get('minutes_since_session_start', 0.0))
        mins_since_norm = min(mins_since / 480.0, 1.0)

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
            # ML quality [16] ⭐ — หลัก signal สำหรับ RL เพราะ AUC 0.59
            float(np.clip(ml_score_norm, -1.0, 1.0)),
            # Portfolio [17-23]
            float(np.clip(total_dd_n, -5.0, 0.0)),
            float(np.clip(daily_dd_n, -5.0, 0.0)),
            float(np.clip(progress_n, -1.0, 2.0)),
            float(np.clip(day_progress, 0.0, 1.0)),
            float(np.clip(trades_today_n, 0.0, 1.0)),
            float(np.clip(recent_wr_norm, -1.0, 1.0)),
            float(np.clip(consec_norm, 0.0, 1.0)),
            # v6 Cost/Flip/HTF [24-26]
            float(np.clip(spread_pct_of_atr, 0.0, 3.0)),
            float(recent_opposite),
            float(np.clip(htf_align, -1.0, 1.0)),
            # v7 Chronos forecast [27-28]
            float(np.clip(chronos_align, -1.0, 1.0)),
            float(np.clip(chronos_unc, 0.0, 3.0)),
            # v7.1 Portfolio realtime + session timing [29-31]
            float(np.clip(floating_pnl_norm, -3.0, 1.0)),
            float(np.clip(open_losing_count_norm, 0.0, 1.0)),
            float(np.clip(mins_since_norm, 0.0, 1.0)),
            # v8.0.74 Keltner / ATR Band features [32-34]
            float(np.clip(sig.get("kc_distance_norm", 0.0), -1.0, 1.0)),
            float(np.clip(sig.get("ema_slope_norm", 0.0), -1.0, 1.0)),
            float(np.clip(sig.get("band_squeeze_ratio", 0.5), 0.0, 1.0)),
        ], dtype=np.float32)
        return obs

    def _update_daily_dd(self):
        day_end_loss = max(0.0, self.daily_start_balance - self.balance)
        self.daily_dd_pct = day_end_loss / max(self.daily_start_balance, 1.0)

    def _start_new_day(self):
        self.daily_start_balance = self.balance
        self.daily_dd_pct = 0.0
        self._trades_today = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        rng = self._get_rng()
        symbol = None

        # ─── Fast path: sample from pre-built pool (reset <1ms) ─────
        if self._signal_pool:
            idx = int(rng.integers(0, len(self._signal_pool)))
            # Shallow copy dicts เพื่อป้องกัน mutation ข้าม episode
            self._signals = [dict(s) for s in self._signal_pool[idx]]
            symbol = f"pool_{idx}"
        else:
            # ─── Slow fallback: generate on-the-fly ─────────────────
            symbol, start_bar = self._pick_episode(rng)
            with self._suppress_stdout():
                self._signals = self._backtester.generate_episode_signals(
                    symbol=symbol,
                    m15_start_bar=start_bar,
                    num_days=self.max_days,
                    rng=rng,
                )

        # ─── ML Pre-filter (Option F: Hybrid Architecture) ──────────
        # กรอง signals ที่ ml_score ต่ำกว่า threshold ก่อน agent เห็น
        # → agent เรียน "quality signals เมื่อไรควรเทรด" แทน "filter เอง"
        # Expected: 19% signals at threshold 0.36 / 5% at 0.40
        if self.ml_filter_threshold > 0.0:
            self._signals = [
                s for s in self._signals
                if s.get('ml_score', 0.0) >= self.ml_filter_threshold
            ]

        if len(self._signals) == 0:
            self._signals = [self._dummy_signal()]

        # v6.3: inject spread noise ±50% around typical (news/illiquidity realism)
        # ทำครั้งเดียวต่อ episode → obs[24] + reward.spread_cost ใช้ค่าเดียวกัน
        for s in self._signals:
            base = float(s.get('spread_pips', 0.0))
            if base > 0:
                s['spread_pips'] = base * float(rng.uniform(0.7, 1.5))

        self._signal_idx = 0
        self._current_day = self._signals[0]['day']

        info = {
            'total_signals': len(self._signals),
            'symbol': symbol,
            'initial_balance': self.INITIAL_BALANCE,
        }
        return self._get_obs(), info

    # ─── Risk Guard ──────────────────────────────────────────────────
    def _clamp_pnl_to_guard(self, raw_pnl: float) -> float:
        """
        Rule-Based Risk Guard: ถ้าเทรดครั้งนี้จะทำให้ DD เกิน guard line
        → ตัด PnL ให้หยุดที่ guard line แทนที่จะปล่อยให้ breach
        Agent จะเห็นว่าเทรดแพ้แต่ "ไม่ตาย" → เรียนรู้ต่อได้

        FTMO Rule Compliance (สำคัญ — denominator ต่างกันโดยตั้งใจ):
          • Total (Max Loss) 10%  → วัดจาก INITIAL_BALANCE (absolute floor)
          • Daily Loss 5%         → วัดจาก daily_start_balance ณ 00:00 EET ทุกวัน
          Guard ที่ 4% daily / 8.5% total ให้ buffer safety margin
        """
        if raw_pnl >= 0:
            return raw_pnl

        new_balance = self.balance + raw_pnl

        # Total DD guard: absolute floor จาก INITIAL_BALANCE (FTMO Max Loss Rule)
        total_floor = self.INITIAL_BALANCE * (1.0 - self.TOTAL_DD_GUARD)
        if new_balance < total_floor:
            raw_pnl = total_floor - self.balance

        # Daily DD guard: floor จาก daily_start_balance (FTMO Daily Loss Rule)
        daily_floor = self.daily_start_balance * (1.0 - self.DAILY_DD_GUARD)
        if (self.balance + raw_pnl) < daily_floor:
            raw_pnl = daily_floor - self.balance

        return min(raw_pnl, 0.0)

    # ─── Continuous DD Penalty ────────────────────────────────────
    def _dd_penalty(self) -> float:
        """
        Continuous Soft Penalty: ไม่มี cliff-edge
        Softened (2026-04-18): signal เป็น -EV โดยธรรมชาติ (WR ~30%)
        penalty เดิมแรงเกินไปจนทำให้ agent เลือก SKIP-all = local optimum แย่
        """
        penalty = 0.0

        if self.total_dd_pct > self.TOTAL_DD_SAFE:
            ratio = (self.total_dd_pct - self.TOTAL_DD_SAFE) / \
                    (self.TOTAL_DD_LIMIT - self.TOTAL_DD_SAFE)
            ratio = min(ratio, 1.0)
            penalty += -0.25 * (np.exp(2.0 * ratio) - 1.0)

        if self.daily_dd_pct > self.DAILY_DD_SAFE:
            ratio = (self.daily_dd_pct - self.DAILY_DD_SAFE) / \
                    (self.DAILY_DD_LIMIT - self.DAILY_DD_SAFE)
            ratio = min(ratio, 1.0)
            penalty += -0.15 * (np.exp(2.0 * ratio) - 1.0)

        return penalty

    # ─── Step ─────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        take = float(action[0]) > 0.0

        sig = self._signals[self._signal_idx]
        pnl = 0.0
        was_clamped = False

        # Handle day transition
        if sig['day'] != self._current_day:
            self._update_daily_dd()
            self._current_day = sig['day']
            self._start_new_day()

        # v7.0.2: drop stale open positions (~hold ~1 signal slot ≈ live trade duration)
        if self._open_positions:
            cutoff = self._signal_idx - self.HOLD_SIGNALS_APPROX
            self._open_positions = [
                op for op in self._open_positions if op['opened_at_idx'] >= cutoff
            ]

        # v7.2.1 (2026-05-06) — LEAK FIX: zero-out floating PnL state in training env.
        # v7.1 used `op['outcome_partial']` (pre-resolved final outcome × decay) which
        # leaked future direction (kept sign constant from open → close). Live env
        # uses true unrealized PnL from RiskManager that oscillates with price path.
        # Mismatch → agent learned fake "concurrent risk" pattern. v7.2 eval Pass 6.3%
        # (vs baseline 9.7%) confirmed leak hurt rather than helped.
        # Train obs[29/30] = 0 → live obs[29/30] = real values, but agent ignores them
        # (effectively no-info dims). Concurrent risk awareness comes from reward
        # penalty + Chronos disagreement, not from these obs dims.
        self._floating_pnl_norm = 0.0
        self._open_losing_count_norm = 0.0

        # v7.0.2: correlation block — forced SKIP ถ้า live executor จะ reject
        # นี่ทำให้ training distribution ตรงกับ live (live block 96.3% ของ TAKE จาก correlation)
        correlation_forced_skip = False
        if take and self._is_correlation_blocked(sig):
            take = False
            correlation_forced_skip = True
            self._correlation_forced_skips += 1

        # v8.0.55: Entry confirmation forced SKIP — mirror live pre-execution gate
        # Backtester ตั้ง entry_confirm_passed=False เมื่อ first future bar swing สวนทิศ signal
        # ถ้า RL อยาก TAKE แต่ entry confirm fail → force SKIP (เหมือน live executor reject)
        # → training distribution ตรงกับ live (Pass rate ที่วัดสะท้อนพฤติกรรมจริง)
        entry_confirm_forced_skip = False
        if take and not sig.get('entry_confirm_passed', True):
            take = False
            entry_confirm_forced_skip = True
            self._entry_confirm_forced_skips += 1

        outcome = float(sig.get('outcome_pnl_ratio', 0.0))
        # v8.0.47: removed trail reward cap — let runners flow naturally
        # v8.0.45 capped @1.5R+0.5x decay → agent ฝึก "พอใจ 1.5R" → no big winners → Pass 52%
        # คืน raw outcome → big runners 2R+/3R+ ให้ reward เต็ม → agent ไล่ trail trades

        # ML quality score (จาก GBM) — ตัวทำนาย win ที่แม่นกว่า confluence
        # ใช้เป็น primary signal สำหรับ reward shaping (แทน confluence ที่ uninformative)
        ml_score = float(sig.get('ml_score', 0.5))

        # ═══ 1. TAKE — execute trade with Risk Guard ═══
        if take and self._trades_today < self.MAX_TRADES_PER_DAY:
            risk_amount = self.balance * self.risk_per_trade

            # Outcome perturbation — กัน agent memorize pool (ถ้ามี pool)
            # Gaussian noise 5% เหมือน slippage variance ใน live
            if self._signal_pool and self.outcome_noise_std > 0:
                noise = float(self.np_random.normal(0.0, self.outcome_noise_std))
                outcome_perturbed = outcome * (1.0 + noise)
            else:
                outcome_perturbed = outcome

            raw_pnl = risk_amount * outcome_perturbed

            if self.enable_risk_penalty:
                pnl = self._clamp_pnl_to_guard(raw_pnl)
                was_clamped = (pnl != raw_pnl)
            else:
                pnl = raw_pnl

            self.balance += pnl
            self._trades_today += 1
            self._trade_results.append(pnl)
            self._total_takes += 1

            # v6: track last closed direction + step for flip-lock observation
            self._last_closed_direction = float(sig.get('direction', 0.0))
            self._last_closed_signal_step = int(self._signal_idx)

            # v7.0.2: register open position สำหรับ correlation simulator
            # ใช้ตอน step ถัดไปเป็น "exposure" ที่ block correlated entries
            # (live: position ค้างอยู่จน hit SL/TP/timeout. ตรงนี้ approximate hold ≈ HOLD_SIGNALS_APPROX)
            # v7.2.1: ลบ outcome_partial — เคย feed obs[29/30] เป็น future leak (ดู _floating_pnl_norm block)
            self._open_positions.append({
                'symbol': sig.get('symbol', ''),
                'direction': 1 if sig.get('direction', 0) > 0 else -1,
                'opened_at_idx': self._signal_idx,
            })

            if pnl > 0:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

            self.peak_balance = max(self.peak_balance, self.balance)
            total_loss = max(0.0, self.INITIAL_BALANCE - self.balance)
            self.total_dd_pct = total_loss / self.INITIAL_BALANCE

            day_loss = max(0.0, self.daily_start_balance - self.balance)
            self.daily_dd_pct = day_loss / max(self.daily_start_balance, 1.0)

            self.min_balance = min(self.min_balance, self.balance)
            self.max_daily_dd_pct = max(self.max_daily_dd_pct, self.daily_dd_pct)

            profit = self.balance - self.INITIAL_BALANCE
            target_amount = self.INITIAL_BALANCE * self.TARGET_PCT
            self.target_progress_pct = (profit / target_amount) * 100.0

            # ═══ Reward (TAKE) ═══
            # Base: PnL in risk units (asymmetric: win กว้างกว่า loss
            # เพราะ signal เป็น -EV → ต้องให้ winner คุ้มกับ loser ถึงจะกล้าเทรด)
            pnl_norm = pnl / max(risk_amount, 1.0)

            # v6: หัก spread cost เป็น R units → agent เรียนหลีกเลี่ยง cost สูง
            # GBPJPY: spread 30 / SL 50 = 0.6R × 0.5 weight = 0.3R deduction
            # v6.3: clamp spread_cost_R ที่ 1.0 (ห้าม penalty > -1R บน news-spike spread)
            spread_pips_trade = sig.get('spread_pips', 0.0)
            sl_pips_trade = sig.get('sl_distance_pips', 0.0)
            if sl_pips_trade > 0 and spread_pips_trade > 0:
                spread_cost_R = min(spread_pips_trade / sl_pips_trade, 1.0)
                # Weight 0.5 — spread affects entry/exit แต่ TP distance ดูดกลับบางส่วน
                pnl_norm -= spread_cost_R * 0.5

            reward = float(np.clip(pnl_norm, -1.0, 3.0))

            # ML quality bonus — v6.13: Equalize @ ml >= 0.36
            # เดิม: +0.35 ถ้า ml >= 0.40, +0.15 ถ้า ml ∈ [0.36, 0.40) → bias หนักไป >0.40
            # ใหม่: +0.30 uniform ทุก signal ที่ ml >= 0.36 → agent take ทุก signal ที่ผ่าน gate
            if pnl_norm > 0 and ml_score >= 0.36:
                reward += 0.30           # Win + ml >= threshold → uniform bonus
            elif pnl_norm < -0.3 and ml_score < 0.35:
                reward -= 0.35           # Low-ml + loss → safety branch (rare ที่ gate 0.36)

            # v7.1 — Chronos disagreement penalty (v7.1.6 revert to v7.1.2 levels)
            # v7.1.5 eval Pass 0.8% (worst): combo (0.40 threshold + -0.90 missed-winner +
            # softened Chronos/concurrent) คือ 3-way conflict ทำให้ entropy collapse
            # v7.1.6: revert reward back to v7.1.2 (Pass 3% proven) + keep pool 10k diversity
            chronos_align_v71 = float(sig.get('chronos_alignment', 0.0))
            if chronos_align_v71 < 0:
                if ml_score < 0.55:
                    reward -= 0.30        # v7.1.6: 0.20 → 0.30 (revert v7.1.2 level)
                else:
                    reward -= 0.10        # v7.1.6: 0.05 → 0.10 (revert v7.1.2 level)

            # v7.1 — Concurrent loss penalty (v7.1.6 revert to v7.1.2 level)
            # v7.1.6: revert 0.15 → 0.20 (back to v7.1.2 medium guard)
            if self._floating_pnl_norm < -0.01:  # floating < -1% ของ daily start
                reward -= 0.20            # v7.1.6: 0.15 → 0.20 (revert v7.1.2 level)

            # Option B (P1 only): Quality-first responsibility shaping
            # ให้ P1 เรียน "เลือก winner" โดยไม่ต้องพึ่ง DD penalty ของ P2
            # → แทนที่จะ push กิจกรรม ให้ reward เอียงไปทางคุณภาพ
            if not self.enable_risk_penalty:
                # P1: กด activity bonus ลง + เพิ่ม quality shaping
                reward += 0.02           # activity nudge ลดลงจาก 0.04
                if pnl_norm > 0.3:
                    reward += 0.20       # clear winner → bonus quality TAKE
                elif pnl_norm < -0.3:
                    reward -= 0.20       # clear loser → responsibility penalty
            else:
                # P2: เหมือนเดิม — activity nudge เดิม + DD penalty
                reward += 0.04
                reward += self._dd_penalty()
                if was_clamped:
                    reward -= 0.25

        # ═══ 2. SKIP — Oracle feedback (chart-reading label) ═══
        else:
            self._total_skips += 1
            reward = 0.0

            # Oracle: agent ไม่เห็น outcome ใน obs แต่ reward ใช้ future info ได้ตอน train
            # Option B: P1 ลด "missed opportunity penalty" เพื่อไม่ push TAKE มั่ว
            # P2 คงเดิม (เพราะ DD penalty เป็นตัว enforce quality แล้ว)
            # v6.13: Push agent take more (especially in P2) — agent undertrade ที่ orders/ep 6.1
            #        เพิ่ม penalty missed-winner + เพิ่ม reward smart-skip → bias ไป TAKE คุณภาพ
            is_p1 = not self.enable_risk_penalty
            # v7.1.6 (2026-05-05) — revert to v7.1.2 levels (Pass 3.0% proven config)
            # v7.0.x: P2 -0.90 → Pass 11.1% but wipeout 2026-05-04 ใน LIVE
            # v7.1.2: P2 -0.85 → Pass 3.0% (sweet spot, no wipeout)
            # v7.1.4: P2 -0.90 (restored aggression) → never trained directly
            # v7.1.5: combo (10k + 0.40 + -0.90) → Pass 0.8% (entropy collapse)
            # v7.1.6: ใช้ pool 10k (diversity) + revert reward to v7.1.2 (-0.85) +
            #         revert ml_threshold 0.36 = combine "best of all worlds"
            # v7.1.11 push (Pass 5.4%) ไม่ดีกว่า v7.1.9 baseline (Pass 6.1%)
            # → restore v7.1.9 reward levels (proven best ใต้ FTMO 1% rule)
            if outcome >= 0.5:
                # Would win big → missed opportunity (v7.1.2 sweet spot level, v7.1.9 confirmed)
                reward -= 0.30 if is_p1 else 0.85
                if ml_score >= 0.40:
                    reward -= 0.25 if is_p1 else 0.50
                elif ml_score < 0.35:
                    reward += 0.15
            elif outcome >= 0.1:
                reward -= 0.10 if is_p1 else 0.20
            elif outcome <= -0.5:
                # Smart skip
                reward += 0.30 if is_p1 else 0.35
                if ml_score < 0.36:
                    reward += 0.15
            elif outcome <= -0.1:
                reward += 0.10 if is_p1 else 0.10        # symmetry

            # Passive SKIP cost — สะสมต่อ step
            # v6.3 B1: ลดจาก -0.015 → -0.010 ให้ room SKIP signals คุณภาพต่ำ
            # 133 signals/ep × 0.010 = -1.33 ถ้า SKIP ทั้งหมด → ยังดันออกจาก "do nothing" trap
            reward -= 0.010

        # ═══ 3. Progress shaping + target bonus ═══
        # v6.3 B1v2: เพิ่ม multiplier 0.02 → 0.05 (2.5x) ให้ reward สะท้อน progress ชัดขึ้น
        # ตัวอย่าง: win 1R (risk 0.7%) ที่ balance 100k → profit $700 → progress +7%
        # → bonus 0.05 × 7 = +0.35 (เดิม +0.14)
        progress_delta = self.target_progress_pct - self._last_progress
        if progress_delta > 0:
            reward += 0.05 * progress_delta
        elif progress_delta < -5.0:
            reward += 0.005 * progress_delta
        self._last_progress = self.target_progress_pct

        # v6.3 B1v2: Mid-episode undertrading checks (P2 only, sticky)
        # ยิง penalty "ระหว่าง episode" ไม่ใช่แค่ตอนจบ → credit assignment เร็วขึ้น
        # agent เรียนว่า "ถ้า day 20 ยังไม่ take พอ = ลงโทษ" → push TAKE ตั้งแต่ต้น
        if self.enable_risk_penalty:
            # v6.13: เพิ่ม day 10 early check — กระตุ้น take ตั้งแต่ต้น episode
            if (
                not self._mid_check_day10_fired
                and self._current_day >= 10
                and self.target_progress_pct < 20.0
                and self._total_takes < 3
            ):
                reward -= 0.2
                self._mid_check_day10_fired = True
            if (
                not self._mid_check_day20_fired
                and self._current_day >= 20
                and self.target_progress_pct < 40.0
                and self._total_takes < 6
            ):
                reward -= 0.3
                self._mid_check_day20_fired = True
            if (
                not self._mid_check_day35_fired
                and self._current_day >= 35
                and self.target_progress_pct < 60.0
                and self._total_takes < 12
            ):
                reward -= 0.7
                self._mid_check_day35_fired = True

        # v6.3 B1: Milestone bonuses — sticky bonus เมื่อถึง 30/60/90 %
        # Agent เรียนว่า "progress = สิ่งที่ต้องไปถึง ไม่ใช่แค่ target สุดท้าย"
        if self.target_progress_pct >= 30.0 and not self._milestone_30_given:
            reward += 0.5
            self._milestone_30_given = True
        if self.target_progress_pct >= 60.0 and not self._milestone_60_given:
            reward += 1.0
            self._milestone_60_given = True
        if self.target_progress_pct >= 90.0 and not self._milestone_90_given:
            reward += 1.5
            self._milestone_90_given = True

        # Target hit logic — P1 vs P2 ต่างกัน
        # P1 (Alpha): ไม่ terminate — ให้ agent เรียนต่อเพื่อ maximize profit
        # P2 (Risk):  terminate ที่ 10% ตาม FTMO rule (challenge จบ)
        terminated = False
        if self.target_progress_pct >= 100.0:
            # Sticky mark: ครั้งแรกที่ถึง target
            # v6.3 B1: bonus เพิ่มจาก +2.0 → +4.0 (max sticky รวม milestone = 7.0)
            if not self._target_bonus_given:
                reward += 4.0
                self._target_bonus_given = True
                self._peak_passed = True
            # เฉพาะ P2 ที่จบ episode — P1 เรียนต่อ
            if self.enable_risk_penalty:
                terminated = True

        # ═══ 4. Episode end ═══
        self._signal_idx += 1

        truncated = False
        if self._signal_idx >= len(self._signals):
            truncated = True
            take_rate = self._total_takes / max(self._total_takes + self._total_skips, 1)
            # Activity floor — ลดจาก -5.0 (2026-04-18)
            # เดิมแรงเกินไปจน agent ติด "SKIP-all" local optimum (std=0, reward=-2.0)
            # ตอนนี้ passive SKIP cost (-0.015/step) เป็นตัวกดดันหลัก terminal แค่ top-up
            if self.enable_risk_penalty:
                if take_rate < 0.05:
                    reward -= 1.0
                elif take_rate < 0.12:
                    reward -= 0.3
                elif self.target_progress_pct < 30.0:
                    reward -= 0.3

                # v6.3 B1v2: Undertrading penalty (terminal) — agent เทรดน้อยเกิน
                # ถ้า ep ใช้เวลาเต็ม (>= 40 วัน) แต่เทรดรวม < 15 ตัว → push ให้ take
                # ลด threshold 20 → 15 (pool ที่ th0.36 มีแค่ ~16 signals/ep → 20 เกินจริง)
                # Calculation: 15 trades × 0.52R avg = 7.8R ≈ 5.5% ที่ 0.7% risk
                if (
                    self._current_day >= 40
                    and (self._total_takes < 15)
                    and not self._peak_passed
                ):
                    reward -= 1.0
            else:
                # Phase 1: กรณีสุดขั้ว SKIP ทั้งหมด → penalty เบา ๆ กันไม่ให้ลงมา degenerate
                if take_rate < 0.03:
                    reward -= 0.8

        info = {
            'action': 'TAKE' if take else 'SKIP',
            'signal_confluence': sig['confluence_score'],
            'signal_rr': sig['rr_ratio'],
            'pnl': pnl,
            'balance': self.balance,
            'total_dd_pct': self.total_dd_pct,
            'daily_dd_pct': self.daily_dd_pct,
            'progress_pct': self.target_progress_pct,
            'day': sig['day'],
            'total_takes': self._total_takes,
            'total_skips': self._total_skips,
            'clamped': was_clamped,
            # v6.9 E2: aux regression target = signal outcome (R units)
            # AuxAwarePPO reads this from infos, stores in AuxRolloutBuffer
            'aux_target': float(sig.get('outcome_pnl_ratio', 0.0)),
        }

        if terminated or truncated:
            stats = self.get_stats()
            profit = self.balance - self.INITIAL_BALANCE
            info['episode_summary'] = {
                'balance': self.balance,
                'profit': profit,
                'profit_pct': profit / self.INITIAL_BALANCE * 100,
                'total_dd_pct': self.total_dd_pct,
                'peak_balance': self.peak_balance,
                'total_takes': self._total_takes,
                'total_skips': self._total_skips,
                'take_rate': stats['take_rate'],
                'win_rate': stats['win_rate'],
                'total_trades': stats['total_trades'],
                'target_progress_pct': self.target_progress_pct,
                'breached': self.total_dd_pct >= self.TOTAL_DD_LIMIT or self.daily_dd_pct >= self.DAILY_DD_LIMIT,
                # 'passed' = sticky: เคยถึง target ณ จุดใดจุดหนึ่ง (ไม่ใช่แค่ตอนจบ)
                # P1 ไม่ terminate ที่ target → อาจ drop กลับก่อน ep จบ ต้องใช้ peak
                'passed': self._peak_passed or self.target_progress_pct >= 100.0,
                'days_traded': self._current_day + 1,
                'daily_dd_pct': self.daily_dd_pct,
                'min_balance': self.min_balance,
                'min_profit': self.min_balance - self.INITIAL_BALANCE,
                'max_profit': self.peak_balance - self.INITIAL_BALANCE,
                'max_daily_dd_pct': self.max_daily_dd_pct,
                # v7.0.2: correlation simulator telemetry
                'correlation_forced_skips': self._correlation_forced_skips,
                # v8.0.55: entry confirmation forced skips
                'entry_confirm_forced_skips': self._entry_confirm_forced_skips,
            }

        # v7.0.2: per-step flag สำหรับ debug logger ภายนอก
        info['correlation_forced_skip'] = correlation_forced_skip
        # v8.0.55: per-step flag for entry confirmation skip
        info['entry_confirm_forced_skip'] = entry_confirm_forced_skip
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_rng(self):
        if hasattr(self, 'np_random') and self.np_random is not None:
            return self.np_random
        return np.random.default_rng()

    def _dummy_signal(self) -> Dict:
        return {
            'day': 0,
            'signal_type': 'BUY',
            'confluence_score': 50.0,
            'rr_ratio': 1.5,
            'atr_value': 0.001,
            'atr_pips': 10.0,
            'ob_score': 0.0,
            'market_bias': 0,
            'trend': 0,
            'direction': 1.0,
            'bias_alignment': 0.0,
            'sl_distance_atr': 1.0,
            'outcome_pnl_ratio': -1.0,
            'pip_size': 0.0001,
            'rsi_value': 50.0,
            'trend_strength': 0.0,
            'macd_histogram': 0.0,
            'ob_size_atr': 0.0,
            'adx': 0.0,
            'stoch_k': 50.0,
            'bb_pctb': 0.5,
            'atr_change_ratio': 0.0,
            'price_roc': 0.0,
            'ml_score': 0.5,
        }

    def get_stats(self) -> Dict:
        total = self._total_takes + self._total_skips
        return {
            'balance': self.balance,
            'total_dd_pct': self.total_dd_pct,
            'daily_dd_pct': self.daily_dd_pct,
            'target_progress_pct': self.target_progress_pct,
            'total_takes': self._total_takes,
            'total_skips': self._total_skips,
            'take_rate': self._total_takes / max(total, 1),
            'total_trades': len(self._trade_results),
            'win_rate': (
                sum(1 for t in self._trade_results if t > 0) / len(self._trade_results)
                if self._trade_results else 0.0
            ),
        }

    def render(self, mode='console'):
        if mode in ('console', 'human'):
            stats = self.get_stats()
            print(
                f"Signal {self._signal_idx}/{len(self._signals)} | "
                f"Balance: ${self.balance:,.2f} | "
                f"DD: {self.total_dd_pct:.2%} | "
                f"Progress: {self.target_progress_pct:.1f}% | "
                f"Takes: {stats['total_takes']} Skips: {stats['total_skips']} | "
                f"WR: {stats['win_rate']:.0%}"
            )

    @staticmethod
    def obs_dim() -> int:
        return 27
