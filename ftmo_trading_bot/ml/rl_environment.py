"""
===============================================================================
FTMO Trading Bot — RL Environment (สภาพแวดล้อมฝึกซ้อมแบบ Real Backtest)
===============================================================================
Gymnasium Environment สำหรับ Optimize พารามิเตอร์ FTMO Bot
ใช้ข้อมูล OHLCV จริง (หรือข้อมูลจำลองที่สอดคล้องกับตลาด Forex จริง)
ในการจำลองการเทรดแบบ Day-by-day ตลอด 30 วันของ FTMO Challenge

ขั้นตอน 1 Episode = 1 FTMO Challenge (30 วัน)
ขั้นตอน 1 Step   = 1 วันเทรด

การทำงานของ step():
1. ถอด Action เป็นพารามิเตอร์เทรด (risk, confluence, atr_sl, rr)
2. ดึงข้อมูลตลาดของวันนั้น (ATR, Regime) จาก OHLCV จริงหรือจำลอง
3. คำนวณจำนวน Setup ที่จะเกิด (จาก confluence threshold)
4. คำนวณ Win Rate ที่เกิดขึ้นจริง (ตาม confluence, RR, ATR mult)
5. Simulate แต่ละเทรดตามผลลัพธ์เชิงสถิติ (ไม่ใช่ random ล้วน)
6. อัพเดท Balance, Drawdown, Progress
7. คำนวณ Reward ผ่าน FTMORewardCalculator
8. ตรวจสอบการจบ Episode (FTMO Fail / Pass / หมด 30 วัน)
===============================================================================
"""

import os
import io
import math
import contextlib
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional

from ml.reward_function import FTMORewardCalculator
from ml.strategy_backtester import StrategyBacktester


class FTMOOptimizationEnv(gym.Env):
    """
    Environment Gymnasium สำหรับเรียนรู้พารามิเตอร์ FTMO Challenge ที่ดีที่สุด

    ปรัชญา:
    - Agent ต้องเรียนรู้ว่า "เสี่ยงน้อยแต่สม่ำเสมอ" > "เสี่ยงเยอะครั้งเดียวแล้วพัง"
    - Reward Function ลงโทษ DD แบบ Exponential ใกล้ 4%/8%
    - Mini-Backtest ใช้ ATR และ Market Regime จริงเพื่อกำหนดความสำเร็จ
    """

    metadata = {'render_modes': ['human', 'console']}

    # ค่าคงที่สำหรับ FTMO Challenge
    INITIAL_BALANCE: float = 100_000.0
    CHALLENGE_DAYS: int = 45
    MAX_POSITIONS_PER_DAY: int = 3  # จำกัดที่ 3 ตาม bot_config.ftmo.MAX_OPEN_POSITIONS

    # Market Regime (สภาวะตลาด) — สัดส่วนถ่วงน้ำหนักจากการสังเกตตลาด Forex จริง
    # win_rate_base = Win Rate พื้นฐานของกลยุทธ์ SMC ภายใต้สภาวะนั้น
    # atr_mult      = ตัวคูณ ATR จากค่าเฉลี่ย (regime ผันผวนจะมีค่าสูง)
    # setups_per_day = จำนวน Setup เฉลี่ยที่ Confluence >= 50 ผ่านได้ต่อวัน
    # weight        = ความถี่ที่ regime นี้จะเกิด
    MARKET_PROFILES: Dict[str, Dict[str, float]] = {
        'trending':  {'win_rate_base': 0.55, 'atr_mult': 1.2, 'setups_per_day': 2.5, 'weight': 0.35},
        'ranging':   {'win_rate_base': 0.42, 'atr_mult': 0.7, 'setups_per_day': 3.5, 'weight': 0.40},
        'volatile':  {'win_rate_base': 0.48, 'atr_mult': 1.8, 'setups_per_day': 1.5, 'weight': 0.15},
        'quiet':     {'win_rate_base': 0.50, 'atr_mult': 0.4, 'setups_per_day': 0.5, 'weight': 0.10},
    }

    def __init__(
        self,
        data_dir: Optional[str] = None,
        excel_log_path: str = "logs/trading_log.xlsx",
        max_steps: int = 45,
        symbols: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.max_steps = min(int(max_steps), self.CHALLENGE_DAYS)
        self.excel_path = excel_log_path
        self.verbose = verbose  # False = ซ่อน log วิเคราะห์กราฟตอน train
        self.symbols = symbols or [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"
        ]

        # Reward function เฉพาะของ FTMO
        self.reward_calculator = FTMORewardCalculator()

        # โหลดข้อมูล OHLCV จากไฟล์ CSV (ถ้ามี)
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "ohlcv"
            )
        self._data_dir = data_dir
        self._load_ohlcv_data()

        # Strategy Backtester — ใช้ SMC Strategy จริงแทน statistical simulation
        # ซ่อน log ระหว่าง init ถ้า verbose=False
        try:
            with self._suppress_stdout():
                self._backtester = StrategyBacktester(data_dir, self.symbols)
            if self._backtester.is_available:
                print(f"🎯 [RL Env] Strategy Backtester พร้อม — ฝึกจาก SMC Strategy จริง")
            else:
                self._backtester = None
                print(f"📊 [RL Env] ไม่มีข้อมูล 3 TFs — ใช้ statistical simulation")
        except Exception as e:
            self._backtester = None
            print(f"⚠️ [RL Env] Backtester init ล้มเหลว ({e}) — fallback statistical")

        # โหลดประวัติเทรดจริง (สำหรับคำนวณ win-rate เชิงสถิติ)
        self._trade_history: Optional[pd.DataFrame] = None
        self._load_trade_history()

        # ==========================================================
        # Observation Space (16 ค่า, Clip ในช่วง [-5, 5]) — V3: T16 expanded
        # ==========================================================
        # Portfolio metrics:
        # [0] total_dd_norm        : Total DD / 10%             (negative value)
        # [1] daily_dd_norm        : Daily DD / 5%              (negative value)
        # [2] progress_norm        : Progress toward 10%        (0 to ~1.0)
        # [3] sortino_ratio        : Risk-adjusted return       (Clip ±5)
        # [4] trades_today_ratio   : จำนวนเทรดวันนี้ / 5    (0 to 1)
        # [5] volatility_norm      : ATR Regime level           (-2 to 2)
        # [6] day_progress         : current_step / 30          (0 to 1)
        # [7] recent_win_rate      : Win Rate ล่าสุด            (-1 to 1)
        # Market awareness (V2):
        # [8]  regime_trend_norm   : +1 trending, -1 ranging, 0 volatile/quiet
        # [9]  atr_zscore          : (atr - rolling_mean) / std  (Clip ±3)
        # [10] day_of_week_norm    : (step % 5) / 4              (0 = Mon, 1 = Fri)
        # [11] regime_consistency  : เศษของ 5 วันล่าสุดที่ regime ตรงกัน (0 to 1)
        # [12] cumulative_pnl_norm : (balance/INITIAL - 1)       (Clip ±0.15)
        # T16 expanded features (V3):
        # [13] avg_rr_achieved     : empirical RR ratio จากเทรดล่าสุด (0 to 3)
        # [14] dd_velocity         : อัตราเปลี่ยนแปลง DD ใน 3 วันล่าสุด (-1 to 1)
        # [15] profit_factor       : total_wins / total_losses normalized (0 to 3)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(16,), dtype=np.float32
        )

        # ==========================================================
        # Action Space (5 ค่า continuous, [-1, 1]) — V3: T16 expanded
        # ==========================================================
        # [0] risk_per_trade_pct        : [-1, 1] → [0.15%, 0.5%]
        # [1] min_confluence_score      : [-1, 1] → [65, 85]
        # [2] atr_sl_multiplier         : [-1, 1] → [1.2, 2.5]
        # [3] preferred_risk_reward     : [-1, 1] → [1.5, 4.0]
        # [4] max_daily_trades          : [-1, 1] → [1, 3]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # สถานะภายใน (รีเซ็ตใน reset())
        self._reset_state()

    # =========================================================================
    # Utility
    # =========================================================================

    def _suppress_stdout(self):
        """Context manager: ซ่อน print ถ้า verbose=False (ใช้ตอน train)"""
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _load_ohlcv_data(self):
        """โหลดข้อมูล OHLCV จากไฟล์ CSV (ถ้ามี) — ข้อมูลจริงทำให้ training สมจริง"""
        if not os.path.isdir(self._data_dir):
            return

        for symbol in self.symbols:
            # รองรับหลายรูปแบบชื่อไฟล์: EURUSD_M15.csv, EURUSD_H1.csv, EURUSD.csv
            patterns = [
                os.path.join(self._data_dir, f"{symbol}_M15.csv"),
                os.path.join(self._data_dir, f"{symbol}_H1.csv"),
                os.path.join(self._data_dir, f"{symbol}_H4.csv"),
                os.path.join(self._data_dir, f"{symbol}.csv"),
            ]
            for filepath in patterns:
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        # Normalize column names เป็นตัวเล็ก
                        df.columns = [c.lower().strip() for c in df.columns]
                        required = {'open', 'high', 'low', 'close'}
                        if required.issubset(set(df.columns)) and len(df) >= 100:
                            self._ohlcv_cache[symbol] = df
                            break
                    except Exception:
                        continue

    def _load_trade_history(self):
        """โหลดประวัติเทรดจริงจาก Excel (ถ้ามี) สำหรับ calibrate win rate"""
        if os.path.exists(self.excel_path):
            try:
                df = pd.read_excel(self.excel_path, sheet_name='Trades')
                if len(df) >= 10:  # ต้องมีเทรดพอเพื่อให้ได้ sample ที่มีความหมาย
                    self._trade_history = df
            except Exception:
                self._trade_history = None

    @property
    def has_real_data(self) -> bool:
        """มีข้อมูล OHLCV จริงหรือไม่"""
        return len(self._ohlcv_cache) > 0

    @property
    def has_trade_history(self) -> bool:
        """มีประวัติเทรดจริงหรือไม่"""
        return self._trade_history is not None and len(self._trade_history) >= 10

    def _get_historical_win_rate(self) -> Optional[float]:
        """คำนวณ win rate จากประวัติเทรดจริง (ถ้ามี) เพื่อใช้ calibrate simulation"""
        if not self.has_trade_history:
            return None
        try:
            profits = pd.to_numeric(self._trade_history.get('profit', []), errors='coerce').dropna()
            if len(profits) < 10:
                return None
            wins = (profits > 0).sum()
            return float(wins / len(profits))
        except Exception:
            return None

    # =========================================================================
    # Market Regime Generator
    # =========================================================================

    def _generate_market_sequence(self):
        """สุ่ม Market Regime สำหรับทั้ง Episode (30 วัน)"""
        regimes = list(self.MARKET_PROFILES.keys())
        weights = np.array([self.MARKET_PROFILES[r]['weight'] for r in regimes])
        weights = weights / weights.sum()  # Normalize

        rng = self._get_rng()
        self._daily_market_regimes = rng.choice(
            regimes, size=self.max_steps, p=weights
        ).tolist()

    def _pick_episode_sequence(self):
        """T14/T15: เลือก symbol + ตำแหน่งเริ่มต้นสำหรับ chronological episode"""
        rng = self._get_rng()
        candles_per_day = 96  # M15

        # ลำดับความสำคัญ: backtester sequential > OHLCV cache > synthetic
        if self._backtester is not None:
            seq_symbols = self._backtester.get_sequential_symbols(self.max_steps)
            if seq_symbols:
                self._episode_symbol = seq_symbols[int(rng.integers(0, len(seq_symbols)))]
                m15_len = len(self._backtester._m15_cache[self._episode_symbol])
                needed = self._backtester.get_min_bars_for_episode(self.max_steps)
                max_start = m15_len - needed
                if max_start > 0:
                    self._episode_start_bar = int(rng.integers(
                        self._backtester.MIN_M15_BARS, max_start
                    ))
                    return

        if self.has_real_data:
            symbols_avail = list(self._ohlcv_cache.keys())
            needed = candles_per_day * (self.max_steps + 1) + 20
            valid_symbols = [s for s in symbols_avail if len(self._ohlcv_cache[s]) >= needed]
            if valid_symbols:
                self._episode_symbol = valid_symbols[int(rng.integers(0, len(valid_symbols)))]
                max_start = len(self._ohlcv_cache[self._episode_symbol]) - needed
                if max_start > 20:
                    self._episode_start_bar = int(rng.integers(20, max_start))
                    return

        self._episode_symbol = None
        self._episode_start_bar = None

    def _get_rng(self):
        """คืน random number generator (ใช้ของ gym ถ้ามี seed)"""
        if hasattr(self, 'np_random') and self.np_random is not None:
            return self.np_random
        return np.random.default_rng()

    def _get_day_market_data(self, day_index: int) -> Dict:
        """
        ดึงข้อมูลตลาดสำหรับวันที่ระบุ
        T14/T15: ใช้ chronological mode (same symbol, sequential bars) ถ้ามี
        Fallback: random sampling หรือ synthetic data
        """
        regime = self._daily_market_regimes[day_index]
        profile = self.MARKET_PROFILES[regime]
        rng = self._get_rng()
        candles_per_day = 96

        # T14/T15: Chronological mode — ใช้ episode symbol + sequential bars
        if self._episode_symbol and self._episode_start_bar is not None:
            df = self._ohlcv_cache.get(self._episode_symbol)
            if df is not None:
                start = self._episode_start_bar + day_index * candles_per_day
                end = start + candles_per_day
                if end <= len(df):
                    day_df = df.iloc[start:end]
                    result = self._compute_market_data_from_df(
                        day_df, self._episode_symbol, regime, profile
                    )
                    if result is not None:
                        return result

        # Fallback: random sampling (เดิม)
        if self.has_real_data:
            symbols_avail = list(self._ohlcv_cache.keys())
            symbol = symbols_avail[rng.integers(0, len(symbols_avail))]
            df = self._ohlcv_cache[symbol]

            if len(df) > candles_per_day + 20:
                start = int(rng.integers(20, len(df) - candles_per_day))
                day_df = df.iloc[start:start + candles_per_day]
                result = self._compute_market_data_from_df(
                    day_df, symbol, regime, profile
                )
                if result is not None:
                    return result

        # Fallback: Synthetic Data
        base_atr = float(rng.normal(12.0, 4.0))
        base_atr = max(3.0, min(base_atr, 40.0))
        atr_pips = base_atr * profile['atr_mult']

        return {
            'regime': regime,
            'atr_pips': atr_pips,
            'daily_range_pips': atr_pips * float(rng.uniform(3.0, 6.0)),
            'base_win_rate': profile['win_rate_base'],
            'base_setups': profile['setups_per_day'],
            'pip_size': 0.0001,
            'price': 1.1000,
            'symbol': 'SYNTHETIC',
            'data_source': 'synthetic',
        }

    def _compute_market_data_from_df(
        self, day_df: pd.DataFrame, symbol: str, regime: str, profile: Dict
    ) -> Optional[Dict]:
        """คำนวณ market data จาก DataFrame ของ 1 วัน"""
        if len(day_df) < 10:
            return None

        high = day_df['high'].values
        low = day_df['low'].values
        close = day_df['close'].values

        high_low = high - low
        high_prev = np.abs(high[1:] - close[:-1])
        low_prev = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(high_low[1:], high_prev), low_prev)
        atr = float(np.mean(tr)) if len(tr) > 0 else 0.0001

        mean_price = float(np.mean(close))
        pip_size = 0.01 if mean_price > 50 else 0.0001
        atr_pips = atr / pip_size
        day_range_pips = (float(high.max()) - float(low.min())) / pip_size

        return {
            'regime': regime,
            'atr_pips': max(3.0, min(atr_pips, 80.0)),
            'daily_range_pips': max(atr_pips * 2, day_range_pips),
            'base_win_rate': profile['win_rate_base'],
            'base_setups': profile['setups_per_day'],
            'pip_size': pip_size,
            'price': mean_price,
            'symbol': symbol,
            'data_source': 'real',
        }

    # =========================================================================
    # Trading Simulation (Mini-Backtest Engine)
    # =========================================================================

    def _simulate_trading_day(self, params: Dict, market_data: Dict) -> Dict:
        """
        จำลองการเทรด 1 วันด้วยพารามิเตอร์และสภาพตลาดที่กำหนด

        กลไก:
        1. คำนวณจำนวน Setup ที่จะผ่าน Confluence filter
        2. คำนวณ Win Rate ที่ adjusted จากพารามิเตอร์
        3. สุ่ม outcome แต่ละเทรดตาม probability
        4. ติดตาม Intraday Drawdown (กรณี loss ก่อน win ในวัน)

        Returns:
            Dict ที่มี pnl, trades_taken, wins, losses, max_intraday_dd, trades (list of PnLs)
        """
        rng = self._get_rng()

        risk_pct = params['risk_per_trade_pct']
        min_confluence = params['min_confluence_score']
        atr_sl_mult = params['atr_sl_multiplier']
        rr_ratio = params['preferred_risk_reward_ratio']

        # T16: Agent ควบคุมจำนวนเทรดต่อวันได้
        max_trades = params.get('max_daily_trades', self.MAX_POSITIONS_PER_DAY)

        # === 1. จำนวน Setup ที่ผ่าน Confluence ===
        confluence_filter = max(0.1, 1.0 - (min_confluence - 50) / 40.0)
        expected_setups = market_data['base_setups'] * confluence_filter

        # Poisson-distributed จำนวนเทรด (capped at max_trades from agent)
        num_trades = int(min(
            max_trades,
            rng.poisson(max(0.05, expected_setups))
        ))

        # Friday late-session hint: ลด setups ลง 50% ใน "Friday" ของ episode
        # (step % 5 == 4 → วันสุดท้ายของสัปดาห์เทรด)
        # เหตุผล: live bot มี friday_cutoff 15:00 UTC + force_close 20:45 EET
        # → ฝึก agent ให้คุ้นชินว่า Friday เทรดน้อย ป้องกัน weekend gap
        if (self.current_step - 1) % 5 == 4:
            num_trades = int(num_trades * 0.5)

        # === 2. คำนวณ Win Rate ที่ปรับแล้ว ===
        base_wr = market_data['base_win_rate']

        # ถ้ามีประวัติเทรดจริง ให้ blend กับ base rate เพื่อ ground truth
        hist_wr = self._get_historical_win_rate()
        if hist_wr is not None:
            base_wr = 0.5 * base_wr + 0.5 * hist_wr

        # Confluence สูง → คุณภาพ setup ดี → WR สูง (+0.0 ถึง +0.15)
        confluence_bonus = ((min_confluence - 50) / 30.0) * 0.15
        # RR สูง → TP ไกล → WR ต่ำ (ตาม Random Walk Theory)
        rr_penalty = ((rr_ratio - 1.5) / 1.5) * 0.12
        # ATR SL Multiplier สูง → SL กว้าง → โดน Stopout น้อยลง (+0.0 ถึง +0.05)
        sl_bonus = ((atr_sl_mult - 1.0) / 1.5) * 0.05

        win_rate = base_wr + confluence_bonus - rr_penalty + sl_bonus
        win_rate = float(np.clip(win_rate, 0.20, 0.75))

        # === 3. Simulate แต่ละเทรด ===
        trades: List[float] = []
        wins = 0
        losses = 0
        running_pnl = 0.0
        max_intraday_dd = 0.0
        min_running = 0.0

        for _ in range(num_trades):
            risk_amount = self.balance * risk_pct

            if rng.random() < win_rate:
                # WIN: ได้ RR × risk (อาจเลื่อน Partial TP → ได้ไม่เต็ม)
                # Actual RR มี variance เพราะ Partial Close (50% ที่ 1:1) + Trailing
                # Expected RR จริง ≈ 0.5 × 1.0 + 0.5 × rr_ratio × (0.8 ~ 1.1)
                effective_rr = 0.5 * 1.0 + 0.5 * rr_ratio * float(rng.uniform(0.8, 1.1))
                pnl = risk_amount * effective_rr
                wins += 1
                self.last_trade_result = 1
            else:
                # LOSS: เสียใกล้ risk amount
                # 15% โอกาสที่ Break-even move จะปกป้อง (SL moved to BE+1pip)
                if rng.random() < 0.15:
                    pnl = -risk_amount * float(rng.uniform(0.0, 0.3))
                else:
                    # SL โดน + slippage 0-5%
                    slippage = float(rng.uniform(1.0, 1.05))
                    pnl = -risk_amount * slippage
                losses += 1
                self.last_trade_result = -1

            trades.append(pnl)
            running_pnl += pnl
            # Track intraday drawdown (worst negative excursion during the day)
            min_running = min(min_running, running_pnl)

        max_intraday_dd = abs(min_running)
        daily_pnl = sum(trades)

        return {
            'pnl': daily_pnl,
            'trades_taken': num_trades,
            'wins': wins,
            'losses': losses,
            'max_intraday_dd': max_intraday_dd,
            'trades': trades,
            'win_rate': win_rate,
            'regime': market_data['regime'],
            'atr_pips': market_data['atr_pips'],
        }

    # =========================================================================
    # State Management
    # =========================================================================

    def _reset_state(self):
        """รีเซ็ตสถานะภายในทั้งหมด"""
        self.current_step = 0
        self.balance = self.INITIAL_BALANCE
        self.peak_balance = self.INITIAL_BALANCE
        self.daily_start_balance = self.INITIAL_BALANCE

        self.total_dd_pct = 0.0
        self.daily_dd_pct = 0.0
        self.target_progress_pct = 0.0

        self.trade_results: List[float] = []
        self.daily_pnl_history: List[float] = []
        self.last_trade_result = 0
        self.current_volatility = 0.0

        self._daily_market_regimes: List[str] = []
        self._regime_history: List[str] = []
        self._atr_history: List[float] = []
        self._dd_history: List[float] = []      # T16: DD history สำหรับ dd_velocity
        self._win_amounts: List[float] = []     # T16: win PnLs สำหรับ avg_rr
        self._loss_amounts: List[float] = []    # T16: loss PnLs สำหรับ avg_rr
        self._generate_market_sequence()

        # T14/T15: Chronological mode — เลือก symbol + start bar สำหรับทั้ง episode
        self._episode_symbol: Optional[str] = None
        self._episode_start_bar: Optional[int] = None
        self._pick_episode_sequence()

    def _compute_sortino(self) -> float:
        """คำนวณ Sortino Ratio (rolling — จาก daily returns ที่ผ่านมา)"""
        if len(self.daily_pnl_history) < 2:
            return 0.0

        returns = np.array(self.daily_pnl_history) / self.INITIAL_BALANCE
        neg_returns = returns[returns < 0]
        mean_return = float(np.mean(returns))

        if len(neg_returns) == 0:
            return 3.0 if mean_return > 0 else 0.0  # ไม่มี downside = ดีมาก

        downside_std = float(np.std(neg_returns))
        if downside_std < 1e-6:
            return 3.0 if mean_return > 0 else 0.0

        # Annualized (252 trading days)
        sortino = mean_return / downside_std * math.sqrt(252)
        return float(np.clip(sortino, -5.0, 5.0))

    def _get_stats(self) -> Dict:
        """คืน stats dict สำหรับ Reward Calculator"""
        trading_days = sum(1 for p in self.daily_pnl_history if p != 0.0)

        consecutive_losses = 0
        for t in reversed(self.trade_results[-5:]):
            if t < 0:
                consecutive_losses += 1
            else:
                break

        # T12: คำนวณ recent win rate (10 เทรดล่าสุด)
        recent = self.trade_results[-10:] if self.trade_results else []
        if len(recent) > 0:
            recent_win_rate = sum(1 for t in recent if t > 0) / len(recent)
        else:
            recent_win_rate = 0.5

        return {
            'daily_dd_pct': self.daily_dd_pct,
            'total_dd_pct': self.total_dd_pct,
            'sortino_ratio': self._compute_sortino(),
            'target_progress_pct': self.target_progress_pct,
            'balance': self.balance,
            'last_trade_win': self.last_trade_result,
            'trades_today': getattr(self, 'trades_today', 0),
            'trading_days': trading_days,
            'consecutive_losses': consecutive_losses,
            'is_final_step': self.current_step >= self.max_steps,
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'intraday_excursion_pct': getattr(self, 'intraday_excursion_pct', 0.0),
            'day_end_loss_pct': getattr(self, 'day_end_loss_pct', 0.0),
            'recent_win_rate': recent_win_rate,
        }

    def _get_obs(self) -> np.ndarray:
        """สร้าง Observation Array จากสถานะปัจจุบัน (16 dims — V3 T16)"""
        # Recent win rate (last 10 trades) → normalize to [-1, 1]
        recent = self.trade_results[-10:] if self.trade_results else []
        if len(recent) > 0:
            win_pct = sum(1 for t in recent if t > 0) / len(recent)
            recent_wr_norm = win_pct * 2.0 - 1.0
        else:
            recent_wr_norm = 0.0

        # === Market awareness features (V2) ===
        regime_map = {'trending': 1.0, 'ranging': -1.0, 'volatile': 0.0, 'quiet': 0.0}
        last_regime = self._regime_history[-1] if self._regime_history else 'ranging'
        regime_trend_norm = regime_map.get(last_regime, 0.0)

        if len(self._atr_history) >= 5:
            atr_arr = np.array(self._atr_history[-20:])
            mu = float(atr_arr.mean())
            sd = float(atr_arr.std()) or 1.0
            atr_zscore = (self._atr_history[-1] - mu) / sd
        else:
            atr_zscore = 0.0

        day_of_week_norm = (self.current_step % 5) / 4.0

        if len(self._regime_history) >= 2:
            recent_regimes = self._regime_history[-5:]
            most_common = max(set(recent_regimes), key=recent_regimes.count)
            regime_consistency = recent_regimes.count(most_common) / len(recent_regimes)
        else:
            regime_consistency = 0.0

        cum_pnl_norm = (self.balance / self.INITIAL_BALANCE) - 1.0

        # === T16 expanded features (V3) ===
        # [13] avg_rr_achieved: empirical RR = avg_win / avg_loss
        if self._win_amounts and self._loss_amounts:
            avg_win = np.mean(self._win_amounts[-20:])
            avg_loss = np.mean(self._loss_amounts[-20:])
            avg_rr = avg_win / max(avg_loss, 1.0)
        else:
            avg_rr = 0.0

        # [14] dd_velocity: อัตราเปลี่ยนแปลง DD ใน 3 วันล่าสุด
        if len(self._dd_history) >= 3:
            dd_velocity = self._dd_history[-1] - self._dd_history[-3]
        elif len(self._dd_history) >= 2:
            dd_velocity = self._dd_history[-1] - self._dd_history[-2]
        else:
            dd_velocity = 0.0

        # [15] profit_factor: total_wins / total_losses
        total_wins = sum(self._win_amounts) if self._win_amounts else 0.0
        total_losses = sum(self._loss_amounts) if self._loss_amounts else 0.0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = 2.0 if total_wins > 0 else 0.0

        obs = np.array([
            float(np.clip(-self.total_dd_pct / 0.10, -5.0, 0.0)),
            float(np.clip(-self.daily_dd_pct / 0.05, -5.0, 0.0)),
            float(np.clip(self.target_progress_pct / 100.0, -1.0, 2.0)),
            float(np.clip(self._compute_sortino(), -5.0, 5.0)),
            float(min(getattr(self, 'trades_today', 0), 5) / 5.0),
            float(np.clip(self.current_volatility, -2.0, 2.0)),
            float(self.current_step / max(1, self.max_steps)),
            float(np.clip(recent_wr_norm, -1.0, 1.0)),
            float(regime_trend_norm),
            float(np.clip(atr_zscore, -3.0, 3.0)),
            float(day_of_week_norm),
            float(regime_consistency),
            float(np.clip(cum_pnl_norm, -0.15, 0.15)),
            # T16 expanded:
            float(np.clip(avg_rr, 0.0, 3.0)),
            float(np.clip(dd_velocity, -1.0, 1.0)),
            float(np.clip(profit_factor, 0.0, 3.0)),
        ], dtype=np.float32)

        return obs

    # =========================================================================
    # Gym Interface
    # =========================================================================

    def reset(self, seed=None, options=None):
        """เริ่ม Episode ใหม่ (FTMO Challenge 30 วันใหม่)"""
        super().reset(seed=seed)
        self._reset_state()

        info = {
            'data_source': 'real' if self.has_real_data else 'synthetic',
            'has_trade_history': self.has_trade_history,
            'initial_balance': self.INITIAL_BALANCE,
            'challenge_days': self.max_steps,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        """
        ดำเนินการ 1 วันเทรด (1 step)

        Args:
            action: np.ndarray shape (5,) ใน [-1, 1] (V3 T16)

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        previous_stats = self._get_stats()

        # 1. ถอด Action → Parameters
        params = self.map_actions_to_parameters(action)

        # 2. ดึงข้อมูลตลาดของวันนี้
        market_data = self._get_day_market_data(self.current_step - 1)
        self.current_volatility = (market_data['atr_pips'] - 12.0) / 6.0

        self._regime_history.append(market_data['regime'])
        self._atr_history.append(float(market_data['atr_pips']))

        # 3. Simulate trading day
        if params['risk_per_trade_pct'] <= 0.0001:
            day_result = {
                'pnl': 0.0, 'trades_taken': 0, 'wins': 0, 'losses': 0,
                'max_intraday_dd': 0.0, 'trades': [], 'win_rate': 0.0,
                'regime': market_data['regime'], 'atr_pips': market_data['atr_pips'],
            }
        else:
            with self._suppress_stdout():
                # T14/T15: ใช้ sequential backtester ถ้ามี episode symbol
                if (self._backtester is not None
                        and self._episode_symbol is not None
                        and self._episode_start_bar is not None
                        and self._episode_symbol in self._backtester._available_symbols):
                    m15_day_start = (
                        self._episode_start_bar
                        + (self.current_step - 1) * 96
                    )
                    day_result = self._backtester.simulate_day_sequential(
                        params, self._episode_symbol, m15_day_start, self._get_rng()
                    )
                elif self._backtester is not None:
                    day_result = self._backtester.simulate_day_with_strategy(
                        params, self.current_step - 1, self._get_rng()
                    )
                else:
                    day_result = self._simulate_trading_day(params, market_data)

        # 4. อัพเดทสถานะ
        self.daily_start_balance = self.balance
        self.balance += day_result['pnl']
        self.trade_results.extend(day_result['trades'])
        self.trades_today = len(day_result.get('trades', []))
        self.daily_pnl_history.append(day_result['pnl'])

        # T16: Track win/loss amounts สำหรับ avg_rr + profit_factor obs
        for t in day_result.get('trades', []):
            if t > 0:
                self._win_amounts.append(abs(t))
            elif t < 0:
                self._loss_amounts.append(abs(t))

        # Peak balance (ใช้สำหรับคำนวณ Total DD from peak)
        self.peak_balance = max(self.peak_balance, self.balance)

        # Total DD จาก peak (อิง initial balance ตามกฎ FTMO)
        # FTMO: Max DD = 10% ของ balance เริ่มต้น (ไม่ใช่ peak)
        total_loss = self.INITIAL_BALANCE - self.balance
        self.total_dd_pct = max(0.0, total_loss / self.INITIAL_BALANCE)
        self._dd_history.append(self.total_dd_pct)  # T16: dd_velocity obs

        # Daily DD (worst case: end-of-day + intraday excursion)
        # ใช้ daily_start_balance เป็น denominator ให้ตรงกับ RiskManager live
        # (FTMO วัด daily loss เทียบกับ equity ต้นวัน ไม่ใช่ initial balance)
        day_end_loss = max(0.0, self.daily_start_balance - self.balance)
        intraday_dd = day_result['max_intraday_dd']
        daily_denom = max(self.daily_start_balance, 1.0)
        self.daily_dd_pct = max(day_end_loss, intraday_dd) / daily_denom

        # เก็บ intraday excursion เพื่อใช้ใน L2 swing penalty (ปิดช่อง "hold loser แล้วโชคดี")
        self.intraday_excursion_pct = intraday_dd / daily_denom
        self.day_end_loss_pct = day_end_loss / daily_denom

        # Target progress (% ของเป้า 10%)
        profit = self.balance - self.INITIAL_BALANCE
        target_amount = self.INITIAL_BALANCE * self.reward_calculator.target
        self.target_progress_pct = (profit / target_amount) * 100.0 if target_amount > 0 else 0.0

        # 5. คำนวณ Reward
        current_stats = self._get_stats()
        reward = self.reward_calculator.calculate_reward(current_stats, previous_stats)

        # 6. ตรวจสอบการจบ Episode
        terminated = False
        truncated = False

        # FTMO Fail: Daily DD > 4% หรือ Total DD > 8%
        if self.daily_dd_pct >= self.reward_calculator.daily_limit:
            terminated = True
        elif self.total_dd_pct >= self.reward_calculator.total_limit:
            terminated = True
        # FTMO Pass: โตถึง 10%
        elif self.target_progress_pct >= 100.0:
            terminated = True

        # หมดวัน Challenge
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            'day': self.current_step,
            'balance': self.balance,
            'pnl': day_result['pnl'],
            'trades': day_result['trades_taken'],
            'wins': day_result['wins'],
            'losses': day_result['losses'],
            'regime': day_result['regime'],
            'atr_pips': day_result['atr_pips'],
            'total_dd_pct': self.total_dd_pct,
            'daily_dd_pct': self.daily_dd_pct,
            'progress_pct': self.target_progress_pct,
            'params': params,
            'win_rate': day_result['win_rate'],
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode='console'):
        """แสดงสถานะปัจจุบันใน Console"""
        if mode in ('console', 'human'):
            total_trades = len(self.trade_results)
            wins = sum(1 for t in self.trade_results if t > 0)
            wr = (wins / total_trades * 100) if total_trades > 0 else 0.0
            print(
                f"Day {self.current_step}/{self.max_steps} | "
                f"Balance: ${self.balance:,.2f} | "
                f"Total DD: {self.total_dd_pct:.2%} | "
                f"Daily DD: {self.daily_dd_pct:.2%} | "
                f"Progress: {self.target_progress_pct:.1f}% | "
                f"WR: {wr:.0f}% ({total_trades}t)"
            )

    # =========================================================================
    # Action → Parameters Mapping (Static — ไม่ต้องสร้าง env instance ใหม่)
    # =========================================================================

    @staticmethod
    def map_actions_to_parameters(action: np.ndarray) -> Dict:
        """
        แปลง Action Space [-1, 1] → Trading Parameters จริง (V3: 5 actions)

        Args:
            action: ndarray shape (5,) ใน [-1, 1]

        Returns:
            Dict ของ 5 parameters พร้อมปัดทศนิยมเรียบร้อย
        """
        a = np.asarray(action, dtype=np.float32).flatten()
        if a.size < 5:
            a = np.concatenate([a, np.zeros(5 - a.size, dtype=np.float32)])

        val_risk = 0.00325 + float(a[0]) * 0.00175    # [-1, 1] → [0.15%, 0.5%]
        val_conf = 75.0 + float(a[1]) * 10.0         # [-1, 1] → [65, 85]
        val_atr = 1.85 + float(a[2]) * 0.65          # [-1, 1] → [1.2, 2.5]
        val_rr = 2.75 + float(a[3]) * 1.25           # [-1, 1] → [1.5, 4.0]
        val_max_trades = 2.0 + float(a[4]) * 1.0     # [-1, 1] → [1, 3]

        val_risk = max(0.0015, min(val_risk, 0.005))
        val_conf = max(65.0, min(val_conf, 85.0))
        val_atr = max(1.2, min(val_atr, 2.5))
        val_rr = max(1.5, min(val_rr, 4.0))
        val_max_trades = max(1, min(int(round(val_max_trades)), 3))

        return {
            'risk_per_trade_pct': round(val_risk, 4),
            'min_confluence_score': int(round(val_conf)),
            'atr_sl_multiplier': round(val_atr, 1),
            'preferred_risk_reward_ratio': round(val_rr, 1),
            'max_daily_trades': val_max_trades,
        }
