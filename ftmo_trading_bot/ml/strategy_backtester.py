"""
===============================================================================
FTMO Trading Bot — Strategy Backtester (Backtest Engine สำหรับ RL Training)
===============================================================================
เชื่อม SMC Strategy เข้ากับ RL Environment เพื่อให้ agent ฝึกจากผลเทรดจริง
แทนที่จะใช้ win rate สุ่ม

วิธีการ:
1. โหลด OHLCV 3 timeframes (M15, H1, H4) จากไฟล์ CSV
2. เลื่อน window ไปข้างหน้าวันละ 96 แท่ง (M15 = 1 วัน)
3. เรียก SMCStrategy.analyze_with_data() เพื่อให้ได้ TradeSignal จริง
4. จำลองผลเทรดจาก signal (entry, SL, TP) กับราคาจริงที่ตามมา
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from strategy.smc_strategy import SMCStrategy
from strategy.indicators import TechnicalIndicators


class _MockConnector:
    """Mock connector สำหรับ backtesting — ไม่ต้องต่อ MT5"""

    def get_symbol_info(self, symbol: str) -> dict:
        is_jpy = "JPY" in symbol.upper()
        return {
            "digits": 3 if is_jpy else 5,
            "point": 0.001 if is_jpy else 0.00001,
            "lot_min": 0.01,
            "lot_max": 100.0,
            "lot_step": 0.01,
        }

    def get_current_price(self, symbol: str) -> None:
        return None

    def get_ohlcv(self, *args, **kwargs):
        return None


class StrategyBacktester:
    """
    Backtest engine ที่ replay ข้อมูล OHLCV ผ่าน SMC Strategy จริง
    ออกแบบให้เรียกจาก RL Environment แทน _simulate_trading_day()
    """

    M15_PER_DAY = 96        # 24h × 4 = 96 แท่ง M15
    H1_PER_DAY = 24
    H4_PER_DAY = 6
    MIN_M15_BARS = 500      # strategy ต้องการอย่างน้อย 200 แท่ง
    MIN_H1_BARS = 500
    MIN_H4_BARS = 300

    def __init__(self, data_dir: str, symbols: Optional[List[str]] = None):
        self.symbols = symbols or [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"
        ]
        self._data_dir = data_dir

        self._m15_cache: Dict[str, pd.DataFrame] = {}
        self._h1_cache: Dict[str, pd.DataFrame] = {}
        self._h4_cache: Dict[str, pd.DataFrame] = {}
        self._load_data()

        self._strategy = SMCStrategy.__new__(SMCStrategy)
        self._init_strategy()

        self._available_symbols = [
            s for s in self.symbols
            if s in self._m15_cache and s in self._h1_cache and s in self._h4_cache
        ]

    def _load_data(self):
        """โหลด OHLCV จาก CSV สำหรับ 3 timeframes"""
        if not os.path.isdir(self._data_dir):
            return

        for symbol in self.symbols:
            for tf, cache in [("M15", self._m15_cache), ("H1", self._h1_cache), ("H4", self._h4_cache)]:
                filepath = os.path.join(self._data_dir, f"{symbol}_{tf}.csv")
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        df.columns = [c.lower().strip() for c in df.columns]
                        required = {'open', 'high', 'low', 'close'}
                        if required.issubset(set(df.columns)) and len(df) >= 100:
                            if 'time' not in df.columns:
                                df['time'] = range(len(df))
                            if 'volume' not in df.columns:
                                df['volume'] = 0
                            cache[symbol] = df
                    except Exception:
                        continue

    def _init_strategy(self):
        """สร้าง SMCStrategy โดยไม่ต้องมี MT5 connector"""
        self._strategy._connector = _MockConnector()
        self._strategy._indicators = TechnicalIndicators()

        from strategy.market_structure import MarketStructure
        from strategy.order_blocks import OrderBlockDetector
        from strategy.fair_value_gaps import FVGDetector
        from strategy.liquidity_sweeps import LiquiditySweepDetector

        self._strategy._structure_mtf = MarketStructure()
        self._strategy._structure_ltf = MarketStructure()
        self._strategy._ob_detector = OrderBlockDetector()
        self._strategy._fvg_detector = FVGDetector()
        self._strategy._sweep_detector = LiquiditySweepDetector()

        self._strategy._htf_data = None
        self._strategy._mtf_data = None
        self._strategy._ltf_data = None
        self._strategy._htf_bias = 0

        from config.settings import bot_config
        self._strategy.MIN_CONFLUENCE_SCORE = getattr(
            bot_config.ftmo, "MIN_CONFLUENCE_SCORE", 70.0
        )

    @property
    def is_available(self) -> bool:
        return len(self._available_symbols) > 0

    def get_min_bars_for_episode(self, max_steps: int = 45) -> int:
        """คำนวณจำนวน M15 bars ขั้นต่ำที่ต้องการสำหรับ 1 episode แบบ sequential"""
        return self.MIN_M15_BARS + max_steps * self.M15_PER_DAY + self.M15_PER_DAY + 48

    def get_sequential_symbols(self, max_steps: int = 45) -> List[str]:
        """คืน symbols ที่มีข้อมูลเพียงพอสำหรับ sequential episode"""
        min_bars = self.get_min_bars_for_episode(max_steps)
        return [
            s for s in self._available_symbols
            if len(self._m15_cache.get(s, [])) >= min_bars
        ]

    def simulate_day_sequential(
        self,
        params: Dict,
        symbol: str,
        m15_day_start: int,
        rng: np.random.Generator,
    ) -> Dict:
        """
        T14/T15: จำลอง 1 วันเทรดแบบ chronological — ใช้ symbol + ตำแหน่งเฉพาะ

        Args:
            params: trading parameters
            symbol: symbol ที่เลือกสำหรับ episode นี้
            m15_day_start: M15 bar index ที่เริ่มวันนี้ (ต้องมี lookback ก่อนหน้า)
            rng: random number generator
        """
        if symbol not in self._m15_cache:
            return self._empty_result()

        m15_df = self._m15_cache[symbol]
        h1_df = self._h1_cache.get(symbol)
        h4_df = self._h4_cache.get(symbol)

        if h1_df is None or h4_df is None:
            return self._empty_result()

        m15_start = max(0, m15_day_start - self.MIN_M15_BARS)
        m15_window_end = m15_day_start

        if m15_window_end >= len(m15_df) - self.M15_PER_DAY:
            return self._empty_result()

        h1_ratio = len(h1_df) / max(len(m15_df), 1)
        h4_ratio = len(h4_df) / max(len(m15_df), 1)
        h1_end = min(int(m15_window_end * h1_ratio), len(h1_df))
        h4_end = min(int(m15_window_end * h4_ratio), len(h4_df))
        h1_start = max(0, h1_end - self.MIN_H1_BARS)
        h4_start = max(0, h4_end - self.MIN_H4_BARS)

        if h1_end - h1_start < 200 or h4_end - h4_start < 200:
            return self._empty_result()

        return self._run_day_scan(
            params, symbol, m15_df, h1_df, h4_df,
            m15_start, m15_window_end, h1_start, h1_end, h4_start, h4_end, rng
        )

    def simulate_day_with_strategy(
        self,
        params: Dict,
        day_offset: int,
        rng: np.random.Generator,
    ) -> Dict:
        """
        จำลอง 1 วันเทรดด้วย SMC Strategy จริง (random sampling mode)

        Args:
            params: {risk_per_trade_pct, min_confluence_score, atr_sl_multiplier, preferred_risk_reward_ratio}
            day_offset: วันที่ใน episode (0-29)
            rng: random number generator

        Returns:
            Dict เหมือน _simulate_trading_day() เดิม:
            {pnl, trades_taken, wins, losses, max_intraday_dd, trades, win_rate, regime, atr_pips}
        """
        symbol = self._available_symbols[rng.integers(0, len(self._available_symbols))]

        m15_df = self._m15_cache[symbol]
        h1_df = self._h1_cache[symbol]
        h4_df = self._h4_cache[symbol]

        max_m15_start = len(m15_df) - self.MIN_M15_BARS - self.M15_PER_DAY
        if max_m15_start <= 0:
            return self._empty_result()

        m15_start = int(rng.integers(0, max(1, max_m15_start)))
        m15_window_end = m15_start + self.MIN_M15_BARS

        h1_ratio = len(h1_df) / max(len(m15_df), 1)
        h4_ratio = len(h4_df) / max(len(m15_df), 1)
        h1_end = min(int(m15_window_end * h1_ratio), len(h1_df))
        h4_end = min(int(m15_window_end * h4_ratio), len(h4_df))
        h1_start = max(0, h1_end - self.MIN_H1_BARS)
        h4_start = max(0, h4_end - self.MIN_H4_BARS)

        if h1_end - h1_start < 200 or h4_end - h4_start < 200:
            return self._empty_result()

        return self._run_day_scan(
            params, symbol, m15_df, h1_df, h4_df,
            m15_start, m15_window_end, h1_start, h1_end, h4_start, h4_end, rng
        )

    def _run_day_scan(
        self,
        params: Dict,
        symbol: str,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        m15_start: int,
        m15_window_end: int,
        h1_start: int,
        h1_end: int,
        h4_start: int,
        h4_end: int,
        rng: np.random.Generator,
    ) -> Dict:
        """Core scan logic — shared by random and sequential modes"""
        risk_pct = params['risk_per_trade_pct']
        min_confluence = params['min_confluence_score']
        atr_sl_mult = params['atr_sl_multiplier']
        rr_ratio = params['preferred_risk_reward_ratio']

        self._strategy.MIN_CONFLUENCE_SCORE = min_confluence

        trades: List[float] = []
        wins = 0
        losses = 0
        running_pnl = 0.0
        min_running = 0.0
        max_trades = params.get('max_daily_trades', 3)

        # สแกนหา signal 4 ครั้งต่อวัน (ทุก 24 แท่ง M15 = ทุก 6 ชั่วโมง)
        scan_points = [0, 24, 48, 72]

        for offset in scan_points:
            if len(trades) >= max_trades:
                break

            scan_idx = m15_start + self.MIN_M15_BARS + offset
            if scan_idx >= len(m15_df) - 20:
                continue

            ltf_slice = m15_df.iloc[scan_idx - self.MIN_M15_BARS + 1:scan_idx + 1].copy()
            h1_slice = h1_df.iloc[h1_start:h1_end].copy()
            h4_slice = h4_df.iloc[h4_start:h4_end].copy()

            if len(ltf_slice) < 200 or len(h1_slice) < 200 or len(h4_slice) < 200:
                continue

            last_close = float(ltf_slice["close"].iloc[-1])
            pip_size = 0.01 if last_close > 50 else 0.0001
            spread = pip_size * 2

            price_info = {
                "bid": last_close,
                "ask": last_close + spread,
                "spread": spread,
            }

            try:
                signal = self._strategy.analyze_with_data(
                    symbol, h4_slice, h1_slice, ltf_slice, price_info
                )
            except Exception:
                continue

            if not signal.is_valid:
                continue

            signal_sl = abs(signal.entry_price - signal.sl_price)
            if signal_sl < pip_size:
                continue

            actual_sl = signal_sl * atr_sl_mult / max(1.0, signal.atr_value / signal_sl)
            actual_tp = actual_sl * rr_ratio

            future_start = scan_idx + 1
            future_end = min(future_start + 48, len(m15_df))
            if future_start >= len(m15_df):
                continue

            future = m15_df.iloc[future_start:future_end]
            if len(future) == 0:
                continue

            balance_for_risk = 100_000.0
            risk_amount = balance_for_risk * risk_pct

            trade_pnl = self._resolve_trade(
                signal, actual_sl, actual_tp, future, risk_amount, pip_size, rng
            )

            trades.append(trade_pnl)
            running_pnl += trade_pnl
            min_running = min(min_running, running_pnl)

            if trade_pnl > 0:
                wins += 1
            else:
                losses += 1

        # คำนวณ ATR จาก M15 data
        m15_day = m15_df.iloc[m15_start:m15_window_end]
        high = m15_day['high'].values
        low = m15_day['low'].values
        close = m15_day['close'].values
        pip_size = 0.01 if float(np.mean(close)) > 50 else 0.0001

        if len(high) > 1:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            )
            atr = float(np.mean(tr))
        else:
            atr = pip_size * 10

        atr_pips = atr / pip_size

        # กำหนด regime จาก slope
        if len(close) >= 20:
            x = np.arange(20)
            y = close[-20:]
            slope = np.polyfit(x, y, 1)[0]
            norm_slope = slope / (atr + 1e-10)
            if abs(norm_slope) > 0.5:
                regime = 'trending'
            elif abs(norm_slope) < 0.1:
                regime = 'quiet'
            else:
                regime = 'ranging'
        else:
            regime = 'ranging'

        total = wins + losses
        return {
            'pnl': sum(trades),
            'trades_taken': len(trades),
            'wins': wins,
            'losses': losses,
            'max_intraday_dd': abs(min_running),
            'trades': trades,
            'win_rate': wins / total if total > 0 else 0.0,
            'regime': regime,
            'atr_pips': max(3.0, min(atr_pips, 80.0)),
        }

    def _resolve_trade(
        self, signal, sl_dist: float, tp_dist: float,
        future_df: pd.DataFrame, risk_amount: float,
        pip_size: float, rng: np.random.Generator,
    ) -> float:
        """
        จำลองผลเทรดจาก signal กับราคาอนาคต

        Fix: เมื่อแท่งเทียนชนทั้ง SL และ TP ในแท่งเดียวกัน
        ใช้ระยะทาง (entry → SL vs entry → TP) เทียบกับ bar open
        เพื่อประมาณว่าราคาชนอะไรก่อน แทนที่จะ bias เป็น SL เสมอ
        """
        entry = signal.entry_price
        is_buy = signal.signal_type.value == "BUY"

        if is_buy:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        for _, row in future_df.iterrows():
            bar_high = row['high']
            bar_low = row['low']
            bar_open = row['open']

            if is_buy:
                hit_sl = bar_low <= sl_price
                hit_tp = bar_high >= tp_price
            else:
                hit_sl = bar_high >= sl_price
                hit_tp = bar_low <= tp_price

            if hit_sl and hit_tp:
                # ทั้ง SL และ TP โดนในแท่งเดียวกัน — ประมาณจาก open direction
                # ถ้า open ใกล้ TP มากกว่า → น่าจะชน TP ก่อน (momentum ไปทาง TP)
                if is_buy:
                    dist_to_sl = abs(bar_open - sl_price)
                    dist_to_tp = abs(bar_open - tp_price)
                else:
                    dist_to_sl = abs(bar_open - sl_price)
                    dist_to_tp = abs(bar_open - tp_price)

                # TP ใกล้กว่า → ชน TP ก่อน, SL ใกล้กว่า → ชน SL ก่อน
                # เท่ากัน → 50/50 random
                if dist_to_tp < dist_to_sl:
                    hit_sl = False  # TP ก่อน
                elif dist_to_sl < dist_to_tp:
                    hit_tp = False  # SL ก่อน
                else:
                    if rng.random() < 0.5:
                        hit_sl = False
                    else:
                        hit_tp = False

            if hit_sl:
                slippage = float(rng.uniform(1.0, 1.05))
                return -risk_amount * slippage

            if hit_tp:
                partial_rr = 0.5 * 1.0 + 0.5 * (tp_dist / max(sl_dist, pip_size))
                return risk_amount * partial_rr * float(rng.uniform(0.85, 1.0))

        # ไม่ชนทั้ง SL/TP ใน 48 แท่ง → ปิดที่ราคาปัจจุบัน
        last_close = float(future_df['close'].iloc[-1])
        if is_buy:
            pnl_pips = (last_close - entry) / pip_size
        else:
            pnl_pips = (entry - last_close) / pip_size

        pnl_ratio = pnl_pips * pip_size / max(sl_dist, pip_size)
        return risk_amount * pnl_ratio

    def _empty_result(self) -> Dict:
        return {
            'pnl': 0.0,
            'trades_taken': 0,
            'wins': 0,
            'losses': 0,
            'max_intraday_dd': 0.0,
            'trades': [],
            'win_rate': 0.0,
            'regime': 'quiet',
            'atr_pips': 10.0,
        }
