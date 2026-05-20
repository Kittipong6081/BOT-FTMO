"""
===============================================================================
FTMO Trading Bot — Backtester Data Infrastructure (v8.0.6 cleanup)
===============================================================================
Base class for offline backtesters. Holds the multi-timeframe OHLCV cache,
indicator pre-computation, and the SL/TP/timeout/gap trade-resolution engine.

History — v8.0.6 (2026-05-07): SMC-specific code removed. The class previously
instantiated `SMCStrategy` and ran SMC-only `generate_episode_signals` /
`_run_day_scan` / `simulate_day_*` methods. With the v8.0 Mean Reversion
pivot those paths are dead. This file now contains only the data + outcome
machinery; strategy logic is the subclass's responsibility.

Subclasses must implement `generate_episode_signals(symbol, m15_start_bar,
num_days, rng)` — see `ml/mean_reversion_backtester.py` for the production
MR implementation.
===============================================================================
"""

import os
from datetime import time as dt_time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from strategy.indicators import TechnicalIndicators


class _MockConnector:
    """Mock connector for backtesting — no MT5 connection required."""

    def get_symbol_info(self, symbol: str) -> dict:
        sym_upper = symbol.upper()
        if "XAU" in sym_upper or "XAG" in sym_upper:
            return {
                "digits": 2, "point": 0.01,
                "lot_min": 0.01, "lot_max": 50.0, "lot_step": 0.01,
                "trade_contract_size": 100,
            }
        if "JPY" in sym_upper:
            return {
                "digits": 3, "point": 0.001,
                "lot_min": 0.01, "lot_max": 100.0, "lot_step": 0.01,
                "trade_contract_size": 100000,
            }
        return {
            "digits": 5, "point": 0.00001,
            "lot_min": 0.01, "lot_max": 100.0, "lot_step": 0.01,
            "trade_contract_size": 100000,
        }

    def get_current_price(self, symbol: str) -> None:
        return None

    def get_ohlcv(self, *args, **kwargs):
        return None


class StrategyBacktester:
    """Backtester base class — data loading, indicator precompute, trade resolution.

    Subclass and override `generate_episode_signals` to implement strategy-
    specific signal generation. The MR pipeline uses
    `ml.mean_reversion_backtester.MeanReversionBacktester(StrategyBacktester)`.
    """

    M15_PER_DAY = 96        # 24h × 4 = 96 M15 bars
    H1_PER_DAY = 24
    H4_PER_DAY = 6
    MIN_M15_BARS = 500
    MIN_H1_BARS = 500
    MIN_H4_BARS = 300

    # Per-symbol typical spread (pips) — used to simulate cost in env reward.
    _TYPICAL_SPREAD_PIPS: Dict[str, float] = {
        "EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.2, "AUDUSD": 1.5,
        "USDCAD": 2.0, "USDCHF": 2.0, "NZDUSD": 2.5,
        "EURJPY": 2.5, "GBPJPY": 5.0,
        "XAUUSD": 35.0,
    }

    def __init__(self, data_dir: str, symbols: Optional[List[str]] = None,
                 ml_model_path: Optional[str] = None):
        self.symbols = symbols or [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD",
        ]
        self._data_dir = data_dir

        # ML quality model (optional). Subclasses load MR-specific GBM
        # (`mr_signal_quality_model.pkl`) by overriding the lookup path.
        self._quality_model = None
        if ml_model_path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "mr_signal_quality_model.pkl",
            )
            if not os.path.exists(default_path):
                # Legacy fallback for SMC-era runs that still reference this path
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "signal_quality_model.pkl",
                )
            if os.path.exists(default_path):
                ml_model_path = default_path
        if ml_model_path and os.path.exists(ml_model_path):
            try:
                from ml.signal_quality import SignalQualityModel
                self._quality_model = SignalQualityModel(ml_model_path)
            except Exception as e:
                print(f"⚠️ [Backtester] ML quality model load failed: {e}")

        # Chronos forecaster (optional — disabled when BOT_DISABLE_CHRONOS=1)
        self._chronos = None
        try:
            from config.settings import bot_config as _bc
            if getattr(_bc.ml, "CHRONOS_ENABLED", True):
                from ml.chronos_forecaster import ChronosForecaster
                self._chronos = ChronosForecaster(
                    model_name=_bc.ml.CHRONOS_MODEL_NAME,
                    device=_bc.ml.CHRONOS_DEVICE,
                    prediction_length=_bc.ml.CHRONOS_PREDICTION_LENGTH,
                    context_length=_bc.ml.CHRONOS_CONTEXT_LENGTH,
                    verbose=1,
                )
        except Exception as e:
            print(f"⚠️ [Backtester] Chronos init failed: {e} → obs[27,28] = 0")
            self._chronos = None

        self._m15_cache: Dict[str, pd.DataFrame] = {}
        self._h1_cache: Dict[str, pd.DataFrame] = {}
        self._h4_cache: Dict[str, pd.DataFrame] = {}
        self._load_data()

        self._available_symbols = [
            s for s in self.symbols
            if s in self._m15_cache and s in self._h1_cache and s in self._h4_cache
        ]

        self._precompute_indicators()

    # ─── Data loading ──────────────────────────────────────────────────

    def _load_data(self):
        """Load OHLCV CSVs for M15/H1/H4 across all symbols."""
        if not os.path.isdir(self._data_dir):
            return

        for symbol in self.symbols:
            for tf, cache in [("M15", self._m15_cache),
                              ("H1", self._h1_cache),
                              ("H4", self._h4_cache)]:
                filepath = os.path.join(self._data_dir, f"{symbol}_{tf}.csv")
                if not os.path.exists(filepath):
                    continue
                try:
                    df = pd.read_csv(filepath)
                    df.columns = [c.lower().strip() for c in df.columns]
                    required = {"open", "high", "low", "close"}
                    if not (required.issubset(set(df.columns)) and len(df) >= 100):
                        continue
                    if "time" not in df.columns:
                        df["time"] = pd.date_range(
                            start="2000-01-01", periods=len(df), freq="15min",
                        )
                    else:
                        df["time"] = pd.to_datetime(df["time"], errors="coerce")
                        df = df.dropna(subset=["time"]).reset_index(drop=True)
                    df = df.sort_values("time").reset_index(drop=True)
                    if "volume" not in df.columns:
                        df["volume"] = 0
                    cache[symbol] = df
                except Exception:
                    continue

    @staticmethod
    def _end_idx_at_or_before(df: pd.DataFrame, ts) -> int:
        """Find last index where time <= ts (end-exclusive)."""
        times = df["time"].values
        return int(np.searchsorted(times, ts, side="right"))

    def _precompute_indicators(self):
        """Compute all indicators once on full DF — avoid recomputing per scan."""
        indicators = TechnicalIndicators()
        for symbol in self._available_symbols:
            for cache in (self._m15_cache, self._h1_cache, self._h4_cache):
                if symbol in cache and len(cache[symbol]) >= 200:
                    cache[symbol] = indicators.calculate_all(cache[symbol])

    @property
    def is_available(self) -> bool:
        return len(self._available_symbols) > 0

    def get_min_bars_for_episode(self, max_steps: int = 45) -> int:
        return self.MIN_M15_BARS + max_steps * self.M15_PER_DAY + self.M15_PER_DAY + 48

    def get_sequential_symbols(self, max_steps: int = 45) -> List[str]:
        min_bars = self.get_min_bars_for_episode(max_steps)
        return [
            s for s in self._available_symbols
            if len(self._m15_cache.get(s, [])) >= min_bars
        ]

    # ─── Trade resolution (SL/TP/timeout/gap, with daily-close handling) ─

    def _resolve_trade(
        self, signal, sl_dist: float, tp_dist: float,
        future_df: pd.DataFrame, risk_amount: float,
        pip_size: float, rng: np.random.Generator,
        enable_trail_after_tp: bool = False,
        trail_sl_behind_r: float = 0.5,
        trail_tp_ahead_r: float = 1.0,
        trail_activation_r: float = 0.9,  # v8.0.47: 0.8→0.9 (balance pre-emptive vs runners)
    ) -> float:
        """Simulate trade outcome against future bars.

        Bar-color heuristic when SL+TP both touched in same bar (no tick data):
          green candle → up-first, red → down-first, doji → 50/50.
        Friction ~0.5% major-pair realistic.

        v8.0.43 Option X (Plan B Trick):
          enable_trail_after_tp=True → หลัง TP hit, switch to trail mode
            SL = best - 0.5R (BUY) / best + 0.5R (SELL) — lock profit
            TP = best + 1R (BUY) / best - 1R (SELL) — chase trend
            Exit when bar hits trail SL (lock-in level)
          Match live trade_manager.py trail logic exactly.
        """
        entry = signal.entry_price
        is_buy = signal.signal_type.value == "BUY"

        if is_buy:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        # v8.0.43: Trail state (activate after price reaches activation level)
        # v8.0.47: activate at 0.9R (balance pre-emptive vs runners; live uses adaptive 1s loop)
        trail_active = False
        best_price = entry
        trail_sl_dist = sl_dist * trail_sl_behind_r  # 0.5R
        trail_tp_dist = sl_dist * trail_tp_ahead_r   # 1R
        trail_activation_dist = sl_dist * trail_activation_r  # 0.9R

        # Daily / Friday close simulation (FTMO zero-overnight rule)
        from config.settings import bot_config
        _enforce_daily = getattr(bot_config.sessions, "enforce_daily_close", False)
        _daily_close_t = getattr(bot_config.sessions, "daily_close_time", dt_time(23, 30))
        _friday_close_t = getattr(bot_config.sessions, "friday_force_close", dt_time(20, 45))

        def _is_force_close_bar(bar_time) -> bool:
            if not _enforce_daily or not hasattr(bar_time, "weekday"):
                return False
            wd = bar_time.weekday()
            bt = bar_time.time() if hasattr(bar_time, "time") else None
            if bt is None:
                return False
            if wd == 4:                       # Friday
                return bt >= _friday_close_t
            if wd in (0, 1, 2, 3):            # Mon-Thu
                return bt >= _daily_close_t
            return False

        for _, row in future_df.iterrows():
            bar_high = row["high"]
            bar_low = row["low"]
            bar_open = row["open"]
            bar_close = row["close"]
            bar_time = row.get("time", None) if hasattr(row, "get") else None
            if bar_time is None:
                try:
                    bar_time = row["time"]
                except (KeyError, IndexError):
                    bar_time = None

            # Force-close at EOD / Friday (works in both phases)
            if bar_time is not None and _is_force_close_bar(bar_time):
                if is_buy:
                    pnl_pips = (bar_open - entry) / pip_size
                else:
                    pnl_pips = (entry - bar_open) / pip_size
                pnl_ratio = pnl_pips * pip_size / max(sl_dist, pip_size)
                slippage = float(rng.uniform(0.998, 1.002))
                return risk_amount * pnl_ratio * slippage

            # ═══ TRAIL MODE (Option X — after first TP hit) ═══
            if trail_active:
                # Update best_price (invariant 3: only forward)
                if is_buy and bar_high > best_price:
                    best_price = bar_high
                    new_sl = best_price - trail_sl_dist
                    new_tp = best_price + trail_tp_dist
                    # Invariant 1: SL only moves up
                    if new_sl > sl_price:
                        sl_price = new_sl
                    # Invariant 2: TP only moves up (further)
                    if new_tp > tp_price:
                        tp_price = new_tp
                elif (not is_buy) and bar_low < best_price:
                    best_price = bar_low
                    new_sl = best_price + trail_sl_dist
                    new_tp = best_price - trail_tp_dist
                    if new_sl < sl_price:
                        sl_price = new_sl
                    if new_tp < tp_price:
                        tp_price = new_tp

                # Check trail SL hit (TP never hits in trail — always 1R ahead of best)
                if is_buy:
                    if bar_low <= sl_price:
                        # Exit at trail SL
                        profit_dist = sl_price - entry
                        profit_ratio = profit_dist / max(sl_dist, pip_size)
                        return risk_amount * profit_ratio
                else:
                    if bar_high >= sl_price:
                        profit_dist = entry - sl_price
                        profit_ratio = profit_dist / max(sl_dist, pip_size)
                        return risk_amount * profit_ratio

                # Bar didn't hit trail SL — continue to next bar
                continue

            # ═══ NORMAL MODE (pre-trail) ═══

            # v8.0.45: Trail activation level (0.8R) — pre-empt TP hit
            if is_buy:
                trail_activation_price = entry + trail_activation_dist
            else:
                trail_activation_price = entry - trail_activation_dist

            # Gap check — bar opens past SL/TP (weekend / news)
            if is_buy:
                if bar_open <= sl_price:
                    slippage = float(rng.uniform(1.0, 1.005))
                    return risk_amount * (bar_open - entry) / max(sl_dist, pip_size) * slippage
                # v8.0.45: activate trail when bar_open >= activation (not TP)
                if enable_trail_after_tp and bar_open >= trail_activation_price:
                    trail_active = True
                    best_price = bar_open
                    sl_price = best_price - trail_sl_dist
                    tp_price = best_price + trail_tp_dist
                    continue
                if bar_open >= tp_price:
                    spread_cost = float(rng.uniform(0.995, 1.0))
                    return risk_amount * (bar_open - entry) / max(sl_dist, pip_size) * spread_cost
            else:
                if bar_open >= sl_price:
                    slippage = float(rng.uniform(1.0, 1.005))
                    return risk_amount * (entry - bar_open) / max(sl_dist, pip_size) * slippage
                # v8.0.45: activate trail when bar_open <= activation (not TP)
                if enable_trail_after_tp and bar_open <= trail_activation_price:
                    trail_active = True
                    best_price = bar_open
                    sl_price = best_price + trail_sl_dist
                    tp_price = best_price - trail_tp_dist
                    continue
                if bar_open <= tp_price:
                    spread_cost = float(rng.uniform(0.995, 1.0))
                    return risk_amount * (entry - bar_open) / max(sl_dist, pip_size) * spread_cost

            # Intra-bar SL / TP / Trail Activation
            if is_buy:
                hit_sl = bar_low <= sl_price
                hit_activation = bar_high >= trail_activation_price  # v8.0.45
                hit_tp = bar_high >= tp_price
            else:
                hit_sl = bar_high >= sl_price
                hit_activation = bar_low <= trail_activation_price  # v8.0.45
                hit_tp = bar_low <= tp_price

            # v8.0.45: If trail enabled AND hit activation level → activate trail
            # (before checking TP exit)
            if enable_trail_after_tp and hit_activation and not hit_sl:
                trail_active = True
                # Best price = activation level (we know price reached at least there)
                if is_buy:
                    best_price = trail_activation_price
                    sl_price = best_price - trail_sl_dist
                    tp_price = best_price + trail_tp_dist
                else:
                    best_price = trail_activation_price
                    sl_price = best_price + trail_sl_dist
                    tp_price = best_price - trail_tp_dist
                continue  # trail mode starts next iteration

            if hit_sl and hit_tp:
                # Bar-color heuristic
                if bar_close > bar_open:
                    if is_buy:
                        hit_sl = False
                    else:
                        hit_tp = False
                elif bar_close < bar_open:
                    if is_buy:
                        hit_tp = False
                    else:
                        hit_sl = False
                else:
                    if rng.random() < 0.5:
                        hit_sl = False
                    else:
                        hit_tp = False

            if hit_sl:
                slippage = float(rng.uniform(1.0, 1.005))
                return -risk_amount * slippage

            if hit_tp:
                if not enable_trail_after_tp:
                    actual_rr = tp_dist / max(sl_dist, pip_size)
                    spread_cost = float(rng.uniform(0.995, 1.0))
                    return risk_amount * actual_rr * spread_cost
                # v8.0.43 Option X: activate trail at TP hit
                trail_active = True
                # Best price = TP level (we know price reached at least there)
                best_price = tp_price if is_buy else tp_price  # same value (TP price)
                if is_buy:
                    sl_price = best_price - trail_sl_dist  # 0.5R behind TP
                    tp_price = best_price + trail_tp_dist  # 1R ahead of TP
                else:
                    sl_price = best_price + trail_sl_dist
                    tp_price = best_price - trail_tp_dist
                # Continue to next bar — trail mode active

        # Timeout — close at last bar
        last_close = float(future_df["close"].iloc[-1])
        if is_buy:
            pnl_pips = (last_close - entry) / pip_size
        else:
            pnl_pips = (entry - last_close) / pip_size

        pnl_ratio = pnl_pips * pip_size / max(sl_dist, pip_size)
        return risk_amount * pnl_ratio

    # ─── Strategy hook (subclass overrides) ─────────────────────────────

    def generate_episode_signals(
        self,
        symbol: str,
        m15_start_bar: int,
        num_days: int = 45,
        rng: np.random.Generator = None,
    ) -> List[Dict]:
        """Strategy-specific signal generation. Subclass must implement.

        See `ml/mean_reversion_backtester.py::MeanReversionBacktester`
        for the production MR implementation.
        """
        raise NotImplementedError(
            "StrategyBacktester is a base class — subclass and implement "
            "`generate_episode_signals` for your strategy. "
            "Use `MeanReversionBacktester` for the v8.0 MR pipeline."
        )

    def _empty_result(self) -> Dict:
        return {
            "pnl": 0.0, "trades_taken": 0, "wins": 0, "losses": 0,
            "max_intraday_dd": 0.0, "trades": [],
            "win_rate": 0.0, "regime": "quiet", "atr_pips": 10.0,
        }
