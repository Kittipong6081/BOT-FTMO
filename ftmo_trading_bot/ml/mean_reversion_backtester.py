"""
===============================================================================
FTMO Trading Bot — Mean Reversion Backtester (v8.0 pool generator)
===============================================================================
Generates the offline signal pool for MR strategy training. Inherits all the
data-loading and trade-resolution machinery from `StrategyBacktester` and only
swaps in `MeanReversionStrategy` as the signal-generation engine.

Key differences vs SMC backtester:
  • Strategy: `MeanReversionStrategy` (BB + RSI + ATR + ADX trend block)
  • RR ratio: fixed 1:1 (per spec — quick TP for capital preservation)
  • Per-signal extra fields: `mr_setup_score`, `bb_extreme`, `bb_band_width_atr`,
    `reversal_wick_ratio`, `bars_to_resolution` (used by RL env for quick-TP
    bonus shaping)
  • Pool dict format keeps SMC keys (ob_score, ob_size_atr → 0) so the existing
    `FTMOSignalFilterEnv._get_obs` schema can read it without errors. The RL
    env that consumes MR pools (`MeanReversionFilterEnv`) reinterprets some
    slots (e.g. obs[4] = bb_extreme score, obs[10] = bb_band_width / ATR).

Output format identical to `StrategyBacktester.generate_episode_signals` so
the GBM training script and pool persistence work unchanged.
===============================================================================
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.strategy_backtester import StrategyBacktester


class MeanReversionBacktester(StrategyBacktester):
    """Pool builder for the mean-reversion strategy.

    Reuses `_load_data`, `_precompute_indicators`, `_resolve_trade`,
    and helpers from the SMC backtester. Replaces the strategy engine
    and `generate_episode_signals` with MR-specific logic.
    """

    # Override scan cadence — MR needs FREQUENT samples because BB extremes are
    # rare + transient (typically last 1-3 bars). SMC scans every 6h (4 per day).
    # MR v8.0.2 scans every 30min (48 per day) so the pool catches enough
    # extremes to give RL diverse training signal.
    MR_SCAN_POINTS_PER_DAY: int = 48
    # Resolution window: 32 M15 bars = ~8 hours. MR with RR 1:1 should resolve
    # quickly; if it hasn't moved in 8h the trade is stale.
    MR_FUTURE_BARS: int = 32
    # Dedup: skip same-direction signals on same symbol within DEDUP_BARS bars
    # of the previous accept — prevents pool from being dominated by a single
    # extreme that persists across multiple scans.
    DEDUP_BARS: int = 4

    def __init__(self, data_dir: str, symbols: Optional[List[str]] = None,
                 ml_model_path: Optional[str] = None):
        # Skip SMC strategy init — replace with MR strategy after parent runs
        super().__init__(data_dir=data_dir, symbols=symbols, ml_model_path=ml_model_path)
        # Replace the SMC strategy slot with MR strategy
        from strategy.mean_reversion_strategy import MeanReversionStrategy
        from strategy.indicators import TechnicalIndicators
        self._mr_strategy = MeanReversionStrategy(
            indicators=getattr(self, "_indicators", TechnicalIndicators())
        )

    # ─── Override pool generator ─────────────────────────────────────────

    def generate_episode_signals(
        self,
        symbol: str,
        m15_start_bar: int,
        num_days: int = 45,
        rng: np.random.Generator = None,
    ) -> List[Dict]:
        """Mirror the SMC builder but use `MeanReversionStrategy`.

        Returns a list of signal dicts. Each dict carries:
          • SMC-schema keys (so `FTMOSignalFilterEnv._get_obs` works)
          • MR-specific keys (`mr_setup_score`, `bb_extreme`,
            `bb_band_width_atr`, `reversal_wick_ratio`, `bars_to_resolution`,
            `is_quick_tp`) used by `MeanReversionFilterEnv` for reward shaping
        """
        if rng is None:
            rng = np.random.default_rng()

        if symbol not in self._m15_cache:
            return []

        m15_df = self._m15_cache[symbol]
        h1_df = self._h1_cache.get(symbol)
        h4_df = self._h4_cache.get(symbol)
        if h1_df is None or h4_df is None:
            return []

        signals: List[Dict] = []
        # Scan every 2 M15 bars (= 30min) → 48 scans/day
        scan_step = max(1, self.M15_PER_DAY // self.MR_SCAN_POINTS_PER_DAY)
        scan_offsets = [scan_step * i for i in range(self.MR_SCAN_POINTS_PER_DAY)]
        # Dedup tracking: last accepted scan_idx per (symbol, direction)
        last_accept_idx: Dict[float, int] = {}  # +1.0 / -1.0 → idx

        for day in range(num_days):
            day_m15_start = m15_start_bar + day * self.M15_PER_DAY
            m15_window_end = day_m15_start

            if m15_window_end >= len(m15_df) - self.M15_PER_DAY:
                break

            m15_end_ts = m15_df["time"].iloc[m15_window_end - 1]
            h1_end = min(self._end_idx_at_or_before(h1_df, m15_end_ts), len(h1_df))
            h4_end = min(self._end_idx_at_or_before(h4_df, m15_end_ts), len(h4_df))
            h1_start = max(0, h1_end - self.MIN_H1_BARS)
            h4_start = max(0, h4_end - self.MIN_H4_BARS)

            if h1_end - h1_start < 200 or h4_end - h4_start < 200:
                continue

            m15_lookback_start = max(0, day_m15_start - self.MIN_M15_BARS)

            for offset in scan_offsets:
                scan_idx = m15_lookback_start + self.MIN_M15_BARS + offset
                if scan_idx >= len(m15_df) - 20:
                    continue

                ltf_slice = m15_df.iloc[
                    scan_idx - self.MIN_M15_BARS + 1: scan_idx + 1
                ].copy()
                h1_slice = h1_df.iloc[h1_start:h1_end].copy()
                h4_slice = h4_df.iloc[h4_start:h4_end].copy()

                if len(ltf_slice) < 200 or len(h1_slice) < 200:
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
                    signal = self._mr_strategy.analyze_with_data(
                        symbol, h4_slice, h1_slice, ltf_slice, price_info
                    )
                except Exception:
                    continue

                if not signal.is_valid:
                    continue

                signal_sl = abs(signal.entry_price - signal.sl_price)
                if signal_sl < pip_size:
                    continue

                # Dedup — skip if same-direction signal accepted within DEDUP_BARS
                is_buy_check = signal.signal_type.value == "BUY"
                dir_key = 1.0 if is_buy_check else -1.0
                last_idx = last_accept_idx.get(dir_key, -10**9)
                if scan_idx - last_idx < self.DEDUP_BARS:
                    continue
                last_accept_idx[dir_key] = scan_idx

                atr_val = max(signal.atr_value, pip_size)
                sl_distance_atr = signal_sl / atr_val
                actual_sl = signal_sl
                rr_ratio = self._mr_strategy.RR_RATIO
                actual_tp = actual_sl * rr_ratio

                future_start = scan_idx + 1
                future_end = min(future_start + self.MR_FUTURE_BARS, len(m15_df))
                if future_start >= len(m15_df):
                    continue
                future = m15_df.iloc[future_start:future_end]
                if len(future) == 0:
                    continue

                # Resolve PnL using SMC backtester's machinery — same SL/TP rules
                risk_amount = 1.0
                trade_pnl = self._resolve_trade(
                    signal, actual_sl, actual_tp, future, risk_amount, pip_size, rng
                )

                # Detect quick TP: figure out how many bars it took to hit TP
                bars_to_resolution = self._bars_to_first_hit(
                    signal, actual_sl, actual_tp, future, pip_size
                )
                # "Quick TP" = winner that resolved in <= 5 bars (~75min M15)
                is_quick_tp = (trade_pnl > 0 and bars_to_resolution <= 5)

                is_buy = signal.signal_type.value == "BUY"
                direction = 1.0 if is_buy else -1.0
                bias_alignment = direction * float(signal.market_bias)

                typical_spread_pips = self._TYPICAL_SPREAD_PIPS.get(symbol, 1.5)
                sl_distance_pips = sl_distance_atr * (atr_val / pip_size)

                # Build SMC-compat dict + MR extras
                sig_dict: Dict = {
                    # Episode metadata
                    "day": day,
                    "symbol": symbol,
                    # SMC schema (filled with MR-equivalent values)
                    "signal_type": signal.signal_type.value,
                    "confluence_score": signal.confluence_score,
                    "rr_ratio": rr_ratio,
                    "atr_value": atr_val,
                    "atr_pips": atr_val / pip_size,
                    "ob_score": 0.0,                     # MR has no OB
                    "market_bias": signal.market_bias,
                    "trend": signal.trend,
                    "direction": direction,
                    "bias_alignment": bias_alignment,
                    "sl_distance_atr": sl_distance_atr,
                    "sl_distance_pips": float(sl_distance_pips),
                    "outcome_pnl_ratio": float(trade_pnl),
                    "pip_size": pip_size,
                    "rsi_value": signal.rsi_value,
                    "trend_strength": signal.trend_strength,
                    "macd_histogram": signal.macd_histogram,
                    "ob_size_atr": 0.0,                  # MR has no OB
                    "adx": signal.adx,
                    "stoch_k": signal.stoch_k,
                    "bb_pctb": signal.bb_pctb,
                    "atr_change_ratio": signal.atr_change_ratio,
                    "price_roc": signal.price_roc,
                    # Cost / spread / HTF align
                    "spread_pips": float(typical_spread_pips),
                    "htf_trend_alignment": float(bias_alignment),
                    # Chronos: not used by MR by default — keep zeros for env compat
                    "chronos_alignment": 0.0,
                    "chronos_uncertainty_norm": 0.0,
                    # MR-specific extras (env reads for shaping + obs reinterpretation)
                    "mr_setup_score": signal.mr_setup_score,
                    "bb_extreme": signal.bb_extreme,
                    "bb_band_width_atr": signal.bb_band_width_atr,
                    "reversal_wick_ratio": signal.reversal_wick_ratio,
                    "bars_to_resolution": int(bars_to_resolution),
                    "is_quick_tp": bool(is_quick_tp),
                }

                # v7.1 temporal/regime features (so GBM trainer can reuse the same
                # feature list) — best-effort import
                try:
                    from ml.signal_quality import compute_temporal_features
                    from config.settings import get_symbol_config
                    is_metal = "XAU" in symbol.upper() or "XAG" in symbol.upper()
                    _default_floor = 100.0 if is_metal else 8.0
                    _atr_floor_pips = float(get_symbol_config(
                        symbol, "atr_floor_pips", _default_floor
                    ))
                    try:
                        scan_ts = pd.to_datetime(ltf_slice["time"].iloc[-1])
                    except Exception:
                        scan_ts = None
                    temporal_feats = compute_temporal_features(
                        timestamp=scan_ts,
                        ltf_df=ltf_slice,
                        atr_floor_pips=_atr_floor_pips,
                        pip_size=pip_size,
                    )
                    sig_dict.update(temporal_feats)
                except Exception:
                    # If temporal helper unavailable, skip — GBM will use 0 for missing
                    pass

                signals.append(sig_dict)

        # Score with quality model if available (same as SMC pool)
        if signals and getattr(self, "_quality_model", None) is not None:
            try:
                scores = self._quality_model.score_batch(signals)
                for sig, s in zip(signals, scores):
                    sig["ml_score"] = float(s)
            except Exception:
                for sig in signals:
                    sig["ml_score"] = 0.5
        else:
            for sig in signals:
                sig.setdefault("ml_score", 0.5)

        return signals

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _bars_to_first_hit(
        signal,
        sl_dist: float,
        tp_dist: float,
        future_df: pd.DataFrame,
        pip_size: float,
    ) -> int:
        """How many M15 bars until SL or TP is touched.

        Returns the count (1-indexed: first future bar = 1). If neither is
        touched within the window, returns the window length (= "stale exit").
        Used by env to detect "quick TP" winners for reward shaping.
        """
        entry = float(signal.entry_price)
        is_buy = signal.signal_type.value == "BUY"
        if is_buy:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        for i, (_, row) in enumerate(future_df.iterrows(), start=1):
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_open = float(row["open"])
            if is_buy:
                if bar_open <= sl_price or bar_open >= tp_price:
                    return i
                if bar_low <= sl_price or bar_high >= tp_price:
                    return i
            else:
                if bar_open >= sl_price or bar_open <= tp_price:
                    return i
                if bar_high >= sl_price or bar_low <= tp_price:
                    return i
        return len(future_df)
