"""
===============================================================================
FTMO Trading Bot — Trend Following Backtester (v8.1 Phase 3 pool generator)
===============================================================================
Builds the offline signal pool for TF strategy training. Inherits all data
loading + trade-resolution machinery from `StrategyBacktester` and swaps in
`TrendFollowingStrategy` as the signal engine.

Key differences vs MR backtester:
  • Entry timeframe = **H1** (not M15) — TF holds longer, scans hourly-ish.
  • Trend filter = H4 (+ D1 bias resampled from H4, since no D1 CSV exists).
  • RR ratio WIDE (2.5) + trail config that lets winners RUN (Stage-2 1.5R cap
    DISABLED via tp_step_trigger_r=99) — opposite of MR's quick 1:1 TP.
  • Pool dict keeps the SMC/MR schema keys (so FTMOSignalFilterEnv._get_obs
    reads it) + TF extras (`trend_age_bars`, `pullback_depth_atr`, `adx_at_entry`)
    consumed by `TrendFollowingFilterEnv` for reward shaping + obs reinterpretation.

Output format identical to the MR builder so the GBM trainer + pool persistence
work unchanged.
===============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.strategy_backtester import StrategyBacktester


class TrendFollowingBacktester(StrategyBacktester):
    """Pool builder for the trend-following strategy (entry on H1)."""

    # Scan every 2 H1 bars (= 2h) → 12 scans/day. TF setups (pullback-resume in a
    # trend) are less transient than MR extremes, so coarser than MR's 48/day.
    TF_SCAN_POINTS_PER_DAY: int = 12
    # Resolution window: 180 H1 bars = ~7.5 trading days — runners need room.
    # (v8.1 runner-tune: 120→180 so long trends can fully develop before timeout.)
    TF_FUTURE_BARS: int = 180
    # Dedup same-direction within 6 H1 bars.
    DEDUP_BARS: int = 6
    # H1 lookback fed to the strategy (calculate_all needs >= 200).
    TF_LOOKBACK_H1: int = 300

    def __init__(self, data_dir: str, symbols: Optional[List[str]] = None,
                 ml_model_path: Optional[str] = None):
        super().__init__(data_dir=data_dir, symbols=symbols, ml_model_path=ml_model_path)
        from strategy.trend_following_strategy import TrendFollowingStrategy
        from strategy.indicators import TechnicalIndicators
        from config.settings import bot_config as _bc
        self._tf_strategy = TrendFollowingStrategy(
            indicators=getattr(self, "_indicators", TechnicalIndicators())
        )
        # Use TRAINING confluence floor (wide pool for diversity) — mirror MR pattern
        _tf_cfg = getattr(_bc, "tf", None)
        if _tf_cfg is not None:
            self._tf_strategy.MIN_CONFLUENCE = float(
                getattr(_tf_cfg, "min_confluence_score_training",
                        self._tf_strategy.MIN_CONFLUENCE)
            )

    # ─── D1 derivation (no D1 CSV → roll up 6 H4 bars = 1 trading day) ──────

    @staticmethod
    def _resample_h4_to_d1(h4_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Approximate D1 by grouping every 6 H4 bars. Used for the soft D1 bias.

        Live uses the broker's real D1; training has only H4 CSVs → this rollup
        keeps the D1-bias feature roughly parity-aligned (it is a soft ±5
        confluence bonus, not a hard gate, so minor differences are tolerable).
        """
        if h4_df is None or len(h4_df) < 12:
            return None
        df = h4_df.reset_index(drop=True)
        grp = df.index // 6
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        cols = {k: v for k, v in agg.items() if k in df.columns}
        d1 = df.groupby(grp).agg(cols)
        return d1.reset_index(drop=True)

    # ─── Override pool generator (H1-anchored) ──────────────────────────────

    def generate_episode_signals(
        self,
        symbol: str,
        h1_start_bar: int,
        num_days: int = 45,
        rng: np.random.Generator = None,
    ) -> List[Dict]:
        if rng is None:
            rng = np.random.default_rng()

        h1_df = self._h1_cache.get(symbol)
        h4_df = self._h4_cache.get(symbol)
        if h1_df is None or h4_df is None:
            return []

        signals: List[Dict] = []
        scan_step = max(1, self.H1_PER_DAY // self.TF_SCAN_POINTS_PER_DAY)  # 24//12 = 2
        scan_offsets = [scan_step * i for i in range(self.TF_SCAN_POINTS_PER_DAY)]
        last_accept_idx: Dict[float, int] = {}

        for day in range(num_days):
            day_h1_start = h1_start_bar + day * self.H1_PER_DAY
            if day_h1_start >= len(h1_df) - self.H1_PER_DAY:
                break
            h1_lookback_start = max(0, day_h1_start - self.TF_LOOKBACK_H1)

            for offset in scan_offsets:
                scan_idx = h1_lookback_start + self.TF_LOOKBACK_H1 + offset
                if scan_idx >= len(h1_df) - 5:
                    continue

                h1_slice = h1_df.iloc[
                    scan_idx - self.TF_LOOKBACK_H1 + 1: scan_idx + 1
                ].copy()
                if len(h1_slice) < 200:
                    continue

                # Anchor H4 by H1 scan timestamp (no future leak: h4_end ≤ scan bar)
                try:
                    h1_scan_ts = h1_df["time"].iloc[scan_idx]
                except Exception:
                    h1_scan_ts = None
                if h1_scan_ts is not None:
                    h4_end = min(self._end_idx_at_or_before(h4_df, h1_scan_ts), len(h4_df))
                else:
                    h4_end = len(h4_df)
                h4_start = max(0, h4_end - self.MIN_H4_BARS)
                h4_slice = h4_df.iloc[h4_start:h4_end].copy() if h4_end > 0 else None
                d1_slice = self._resample_h4_to_d1(h4_slice)

                last_close = float(h1_slice["close"].iloc[-1])
                pip_size = 0.01 if last_close > 50 else 0.0001
                spread = pip_size * 2
                price_info = {"bid": last_close, "ask": last_close + spread, "spread": spread}

                try:
                    signal = self._tf_strategy.analyze_with_data(
                        symbol, d1_slice, h4_slice, h1_slice, price_info
                    )
                except Exception:
                    continue
                if not signal.is_valid:
                    continue

                signal_sl = abs(signal.entry_price - signal.sl_price)
                if signal_sl < pip_size:
                    continue

                is_buy = signal.signal_type.value == "BUY"
                dir_key = 1.0 if is_buy else -1.0
                if scan_idx - last_accept_idx.get(dir_key, -10**9) < self.DEDUP_BARS:
                    continue
                last_accept_idx[dir_key] = scan_idx

                atr_val = max(signal.atr_value, pip_size)
                sl_distance_atr = signal_sl / atr_val
                actual_sl = signal_sl
                rr_ratio = self._tf_strategy.RR_RATIO
                actual_tp = actual_sl * rr_ratio

                future_start = scan_idx + 1
                future_end = min(future_start + self.TF_FUTURE_BARS, len(h1_df))
                if future_start >= len(h1_df):
                    continue
                future = h1_df.iloc[future_start:future_end]
                if len(future) == 0:
                    continue

                # TF resolve: let winners RUN. Disable MR's Stage-2 1.5R cap
                # (tp_step_trigger_r=99). v8.1 runner-tune: LOOSER trail so normal
                # pullbacks don't stop a runner out —
                #   activation 1.0→1.5R (establish trend before trailing),
                #   behind 1.0→2.0R (ride through 2R pullbacks instead of 1R),
                #   floor stays 1.0R (lock +1R once trailing). This shifts the
                #   outcome distribution from "many +1R floor wins" toward fewer
                #   but bigger runners (true TF profile: lower win%, higher avg R).
                # v8.1 Phase4 fix-A: read trail geometry from bot_config.tf.mgmt_*
                # — the SAME source live TradeManager._exit_profile("TF") uses, so
                # train resolve == live exit (no parity drift). tp_step disabled (99).
                from config.settings import bot_config as _bc
                _tf = _bc.tf
                risk_amount = 1.0
                trade_pnl = self._resolve_trade(
                    signal, actual_sl, actual_tp, future, risk_amount, pip_size, rng,
                    enable_trail_after_tp=True,
                    trail_sl_behind_r=float(getattr(_tf, "mgmt_trail_sl_behind_r", 2.0)),
                    trail_tp_ahead_r=float(getattr(_tf, "mgmt_trail_tp_ahead_r", 2.0)),
                    trail_activation_r=float(getattr(_tf, "mgmt_trail_activation_r", 1.5)),
                    tp_step_trigger_r=99.0,   # DISABLE Stage-2 cap (let TF run)
                    trail_sl_floor_r=float(getattr(_tf, "mgmt_trail_sl_floor_r", 1.0)),
                )

                bars_to_resolution = self._bars_to_first_hit(
                    signal, actual_sl, actual_tp, future, pip_size
                )
                # TF "runner" = big winner (>= 2R). is_quick_tp kept for env compat.
                is_runner = (trade_pnl >= 2.0)
                is_quick_tp = (trade_pnl > 0 and bars_to_resolution <= 2)

                bias_alignment = dir_key * float(signal.market_bias)
                typical_spread_pips = self._TYPICAL_SPREAD_PIPS.get(symbol, 1.5)
                sl_distance_pips = sl_distance_atr * (atr_val / pip_size)

                sig_dict: Dict = {
                    "day": day,
                    "symbol": symbol,
                    "signal_type": signal.signal_type.value,
                    "confluence_score": signal.confluence_score,
                    "rr_ratio": rr_ratio,
                    "atr_value": atr_val,
                    "atr_pips": atr_val / pip_size,
                    "ob_score": 0.0,
                    "market_bias": signal.market_bias,
                    "trend": signal.trend,
                    "direction": dir_key,
                    "bias_alignment": bias_alignment,
                    "sl_distance_atr": sl_distance_atr,
                    "sl_distance_pips": float(sl_distance_pips),
                    "outcome_pnl_ratio": float(trade_pnl),
                    "pip_size": pip_size,
                    "rsi_value": signal.rsi_value,
                    "trend_strength": signal.trend_strength,
                    "macd_histogram": signal.macd_histogram,
                    "ob_size_atr": 0.0,
                    "adx": signal.adx,
                    "stoch_k": signal.stoch_k,
                    "bb_pctb": signal.bb_pctb,
                    "atr_change_ratio": signal.atr_change_ratio,
                    "price_roc": signal.price_roc,
                    "spread_pips": float(typical_spread_pips),
                    "htf_trend_alignment": float(bias_alignment),
                    "chronos_alignment": 0.0,
                    "chronos_uncertainty_norm": 0.0,
                    # MR-schema extras kept zero/neutral (env base may read them)
                    "mr_setup_score": 0.0,
                    "bb_extreme": 0.0,
                    "bb_band_width_atr": 0.0,
                    "kc_distance_norm": 0.0,
                    "ema_slope_norm": float(getattr(signal, "ema_slope_norm", 0.0)),
                    "consecutive_outside": 0,
                    "band_squeeze_ratio": float(getattr(signal, "band_squeeze_ratio", 0.5)),  # was hardcoded 0.5 (model never saw real squeeze)
                    "reversal_wick_ratio": 0.0,
                    "bars_to_resolution": int(bars_to_resolution),
                    "is_quick_tp": bool(is_quick_tp),
                    # TF-specific (env reads for obs reinterpretation + reward)
                    "trend_age_bars": int(getattr(signal, "trend_age_bars", 0)),
                    "pullback_depth_atr": float(getattr(signal, "pullback_depth_atr", 0.0)),
                    "adx_at_entry": float(getattr(signal, "adx_at_entry", signal.adx)),
                    "di_spread": float(getattr(signal, "di_spread", 0.0)),      # leading direction (early_entry)
                    "adx_slope": float(getattr(signal, "adx_slope", 0.0)),      # trend building vs exhausting
                    "is_runner": bool(is_runner),
                    # TF has no MR-style entry-confirm gate → never force-skip
                    "entry_confirm_passed": True,
                }

                # Temporal/regime features (GBM reuses same feature list)
                try:
                    from ml.signal_quality import compute_temporal_features
                    from config.settings import get_symbol_config
                    is_metal = "XAU" in symbol.upper() or "XAG" in symbol.upper()
                    _floor = float(get_symbol_config(
                        symbol, "atr_floor_pips", 100.0 if is_metal else 8.0))
                    try:
                        scan_ts = pd.to_datetime(h1_slice["time"].iloc[-1])
                    except Exception:
                        scan_ts = None
                    sig_dict.update(compute_temporal_features(
                        timestamp=scan_ts, ltf_df=h1_slice,
                        atr_floor_pips=_floor, pip_size=pip_size,
                    ))
                except Exception:
                    pass

                signals.append(sig_dict)

        # Score with quality model if available
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

    # ─── Helper (mirror MR; works on any future_df timeframe) ──────────────

    @staticmethod
    def _bars_to_first_hit(signal, sl_dist: float, tp_dist: float,
                           future_df: pd.DataFrame, pip_size: float) -> int:
        entry = float(signal.entry_price)
        is_buy = signal.signal_type.value == "BUY"
        if is_buy:
            sl_price, tp_price = entry - sl_dist, entry + tp_dist
        else:
            sl_price, tp_price = entry + sl_dist, entry - tp_dist
        for i, (_, row) in enumerate(future_df.iterrows(), start=1):
            bh, bl, bo = float(row["high"]), float(row["low"]), float(row["open"])
            if is_buy:
                if bo <= sl_price or bo >= tp_price or bl <= sl_price or bh >= tp_price:
                    return i
            else:
                if bo >= sl_price or bo <= tp_price or bh >= sl_price or bl <= tp_price:
                    return i
        return len(future_df)
