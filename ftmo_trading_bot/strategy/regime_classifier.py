"""
===============================================================================
FTMO Trading Bot — Market Regime Classifier (v8.1 dual-strategy switch)
===============================================================================
Per-symbol regime gate that decides which strategy may arm a symbol:

    RANGING   → Mean Reversion (MR)
    TRENDING  → Trend Following (TF)
    AMBIGUOUS → NEITHER (statistical dead-zone)

ADX H1 is the PRIMARY separator. The band [adx_ranging_max, adx_trending_min]
(default 20–27) is a dead-zone where neither strategy trades — wiki v8.0.51
proved ADX H1 25-30 is a "killer zone" (WR 25%, −$679). Choppiness Index and
EMA-slope-per-ATR are secondary confirmations.

Hysteresis prevents flip-flapping three ways:
  1. The wide AMBIGUOUS band — to flip MR↔TF you must cross the whole dead-zone.
  2. `confirm_bars` — a new regime must repeat N consecutive scans before it
     becomes the active regime.
  3. `min_dwell_sec` — minimum seconds between regime switches per symbol.

This builds on `FTMOTradingBot._compute_symbol_regime` (which already computes
ATR-zscore + slope-per-ATR on H1) and adds ADX + choppiness + hysteresis.
===============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from config.settings import bot_config
from strategy.indicators import TechnicalIndicators


class Regime(Enum):
    RANGING = "RANGING"        # MR arms
    TRENDING = "TRENDING"      # TF arms
    AMBIGUOUS = "AMBIGUOUS"    # neither


# Map confirmed regime → the strategy allowed to arm.
_REGIME_TO_STRATEGY = {
    Regime.RANGING: "MR",
    Regime.TRENDING: "TF",
    Regime.AMBIGUOUS: None,
}


@dataclass
class RegimeResult:
    symbol: str
    regime: Regime           # confirmed regime (after hysteresis)
    raw_regime: Regime       # instantaneous regime (this scan, pre-hysteresis)
    armed: Optional[str]     # "MR" / "TF" / None
    adx: float
    choppiness: float
    slope_per_atr: float
    atr_zscore: float


class MarketRegimeClassifier:
    """Classifies each symbol's market regime and applies per-symbol hysteresis."""

    def __init__(self, connector, config=None, indicators: Optional[TechnicalIndicators] = None):
        self._connector = connector
        self._cfg = config or bot_config.regime
        self._indicators = indicators or TechnicalIndicators()
        # Per-symbol hysteresis state.
        # {symbol: {"current": Regime, "pending": Regime, "pending_count": int,
        #           "last_switch_ts": float}}
        self._state: Dict[str, dict] = {}

    # ─── Public API ───────────────────────────────────────────────────────

    def classify(self, symbol: str) -> Optional[RegimeResult]:
        """Classify one symbol from H1 data. Returns None if data unavailable."""
        feats = self._compute_features(symbol)
        if feats is None:
            return None
        adx, chop, slope_abs, slope_signed, atr_z = feats
        raw = self._raw_regime(adx, chop, slope_abs)
        confirmed = self._apply_hysteresis(symbol, raw)
        return RegimeResult(
            symbol=symbol,
            regime=confirmed,
            raw_regime=raw,
            armed=_REGIME_TO_STRATEGY[confirmed],
            adx=adx,
            choppiness=chop,
            slope_per_atr=slope_signed,
            atr_zscore=atr_z,
        )

    def classify_all(
        self,
        symbols: List[str],
        open_owner: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Optional[str]]:
        """Return {symbol: armed_strategy ("MR"/"TF"/None)} for every symbol.

        `open_owner` (optional) maps symbols with an OPEN position to the
        strategy that owns it — those symbols are held by their owner (regime
        lock) regardless of the current regime, so a regime flip mid-trade can't
        hand the symbol to the other strategy.
        """
        open_owner = open_owner or {}
        out: Dict[str, Optional[str]] = {}
        for sym in symbols:
            if sym in open_owner:
                out[sym] = open_owner[sym]      # regime lock while position open
                # keep the classifier state warm so hysteresis is current on unlock
                self.classify(sym)
                continue
            res = self.classify(sym)
            out[sym] = res.armed if res is not None else None
        return out

    def get_result(self, symbol: str) -> Optional[RegimeResult]:
        """Convenience — classify and return the full RegimeResult (for logging)."""
        return self.classify(symbol)

    # ─── Internals ────────────────────────────────────────────────────────

    def _compute_features(self, symbol: str):
        """Fetch H1 and compute (adx, chop, |slope/atr|, slope/atr, atr_zscore)."""
        try:
            df = self._connector.get_ohlcv(symbol, "H1", count=self._cfg.h1_bars)
        except Exception:
            return None
        if df is None or len(df) < 50:
            return None
        try:
            close = df["close"].values.astype(np.float64)
            high = df["high"].values.astype(np.float64)
            low = df["low"].values.astype(np.float64)
            n = len(close)

            # ADX (reuse indicators) — compute on a copy to avoid mutating cache
            adx_df = self._indicators.calculate_adx(df.copy())
            adx = float(adx_df["adx"].iloc[-1])
            if np.isnan(adx):
                return None

            # Choppiness Index
            chop = float(self._indicators.calculate_choppiness(df, period=14))

            # ATR (Wilder TR mean over 14) + z-score vs full window — mirrors
            # FTMOTradingBot._compute_symbol_regime so live obs + regime agree.
            hl = high[1:] - low[1:]
            hc = np.abs(high[1:] - close[:-1])
            lc = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(hl, hc), lc)
            atr_recent = float(np.mean(tr[-14:]))
            atr_baseline = float(np.mean(tr))
            atr_std = float(np.std(tr)) or 1e-8
            atr_zscore = (atr_recent - atr_baseline) / atr_std

            # Trend slope normalized to ATR/bar (scale-invariant across symbols)
            window = min(self._cfg.slope_window, n)
            y = close[-window:]
            x = np.arange(window, dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0])
            slope_per_atr = slope / max(atr_recent, 1e-8)

            return adx, chop, abs(slope_per_atr), slope_per_atr, atr_zscore
        except Exception:
            return None

    def _raw_regime(self, adx: float, chop: float, slope_abs: float) -> Regime:
        """Instantaneous regime — ADX primary gate + chop/slope confirmation."""
        cfg = self._cfg
        # Dead-zone is absolute: ADX inside [ranging_max, trending_min] → AMBIGUOUS.
        if cfg.adx_ranging_max <= adx <= cfg.adx_trending_min:
            return Regime.AMBIGUOUS
        if adx > cfg.adx_trending_min:
            if chop < cfg.chop_trending_max or slope_abs > cfg.slope_trending_min:
                return Regime.TRENDING
            return Regime.AMBIGUOUS
        # adx < adx_ranging_max → candidate RANGING
        if chop > cfg.chop_ranging_min or slope_abs < cfg.slope_ranging_max:
            return Regime.RANGING
        return Regime.AMBIGUOUS

    def _apply_hysteresis(self, symbol: str, raw: Regime) -> Regime:
        """Debounce regime switches (confirm_bars + min_dwell_sec).

        New symbols start AMBIGUOUS (conservative — no trading until a regime is
        confirmed for `confirm_bars` consecutive scans).
        """
        cfg = self._cfg
        st = self._state.get(symbol)
        if st is None:
            st = {"current": Regime.AMBIGUOUS, "pending": raw,
                  "pending_count": 1, "last_switch_ts": 0.0}
            self._state[symbol] = st

        if raw == st["current"]:
            # already in this regime — reset any pending switch
            st["pending"] = raw
            st["pending_count"] = 0
            return st["current"]

        # raw differs from current → accumulate confirmation
        if raw == st["pending"]:
            st["pending_count"] += 1
        else:
            st["pending"] = raw
            st["pending_count"] = 1

        now = time.time()
        if (st["pending_count"] >= cfg.confirm_bars
                and (now - st["last_switch_ts"]) >= cfg.min_dwell_sec):
            st["current"] = raw
            st["last_switch_ts"] = now
            st["pending_count"] = 0
        return st["current"]
