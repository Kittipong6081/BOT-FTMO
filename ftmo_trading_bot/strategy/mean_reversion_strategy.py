"""
===============================================================================
FTMO Trading Bot — Mean Reversion Strategy (v8.0 pivot)
===============================================================================
High-Win-Rate Hybrid Strategy: Mean Reversion entry + Trend Filter veto.

Rules (entry on M15):
  1. ATR floor          → atr_pips >= per-symbol floor (skip dead market)
  2. ADX H1 trend block → adx_h1 > ADX_TREND_BLOCK (default 25) → no MR entry
                          (mean-reversion fails on strong trends)
  3. BB %B extreme      → BUY when bb_pctb <= BB_OVERSOLD (default 0.10)
                          SELL when bb_pctb >= BB_OVERBOUGHT (default 0.90)
  4. RSI confirmation   → BUY when rsi <= RSI_OVERSOLD (default 30)
                          SELL when rsi >= RSI_OVERBOUGHT (default 70)
  5. Reversal candle    → BUY needs bullish-rejection wick (lower wick > body)
                          SELL needs bearish-rejection wick (upper wick > body)

Stop Loss / Take Profit:
  • SL = ATR × SL_ATR_MULT (default 1.0 — TIGHT for capital preservation)
  • TP = SL × RR_RATIO   (default 1.0 — quick-TP target = 1:1 RR)
  • Output mimics `TradeSignal` interface so existing pool format / RL env
    work unchanged.

Trend Filter Veto:
  When `adx_h1 > ADX_TREND_BLOCK` strategy returns NO_SIGNAL even if BB+RSI
  agree — preserves capital from being against strong momentum.

Output:
  `MRSignal` dataclass with the same field names as
  `strategy.smc_strategy.TradeSignal` (signal_type, entry_price, sl_price,
  confluence_score, atr_value, rsi_value, adx, bb_pctb, ...) so
  `StrategyBacktester._resolve_trade` can consume it without changes.
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from config.settings import bot_config, get_symbol_config
from strategy.indicators import TechnicalIndicators


_QUIET = os.environ.get("SMC_QUIET", "0") == "1"


def _qprint(*args, **kwargs):
    if not _QUIET:
        print(*args, **kwargs)


class MRSignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_SIGNAL = "NO_SIGNAL"


# v8.0.6: legacy aliases — MRSignal/MRSignalType are the canonical names.
# Kept for files that historically imported `TradeSignal`/`SignalType` from
# `strategy.smc_strategy`. Type-equivalent: MRSignal mimics the old
# TradeSignal field set exactly (signal_type, entry_price, sl_price,
# confluence_score, atr_value, rsi_value, bb_pctb, adx, ...).
SignalType = MRSignalType    # noqa: E305 — alias for legacy code paths


@dataclass
class MRSignal:
    """Mean-reversion signal — field-compatible with strategy.smc_strategy.TradeSignal.

    Backtester reads `signal_type.value`, `entry_price`, `sl_price`,
    `confluence_score`, `atr_value`, `rsi_value`, `bb_pctb`, `adx`, etc.
    Mirroring the SMC schema lets us reuse `_resolve_trade` and pool format.
    """

    signal_type: MRSignalType
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    confluence_score: float
    atr_value: float
    sl_distance: float
    market_bias: int = 0
    trend: int = 0
    rsi_value: float = 50.0
    trend_strength: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    stoch_k: float = 50.0
    bb_pctb: float = 0.5
    atr_change_ratio: float = 0.0
    price_roc: float = 0.0
    ob_score: float = 0.0
    ob_high: Optional[float] = None
    ob_low: Optional[float] = None

    # MR-specific telemetry (not used by env directly but logged for analysis)
    mr_setup_score: float = 0.0
    bb_extreme: float = 0.0
    bb_band_width_atr: float = 0.0
    reversal_wick_ratio: float = 0.0
    adx_block_active: bool = False

    reasons: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.signal_type != MRSignalType.NO_SIGNAL

    # ─── Derived properties (v7.2.2 parity with legacy TradeSignal) ────
    # These are needed by ML/RL feature extractors that historically read
    # them as raw attributes off TradeSignal. Without them, GBM features
    # would silently get 0.0 and KS-drift would spike to 1.0.

    @property
    def atr_pips(self) -> float:
        if self.atr_value <= 0:
            return 0.0
        sym = self.symbol.upper()
        is_metal = "XAU" in sym or "XAG" in sym
        is_jpy = "JPY" in sym
        pip_size = 0.01 if (is_metal or is_jpy) else 0.0001
        return self.atr_value / pip_size

    @property
    def sl_distance_atr(self) -> float:
        return self.sl_distance / max(self.atr_value, 1e-9)

    @property
    def bias_alignment(self) -> float:
        direction = 1.0 if self.signal_type == MRSignalType.BUY else -1.0
        return direction * float(self.market_bias)

    @property
    def ob_size_atr(self) -> float:
        if self.ob_high is None or self.ob_low is None:
            return 0.0
        return abs(self.ob_high - self.ob_low) / max(self.atr_value, 1e-9)

    @property
    def direction(self) -> float:
        return 1.0 if self.signal_type == MRSignalType.BUY else -1.0


# v8.0.6: legacy alias — `TradeSignal` is the historical name. MRSignal
# already mimics its field set + adds MR-specific extras (bb_extreme,
# bb_band_width_atr, etc.). Code that imports `TradeSignal` from the old
# location should now import from `mean_reversion_strategy`.
TradeSignal = MRSignal    # noqa: E305


class MeanReversionStrategy:
    """Mean-reversion entry engine with ADX trend-filter veto.

    Lightweight: relies only on `TechnicalIndicators` + ADX H1 from upstream.
    No order-block / FVG / sweep / market-structure dependencies.
    """

    # ─── Tunables (relaxed v8.0.2 after 2 pilots: target ≥ 25 sig/ep) ──────
    # RL must see enough variety to learn TAKE/SKIP discrimination. Wider entry
    # gates here + RL/ML filter quality ที่ training time.
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 40.0        # was 30 — looser oversold
    RSI_OVERBOUGHT: float = 60.0      # was 70 — looser overbought
    BB_OVERSOLD: float = 0.30         # was 0.10 — entry on lower-half pullback
    BB_OVERBOUGHT: float = 0.70       # was 0.90 — entry on upper-half pullback
    ADX_TREND_BLOCK: float = 30.0     # was 25 — only block extreme trends
    SL_ATR_MULT: float = 1.0          # tight SL — capital preservation
    RR_RATIO: float = 1.0             # 1:1 quick-TP target
    MIN_REVERSAL_WICK_RATIO: float = 0.4  # was 1.2 — RL learns to filter weak wicks
    MIN_CONFLUENCE_SCORE: float = 30.0    # was 50 — wider acceptance, RL filters

    def __init__(self, indicators: Optional[TechnicalIndicators] = None):
        self._indicators = indicators or TechnicalIndicators()
        # pull MR config overrides if present (config-aware tuning)
        mr_cfg = getattr(bot_config, "mr", None)
        if mr_cfg is not None:
            self.BB_PERIOD = int(getattr(mr_cfg, "bb_period", self.BB_PERIOD))
            self.BB_STD = float(getattr(mr_cfg, "bb_std", self.BB_STD))
            self.RSI_OVERSOLD = float(getattr(mr_cfg, "rsi_oversold", self.RSI_OVERSOLD))
            self.RSI_OVERBOUGHT = float(getattr(mr_cfg, "rsi_overbought", self.RSI_OVERBOUGHT))
            self.BB_OVERSOLD = float(getattr(mr_cfg, "bb_oversold", self.BB_OVERSOLD))
            self.BB_OVERBOUGHT = float(getattr(mr_cfg, "bb_overbought", self.BB_OVERBOUGHT))
            self.ADX_TREND_BLOCK = float(getattr(mr_cfg, "adx_trend_block", self.ADX_TREND_BLOCK))
            self.SL_ATR_MULT = float(getattr(mr_cfg, "sl_atr_mult", self.SL_ATR_MULT))
            self.RR_RATIO = float(getattr(mr_cfg, "rr_ratio", self.RR_RATIO))
            self.MIN_REVERSAL_WICK_RATIO = float(
                getattr(mr_cfg, "min_reversal_wick_ratio", self.MIN_REVERSAL_WICK_RATIO)
            )

    # ─── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _digits_for(symbol: str) -> int:
        sym = symbol.upper()
        if "XAU" in sym or "XAG" in sym:
            return 2
        if "JPY" in sym:
            return 3
        return 5

    @staticmethod
    def _pip_size_for(symbol: str, last_close: float) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "XAG" in sym:
            return 0.01
        if "JPY" in sym:
            return 0.01
        # autodetect: JPY-style price (>50) → 0.01, otherwise 0.0001
        return 0.01 if last_close > 50 else 0.0001

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute indicators if not already present."""
        needed = {"atr", "rsi", "bb_pctb", "adx", "stoch_k",
                  "macd_histogram", "atr_change_ratio", "price_roc"}
        if not needed.issubset(set(df.columns)):
            df = self._indicators.calculate_all(df)
        return df

    @staticmethod
    def _no_signal(symbol: str, atr_value: float, reasons: List[str]) -> MRSignal:
        return MRSignal(
            signal_type=MRSignalType.NO_SIGNAL,
            symbol=symbol,
            entry_price=0.0,
            sl_price=0.0,
            tp_price=0.0,
            confluence_score=0.0,
            atr_value=atr_value,
            sl_distance=0.0,
            reasons=reasons,
        )

    @staticmethod
    def _reversal_wick_ratio(bar: pd.Series, direction: str) -> float:
        """How big the rejection wick is vs the body.

        BUY  → lower wick = (open|close min) - low
        SELL → upper wick = high - (open|close max)
        Returns 0 if body is essentially zero (doji = no clear rejection).
        """
        body = abs(float(bar["close"]) - float(bar["open"]))
        if body < 1e-12:
            return 0.0
        if direction == "BUY":
            wick = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
        else:
            wick = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
        return max(0.0, wick) / body

    # ─── Main entry ────────────────────────────────────────────────────

    def analyze_with_data(
        self,
        symbol: str,
        h4_df: pd.DataFrame,    # unused (kept for interface compat)
        h1_df: pd.DataFrame,    # used for ADX H1 trend filter
        ltf_df: pd.DataFrame,   # M15 entry timeframe
        price_info: Dict,
    ) -> MRSignal:
        """Scan one bar for MR setup. Returns NO_SIGNAL if filter blocks."""
        if ltf_df is None or len(ltf_df) < max(self.BB_PERIOD, 200):
            return self._no_signal(symbol, 0.0, ["ltf_df too short"])

        ltf_df = self._ensure_indicators(ltf_df)

        last = ltf_df.iloc[-1]
        last_close = float(last["close"])
        atr_value = float(last.get("atr", 0.0))
        if atr_value <= 0 or not np.isfinite(atr_value):
            return self._no_signal(symbol, atr_value, ["ATR invalid"])

        pip_size = self._pip_size_for(symbol, last_close)
        atr_pips = atr_value / pip_size

        # 1. ATR floor — skip dead market
        is_metal = "XAU" in symbol.upper() or "XAG" in symbol.upper()
        default_floor = 100.0 if is_metal else 5.0
        atr_floor_pips = float(get_symbol_config(symbol, "atr_floor_pips", default_floor))
        if atr_pips < atr_floor_pips:
            return self._no_signal(
                symbol, atr_value,
                [f"ATR {atr_pips:.1f} pips < floor {atr_floor_pips}"],
            )

        # 2. ADX H1 trend filter — block MR entries when trending hard
        adx_h1 = 0.0
        if h1_df is not None and len(h1_df) > 20:
            if "adx" not in h1_df.columns:
                h1_df = self._indicators.calculate_adx(h1_df.copy())
            adx_h1 = float(h1_df["adx"].iloc[-1] or 0.0)
        adx_block = adx_h1 > self.ADX_TREND_BLOCK
        if adx_block:
            return self._no_signal(
                symbol, atr_value,
                [f"ADX H1 {adx_h1:.1f} > {self.ADX_TREND_BLOCK} → trend block"],
            )

        # 3. BB %B + 4. RSI — pick direction
        bb_pctb = float(last.get("bb_pctb", 0.5))
        rsi_value = float(last.get("rsi", 50.0))
        adx_m15 = float(last.get("adx", 0.0))

        direction: Optional[str] = None
        bb_extreme = 0.0
        rsi_extreme = 0.0
        if bb_pctb <= self.BB_OVERSOLD and rsi_value <= self.RSI_OVERSOLD:
            direction = "BUY"
            bb_extreme = (self.BB_OVERSOLD - bb_pctb) / max(self.BB_OVERSOLD, 1e-6)
            rsi_extreme = (self.RSI_OVERSOLD - rsi_value) / max(self.RSI_OVERSOLD, 1e-6)
        elif bb_pctb >= self.BB_OVERBOUGHT and rsi_value >= self.RSI_OVERBOUGHT:
            direction = "SELL"
            bb_extreme = (bb_pctb - self.BB_OVERBOUGHT) / max(1.0 - self.BB_OVERBOUGHT, 1e-6)
            rsi_extreme = (rsi_value - self.RSI_OVERBOUGHT) / max(100.0 - self.RSI_OVERBOUGHT, 1e-6)

        if direction is None:
            return self._no_signal(
                symbol, atr_value,
                [f"No MR extreme (bb_pctb={bb_pctb:.2f}, rsi={rsi_value:.1f})"],
            )

        # 5. Rejection-wick confirmation on the last bar
        wick_ratio = self._reversal_wick_ratio(last, direction)
        if wick_ratio < self.MIN_REVERSAL_WICK_RATIO:
            return self._no_signal(
                symbol, atr_value,
                [f"Reversal wick {wick_ratio:.2f} < {self.MIN_REVERSAL_WICK_RATIO}"],
            )

        # ─── Build entry / SL / TP ─────────────────────────────────────
        sl_atr_mult = float(get_symbol_config(
            symbol, "mr_sl_atr_multiplier", self.SL_ATR_MULT
        ))
        sl_distance = atr_value * sl_atr_mult

        # Per-symbol min_sl_pips guard (reuse SMC config — same broker rules)
        min_sl_pips = float(get_symbol_config(
            symbol,
            "min_sl_pips",
            (1000.0 if is_metal else 10.0),
        ))
        if sl_distance < min_sl_pips * pip_size:
            sl_distance = min_sl_pips * pip_size

        rr = float(get_symbol_config(symbol, "mr_rr_ratio", self.RR_RATIO))
        tp_distance = sl_distance * rr

        digits = self._digits_for(symbol)
        if direction == "BUY":
            entry_price = float(price_info.get("ask", last_close))
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
            signal_type = MRSignalType.BUY
            direction_int = 1
        else:
            entry_price = float(price_info.get("bid", last_close))
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)
            signal_type = MRSignalType.SELL
            direction_int = -1

        # ─── Confluence score (0-100, MR-specific) ─────────────────────
        # 40 base for valid setup + up to 30 from BB extremity + up to 20 from
        # RSI extremity + up to 10 from wick strength. ADX H1 already filtered.
        score = 40.0
        score += 30.0 * float(np.clip(bb_extreme, 0.0, 1.0))
        score += 20.0 * float(np.clip(rsi_extreme, 0.0, 1.0))
        score += 10.0 * float(np.clip((wick_ratio - self.MIN_REVERSAL_WICK_RATIO)
                                       / 2.0, 0.0, 1.0))
        score = float(np.clip(score, 0.0, 100.0))

        # MR is contrarian: when last_close above mid (above bb_pctb=0.5) and we
        # are SELLing, market_bias = -1 (we sell into resistance). Map to ±1.
        market_bias = -direction_int   # contrarian — we trade against momentum

        # BB band width / ATR (regime info — wide bands = high vol)
        bb_period = self.BB_PERIOD
        if len(ltf_df) >= bb_period:
            sma = float(ltf_df["close"].iloc[-bb_period:].mean())
            std = float(ltf_df["close"].iloc[-bb_period:].std() or 0.0)
            band_width = 2.0 * self.BB_STD * std
            bb_band_width_atr = band_width / max(atr_value, 1e-9)
        else:
            bb_band_width_atr = 0.0

        if score < self.MIN_CONFLUENCE_SCORE:
            return self._no_signal(
                symbol, atr_value,
                [f"Score {score:.1f} < {self.MIN_CONFLUENCE_SCORE}"],
            )

        return MRSignal(
            signal_type=signal_type,
            symbol=symbol,
            entry_price=round(entry_price, digits),
            sl_price=sl_price,
            tp_price=tp_price,
            confluence_score=score,
            atr_value=atr_value,
            sl_distance=sl_distance,
            market_bias=market_bias,
            trend=int(last.get("trend", 0) or 0),
            rsi_value=rsi_value,
            trend_strength=float(last.get("trend_strength", 0.0) or 0.0),
            macd_histogram=float(last.get("macd_histogram", 0.0) or 0.0),
            adx=adx_m15,
            stoch_k=float(last.get("stoch_k", 50.0) or 50.0),
            bb_pctb=bb_pctb,
            atr_change_ratio=float(last.get("atr_change_ratio", 0.0) or 0.0),
            price_roc=float(last.get("price_roc", 0.0) or 0.0),
            ob_score=0.0,
            ob_high=None,
            ob_low=None,
            mr_setup_score=score,
            bb_extreme=bb_extreme,
            bb_band_width_atr=bb_band_width_atr,
            reversal_wick_ratio=wick_ratio,
            adx_block_active=False,
            reasons=[
                f"BB%B={bb_pctb:.2f} (extreme={bb_extreme:.2f})",
                f"RSI={rsi_value:.1f} (extreme={rsi_extreme:.2f})",
                f"Wick ratio={wick_ratio:.2f}",
                f"ADX H1={adx_h1:.1f} (block={self.ADX_TREND_BLOCK})",
                f"SL={sl_distance:.5f} TP={tp_distance:.5f} RR={rr:.2f}",
            ],
        )


# ─── Live MR Scanner — drop-in replacement for SMCStrategy in main.py ─────


class LiveMRScanner:
    """Live MR scanner — SMC-compatible interface (`scan_all_symbols`).

    Wraps `MeanReversionStrategy` and exposes the same surface that
    `FTMOTradingBot` expects from `SMCStrategy`:

      - `scan_all_symbols() -> List[MRSignal]`
      - `_ltf_data` (last M15 dataframe; used by Chronos forecaster)
      - `_mtf_data`, `_htf_data`, `_htf_bias` (placeholders for context)
      - `MIN_CONFLUENCE_SCORE` class attr (mirrors live config)

    Designed to be drop-in: `main.py` only needs to import this instead of
    `SMCStrategy`. `MRSignal` already mimics `TradeSignal` field names so the
    rest of the live path (executor, logger, observation builder) does not
    need to change shapes — only semantics on a few obs slots.
    """

    MIN_CONFLUENCE_SCORE: float = 30.0
    SCAN_BARS_M15: int = 200       # M15 bars needed for indicators
    SCAN_BARS_H1: int = 200
    SCAN_BARS_H4: int = 100        # MR ignores H4 but kept for parity

    def __init__(self, connector):
        self._connector = connector
        self._indicators = TechnicalIndicators()
        self._strategy = MeanReversionStrategy(indicators=self._indicators)
        # Internal state mirrors SMCStrategy attrs that main.py reads
        self._ltf_data = None      # last M15 dataframe (for Chronos)
        self._mtf_data = None      # last H1
        self._htf_data = None      # last H4
        self._htf_bias = 0
        self._d1_bias_cache = {}
        # Accept the symbols list from settings — same as SMC
        from config.settings import bot_config
        self._symbols = list(bot_config.symbols.symbols)

    # ─── Public API ───────────────────────────────────────────────────

    def scan_all_symbols(self) -> List["MRSignal"]:
        """Scan every configured symbol for an MR setup.

        Returns a list of valid `MRSignal` objects (NO_SIGNAL filtered out).
        Mirrors the contract of `SMCStrategy.scan_all_symbols`.
        """
        results: List[MRSignal] = []
        for symbol in self._symbols:
            sig = self._scan_one_symbol(symbol)
            if sig is not None and sig.is_valid:
                results.append(sig)
        # Sort by confluence so executor sees best setups first
        results.sort(key=lambda s: s.confluence_score, reverse=True)
        return results

    def _scan_one_symbol(self, symbol: str) -> Optional["MRSignal"]:
        """Fetch OHLCV + run MR strategy for one symbol."""
        try:
            m15 = self._connector.get_ohlcv(symbol, "M15", self.SCAN_BARS_M15)
            h1 = self._connector.get_ohlcv(symbol, "H1", self.SCAN_BARS_H1)
            h4 = self._connector.get_ohlcv(symbol, "H4", self.SCAN_BARS_H4)
        except Exception:
            return None

        if m15 is None or h1 is None or len(m15) < self.SCAN_BARS_M15:
            return None

        # Compute indicators on M15 + H1 (H4 ignored for MR — keep nominal)
        try:
            m15 = self._indicators.calculate_all(m15)
            if "adx" not in h1.columns:
                h1 = self._indicators.calculate_adx(h1.copy())
        except Exception:
            return None

        # Cache for context (Chronos forecaster + obs builders)
        self._ltf_data = m15
        self._mtf_data = h1
        self._htf_data = h4

        # Build price_info from the freshest bid/ask if available
        try:
            price_info = self._connector.get_current_price(symbol) or {}
        except Exception:
            price_info = {}
        if not price_info:
            last_close = float(m15["close"].iloc[-1])
            pip_size = self._strategy._pip_size_for(symbol, last_close)
            price_info = {
                "bid": last_close,
                "ask": last_close + pip_size * 2,
                "spread": pip_size * 2,
            }

        return self._strategy.analyze_with_data(symbol, h4, h1, m15, price_info)

    # ─── Methods main.py expects from SMCStrategy ───────────────────

    def _get_d1_bias(self, symbol: str) -> int:
        """MR doesn't use D1 bias as a hard filter — return 0 (neutral).

        Kept for `_build_live_context` compatibility.
        """
        return self._d1_bias_cache.get(symbol, 0)


class _MRStructureProxy:
    """Stub returned by `LiveMRScanner._structure_mtf` to avoid AttributeError
    in contexts that call `get_current_bias()`. MR doesn't track structure
    biases; returns 0 (neutral)."""

    @staticmethod
    def get_current_bias() -> int:
        return 0


# Attach as class attr after class definition (avoids fwd-ref headache)
LiveMRScanner._structure_mtf = _MRStructureProxy()
