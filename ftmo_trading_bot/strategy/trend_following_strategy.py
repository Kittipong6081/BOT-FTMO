"""
===============================================================================
FTMO Trading Bot — Trend Following Strategy (v8.1 — parallel to MR)
===============================================================================
Trend-continuation strategy that runs in PARALLEL with Mean Reversion (MR),
gated by the per-symbol `MarketRegimeClassifier` (TF arms only on TRENDING).

Rules (entry on H1, trend filter from H4/D1):
  1. Strong trend     → ADX H1 >= adx_entry_min (default 27)
  2. EMA alignment    → EMA21>50>200 + close>EMA200 (BUY) / mirror (SELL)
  3. HTF agreement    → H4 trend agrees (or neutral); D1 bias soft-confluence
  4. Pullback-resume  → price pulled back toward EMA21 (within pullback_max_atr
                        × ATR) and is resuming in the trend direction; RSI not
                        stretched against us (BUY: RSI < rsi_pullback_buy_max)
  5. Momentum confirm → MACD histogram sign agrees (when macd_confirm)

Stop / Target (WIDE — opposite of MR's tight quick-TP, let winners run):
  • SL = ATR × sl_atr_mult (default 2.0, XAU 2.5)
  • TP = SL × rr_ratio (default 2.5)

Output: `TFSignal` — a subclass of `MRSignal` (so the executor/logger/obs path
consumes it unchanged) carrying `strategy_id="TF"` + TF telemetry
(trend_age_bars, pullback_depth_atr, adx_at_entry).
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import bot_config, get_symbol_config
from strategy.indicators import TechnicalIndicators
from strategy.strategy_base import StrategyBase
from strategy.mean_reversion_strategy import MRSignal, MRSignalType


_QUIET = os.environ.get("SMC_QUIET", "0") == "1"


@dataclass
class TFSignal(MRSignal):
    """Trend-following signal — field-compatible with MRSignal (== TradeSignal)
    so the live executor/logger/observation path needs no shape change.

    Adds TF-specific telemetry consumed by the Phase 3 TF env/obs and overrides
    `strategy_id` to "TF".
    """
    # TF-specific telemetry
    trend_age_bars: int = 0          # consecutive H1 bars in the current trend
    pullback_depth_atr: float = 0.0  # how deep the pullback dipped (in ATR units)
    adx_at_entry: float = 0.0        # ADX H1 at entry (trend strength)
    htf_align: int = 0               # +1 H4 agrees, 0 neutral, -1 opposes (gated out)
    d1_align: int = 0                # +1/0/-1 D1 bias agreement (soft confluence)
    di_spread: float = 0.0           # +DI − −DI at entry (leading directional signal; early_entry)
    adx_slope: float = 0.0           # ADX change over early_adx_slope_lookback bars (rising = building)

    # Override MR default
    strategy_id: str = "TF"


class TrendFollowingStrategy:
    """Pure trend-following rules engine. Stateless — feed it H1/H4/D1 frames."""

    def __init__(self, indicators: Optional[TechnicalIndicators] = None):
        self._indicators = indicators or TechnicalIndicators()
        cfg = bot_config.tf
        self.ADX_ENTRY_MIN = float(cfg.adx_entry_min)
        self.RR_RATIO = float(cfg.rr_ratio)
        self.SL_ATR_MULT = float(cfg.sl_atr_mult)
        self.SL_ATR_MULT_XAU = float(cfg.sl_atr_mult_xau)
        self.PULLBACK_EMA = int(cfg.pullback_ema)
        self.PULLBACK_MAX_ATR = float(cfg.pullback_max_atr)
        self.RSI_BUY_MAX = float(cfg.rsi_pullback_buy_max)
        self.RSI_SELL_MIN = float(cfg.rsi_pullback_sell_min)
        self.MACD_CONFIRM = bool(cfg.macd_confirm)
        self.MIN_CONFLUENCE = float(cfg.min_confluence_score)
        # Early-entry redesign (flag default OFF → legacy behaviour byte-identical)
        self.EARLY_ENTRY = bool(getattr(cfg, "early_entry", False))
        self.EARLY_DI_SPREAD_MIN = float(getattr(cfg, "early_di_spread_min", 2.0))
        self.EARLY_ADX_SLOPE_LOOKBACK = int(getattr(cfg, "early_adx_slope_lookback", 3))
        self.EARLY_ADX_FLOOR = float(getattr(cfg, "early_adx_floor", 22.0))
        # Fix#2 (2026-06-04) — concentrate on the post-spread +EV band (validation 250-ep pool:
        # +EV only at ADX 22-30 + adx_slope≥~0.5; ADX>33 and weak-slope flip −EV after spread).
        self.EARLY_ADX_SLOPE_MIN = float(getattr(cfg, "early_adx_slope_min", 0.4))  # require trend BUILDING, not barely rising
        self.EARLY_ADX_MAX = float(getattr(cfg, "early_adx_max", 33.0))             # soft upper cap — skip EXHAUSTED trends (NOT a late-entry floor)

    # ─── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pip_size_for(symbol: str, last_close: float) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "XAG" in sym or "JPY" in sym:
            return 0.01
        return 0.01 if last_close > 50 else 0.0001

    @staticmethod
    def _no_signal(symbol: str, atr_value: float, reasons: List[str]) -> TFSignal:
        return TFSignal(
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
    def _trend_age(trend_series: pd.Series, direction: int) -> int:
        """Count consecutive most-recent bars whose trend == direction."""
        age = 0
        for v in reversed(trend_series.tolist()):
            if int(v) == direction:
                age += 1
            else:
                break
        return age

    def _htf_trend(self, htf_df: Optional[pd.DataFrame]) -> int:
        """Trend on a higher timeframe via EMA alignment. 0 if unavailable."""
        if htf_df is None or len(htf_df) < 60:
            return 0
        try:
            df = self._indicators.calculate_all_emas(htf_df.copy())
            last = df.iloc[-1]
            if last["ema_fast"] > last["ema_medium"] > last["ema_slow"]:
                return 1
            if last["ema_fast"] < last["ema_medium"] < last["ema_slow"]:
                return -1
        except Exception:
            return 0
        return 0

    def _d1_bias(self, d1_df: Optional[pd.DataFrame]) -> int:
        """D1 directional bias = sign(close − EMA50). 0 if unavailable."""
        if d1_df is None or len(d1_df) < 55:
            return 0
        try:
            df = self._indicators.calculate_ema(d1_df.copy(), period=50, column_name="ema50")
            last = df.iloc[-1]
            if last["close"] > last["ema50"]:
                return 1
            if last["close"] < last["ema50"]:
                return -1
        except Exception:
            return 0
        return 0

    # ─── main entry ───────────────────────────────────────────────────────

    def analyze_with_data(
        self,
        symbol: str,
        d1_df: Optional[pd.DataFrame],
        h4_df: Optional[pd.DataFrame],
        h1_df: pd.DataFrame,
        price_info: Dict,
    ) -> TFSignal:
        if h1_df is None or len(h1_df) < 200:
            return self._no_signal(symbol, 0.0, ["H1 data insufficient"])

        try:
            h1 = self._indicators.calculate_all(h1_df)
        except Exception as e:
            return self._no_signal(symbol, 0.0, [f"indicator error: {e}"])

        last = h1.iloc[-1]
        prev = h1.iloc[-2]
        atr = float(last.get("atr", 0.0))
        if atr <= 0:
            return self._no_signal(symbol, atr, ["ATR <= 0"])

        adx = float(last.get("adx", 0.0))
        trend = int(last.get("trend", 0))
        rsi = float(last.get("rsi", 50.0))
        macd_h = float(last.get("macd_histogram", 0.0))
        ema21 = float(last.get("ema_fast", last["close"]))
        close = float(last["close"])

        reasons: List[str] = []

        # Leading directional/strength signals — computed from the SAME H1 slice the
        # backtester passes, so they are PARITY-TRIVIAL (no regime state machine needed).
        plus_di = float(last.get("plus_di", 0.0))
        minus_di = float(last.get("minus_di", 0.0))
        di_spread = float(last.get("di_spread", plus_di - minus_di))
        ema_slope = float(last.get("ema_slope_norm", 0.0))
        band_squeeze = float(last.get("band_squeeze_ratio", 0.5))
        ema50 = float(last.get("ema_medium", close))
        # Fix#2 (2026-06-04): 1-day signed efficiency (obs[35]) — mirror MeanReversion/TF
        # backtester (calculate_signed_efficiency(h1, 24)) so live TF obs is 36-dim too.
        trend_eff_h1_val = float(TechnicalIndicators.calculate_signed_efficiency(h1, 24))
        _k = self.EARLY_ADX_SLOPE_LOOKBACK
        adx_slope = adx - (float(h1["adx"].iloc[-1 - _k]) if len(h1) > _k else adx)
        htf_align_raw = self._htf_trend(h4_df)

        if self.EARLY_ENTRY:
            # ── LEADING entry: catch the trend at BIRTH ──────────────────────
            # Direction from +DI/-DI dominance (leads the lagging EMA stack; reflects the
            # ACTUAL current directional pressure — would reject a BUY when -DI>+DI in a crash).
            if di_spread >= self.EARLY_DI_SPREAD_MIN:
                trend, direction = 1, "BUY"
            elif di_spread <= -self.EARLY_DI_SPREAD_MIN:
                trend, direction = -1, "SELL"
            else:
                return self._no_signal(symbol, atr, [f"DI flat (spread {di_spread:+.1f}) — no clear direction"])
            # Trend must be BUILDING (ADX rising meaningfully) — Fix#2: ≥ slope_min, not just >0
            # (post-spread validation: slope 0-0.5 = −EV, slope ≥0.5 = +EV).
            if adx_slope < self.EARLY_ADX_SLOPE_MIN:
                return self._no_signal(symbol, atr, [f"ADX rising too weak (Δ{adx_slope:+.2f} < {self.EARLY_ADX_SLOPE_MIN}) — trend not building"])
            # ADX band: above chop floor, below exhaustion cap. Fix#2 — the cap is a SOFT
            # upper bound to skip EXHAUSTED trends (post-spread: ADX>33 = −EV); it is NOT a
            # late-entry LOWER bound (the old bug). di_spread/adx_slope still in obs for fine-tune.
            if adx < self.EARLY_ADX_FLOOR:
                return self._no_signal(symbol, atr, [f"ADX {adx:.1f} < floor {self.EARLY_ADX_FLOOR} — chop"])
            if adx > self.EARLY_ADX_MAX:
                return self._no_signal(symbol, atr, [f"ADX {adx:.1f} > cap {self.EARLY_ADX_MAX} — trend exhausted"])
            # Fast EMA confirm (EMA21 vs EMA50 + slope) — NOT the slow EMA200 stack.
            fast_ok = (ema21 > ema50 and ema_slope > 0) if trend > 0 else (ema21 < ema50 and ema_slope < 0)
            if not fast_ok:
                return self._no_signal(symbol, atr, ["fast EMA21/EMA50/slope not aligned with DI direction"])
            # EMA200 full-stack + H4 + D1 are SOFT confluence here (scored below, no hard veto).
        else:
            # ── LEGACY entry (lagging level/stack gates) — UNCHANGED ─────────
            # 1) strong-trend gate
            if adx < self.ADX_ENTRY_MIN:
                return self._no_signal(symbol, atr, [f"ADX {adx:.1f} < {self.ADX_ENTRY_MIN}"])
            # 2) EMA-alignment direction
            if trend == 0:
                return self._no_signal(symbol, atr, ["EMA not aligned (no trend)"])
            direction = "BUY" if trend > 0 else "SELL"
            # 3) HTF agreement (H4 must agree or be neutral; D1 soft confluence)
            if htf_align_raw != 0 and htf_align_raw != trend:
                return self._no_signal(symbol, atr, [f"H4 opposes ({htf_align_raw} vs {trend})"])

        htf_align = 1 if htf_align_raw == trend else 0
        d1_raw = self._d1_bias(d1_df)
        d1_align = 1 if d1_raw == trend else (-1 if d1_raw == -trend else 0)

        # 4) pullback toward EMA21 (HARD) + RSI not stretched (HARD).
        # `resuming` (last bar ticking back in trend dir) and MACD agreement are
        # SOFT confluence (not hard gates) — a strict conjunction yields almost no
        # signals (ADX>27 ∧ EMA-stack ∧ pullback ∧ resume ∧ MACD); making resume +
        # MACD confluence bonuses keeps the population wide for both train + live
        # (same gate structure → no train/live distribution gap) and lets the GBM
        # + RL filter rank them. Thresholds are Phase-4 tunable.
        dist_to_ema_atr = abs(close - ema21) / max(atr, 1e-9)
        if dist_to_ema_atr > self.PULLBACK_MAX_ATR:
            return self._no_signal(
                symbol, atr,
                [f"no pullback (dist {dist_to_ema_atr:.2f}×ATR > {self.PULLBACK_MAX_ATR})"],
            )
        if direction == "BUY":
            resuming = close > float(prev["close"])
            rsi_ok = rsi < self.RSI_BUY_MAX
            pullback_depth = (float(last["low"]) - ema21) / max(atr, 1e-9)
            macd_ok = macd_h > 0
        else:
            resuming = close < float(prev["close"])
            rsi_ok = rsi > self.RSI_SELL_MIN
            pullback_depth = (ema21 - float(last["high"])) / max(atr, 1e-9)
            macd_ok = macd_h < 0
        if not rsi_ok:
            return self._no_signal(symbol, atr, [f"RSI {rsi:.1f} stretched against entry"])

        # ─── SL / TP (wide) ───
        sl_mult = self.SL_ATR_MULT_XAU if "XAU" in symbol.upper() else self.SL_ATR_MULT
        sl_distance = atr * sl_mult
        bid = float(price_info.get("bid", close))
        ask = float(price_info.get("ask", close))
        if direction == "BUY":
            entry = ask
            sl_price = entry - sl_distance
            tp_price = entry + sl_distance * self.RR_RATIO
        else:
            entry = bid
            sl_price = entry + sl_distance
            tp_price = entry - sl_distance * self.RR_RATIO

        # ─── confluence score ───
        score = 40.0
        score += min(max(adx - self.ADX_ENTRY_MIN, 0.0), 20.0)        # trend strength up to +20
        score += min(float(last.get("trend_strength", 0.0)) * 0.2, 10.0)  # EMA spread up to +10
        score += 10.0 if htf_align == 1 else 0.0                       # H4 agree
        score += 5.0 if d1_align == 1 else (-5.0 if d1_align == -1 else 0.0)  # D1 bias
        # tighter pullback (closer to EMA) = better entry → up to +10
        score += max(0.0, (self.PULLBACK_MAX_ATR - dist_to_ema_atr)) / max(self.PULLBACK_MAX_ATR, 1e-9) * 10.0
        score += 8.0 if resuming else 0.0                             # SOFT: resuming in trend dir
        score += 7.0 if macd_ok else 0.0                              # SOFT: MACD momentum agrees
        score = float(np.clip(score, 0.0, 100.0))

        if score < self.MIN_CONFLUENCE:
            return self._no_signal(symbol, atr, [f"Score {score:.1f} < {self.MIN_CONFLUENCE}"])

        trend_age = self._trend_age(h1["trend"].tail(60), trend)
        reasons.append(
            f"TF {direction} ADX={adx:.1f} trend_age={trend_age} "
            f"pullback={dist_to_ema_atr:.2f}×ATR H4={htf_align} D1={d1_align}"
        )

        return TFSignal(
            signal_type=MRSignalType.BUY if direction == "BUY" else MRSignalType.SELL,
            symbol=symbol,
            entry_price=entry,
            sl_price=round(sl_price, 5),
            tp_price=round(tp_price, 5),
            confluence_score=score,
            atr_value=atr,
            sl_distance=sl_distance,
            market_bias=trend,
            trend=trend,
            rsi_value=rsi,
            trend_strength=float(last.get("trend_strength", 0.0)),
            macd_histogram=macd_h,
            adx=adx,
            stoch_k=float(last.get("stoch_k", 50.0)),
            bb_pctb=float(last.get("bb_pctb", 0.5)),
            atr_change_ratio=float(last.get("atr_change_ratio", 0.0)),
            price_roc=float(last.get("price_roc", 0.0)),
            trend_age_bars=trend_age,
            pullback_depth_atr=float(pullback_depth),
            adx_at_entry=adx,
            htf_align=htf_align,
            d1_align=d1_align,
            di_spread=di_spread,
            adx_slope=adx_slope,
            trend_eff_h1=trend_eff_h1_val,
            ema_slope_norm=ema_slope,          # real value (was MRSignal default 0.0 for TF)
            band_squeeze_ratio=band_squeeze,   # real value (fixes hardcoded 0.5 in TF pool)
            reasons=reasons,
        )


class TrendFollowingScanner(StrategyBase):
    """Live TF scanner — StrategyBase interface (parallel to LiveMRScanner).

    Owns its OWN per-symbol caches (H1 entry / H4 mtf / D1 htf) — never reads
    MR's M15/H1/H4 caches. `get_ltf_data` returns H1 (TF entry timeframe).
    """

    # StrategyBase identity
    STRATEGY_ID: str = "TF"
    MAGIC_NUMBER: int = 123457
    OBS_LAYOUT_ID: str = "tf_v2"   # v2 (2026-06-02): obs[27]/[28] = di_spread/adx_slope (early-entry)

    def __init__(self, connector):
        self._connector = connector
        self._indicators = TechnicalIndicators()
        self._strategy = TrendFollowingStrategy(indicators=self._indicators)
        cfg = bot_config.tf
        self.SCAN_BARS_H1 = int(cfg.scan_bars_h1)
        self.SCAN_BARS_H4 = int(cfg.scan_bars_h4)
        self.SCAN_BARS_D1 = int(cfg.scan_bars_d1)
        # per-symbol caches (own — no cross-strategy read)
        self._ltf_by_symbol: Dict[str, pd.DataFrame] = {}   # H1 (entry)
        self._mtf_by_symbol: Dict[str, pd.DataFrame] = {}   # H4
        self._htf_by_symbol: Dict[str, pd.DataFrame] = {}   # D1
        from config.settings import bot_config as _bc
        blocked = set(getattr(_bc.symbols, "blocked_symbols", []))
        self._symbols = [s for s in _bc.symbols.symbols if s not in blocked]

    def scan_all_symbols(self, allowed_symbols: Optional[set] = None) -> List["TFSignal"]:
        results: List[TFSignal] = []
        for symbol in self._symbols:
            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue
            sig = self._scan_one_symbol(symbol)
            if sig is not None and sig.is_valid:
                results.append(sig)
        results.sort(key=lambda s: s.confluence_score, reverse=True)
        return results

    def _scan_one_symbol(self, symbol: str) -> Optional["TFSignal"]:
        try:
            h1 = self._connector.get_ohlcv(symbol, "H1", self.SCAN_BARS_H1)
            h4 = self._connector.get_ohlcv(symbol, "H4", self.SCAN_BARS_H4)
            d1 = self._connector.get_ohlcv(symbol, "D1", self.SCAN_BARS_D1)
        except Exception:
            return None
        if h1 is None or len(h1) < 200:
            return None
        try:
            h1 = self._indicators.calculate_all(h1)
        except Exception:
            return None
        # cache per-symbol (own caches)
        self._ltf_by_symbol[symbol] = h1
        self._mtf_by_symbol[symbol] = h4
        self._htf_by_symbol[symbol] = d1
        try:
            price_info = self._connector.get_current_price(symbol) or {}
        except Exception:
            price_info = {}
        if not price_info:
            last_close = float(h1["close"].iloc[-1])
            pip = self._strategy._pip_size_for(symbol, last_close)
            price_info = {"bid": last_close, "ask": last_close + pip * 2, "spread": pip * 2}
        return self._strategy.analyze_with_data(symbol, d1, h4, h1, price_info)

    # ─── per-symbol cache accessors (own caches; semantic: LTF=H1) ───
    def get_ltf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._ltf_by_symbol.get(symbol)

    def get_mtf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._mtf_by_symbol.get(symbol)

    def get_htf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._htf_by_symbol.get(symbol)

    def _get_d1_bias(self, symbol: str) -> int:
        return self._strategy._d1_bias(self._htf_by_symbol.get(symbol))
