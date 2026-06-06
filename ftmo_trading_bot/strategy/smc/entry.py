"""
===============================================================================
EntryEngine — strict sequential SMC entry state machine
===============================================================================
This is the fix for the old SMC's #1 failure (a confluence-points soup). Every
stage is a PREREQUISITE checked IN ORDER; if any fails, no signal:

  1. HTF bias clear            (BiasEngine on D1) — trade only WITH it
  2. price in discount/premium (HTF dealing range) aligned with bias
  3. liquidity sweep on LTF    (wick beyond a level, close back in) in the
                                 reversal direction == HTF bias
  4. LTF CHOCH/BOS after the sweep, in the bias direction (the MSS confirmation)
  5. entry zone: an unmitigated LTF OB or FVG (bias direction) price is retracing
                 into, sitting in discount/premium
  6. RR gate: SL behind the swept level / OB, TP at the HTF liquidity target

SL/TP are STRUCTURAL (not fixed pips). The expectancy fix vs old SMC: tight
structural SL + liquidity-based TP + the sweep+CHOCH filter (raises base win
rate), validated later on the realistic FTMO sim.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .bias import BiasEngine
from .fvg import FVGEngine
from .liquidity import LiquidityEngine
from .order_block import OrderBlockEngine
from .structure import StructureEngine
from .swing import SwingEngine, _atr
from .types import (
    Direction,
    SMCSignal,
    SMCSignalType,
    StructureKind,
)


@dataclass
class SMCEntryConfig:
    sweep_window: int = 12       # LTF bars: sweep must be this recent
    choch_window: int = 12       # LTF bars: CHOCH must follow the sweep within this
    zone_tol_atr: float = 0.25   # price must be within this ×ATR of the entry zone
    sl_buffer_atr: float = 0.10  # SL placed this ×ATR beyond the structural level
    min_rr: float = 1.5
    atr_period: int = 14
    require_ote: bool = False     # if True, entry must be in the 0.62–0.79 OTE band
    min_ltf_bars: int = 40
    min_htf_bars: int = 20


@dataclass
class EntryResult:
    signal: Optional[SMCSignal] = None
    stage: str = ""              # deepest stage reached
    reason: str = ""             # rejection reason (empty if signal produced)
    confluences: List[str] = field(default_factory=list)


class EntryEngine:
    def __init__(self, config: Optional[SMCEntryConfig] = None,
                 swing: Optional[SwingEngine] = None) -> None:
        self.cfg = config or SMCEntryConfig()
        self.swing = swing or SwingEngine()
        self.structure = StructureEngine()
        self.liquidity = LiquidityEngine()
        self.fvg = FVGEngine()
        self.order_block = OrderBlockEngine()
        self.bias_engine = BiasEngine(self.swing)

    def evaluate(self, symbol: str, htf_df: pd.DataFrame, ltf_df: pd.DataFrame,
                 *, pip_size: float) -> EntryResult:
        res = EntryResult()
        if (ltf_df is None or len(ltf_df) < self.cfg.min_ltf_bars
                or htf_df is None or len(htf_df) < self.cfg.min_htf_bars):
            res.reason = "insufficient_data"
            return res

        # ── stage 1: HTF bias ──────────────────────────────────────────
        mb = self.bias_engine.analyze(htf_df)
        res.stage = "htf_bias"
        if mb.direction is None or mb.dealing_range is None:
            res.reason = "no_htf_bias"
            return res
        bias = mb.direction
        price = float(ltf_df["close"].iloc[-1])

        # ── stage 2: discount / premium aligned with bias ──────────────
        res.stage = "zone"
        if bias is Direction.BULLISH and not mb.dealing_range.in_discount(price):
            res.reason = "price_not_in_discount"
            return res
        if bias is Direction.BEARISH and not mb.dealing_range.in_premium(price):
            res.reason = "price_not_in_premium"
            return res
        if self.cfg.require_ote and not mb.dealing_range.in_ote(price, bias):
            res.reason = "not_in_ote"
            return res

        # ── LTF analysis ───────────────────────────────────────────────
        atr = _atr(ltf_df, self.cfg.atr_period)
        atr_now = float(atr[-1])
        swings = self.swing.analyze(ltf_df)
        struct = self.structure.analyze(ltf_df, swings)
        sweeps = self.liquidity.sweeps(ltf_df, swings)
        last_i = len(ltf_df) - 1

        # ── stage 3: recent liquidity sweep in the bias direction ──────
        res.stage = "sweep"
        sweep = next((s for s in reversed(sweeps)
                      if s.direction is bias and last_i - s.index <= self.cfg.sweep_window), None)
        if sweep is None:
            res.reason = "no_recent_sweep"
            return res
        res.confluences.append("sweep")

        # ── stage 4: CHOCH/BOS after the sweep, in the bias direction ──
        res.stage = "choch"
        mss = next((e for e in reversed(struct.events)
                    if e.direction is bias and e.index >= sweep.index
                    and last_i - e.index <= self.cfg.choch_window), None)
        if mss is None:
            res.reason = "no_mss_after_sweep"
            return res
        res.confluences.append(mss.kind.value)

        # ── stage 5: entry zone (OB preferred, FVG fallback), unmitigated ──
        res.stage = "entry_zone"
        obs = [o for o in self.order_block.detect(ltf_df, struct)
               if o.direction is bias and not o.mitigated and o.confirm_index >= sweep.index]
        fvgs = [f for f in self.fvg.detect(ltf_df)
                if f.direction is bias and not f.filled and f.index >= sweep.index]
        zone_top = zone_bottom = None
        ob_grade = 0.0
        fvg_mit = 0.0
        has_ob = has_fvg = False
        if obs:
            ob = max(obs, key=lambda o: o.grade)
            zone_top, zone_bottom, ob_grade, has_ob = ob.top, ob.bottom, ob.grade, True
            res.confluences.append("OB")
        elif fvgs:
            f = fvgs[-1]
            zone_top, zone_bottom, fvg_mit, has_fvg = f.top, f.bottom, f.mitigation, True
            res.confluences.append("FVG")
        else:
            res.reason = "no_entry_zone"
            return res

        # price must be retracing INTO the zone (within tolerance)
        tol = self.cfg.zone_tol_atr * atr_now
        if not (zone_bottom - tol <= price <= zone_top + tol):
            res.reason = "price_not_at_zone"
            return res

        # ── stage 6: structural SL/TP + RR gate ────────────────────────
        res.stage = "rr"
        buf = self.cfg.sl_buffer_atr * atr_now
        if bias is Direction.BULLISH:
            sl = min(zone_bottom, sweep.level) - buf
            entry = price
            tp = mb.target_liquidity.price if mb.target_liquidity else self._fallback_tp(
                swings, bias, entry)
            risk = entry - sl
            reward = tp - entry
        else:
            sl = max(zone_top, sweep.level) + buf
            entry = price
            tp = mb.target_liquidity.price if mb.target_liquidity else self._fallback_tp(
                swings, bias, entry)
            risk = sl - entry
            reward = entry - tp

        if risk <= 0 or reward <= 0:
            res.reason = "bad_geometry"
            return res
        rr = reward / risk
        if rr < self.cfg.min_rr:
            res.reason = f"rr_too_low({rr:.2f})"
            return res

        # ── build signal ───────────────────────────────────────────────
        sig = SMCSignal(
            signal_type=SMCSignalType.BUY if bias is Direction.BULLISH else SMCSignalType.SELL,
            symbol=symbol,
            entry_price=round(entry, 6),
            sl_price=round(sl, 6),
            tp_price=round(tp, 6),
            confluence_score=self._confluence_score(res.confluences, ob_grade, rr),
            atr_value=atr_now,
            sl_distance=abs(entry - sl),
            rr_ratio=round(rr, 3),
            market_bias=bias.sign,
            htf_bias=bias.sign,
            structure_event=mss.kind.value,
            swept_liquidity=sweep.level,
            sweep_age_bars=last_i - sweep.index,
            ob_grade=ob_grade,
            fvg_mitigation=fvg_mit,
            entry_zone_pos=mb.dealing_range.position(price),
            has_ob=has_ob,
            has_fvg=has_fvg,
            mss_confirmed=mss.kind is StructureKind.CHOCH,
            timestamp=ltf_df["time"].iloc[-1] if "time" in ltf_df.columns else None,
            reasons=list(res.confluences),
        )
        res.signal = sig
        res.reason = ""
        return res

    @staticmethod
    def _fallback_tp(swings, bias: Direction, entry: float) -> float:
        """Next opposing-side swing as a liquidity proxy when HTF target absent."""
        from .types import SwingType
        if bias is Direction.BULLISH:
            highs = [s.price for s in swings if s.swing_type is SwingType.HIGH and s.price > entry]
            return min(highs) if highs else entry * 1.01
        lows = [s.price for s in swings if s.swing_type is SwingType.LOW and s.price < entry]
        return max(lows) if lows else entry * 0.99

    @staticmethod
    def _confluence_score(confluences: List[str], ob_grade: float, rr: float) -> float:
        """0–100 quality score from how much of the sequence aligned.
        Not used as an entry gate (the state machine is) — only ranking + obs."""
        score = 30.0
        if "sweep" in confluences:
            score += 15
        if "CHOCH" in confluences:
            score += 20
        elif "BOS" in confluences:
            score += 12
        if "OB" in confluences:
            score += 10 + 15 * ob_grade
        if "FVG" in confluences:
            score += 8
        score += min(15.0, (rr - 1.0) * 7.5)
        return round(min(100.0, score), 1)
