"""
===============================================================================
BiasEngine — multi-timeframe (HTF) directional bias + liquidity target (SMC)
===============================================================================
Top-down step 1: on the higher timeframe (D1 then H4) establish the institutional
bias from market structure, the dealing range (premium/discount frame), and the
next major liquidity pool the move is likely drawn toward.

The EntryEngine consumes this: it only arms longs when HTF bias is bullish (and
price is in discount), shorts when bearish (price in premium). Trading WITH the
HTF bias — not the old SMC's counter-Daily hard veto — is the fix for the
"D1 veto cut 80% of signals" failure mode.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .liquidity import LiquidityEngine
from .structure import StructureEngine
from .swing import SwingEngine
from .types import DealingRange, Direction, LiquidityLevel, SwingType


@dataclass
class MTFBias:
    direction: Optional[Direction]          # None = no clear bias → stand aside
    dealing_range: Optional[DealingRange]
    target_liquidity: Optional[LiquidityLevel]  # next pool the move targets


class BiasEngine:
    def __init__(self, swing: Optional[SwingEngine] = None) -> None:
        self.swing = swing or SwingEngine()
        self.structure = StructureEngine()
        self.liquidity = LiquidityEngine()

    def analyze(self, htf_df: pd.DataFrame) -> MTFBias:
        swings = self.swing.analyze(htf_df)
        if not swings:
            return MTFBias(None, None, None)
        state = self.structure.analyze(htf_df, swings)
        bias = state.bias
        dr = self.liquidity.dealing_range(swings)
        target = self._target(htf_df, swings, bias)
        return MTFBias(bias, dr, target)

    def _target(self, htf_df: pd.DataFrame, swings, bias: Optional[Direction]
                ) -> Optional[LiquidityLevel]:
        if bias is None:
            return None
        price = float(htf_df["close"].iloc[-1])
        levels = self.liquidity.levels(htf_df, swings)
        if bias is Direction.BULLISH:
            # next buy-side pool ABOVE price (a swing high to draw toward)
            above = [lv for lv in levels
                     if lv.direction is Direction.BULLISH and lv.price > price]
            return min(above, key=lambda lv: lv.price) if above else None
        below = [lv for lv in levels
                 if lv.direction is Direction.BEARISH and lv.price < price]
        return max(below, key=lambda lv: lv.price) if below else None
