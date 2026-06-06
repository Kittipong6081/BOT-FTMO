"""
===============================================================================
StructureEngine — BOS / CHOCH detection (SMC)
===============================================================================
Consumes swings from `SwingEngine` and the candle CLOSES to detect structure
breaks. This is where the BODY-CLOSE confirmation lives (swings are found on
wicks; breaks are confirmed on closes).

Definitions (unambiguous):
  • BOS (Break of Structure) = a candle CLOSES beyond the most recent swing in
    the SAME direction as the prevailing trend → continuation.
  • CHOCH (Change of Character) = the FIRST candle CLOSE beyond the protected
    swing AGAINST the prevailing trend → potential reversal. CHOCH flips bias.

No-lookahead: a swing only becomes an eligible break reference at its
`confirm_index`. A reference can be broken once; a fresh swing of that type must
form before it can be broken again.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .types import Direction, StructureEvent, StructureKind, SwingPoint, SwingType


@dataclass
class StructureState:
    events: List[StructureEvent] = field(default_factory=list)
    bias: Optional[Direction] = None

    @property
    def last_event(self) -> Optional[StructureEvent]:
        return self.events[-1] if self.events else None

    def last_of(self, kind: StructureKind) -> Optional[StructureEvent]:
        for ev in reversed(self.events):
            if ev.kind is kind:
                return ev
        return None


class StructureEngine:
    def analyze(self, df: pd.DataFrame, swings: List[SwingPoint]) -> StructureState:
        state = StructureState()
        if not swings:
            return state

        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        ordered = sorted(swings, key=lambda s: s.confirm_index)
        p = 0

        cur_high: Optional[SwingPoint] = None
        cur_low: Optional[SwingPoint] = None
        high_broken = False
        low_broken = False
        bias: Optional[Direction] = None

        for i in range(n):
            # publish swings as they become knowable
            while p < len(ordered) and ordered[p].confirm_index <= i:
                s = ordered[p]
                p += 1
                if s.swing_type is SwingType.HIGH:
                    if cur_high is None or s.index > cur_high.index:
                        cur_high = s
                        high_broken = False
                else:
                    if cur_low is None or s.index > cur_low.index:
                        cur_low = s
                        low_broken = False

            c = close[i]
            if cur_high is not None and not high_broken and c > cur_high.price:
                kind = StructureKind.CHOCH if bias is Direction.BEARISH else StructureKind.BOS
                state.events.append(
                    StructureEvent(i, kind, Direction.BULLISH, cur_high.price, cur_high.index)
                )
                bias = Direction.BULLISH
                high_broken = True
            elif cur_low is not None and not low_broken and c < cur_low.price:
                kind = StructureKind.CHOCH if bias is Direction.BULLISH else StructureKind.BOS
                state.events.append(
                    StructureEvent(i, kind, Direction.BEARISH, cur_low.price, cur_low.index)
                )
                bias = Direction.BEARISH
                low_broken = True

        state.bias = bias
        return state
