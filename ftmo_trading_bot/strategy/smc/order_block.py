"""
===============================================================================
OrderBlockEngine — order block detection + quality grade + mitigation (SMC)
===============================================================================
An Order Block is the LAST opposing candle before a displacement that caused a
Break of Structure:
  • Bullish OB (demand): last bearish candle before an up-displacement that BOS'd
    a swing high. Zone = [low, high] of that candle.
  • Bearish OB (supply): last bullish candle before a down-displacement BOS.

Quality grade [0,1] rewards: structural confirmation (it caused a BOS),
displacement away (>= displacement_atr × ATR), and being unmitigated. We only
keep OBs tied to a real structure break — that is the fix for the old SMC, which
graded internal/consolidation OBs as if they were decisional.

Leakage-safe: an OB's `confirm_index` is the BOS bar; it is only knowable then.
===============================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .structure import StructureState
from .swing import _atr
from .types import Direction, OrderBlock


class OrderBlockEngine:
    def __init__(self, atr_period: int = 14, scan_back: int = 12,
                 displacement_atr: float = 1.0) -> None:
        self.atr_period = atr_period
        self.scan_back = scan_back
        self.displacement_atr = displacement_atr

    def detect(self, df: pd.DataFrame, structure: StructureState) -> List[OrderBlock]:
        open_ = df["open"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        atr = _atr(df, self.atr_period)
        n = len(df)
        out: List[OrderBlock] = []

        for ev in structure.events:
            bull = ev.direction is Direction.BULLISH
            ob_idx = self._last_opposing(open_, close, ev.index, want_bearish=bull)
            if ob_idx is None:
                continue
            top, bottom = float(high[ob_idx]), float(low[ob_idx])
            direction = Direction.BULLISH if bull else Direction.BEARISH
            ob = OrderBlock(direction, top, bottom, index=ob_idx, confirm_index=ev.index)
            self._grade(ob, close, high, low, atr, n)
            out.append(ob)
        return out

    def _last_opposing(self, open_, close, break_idx: int, want_bearish: bool):
        lo = max(0, break_idx - self.scan_back)
        for j in range(break_idx, lo - 1, -1):
            is_bearish = close[j] < open_[j]
            is_bullish = close[j] > open_[j]
            if (want_bearish and is_bearish) or (not want_bearish and is_bullish):
                return j
        return None

    def _grade(self, ob: OrderBlock, close, high, low, atr, n: int) -> None:
        grade = 0.5  # caused a BOS (structural confirmation)
        # displacement away from the OB
        move = abs(float(close[ob.confirm_index]) - float(close[ob.index]))
        if move >= self.displacement_atr * float(atr[ob.confirm_index]):
            grade += 0.25
        # mitigation after confirmation
        start = ob.confirm_index + 1
        size = ob.top - ob.bottom
        if start < n and size > 0:
            after_low = low[start:]
            after_high = high[start:]
            after_close = close[start:]
            if ob.direction is Direction.BULLISH:
                depth = ob.top - float(after_low.min())
                ob.mitigated = bool((after_close < ob.bottom).any())
            else:
                depth = float(after_high.max()) - ob.bottom
                ob.mitigated = bool((after_close > ob.top).any())
            ob.mitigation = float(min(1.0, max(0.0, depth / size)))
        if not ob.mitigated:
            grade += 0.25
        ob.grade = round(grade, 3)
