"""
===============================================================================
FVGEngine — Fair Value Gap detection + mitigation (SMC)
===============================================================================
A Fair Value Gap is a 3-candle imbalance (candles A, B, C):
  • Bullish FVG: A.high < C.low  → gap zone [A.high, C.low] (a discount to revisit)
  • Bearish FVG: A.low  > C.high → gap zone [C.high, A.low] (a premium to revisit)
The gap is knowable only at candle C (`FVG.index = C`).

Mitigation states (deterministic):
  • unfilled   : price has not returned to the zone           (mitigation == 0)
  • partial    : price has tapped the zone but not closed through (0 < mitigation < 1)
  • filled     : a later candle CLOSED through the far edge    (filled == True)

Optional `min_disp_atr` filters out trivial gaps smaller than a fraction of ATR
(noise). Default 0.0 = pure geometric detection (keeps it testable).
===============================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .swing import _atr
from .types import Direction, FVG


class FVGEngine:
    def __init__(self, min_disp_atr: float = 0.0, atr_period: int = 14) -> None:
        self.min_disp_atr = min_disp_atr
        self.atr_period = atr_period

    def detect(self, df: pd.DataFrame) -> List[FVG]:
        n = len(df)
        if n < 3:
            return []
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        atr = _atr(df, self.atr_period) if self.min_disp_atr > 0 else None

        fvgs: List[FVG] = []
        for i in range(1, n - 1):
            a, c = i - 1, i + 1
            bull = low[c] > high[a]
            bear = high[c] < low[a]
            if not (bull or bear):
                continue
            if bull:
                top, bottom, direction = low[c], high[a], Direction.BULLISH
            else:
                top, bottom, direction = low[a], high[c], Direction.BEARISH
            if atr is not None:
                thr = self.min_disp_atr * float(atr[i])
                if (top - bottom) < thr:
                    continue
            fvgs.append(FVG(direction, float(top), float(bottom), index=c))

        self._compute_mitigation(fvgs, high, low, close)
        return fvgs

    @staticmethod
    def _compute_mitigation(
        fvgs: List[FVG], high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> None:
        n = len(close)
        for fvg in fvgs:
            start = fvg.index + 1
            if start >= n:
                continue
            size = fvg.top - fvg.bottom
            if size <= 0:
                continue
            after_high = high[start:]
            after_low = low[start:]
            after_close = close[start:]
            if fvg.direction is Direction.BULLISH:
                depth = fvg.top - float(after_low.min())
                fvg.filled = bool((after_close <= fvg.bottom).any())
            else:
                depth = float(after_high.max()) - fvg.bottom
                fvg.filled = bool((after_close >= fvg.top).any())
            fvg.mitigation = float(min(1.0, max(0.0, depth / size)))
            if fvg.filled:
                fvg.mitigation = 1.0
