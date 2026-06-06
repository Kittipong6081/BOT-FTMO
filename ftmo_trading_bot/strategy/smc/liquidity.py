"""
===============================================================================
LiquidityEngine — liquidity pools, sweeps, premium/discount (SMC)
===============================================================================
  • Liquidity levels: every swing high is buy-side liquidity (BSL, stops of
    shorts rest above); every swing low is sell-side liquidity (SSL). Swings at
    ~the same price cluster into equal highs / equal lows (`equal_count >= 2`).
  • Liquidity sweep: a bar whose WICK trades beyond a level but whose CLOSE
    returns inside → engineered stop-hunt that precedes a reversal. Reversal
    direction is OPPOSITE the side swept (sweep of a high → bearish reversal).
  • Dealing range / premium-discount: range between the latest EXTERNAL swing
    high and low; equilibrium = 50%; longs in discount, shorts in premium.

Leakage-safe: a swing is sweepable only from its `confirm_index` onward.
===============================================================================
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .swing import _atr
from .types import DealingRange, Direction, LiquidityLevel, LiquiditySweep, SwingPoint, SwingType


class LiquidityEngine:
    def __init__(self, equal_tol_atr: float = 0.10, atr_period: int = 14,
                 sweep_lookback: int = 30) -> None:
        self.equal_tol_atr = equal_tol_atr
        self.atr_period = atr_period
        self.sweep_lookback = sweep_lookback

    # ─── liquidity pools (equal highs / lows) ─────────────────────────────
    def levels(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[LiquidityLevel]:
        atr = _atr(df, self.atr_period)
        tol = self.equal_tol_atr * float(np.nanmedian(atr)) if len(atr) else 0.0
        highs = sorted((s for s in swings if s.swing_type is SwingType.HIGH),
                       key=lambda s: s.price)
        lows = sorted((s for s in swings if s.swing_type is SwingType.LOW),
                      key=lambda s: s.price)
        return (self._cluster(highs, tol, Direction.BULLISH)
                + self._cluster(lows, tol, Direction.BEARISH))

    @staticmethod
    def _cluster(points: List[SwingPoint], tol: float, side: Direction) -> List[LiquidityLevel]:
        out: List[LiquidityLevel] = []
        bucket: List[SwingPoint] = []
        for sp in points:
            if bucket and abs(sp.price - bucket[-1].price) > tol:
                out.append(LiquidityEngine._level_from(bucket, side))
                bucket = []
            bucket.append(sp)
        if bucket:
            out.append(LiquidityEngine._level_from(bucket, side))
        return out

    @staticmethod
    def _level_from(bucket: List[SwingPoint], side: Direction) -> LiquidityLevel:
        price = sum(s.price for s in bucket) / len(bucket)
        return LiquidityLevel(
            direction=side,
            price=price,
            index=max(s.index for s in bucket),
            equal_count=len(bucket),
        )

    # ─── liquidity sweeps (wick beyond + close back inside) ───────────────
    def sweeps(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[LiquiditySweep]:
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        out: List[LiquiditySweep] = []

        highs = [s for s in swings if s.swing_type is SwingType.HIGH]
        lows = [s for s in swings if s.swing_type is SwingType.LOW]
        swept_h: set = set()
        swept_l: set = set()

        for i in range(n):
            for s in highs:
                if s.confirm_index > i or s.index in swept_h:
                    continue
                if i - s.confirm_index > self.sweep_lookback:
                    continue
                if high[i] > s.price and close[i] < s.price:
                    out.append(LiquiditySweep(Direction.BEARISH, s.price, i,
                                              penetration=float(high[i] - s.price)))
                    swept_h.add(s.index)
            for s in lows:
                if s.confirm_index > i or s.index in swept_l:
                    continue
                if i - s.confirm_index > self.sweep_lookback:
                    continue
                if low[i] < s.price and close[i] > s.price:
                    out.append(LiquiditySweep(Direction.BULLISH, s.price, i,
                                              penetration=float(s.price - low[i])))
                    swept_l.add(s.index)
        out.sort(key=lambda x: x.index)
        return out

    # ─── dealing range (premium / discount) ───────────────────────────────
    def dealing_range(self, swings: List[SwingPoint]) -> Optional[DealingRange]:
        ext_high = self._latest(swings, SwingType.HIGH, external=True)
        ext_low = self._latest(swings, SwingType.LOW, external=True)
        if ext_high is None:
            ext_high = self._latest(swings, SwingType.HIGH, external=False)
        if ext_low is None:
            ext_low = self._latest(swings, SwingType.LOW, external=False)
        if ext_high is None or ext_low is None:
            return None
        return DealingRange(ext_high.price, ext_low.price, ext_high.index, ext_low.index)

    @staticmethod
    def _latest(swings: List[SwingPoint], st: SwingType, external: bool) -> Optional[SwingPoint]:
        cands = [s for s in swings if s.swing_type is st and (s.is_external or not external)]
        return max(cands, key=lambda s: s.index) if cands else None
