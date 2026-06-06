"""
===============================================================================
SwingEngine — fractal swing detection + structure labeling (SMC)
===============================================================================
Foundation engine: every other SMC engine (structure/liquidity/order-block)
references swings produced here.

Method (authoritative ICT/SMC — see plan file sources):
  • Swing high = fractal: bar high strictly greater than `n` bars on each side.
    Swing low  = fractal: bar low strictly lower than `n` bars on each side.
    Detection uses WICKS (high/low). Structure *confirmation* (BOS/CHOCH) uses
    the CLOSE — that lives in StructureEngine, not here.
  • Noise filter: an alternating swing is only kept if it is displaced from the
    previous swing by >= `atr_mult` × ATR (so it is not a naive zigzag).
  • Alternation: consecutive same-type fractals are collapsed to the extreme
    one, yielding a clean H/L/H/L sequence.
  • Internal vs external: a second, wider fractal pass (`external_n`) marks the
    major swings that define the dealing range and act as BOS/CHOCH references.

No-lookahead contract: each `SwingPoint.confirm_index = index + right` — the bar
at which the fractal first becomes knowable. Downstream must not use a swing
before that bar. This is what keeps the training pool leakage-free.
===============================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .types import SwingLabel, SwingPoint, SwingType


def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder-style ATR as a per-bar array (NaN-safe, leakage-free: bar i uses
    only data up to i)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return atr


class SwingEngine:
    def __init__(
        self,
        fractal_n: int = 2,
        external_n: int = 5,
        atr_period: int = 14,
        atr_mult: float = 0.5,
    ) -> None:
        self.fractal_n = fractal_n
        self.external_n = external_n
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    # ─── raw fractal pass ─────────────────────────────────────────────────
    def _raw_fractals(self, df: pd.DataFrame, n: int) -> List[SwingPoint]:
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        m = len(df)
        out: List[SwingPoint] = []
        for i in range(n, m - n):
            left = range(i - n, i)
            right = range(i + 1, i + 1 + n)
            if all(high[i] > high[j] for j in left) and all(high[i] > high[j] for j in right):
                out.append(SwingPoint(i, float(high[i]), SwingType.HIGH, confirm_index=i + n))
            if all(low[i] < low[j] for j in left) and all(low[i] < low[j] for j in right):
                out.append(SwingPoint(i, float(low[i]), SwingType.LOW, confirm_index=i + n))
        out.sort(key=lambda s: (s.index, 0 if s.swing_type is SwingType.HIGH else 1))
        return out

    # ─── alternation + ATR noise filter ───────────────────────────────────
    def _clean(self, raw: List[SwingPoint], atr: np.ndarray) -> List[SwingPoint]:
        cleaned: List[SwingPoint] = []
        for sp in raw:
            if not cleaned:
                cleaned.append(sp)
                continue
            last = cleaned[-1]
            if sp.swing_type is last.swing_type:
                # keep the more extreme of consecutive same-type fractals
                more_extreme = (
                    sp.price > last.price if sp.swing_type is SwingType.HIGH
                    else sp.price < last.price
                )
                if more_extreme:
                    cleaned[-1] = sp
                continue
            # alternating swing: require minimum displacement vs the last swing
            thr = self.atr_mult * float(atr[min(sp.index, len(atr) - 1)])
            if abs(sp.price - last.price) < thr:
                continue
            cleaned.append(sp)
        return cleaned

    # ─── HH/HL/LH/LL labeling ─────────────────────────────────────────────
    @staticmethod
    def _label(swings: List[SwingPoint]) -> None:
        last_high = None
        last_low = None
        for sp in swings:
            if sp.swing_type is SwingType.HIGH:
                if last_high is not None:
                    sp.label = SwingLabel.HH if sp.price > last_high else SwingLabel.LH
                last_high = sp.price
            else:
                if last_low is not None:
                    sp.label = SwingLabel.HL if sp.price > last_low else SwingLabel.LL
                last_low = sp.price

    # ─── public API ───────────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame) -> List[SwingPoint]:
        if len(df) < 2 * self.fractal_n + 1:
            return []
        atr = _atr(df, self.atr_period)
        swings = self._clean(self._raw_fractals(df, self.fractal_n), atr)

        # external pass — wider fractal marks the major (range-defining) swings.
        # A wider fractal extreme is also a narrow one, so we mark by price+type
        # match against the cleaned list.
        ext = self._raw_fractals(df, self.external_n)
        ext_keys = {(s.index, s.swing_type) for s in ext}
        for sp in swings:
            if (sp.index, sp.swing_type) in ext_keys:
                sp.is_external = True

        self._label(swings)
        return swings
