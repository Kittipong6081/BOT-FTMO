"""
===============================================================================
FTMO Trading Bot — Fair Value Gap Detection (ตรวจจับ FVG / Imbalance Zone)
===============================================================================
Fair Value Gap คือ ช่องว่างราคาที่เกิดจากการเคลื่อนไหวรุนแรง (Imbalance)
ระหว่างแท่งเทียน 3 แท่งต่อเนื่อง ราคามักกลับมาเติม gap ก่อนไปต่อ

ลักษณะ FVG:
- Bullish FVG: candle[i-2].high < candle[i].low → gap ขาขึ้น
- Bearish FVG: candle[i-2].low > candle[i].high → gap ขาลง

การใช้งาน:
- ราคากลับมาในโซน FVG ที่ยังไม่ถูก fill → จุด Entry ที่มีโอกาสสูง
- FVG ที่ยังเปิดอยู่ (unfilled) มีความสำคัญมากกว่า
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class FairValueGap:
    """ข้อมูล FVG หนึ่งจุด"""
    fvg_type: str               # "BULLISH" / "BEARISH"
    index: int                  # ตำแหน่งแท่งกลาง (impulse candle)
    time: object                # เวลาเกิด
    gap_high: float             # ขอบบนของ gap
    gap_low: float              # ขอบล่างของ gap
    gap_size: float             # gap_high - gap_low
    impulse_body_ratio: float   # body ratio ของแท่ง impulse
    filled: bool = False        # ราคากลับมาปิด gap แล้วหรือยัง
    filled_pct: float = 0.0     # 0-100% ถูกเติมเท่าไหร่
    strength_score: float = 0.0 # 0-100

    @property
    def zone_mid(self) -> float:
        return (self.gap_high + self.gap_low) / 2

    @property
    def is_active(self) -> bool:
        return not self.filled


class FVGDetector:
    """
    ตรวจจับ Fair Value Gaps จาก Price Action

    วิธีการ:
    1. สแกน 3 แท่งเทียนต่อเนื่อง หาช่องว่างระหว่างแท่ง 1 กับแท่ง 3
    2. ตรวจสอบว่า gap ถูก fill หรือยัง
    3. ให้คะแนนตาม gap size, freshness, proximity
    """

    def __init__(self, max_fvgs: int = 20):
        self._max_fvgs = max_fvgs
        self._bullish_fvgs: List[FairValueGap] = []
        self._bearish_fvgs: List[FairValueGap] = []

    # =========================================================================
    # 🔍 ตรวจจับ FVGs
    # =========================================================================

    def detect_fvgs(self, df: pd.DataFrame) -> pd.DataFrame:
        """หา FVG ทั้งหมดจาก DataFrame"""
        if df is None or len(df) < 5:
            return df

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values
        times = df["time"].values if "time" in df.columns else range(len(df))

        self._bullish_fvgs.clear()
        self._bearish_fvgs.clear()

        body_sizes = np.abs(closes - opens)
        candle_ranges = highs - lows
        candle_ranges[candle_ranges == 0] = 1e-10
        body_ratios = body_sizes / candle_ranges

        for i in range(2, len(df)):
            # === Bullish FVG: แท่ง 1 high < แท่ง 3 low ===
            if highs[i - 2] < lows[i]:
                gap_high = lows[i]
                gap_low = highs[i - 2]
                gap_size = gap_high - gap_low

                if gap_size > 0:
                    self._bullish_fvgs.append(FairValueGap(
                        fvg_type="BULLISH",
                        index=i - 1,
                        time=times[i - 1],
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        impulse_body_ratio=body_ratios[i - 1],
                    ))

            # === Bearish FVG: แท่ง 1 low > แท่ง 3 high ===
            if lows[i - 2] > highs[i]:
                gap_high = lows[i - 2]
                gap_low = highs[i]
                gap_size = gap_high - gap_low

                if gap_size > 0:
                    self._bearish_fvgs.append(FairValueGap(
                        fvg_type="BEARISH",
                        index=i - 1,
                        time=times[i - 1],
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        impulse_body_ratio=body_ratios[i - 1],
                    ))

        # จำกัดจำนวน FVG (เก็บล่าสุด)
        self._bullish_fvgs = self._bullish_fvgs[-self._max_fvgs:]
        self._bearish_fvgs = self._bearish_fvgs[-self._max_fvgs:]

        return df

    # =========================================================================
    # ✅ ตรวจสอบ Fill Status
    # =========================================================================

    def _check_fill_status(self, df: pd.DataFrame):
        """ตรวจว่า FVG ถูก fill หรือยัง"""
        if df is None or len(df) == 0:
            return

        lows = df["low"].values
        highs = df["high"].values

        for fvg in self._bullish_fvgs:
            if fvg.filled:
                continue
            for i in range(fvg.index + 2, len(df)):
                penetration = max(0, fvg.gap_high - lows[i])
                if penetration > 0:
                    fvg.filled_pct = min(100.0, (penetration / fvg.gap_size) * 100)
                    if fvg.filled_pct >= 75.0:
                        fvg.filled = True
                        break

        for fvg in self._bearish_fvgs:
            if fvg.filled:
                continue
            for i in range(fvg.index + 2, len(df)):
                penetration = max(0, highs[i] - fvg.gap_low)
                if penetration > 0:
                    fvg.filled_pct = min(100.0, (penetration / fvg.gap_size) * 100)
                    if fvg.filled_pct >= 75.0:
                        fvg.filled = True
                        break

    # =========================================================================
    # 📊 ให้คะแนน FVGs
    # =========================================================================

    def _score_fvgs(self, df: pd.DataFrame):
        """ให้คะแนน FVG ตาม gap size, freshness, proximity"""
        if df is None or len(df) == 0:
            return

        atr_col = df.get("atr")
        current_atr = atr_col.iloc[-1] if atr_col is not None and len(atr_col) > 0 else None
        current_price = df["close"].iloc[-1]
        total_bars = len(df)

        all_fvgs = self._bullish_fvgs + self._bearish_fvgs
        for fvg in all_fvgs:
            score = 0.0

            # 1. Gap Size เทียบ ATR (max 35)
            if current_atr and current_atr > 0:
                size_ratio = fvg.gap_size / current_atr
                score += min(size_ratio * 35, 35.0)

            # 2. Freshness — ยังไม่ถูก fill (max 30)
            if fvg.filled_pct < 25.0:
                score += 30.0
            elif fvg.filled_pct < 50.0:
                score += 15.0
            elif fvg.filled_pct < 75.0:
                score += 5.0

            # 3. Impulse candle body ratio (max 20)
            score += min(fvg.impulse_body_ratio * 20, 20.0)

            # 4. Proximity to current price (max 15)
            if current_atr and current_atr > 0:
                distance = abs(current_price - fvg.zone_mid)
                proximity = max(0, 1.0 - distance / (current_atr * 3))
                score += proximity * 15.0

            fvg.strength_score = min(score, 100.0)

    # =========================================================================
    # 🎯 ค้นหา FVG ที่ราคาปัจจุบันอยู่ใกล้
    # =========================================================================

    def is_price_at_bullish_fvg(
        self, price: float, tolerance_pips: float = 5, pip_size: float = 0.0001
    ) -> Optional[FairValueGap]:
        """ตรวจว่าราคาอยู่ในโซน Bullish FVG หรือไม่"""
        tolerance = tolerance_pips * pip_size
        best = None
        best_score = 0

        for fvg in self._bullish_fvgs:
            if fvg.filled:
                continue
            if (fvg.gap_low - tolerance) <= price <= (fvg.gap_high + tolerance):
                if fvg.strength_score > best_score:
                    best = fvg
                    best_score = fvg.strength_score

        return best

    def is_price_at_bearish_fvg(
        self, price: float, tolerance_pips: float = 5, pip_size: float = 0.0001
    ) -> Optional[FairValueGap]:
        """ตรวจว่าราคาอยู่ในโซน Bearish FVG หรือไม่"""
        tolerance = tolerance_pips * pip_size
        best = None
        best_score = 0

        for fvg in self._bearish_fvgs:
            if fvg.filled:
                continue
            if (fvg.gap_low - tolerance) <= price <= (fvg.gap_high + tolerance):
                if fvg.strength_score > best_score:
                    best = fvg
                    best_score = fvg.strength_score

        return best

    # =========================================================================
    # 📋 Helper Methods
    # =========================================================================

    def get_active_fvgs(self, fvg_type: str = "BULLISH", min_score: float = 20) -> list:
        """ดึง FVG ที่ยัง active"""
        fvgs = self._bullish_fvgs if fvg_type == "BULLISH" else self._bearish_fvgs
        return [f for f in fvgs if f.is_active and f.strength_score >= min_score]

    def get_fvg_summary(self) -> dict:
        """สรุปสถานะ FVG"""
        active_bull = [f for f in self._bullish_fvgs if f.is_active]
        active_bear = [f for f in self._bearish_fvgs if f.is_active]
        return {
            "total_bullish": len(self._bullish_fvgs),
            "active_bullish": len(active_bull),
            "total_bearish": len(self._bearish_fvgs),
            "active_bearish": len(active_bear),
        }

    # =========================================================================
    # 🚀 Main Entry — เรียกจาก SMCStrategy
    # =========================================================================

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """วิเคราะห์ FVG ทั้งหมด (detect → check fill → score)"""
        self.detect_fvgs(df)
        self._check_fill_status(df)
        self._score_fvgs(df)
        return df
