"""
===============================================================================
FTMO Trading Bot — Liquidity Sweep Detection (ตรวจจับการกวาด Liquidity)
===============================================================================
Liquidity Sweep คือ การที่ราคา sweep ผ่าน Swing High/Low (กวาด Stop Loss)
แล้วกลับตัวรุนแรง — เป็นสัญญาณ entry สำคัญของ SMC

ลักษณะ:
- Bullish Sweep: wick ลงต่ำกว่า swing low → close กลับมาเหนือ swing low
- Bearish Sweep: wick ขึ้นสูงกว่า swing high → close กลับมาต่ำกว่า swing high

การใช้งาน:
- sweep ที่เพิ่งเกิด (1-5 แท่ง) → ยืนยันว่า Smart Money เข้าซื้อ/ขาย
- ยิ่ง reversal candle ใหญ่ ยิ่งน่าเชื่อถือ
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass

from strategy.market_structure import MarketStructure


@dataclass
class LiquiditySweep:
    """ข้อมูล Liquidity Sweep หนึ่งจุด"""
    sweep_type: str             # "BULLISH" / "BEARISH"
    index: int                  # แท่งที่เกิด sweep
    time: object                # เวลาเกิด
    swept_level: float          # ราคา swing point ที่ถูก sweep
    sweep_distance: float       # wick เลย swing point เท่าไหร่
    reversal_size: float        # ขนาด body ของ reversal candle
    strength_score: float = 0.0 # 0-100
    bars_ago: int = 0           # เกิดกี่แท่งก่อน (freshness)


class LiquiditySweepDetector:
    """
    ตรวจจับ Liquidity Sweeps จาก Swing Points + Price Action

    วิธีการ:
    1. ดึง swing highs/lows จาก MarketStructure
    2. ตรวจว่าแท่งเทียนมี wick เกิน swing point แต่ close กลับมาด้านใน
    3. ให้คะแนนตาม sweep distance, reversal size, freshness
    """

    def __init__(self, max_lookback_bars: int = 10):
        self._max_lookback = max_lookback_bars
        self._bullish_sweeps: List[LiquiditySweep] = []
        self._bearish_sweeps: List[LiquiditySweep] = []

    # =========================================================================
    # 🔍 ตรวจจับ Sweeps
    # =========================================================================

    def detect_sweeps(
        self, df: pd.DataFrame, market_structure: MarketStructure
    ) -> None:
        """ตรวจจับ liquidity sweeps จาก swing points"""
        if df is None or len(df) < 10:
            return

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values
        times = df["time"].values if "time" in df.columns else range(len(df))

        self._bullish_sweeps.clear()
        self._bearish_sweeps.clear()

        swing_lows = market_structure.get_all_swing_lows()
        swing_highs = market_structure.get_all_swing_highs()

        total_bars = len(df)
        scan_start = max(0, total_bars - self._max_lookback)

        # === Bullish Sweep: wick < swing low แต่ close > swing low ===
        for sl in swing_lows:
            if sl.broken:
                continue
            sl_price = sl.price

            for i in range(scan_start, total_bars):
                if lows[i] < sl_price and closes[i] > sl_price:
                    sweep_dist = sl_price - lows[i]
                    body = abs(closes[i] - opens[i])

                    self._bullish_sweeps.append(LiquiditySweep(
                        sweep_type="BULLISH",
                        index=i,
                        time=times[i],
                        swept_level=sl_price,
                        sweep_distance=sweep_dist,
                        reversal_size=body,
                        bars_ago=total_bars - 1 - i,
                    ))
                    break

        # === Bearish Sweep: wick > swing high แต่ close < swing high ===
        for sh in swing_highs:
            if sh.broken:
                continue
            sh_price = sh.price

            for i in range(scan_start, total_bars):
                if highs[i] > sh_price and closes[i] < sh_price:
                    sweep_dist = highs[i] - sh_price
                    body = abs(closes[i] - opens[i])

                    self._bearish_sweeps.append(LiquiditySweep(
                        sweep_type="BEARISH",
                        index=i,
                        time=times[i],
                        swept_level=sh_price,
                        sweep_distance=sweep_dist,
                        reversal_size=body,
                        bars_ago=total_bars - 1 - i,
                    ))
                    break

    # =========================================================================
    # 📊 ให้คะแนน Sweeps
    # =========================================================================

    def _score_sweeps(self, df: pd.DataFrame):
        """ให้คะแนน sweep ตาม distance, reversal size, freshness"""
        if df is None or len(df) == 0:
            return

        atr_col = df.get("atr")
        current_atr = atr_col.iloc[-1] if atr_col is not None and len(atr_col) > 0 else None

        body_sizes = np.abs(df["close"].values - df["open"].values)
        avg_body = np.mean(body_sizes[-50:]) if len(body_sizes) >= 50 else np.mean(body_sizes)
        if avg_body == 0:
            avg_body = 1e-10

        all_sweeps = self._bullish_sweeps + self._bearish_sweeps
        for sweep in all_sweeps:
            score = 0.0

            # 1. Sweep distance เทียบ ATR (max 30)
            if current_atr and current_atr > 0:
                dist_ratio = sweep.sweep_distance / current_atr
                score += min(dist_ratio * 60, 30.0)

            # 2. Reversal candle size เทียบ avg body (max 30)
            rev_ratio = sweep.reversal_size / avg_body
            score += min(rev_ratio * 15, 30.0)

            # 3. Freshness (max 25)
            freshness = max(0, 1.0 - sweep.bars_ago / self._max_lookback)
            score += freshness * 25.0

            # 4. Clean sweep bonus — sweep + reversal ในแท่งเดียว (max 15)
            if sweep.bars_ago <= 1:
                score += 15.0

            sweep.strength_score = min(score, 100.0)

    # =========================================================================
    # 🎯 ดึง Sweep ล่าสุด
    # =========================================================================

    def get_recent_bullish_sweep(self, max_bars_ago: int = 5) -> Optional[LiquiditySweep]:
        """ดึง bullish sweep ล่าสุดที่ยังใหม่"""
        best = None
        best_score = 0
        for s in self._bullish_sweeps:
            if s.bars_ago <= max_bars_ago and s.strength_score > best_score:
                best = s
                best_score = s.strength_score
        return best

    def get_recent_bearish_sweep(self, max_bars_ago: int = 5) -> Optional[LiquiditySweep]:
        """ดึง bearish sweep ล่าสุดที่ยังใหม่"""
        best = None
        best_score = 0
        for s in self._bearish_sweeps:
            if s.bars_ago <= max_bars_ago and s.strength_score > best_score:
                best = s
                best_score = s.strength_score
        return best

    # =========================================================================
    # 🚀 Main Entry — เรียกจาก SMCStrategy
    # =========================================================================

    def analyze(self, df: pd.DataFrame, market_structure: MarketStructure) -> pd.DataFrame:
        """วิเคราะห์ Liquidity Sweeps (detect → score)"""
        self.detect_sweeps(df, market_structure)
        self._score_sweeps(df)
        return df
