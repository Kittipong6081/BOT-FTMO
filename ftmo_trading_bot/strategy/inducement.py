"""
===============================================================================
FTMO Trading Bot — Inducement (IDM) Detection
===============================================================================
ตรวจจับ Inducement Moves (IDM) — pattern ของ smart money ที่ "หลอก" liquidity
ฝั่งตรงข้ามก่อนพาราคาวิ่งจริง

หลักการ SMC (Inducement):
- ก่อนที่ smart money จะดันราคาขึ้น (BUY direction): มักมี wick ลงสั้นๆ
  ที่ทะลุ minor swing low แต่ปิดกลับขึ้น → "trapped sellers" ต้องมา cover
  เป็น fuel ให้ราคาวิ่งขึ้น (rejection candle: ปิด > เปิด + wick ใหญ่กว่า body)
- ก่อนดันลง (SELL direction): wick ขึ้น + ปิดลง = "trapped buyers"

การใช้งาน:
- เรียกใน `SMCStrategy._evaluate_buy_signal/_evaluate_sell_signal` หลัง sweep gate
- ถ้าเจอ IDM → +10 confluence bonus (smart money confirmation ดีกว่า pure sweep)
- ถ้าไม่เจอ → -5 penalty (entry ที่ไม่ผ่านการ trap อาจเป็น obvious swing → IDM trap probability สูง)
===============================================================================
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class InducementEvent:
    """ข้อมูล Inducement event 1 จุด"""
    direction: int          # +1 = bullish-rejection (สำหรับ BUY entry), -1 = bearish-rejection (SELL)
    bar_index: int          # ตำแหน่งใน DataFrame
    bars_ago: int           # ห่างกี่แท่งจาก current
    wick_size: float        # ขนาดของ wick ที่ failed
    body_size: float        # ขนาด body
    wick_to_body_ratio: float


class InducementDetector:
    """
    ตรวจจับ Inducement candles (rejection wick patterns) ใน LTF (M15)

    Algorithm:
    1. Scan แท่งเทียน N อันสุดท้าย (default 8)
    2. สำหรับแต่ละแท่ง — คำนวณ wick / body ratio ฝั่งตรงข้ามกับ direction
    3. ถ้า wick ≥ 1.5 × body AND ปิดในทิศทางที่ต้องการ → ถือเป็น IDM
    4. คืน event ที่ใหม่ที่สุด (bars_ago น้อยที่สุด)

    Note: นี่คือ minimal viable IDM detector. Production-grade SMC IDM
    มักรวม sweep level + structure context. v6.11 ใช้ rejection-candle
    เป็นจุดตั้งต้น — ปรับปรุงทีหลังถ้า data confirm edge
    """

    def __init__(self, lookback: int = 8, min_wick_to_body: float = 1.5):
        self._lookback = lookback
        self._min_ratio = min_wick_to_body

    def detect_idm(
        self,
        df: pd.DataFrame,
        direction: int,
    ) -> Optional[InducementEvent]:
        """
        หา IDM candle ที่ใหม่ที่สุดภายใน lookback bars

        Args:
            df: M15 DataFrame ที่มี OHLC
            direction: +1 (BUY entry — หา bearish-failed wick), -1 (SELL — bullish-failed wick)

        Returns:
            Optional[InducementEvent]: latest IDM, หรือ None ถ้าไม่พบ
        """
        if df is None or len(df) < 2:
            return None

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        total_bars = len(df)
        scan_start = max(0, total_bars - self._lookback)

        # Scan จากใหม่ไปเก่า → return เจออันแรก (newest)
        for i in range(total_bars - 1, scan_start - 1, -1):
            o = opens[i]
            h = highs[i]
            low = lows[i]
            c = closes[i]
            body = abs(c - o)
            if body <= 0:
                continue

            if direction == 1:
                # BUY context — หา bullish rejection (wick ลง failed)
                # ปิดสูงกว่าเปิด + wick ล่างใหญ่กว่า body × 1.5
                if c <= o:
                    continue
                lower_wick = min(o, c) - low
                if lower_wick <= 0:
                    continue
                ratio = lower_wick / body
                if ratio >= self._min_ratio:
                    return InducementEvent(
                        direction=1,
                        bar_index=i,
                        bars_ago=total_bars - 1 - i,
                        wick_size=float(lower_wick),
                        body_size=float(body),
                        wick_to_body_ratio=float(ratio),
                    )
            elif direction == -1:
                # SELL context — หา bearish rejection (wick ขึ้น failed)
                if c >= o:
                    continue
                upper_wick = h - max(o, c)
                if upper_wick <= 0:
                    continue
                ratio = upper_wick / body
                if ratio >= self._min_ratio:
                    return InducementEvent(
                        direction=-1,
                        bar_index=i,
                        bars_ago=total_bars - 1 - i,
                        wick_size=float(upper_wick),
                        body_size=float(body),
                        wick_to_body_ratio=float(ratio),
                    )

        return None
