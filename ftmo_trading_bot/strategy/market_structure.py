"""
===============================================================================
FTMO Trading Bot — Market Structure Analysis (วิเคราะห์โครงสร้างตลาด)
===============================================================================
โมดูลหัวใจของ SMC (Smart Money Concepts):
- ตรวจจับ Swing High / Swing Low
- ระบุ BOS (Break of Structure) — การทะลุโครงสร้าง
- ระบุ CHoCH (Change of Character) — การเปลี่ยนแนวโน้ม
- ติดตาม Higher Highs, Higher Lows (Uptrend)
- ติดตาม Lower Highs, Lower Lows (Downtrend)

หลักการ SMC:
- Uptrend: HH → HL → HH → HL (จุดสูงสุดและต่ำสุดเพิ่มขึ้นเรื่อยๆ)
- Downtrend: LL → LH → LL → LH (จุดสูงสุดและต่ำสุดลดลงเรื่อยๆ)
- BOS: ราคาทะลุ Swing High/Low ล่าสุด → ยืนยันแนวโน้ม
- CHoCH: ราคาทะลุ Swing ตรงข้ามแนวโน้ม → สัญญาณกลับตัว
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.settings import bot_config


class StructureType(Enum):
    """ประเภทของ Market Structure Event"""
    BOS_BULLISH = "BOS_BULLISH"         # Break of Structure ขาขึ้น (ทะลุ Swing High)
    BOS_BEARISH = "BOS_BEARISH"         # Break of Structure ขาลง (ทะลุ Swing Low)
    CHOCH_BULLISH = "CHOCH_BULLISH"     # Change of Character → เปลี่ยนเป็นขาขึ้น
    CHOCH_BEARISH = "CHOCH_BEARISH"     # Change of Character → เปลี่ยนเป็นขาลง
    NONE = "NONE"                        # ไม่มีเหตุการณ์


@dataclass
class SwingPoint:
    """จุด Swing High หรือ Swing Low"""
    index: int                  # ตำแหน่งใน DataFrame
    price: float                # ราคาที่จุด Swing
    time: object                # เวลา
    is_high: bool               # True = Swing High, False = Swing Low
    broken: bool = False        # ถูกทะลุแล้วหรือยัง
    
    @property
    def type_label(self) -> str:
        return "Swing High" if self.is_high else "Swing Low"


@dataclass
class StructureEvent:
    """เหตุการณ์ Market Structure (BOS หรือ CHoCH)"""
    event_type: StructureType       # ประเภทเหตุการณ์
    index: int                      # ตำแหน่งที่เกิด
    price: float                    # ราคาที่เกิด
    time: object                    # เวลา
    broken_swing: SwingPoint        # Swing Point ที่ถูกทะลุ
    
    @property
    def is_bullish(self) -> bool:
        return self.event_type in (StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH)
    
    @property
    def is_bos(self) -> bool:
        return self.event_type in (StructureType.BOS_BULLISH, StructureType.BOS_BEARISH)


class MarketStructure:
    """
    วิเคราะห์โครงสร้างตลาดตามหลัก Smart Money Concepts (SMC)
    
    ทำหน้าที่:
    1. หา Swing High / Swing Low จาก Price Action
    2. ตรวจจับ BOS (Break of Structure) — ยืนยันแนวโน้ม
    3. ตรวจจับ CHoCH (Change of Character) — สัญญาณกลับตัว
    4. กำหนด Market Bias ปัจจุบัน (Bullish / Bearish / Neutral)
    """

    def __init__(self):
        """เริ่มต้น Market Structure Analyzer"""
        self._lookback = bot_config.indicators.swing_lookback  # 5 แท่ง
        
        # เก็บ Swing Points ที่ตรวจพบ
        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        
        # เก็บ Structure Events
        self._events: List[StructureEvent] = []
        
        # Market Bias ปัจจุบัน (1=Bullish, -1=Bearish, 0=Neutral)
        self._current_bias: int = 0
        
        print("🏗️ [Market Structure] เริ่มต้นระบบวิเคราะห์โครงสร้างตลาด SMC")

    # =========================================================================
    # 🔍 Swing Point Detection (ตรวจจับจุด Swing)
    # =========================================================================

    def find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = None
    ) -> pd.DataFrame:
        """
        ตรวจจับ Swing High และ Swing Low จาก Price Action
        
        วิธี: Fractal Method
        - Swing High: แท่งเทียนที่ High สูงกว่าแท่งเทียน N แท่งรอบข้าง
        - Swing Low: แท่งเทียนที่ Low ต่ำกว่าแท่งเทียน N แท่งรอบข้าง
        
        N = lookback (ค่าเริ่มต้น 5)
        ยิ่ง N มาก → Swing Points สำคัญมากขึ้น แต่จำนวนน้อยลง
        
        Args:
            df: DataFrame ที่มี ['high', 'low']
            lookback: จำนวนแท่งรอบข้าง (None = ใช้ Config)
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่ม 'swing_high', 'swing_low', 'swing_type'
        """
        if lookback is None:
            lookback = self._lookback

        n = lookback
        highs = df['high'].values
        lows = df['low'].values
        length = len(df)
        
        # สร้าง Arrays สำหรับ Swing Points
        swing_high = np.full(length, np.nan)
        swing_low = np.full(length, np.nan)
        swing_type = np.zeros(length)  # 1 = Swing High, -1 = Swing Low, 0 = ไม่ใช่
        
        # ล้าง Swing Points เดิม
        self._swing_highs = []
        self._swing_lows = []
        
        # ตรวจจับ Swing Points
        for i in range(n, length - n):
            # === ตรวจจับ Swing High ===
            # High ของแท่ง i ต้องสูงกว่า High ของ N แท่งซ้ายและขวา
            is_swing_high = True
            for j in range(1, n + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_high[i] = highs[i]
                swing_type[i] = 1
                
                sp = SwingPoint(
                    index=i,
                    price=highs[i],
                    time=df.index[i],
                    is_high=True
                )
                self._swing_highs.append(sp)
            
            # === ตรวจจับ Swing Low ===
            # Low ของแท่ง i ต้องต่ำกว่า Low ของ N แท่งซ้ายและขวา
            is_swing_low = True
            for j in range(1, n + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_low[i] = lows[i]
                swing_type[i] = -1
                
                sp = SwingPoint(
                    index=i,
                    price=lows[i],
                    time=df.index[i],
                    is_high=False
                )
                self._swing_lows.append(sp)

        # เพิ่มคอลัมน์เข้า DataFrame
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        df['swing_type'] = swing_type
        
        if bot_config.debug_mode:
            print(f"🔍 [Market Structure] พบ {len(self._swing_highs)} Swing Highs, {len(self._swing_lows)} Swing Lows")

        return df

    # =========================================================================
    # 💥 BOS & CHoCH Detection (ตรวจจับการทะลุโครงสร้าง)
    # =========================================================================

    def detect_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ตรวจจับ BOS (Break of Structure) และ CHoCH (Change of Character)
        
        หลักการ:
        
        BOS (ยืนยันแนวโน้ม):
        - BOS Bullish: ราคาทะลุ Swing High ล่าสุดในขณะที่แนวโน้มเป็นขาขึ้น
        - BOS Bearish: ราคาทะลุ Swing Low ล่าสุดในขณะที่แนวโน้มเป็นขาลง
        
        CHoCH (สัญญาณกลับตัว):
        - CHoCH Bullish: ราคาทะลุ Swing High ล่าสุดในขณะที่แนวโน้มเป็นขาลง → กลับเป็นขาขึ้น
        - CHoCH Bearish: ราคาทะลุ Swing Low ล่าสุดในขณะที่แนวโน้มเป็นขาขึ้น → กลับเป็นขาลง
        
        Args:
            df: DataFrame ที่คำนวณ Swing Points แล้ว
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่ม 'structure_event', 'market_bias'
        """
        # ตรวจสอบว่าหา Swing Points แล้วหรือยัง
        if 'swing_high' not in df.columns:
            df = self.find_swing_points(df)

        length = len(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # สร้าง Arrays
        structure_event = np.full(length, '', dtype=object)
        market_bias = np.zeros(length)
        
        # ล้าง Events เดิม
        self._events = []
        
        # ตรวจจับ BOS/CHoCH โดยเทียบกับ Swing Points ล่าสุด
        current_bias = 0  # 1=Bullish, -1=Bearish, 0=Neutral
        
        # ดึง Swing Points ทั้งหมดเรียงตามลำดับเวลา
        all_swings = sorted(
            self._swing_highs + self._swing_lows,
            key=lambda sp: sp.index
        )
        
        if len(all_swings) < 2:
            df['structure_event'] = structure_event
            df['market_bias'] = market_bias
            return df

        # วนลูปตามแท่งเทียน
        last_sh: Optional[SwingPoint] = None    # Swing High ล่าสุด
        last_sl: Optional[SwingPoint] = None    # Swing Low ล่าสุด
        
        for sp in all_swings:
            if sp.is_high:
                last_sh = sp
            else:
                last_sl = sp

        # ตรวจจับ Structure Breaks ที่แท่งเทียนล่าสุดๆ
        recent_sh = None
        recent_sl = None
        
        for sp in all_swings:
            if sp.is_high:
                recent_sh = sp
            else:
                recent_sl = sp
        
        # ตรวจสอบทีละแท่งจากตำแหน่งหลัง Swing Point แรก
        swing_idx = 0
        active_sh: Optional[SwingPoint] = None
        active_sl: Optional[SwingPoint] = None
        
        for i in range(length):
            # อัพเดท Swing Points ที่ Active (เกิดก่อนแท่งปัจจุบัน)
            while swing_idx < len(all_swings) and all_swings[swing_idx].index <= i:
                sp = all_swings[swing_idx]
                if sp.is_high:
                    active_sh = sp
                else:
                    active_sl = sp
                swing_idx += 1
            
            # ตรวจจับ Break ที่แท่งปัจจุบัน
            if active_sh and not active_sh.broken:
                # ตรวจว่า Close ทะลุ Swing High ล่าสุดหรือไม่
                if closes[i] > active_sh.price:
                    active_sh.broken = True
                    
                    if current_bias <= 0:
                        # เปลี่ยนจาก Bearish/Neutral → Bullish = CHoCH
                        event_type = StructureType.CHOCH_BULLISH
                    else:
                        # ยังคงเป็น Bullish = BOS
                        event_type = StructureType.BOS_BULLISH
                    
                    event = StructureEvent(
                        event_type=event_type,
                        index=i,
                        price=closes[i],
                        time=df.index[i],
                        broken_swing=active_sh
                    )
                    self._events.append(event)
                    structure_event[i] = event_type.value
                    current_bias = 1  # Bullish
            
            if active_sl and not active_sl.broken:
                # ตรวจว่า Close ทะลุ Swing Low ล่าสุดหรือไม่
                if closes[i] < active_sl.price:
                    active_sl.broken = True
                    
                    if current_bias >= 0:
                        # เปลี่ยนจาก Bullish/Neutral → Bearish = CHoCH
                        event_type = StructureType.CHOCH_BEARISH
                    else:
                        # ยังคงเป็น Bearish = BOS
                        event_type = StructureType.BOS_BEARISH
                    
                    event = StructureEvent(
                        event_type=event_type,
                        index=i,
                        price=closes[i],
                        time=df.index[i],
                        broken_swing=active_sl
                    )
                    self._events.append(event)
                    structure_event[i] = event_type.value
                    current_bias = -1  # Bearish
            
            market_bias[i] = current_bias

        # อัพเดท Current Bias
        self._current_bias = current_bias
        
        # เพิ่มคอลัมน์
        df['structure_event'] = structure_event
        df['market_bias'] = market_bias
        
        if bot_config.debug_mode:
            bos_count = sum(1 for e in self._events if e.is_bos)
            choch_count = len(self._events) - bos_count
            bias_label = {1: "BULLISH ↑", -1: "BEARISH ↓", 0: "NEUTRAL ↔"}
            print(f"💥 [Market Structure] พบ {bos_count} BOS, {choch_count} CHoCH")
            print(f"   📊 Market Bias ปัจจุบัน: {bias_label.get(self._current_bias, 'UNKNOWN')}")

        return df

    # =========================================================================
    # 📊 ดึงข้อมูลสรุป
    # =========================================================================

    def get_current_bias(self) -> int:
        """
        ดึง Market Bias ปัจจุบัน
        
        Returns:
            int: 1 (Bullish), -1 (Bearish), 0 (Neutral)
        """
        return self._current_bias

    def get_latest_event(self) -> Optional[StructureEvent]:
        """ดึง Structure Event ล่าสุด (BOS หรือ CHoCH)"""
        return self._events[-1] if self._events else None

    def get_recent_swing_high(self) -> Optional[SwingPoint]:
        """ดึง Swing High ล่าสุดที่ยังไม่ถูกทะลุ"""
        for sh in reversed(self._swing_highs):
            if not sh.broken:
                return sh
        return self._swing_highs[-1] if self._swing_highs else None

    def get_recent_swing_low(self) -> Optional[SwingPoint]:
        """ดึง Swing Low ล่าสุดที่ยังไม่ถูกทะลุ"""
        for sl in reversed(self._swing_lows):
            if not sl.broken:
                return sl
        return self._swing_lows[-1] if self._swing_lows else None

    def get_structure_summary(self) -> Dict:
        """
        สรุปโครงสร้างตลาดปัจจุบัน
        
        Returns:
            Dict: ข้อมูลสรุปโครงสร้างตลาด
        """
        bias_labels = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
        latest_event = self.get_latest_event()
        recent_sh = self.get_recent_swing_high()
        recent_sl = self.get_recent_swing_low()
        
        return {
            "bias": self._current_bias,
            "bias_label": bias_labels.get(self._current_bias, "UNKNOWN"),
            "total_swing_highs": len(self._swing_highs),
            "total_swing_lows": len(self._swing_lows),
            "total_bos": sum(1 for e in self._events if e.is_bos),
            "total_choch": sum(1 for e in self._events if not e.is_bos),
            "latest_event": latest_event.event_type.value if latest_event else "NONE",
            "latest_event_price": latest_event.price if latest_event else None,
            "recent_swing_high": recent_sh.price if recent_sh else None,
            "recent_swing_low": recent_sl.price if recent_sl else None,
        }

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        วิเคราะห์โครงสร้างตลาดทั้งหมดในครั้งเดียว
        
        Args:
            df: DataFrame ที่มี OHLCV
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์โครงสร้างตลาด
        """
        df = self.find_swing_points(df)
        df = self.detect_structure_breaks(df)
        return df

    def __repr__(self) -> str:
        return (
            f"MarketStructure(bias={self._current_bias}, "
            f"swings_h={len(self._swing_highs)}, swings_l={len(self._swing_lows)}, "
            f"events={len(self._events)})"
        )
