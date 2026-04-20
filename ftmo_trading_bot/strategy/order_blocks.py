"""
===============================================================================
FTMO Trading Bot — Order Block Detection (ตรวจจับ Order Blocks)
===============================================================================
Order Block คือ โซนราคาที่ Smart Money (สถาบันการเงินขนาดใหญ่) เข้าเปิด
Position จำนวนมาก ทำให้เกิดการเคลื่อนไหวรุนแรง

ลักษณะ Order Block:
- Bullish OB: แท่งเทียนขาลงสุดท้ายก่อนราคาพุ่งขึ้นรุนแรง (Impulsive Move)
- Bearish OB: แท่งเทียนขาขึ้นสุดท้ายก่อนราคาร่วงรุนแรง (Impulsive Move)

การใช้งาน:
- เมื่อราคากลับมาทดสอบ Order Block → จุด Entry ที่มีโอกาสสูง
- OB ที่ยังไม่ถูก Mitigate (ทดสอบ) มีความสำคัญมากกว่า
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from config.settings import bot_config


class OBType(Enum):
    """ประเภทของ Order Block"""
    BULLISH = "BULLISH"     # โซนรับ — ราคาน่าจะเด้งขึ้น
    BEARISH = "BEARISH"     # โซนต้าน — ราคาน่าจะกลับลง


@dataclass
class OrderBlock:
    """
    ข้อมูล Order Block หนึ่งจุด
    
    Order Block ประกอบด้วย:
    - โซนราคา (High - Low ของแท่งเทียน OB)
    - ประเภท (Bullish = รับ, Bearish = ต้าน)
    - สถานะ (Mitigated = ถูกทดสอบแล้ว, Active = ยังไม่ถูกทดสอบ)
    """
    ob_type: OBType             # ประเภท: Bullish หรือ Bearish
    index: int                  # ตำแหน่งใน DataFrame
    time: object                # เวลาเกิด
    high: float                 # ขอบบนของ Order Block
    low: float                  # ขอบล่างของ Order Block
    open_price: float           # ราคาเปิดของแท่ง OB
    close_price: float          # ราคาปิดของแท่ง OB
    body_size: float            # ขนาด Body ของแท่ง OB
    impulse_size: float         # ขนาดของ Impulsive Move ที่ตามมา
    mitigated: bool = False     # ถูก Mitigate (ทดสอบ) แล้วหรือยัง
    mitigated_index: int = -1   # ตำแหน่งที่ถูก Mitigate
    strength_score: float = 0.0 # คะแนนความแข็งแกร่ง (0-100)
    confirmed_index: int = -1   # ตำแหน่งแท่ง impulse ที่ยืนยัน OB (> index — anti-repaint)
    
    @property
    def zone_mid(self) -> float:
        """ราคากลางของ Order Block"""
        return (self.high + self.low) / 2
    
    @property
    def zone_range(self) -> float:
        """ความกว้างของ Order Block (pips)"""
        return self.high - self.low
    
    @property
    def is_active(self) -> bool:
        """ยังเป็น OB ที่ Active อยู่หรือไม่ (ยังไม่ถูก Mitigate)"""
        return not self.mitigated


class OrderBlockDetector:
    """
    ตรวจจับ Order Blocks จาก Price Action
    
    วิธีการตรวจจับ:
    1. หาแท่งเทียนที่สร้าง Impulsive Move (การเคลื่อนไหวรุนแรง)
    2. ดูแท่งเทียนก่อนหน้าที่เป็นทิศตรงข้าม → นั่นคือ Order Block
    3. ตรวจสอบว่า OB ถูก Mitigate แล้วหรือยัง
    4. ให้คะแนนความแข็งแกร่งของ OB
    """

    def __init__(self):
        """เริ่มต้น Order Block Detector"""
        self._config = bot_config.indicators
        self._lookback = self._config.ob_lookback               # 50 แท่ง
        self._min_body_ratio = self._config.ob_min_body_ratio   # 0.5 (50%)
        
        # เก็บ Order Blocks ที่ตรวจพบ
        self._bullish_obs: List[OrderBlock] = []
        self._bearish_obs: List[OrderBlock] = []
        
        print("📦 [Order Blocks] เริ่มต้นระบบตรวจจับ Order Blocks")

    # =========================================================================
    # 🔍 ตรวจจับ Order Blocks
    # =========================================================================

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ตรวจจับ Bullish และ Bearish Order Blocks จาก DataFrame
        
        ขั้นตอน:
        1. คำนวณ Body Size และ Range ของแต่ละแท่ง
        2. หาแท่งที่สร้าง Impulsive Move (Body > 50% ของ Range + ขนาดใหญ่)
        3. ดูแท่งก่อนหน้า — ถ้าเป็นแท่งทิศตรงข้าม → Order Block
        4. ตรวจสอบ Mitigation
        5. ให้คะแนนความแข็งแกร่ง
        
        Args:
            df: DataFrame ที่มี OHLCV
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ OB
        """
        if len(df) < self._lookback:
            print(f"⚠️ [Order Blocks] ข้อมูลไม่เพียงพอ (ต้องการ >= {self._lookback})")
            return df

        # ล้างข้อมูลเดิม
        self._bullish_obs = []
        self._bearish_obs = []
        
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        length = len(df)
        
        # คำนวณ Body Size และ Range
        body_size = np.abs(closes - opens)
        candle_range = highs - lows
        
        # หลีกเลี่ยงการหาร 0
        candle_range = np.where(candle_range == 0, 0.00001, candle_range)
        body_ratio = body_size / candle_range  # อัตราส่วน Body/Range
        
        # คำนวณค่าเฉลี่ย Body Size สำหรับ Filter (ใช้ Rolling Average)
        avg_body = pd.Series(body_size).rolling(20).mean().values
        
        # สร้าง Arrays สำหรับ OB
        ob_bullish_high = np.full(length, np.nan)
        ob_bullish_low = np.full(length, np.nan)
        ob_bearish_high = np.full(length, np.nan)
        ob_bearish_low = np.full(length, np.nan)

        # วนลูปตรวจจับ OB (เริ่มจากแท่งที่ 3 เป็นต้นไป)
        for i in range(3, length - 1):
            # === ตรวจจับ Bullish Order Block ===
            # เงื่อนไข: แท่ง i เป็น Bullish Impulsive Move (ขึ้นแรง)
            is_bullish_impulse = (
                closes[i] > opens[i] and                        # แท่งเขียว (ขึ้น)
                body_ratio[i] >= self._min_body_ratio and       # Body > 50% ของ Range
                body_size[i] > avg_body[i] * 1.5 and           # Body ใหญ่กว่าเฉลี่ย 1.5 เท่า
                not np.isnan(avg_body[i])                       # มีค่าเฉลี่ยแล้ว
            )
            
            if is_bullish_impulse:
                # หาแท่ง Bearish (ขาลง) ก่อนหน้า → นั่นคือ Bullish OB
                for j in range(i - 1, max(i - 4, 0), -1):  # ดูย้อนหลัง 1-3 แท่ง
                    if closes[j] < opens[j]:  # แท่งแดง (ขาลง)
                        ob = OrderBlock(
                            ob_type=OBType.BULLISH,
                            index=j,
                            time=df.index[j],
                            high=highs[j],
                            low=lows[j],
                            open_price=opens[j],
                            close_price=closes[j],
                            body_size=body_size[j],
                            impulse_size=body_size[i],
                        )
                        # Anti-repaint: mark "confirmed at impulse bar i" ไม่ใช่แท่งต้นกำเนิด j
                        # เหตุผล: ใน live เรารู้ว่า j เป็น OB ก็ต่อเมื่อ i ปิดแล้ว
                        # ถ้าเขียนที่ j → backtest reader ที่อ่าน df.iloc[j] จะ "เห็น" OB ก่อนเวลา
                        ob.confirmed_index = i
                        self._bullish_obs.append(ob)
                        ob_bullish_high[i] = highs[j]
                        ob_bullish_low[i] = lows[j]
                        break  # ใช้แท่งแรกที่เจอ
            
            # === ตรวจจับ Bearish Order Block ===
            # เงื่อนไข: แท่ง i เป็น Bearish Impulsive Move (ลงแรง)
            is_bearish_impulse = (
                closes[i] < opens[i] and                        # แท่งแดง (ลง)
                body_ratio[i] >= self._min_body_ratio and       # Body > 50% ของ Range
                body_size[i] > avg_body[i] * 1.5 and           # Body ใหญ่กว่าเฉลี่ย 1.5 เท่า
                not np.isnan(avg_body[i])
            )
            
            if is_bearish_impulse:
                # หาแท่ง Bullish (ขาขึ้น) ก่อนหน้า → นั่นคือ Bearish OB
                for j in range(i - 1, max(i - 4, 0), -1):
                    if closes[j] > opens[j]:  # แท่งเขียว (ขาขึ้น)
                        ob = OrderBlock(
                            ob_type=OBType.BEARISH,
                            index=j,
                            time=df.index[j],
                            high=highs[j],
                            low=lows[j],
                            open_price=opens[j],
                            close_price=closes[j],
                            body_size=body_size[j],
                            impulse_size=body_size[i],
                        )
                        # Anti-repaint: เขียนค่าที่แท่ง confirmation (i) ไม่ใช่แท่ง origin (j)
                        ob.confirmed_index = i
                        self._bearish_obs.append(ob)
                        ob_bearish_high[i] = highs[j]
                        ob_bearish_low[i] = lows[j]
                        break

        # เพิ่มคอลัมน์เข้า DataFrame
        df['ob_bullish_high'] = ob_bullish_high
        df['ob_bullish_low'] = ob_bullish_low
        df['ob_bearish_high'] = ob_bearish_high
        df['ob_bearish_low'] = ob_bearish_low
        
        # ตรวจสอบ Mitigation และให้คะแนน
        self._check_mitigation(df)
        self._score_order_blocks(df)
        
        return df

    # =========================================================================
    # 🔄 Mitigation Check (ตรวจสอบว่า OB ถูกทดสอบแล้วหรือยัง)
    # =========================================================================

    def _check_mitigation(self, df: pd.DataFrame):
        """
        ตรวจสอบว่า Order Block ถูก Mitigate (ราคากลับมาทดสอบแล้ว) หรือยัง

        กฎ (SMC standard):
        - Bullish OB: ถูก Mitigate เมื่อ **body ปิด** ต่ำกว่า OB.low (ทะลุโซนจริง)
          wick ลงไปแตะแล้วเด้งกลับ = OB ยังดีอยู่ (liquidity sweep)
        - Bearish OB: ถูก Mitigate เมื่อ **body ปิด** สูงกว่า OB.high

        OB ที่ถูก Mitigate แล้วจะลดความสำคัญลง
        """
        closes = df['close'].values
        opens = df['open'].values

        # ตรวจ Bullish OBs — body close ต่ำกว่าขอบล่าง OB = ทะลุจริง
        for ob in self._bullish_obs:
            if ob.mitigated:
                continue
            for i in range(ob.index + 1, len(df)):
                candle_body_low = min(closes[i], opens[i])
                if candle_body_low < ob.low:
                    ob.mitigated = True
                    ob.mitigated_index = i
                    break

        # ตรวจ Bearish OBs — body close สูงกว่าขอบบน OB = ทะลุจริง
        for ob in self._bearish_obs:
            if ob.mitigated:
                continue
            for i in range(ob.index + 1, len(df)):
                candle_body_high = max(closes[i], opens[i])
                if candle_body_high > ob.high:
                    ob.mitigated = True
                    ob.mitigated_index = i
                    break

    # =========================================================================
    # ⭐ Order Block Scoring (ให้คะแนนความแข็งแกร่ง)
    # =========================================================================

    def _score_order_blocks(self, df: pd.DataFrame):
        """
        คำนวณคะแนนความแข็งแกร่งของแต่ละ Order Block (0-100)
        
        ปัจจัยที่ใช้ให้คะแนน:
        1. ขนาด Impulsive Move (ยิ่งแรง = ยิ่งดี) — 35 คะแนน
        2. Body Ratio ของ OB (ยิ่ง Body ใหญ่ = ยิ่ง strong) — 20 คะแนน
        3. ยังไม่ถูก Mitigate (Fresh OB) — 25 คะแนน
        4. ความใกล้กับราคาปัจจุบัน — 20 คะแนน
        """
        if len(df) == 0:
            return
            
        current_price = df['close'].iloc[-1]
        
        all_obs = self._bullish_obs + self._bearish_obs
        if not all_obs:
            return
        
        # หาค่าสูงสุดของ Impulse Size สำหรับ Normalize
        max_impulse = max(ob.impulse_size for ob in all_obs) if all_obs else 1
        max_impulse = max(max_impulse, 0.00001)  # ป้องกันหาร 0
        
        for ob in all_obs:
            score = 0.0
            
            # ปัจจัยที่ 1: Impulse Size (35 คะแนน)
            impulse_score = (ob.impulse_size / max_impulse) * 35
            score += impulse_score
            
            # ปัจจัยที่ 2: Body Ratio (20 คะแนน)
            ob_range = ob.high - ob.low
            ob_body_ratio = ob.body_size / ob_range if ob_range > 0 else 0
            score += ob_body_ratio * 20
            
            # ปัจจัยที่ 3: Fresh (ยังไม่ Mitigate) — 25 คะแนน
            if ob.is_active:
                score += 25
            
            # ปัจจัยที่ 4: ความใกล้กับราคาปัจจุบัน (20 คะแนน)
            distance = abs(current_price - ob.zone_mid)
            relative_distance = distance / current_price if current_price > 0 else 1
            proximity_score = max(0, 20 * (1 - relative_distance * 10))  # ยิ่งใกล้ = ยิ่งดี
            score += proximity_score
            
            ob.strength_score = min(score, 100)  # จำกัดที่ 100

    # =========================================================================
    # 📊 ดึงข้อมูล Order Blocks
    # =========================================================================

    def get_active_bullish_obs(self, min_score: float = 30.0) -> List[OrderBlock]:
        """
        ดึง Bullish Order Blocks ที่ Active (ยังไม่ถูก Mitigate) และคะแนนเพียงพอ
        
        Args:
            min_score: คะแนนขั้นต่ำ (ค่าเริ่มต้น: 30)
            
        Returns:
            List[OrderBlock]: รายการ Bullish OB ที่ Active
        """
        return [
            ob for ob in self._bullish_obs
            if ob.is_active and ob.strength_score >= min_score
        ]

    def get_active_bearish_obs(self, min_score: float = 30.0) -> List[OrderBlock]:
        """
        ดึง Bearish Order Blocks ที่ Active
        
        Args:
            min_score: คะแนนขั้นต่ำ
            
        Returns:
            List[OrderBlock]: รายการ Bearish OB ที่ Active
        """
        return [
            ob for ob in self._bearish_obs
            if ob.is_active and ob.strength_score >= min_score
        ]

    def is_price_at_bullish_ob(
        self,
        current_price: float,
        tolerance_pips: float = 5.0,
        pip_size: float = 0.0001
    ) -> Optional[OrderBlock]:
        """
        ตรวจสอบว่าราคาปัจจุบันอยู่ที่ Bullish Order Block หรือไม่
        
        เงื่อนไข:
        - ราคาต้องอยู่ภายในโซน OB (Low - tolerance ถึง High + tolerance)
        - OB ต้อง Active (ยังไม่ถูก Mitigate)
        - ใช้ OB ที่มีคะแนนสูงสุด
        
        Args:
            current_price: ราคาปัจจุบัน
            tolerance_pips: ระยะ tolerance (pips)
            pip_size: ขนาด 1 pip
            
        Returns:
            OrderBlock หรือ None: OB ที่ราคาอยู่ (None ถ้าไม่มี)
        """
        tolerance = tolerance_pips * pip_size
        best_ob = None
        best_score = 0
        
        for ob in self._bullish_obs:
            if not ob.is_active:
                continue
            
            # ตรวจว่าราคาอยู่ในโซน OB หรือไม่
            if (ob.low - tolerance) <= current_price <= (ob.high + tolerance):
                if ob.strength_score > best_score:
                    best_ob = ob
                    best_score = ob.strength_score
        
        return best_ob

    def is_price_at_bearish_ob(
        self,
        current_price: float,
        tolerance_pips: float = 5.0,
        pip_size: float = 0.0001
    ) -> Optional[OrderBlock]:
        """
        ตรวจสอบว่าราคาปัจจุบันอยู่ที่ Bearish Order Block หรือไม่
        
        Args:
            current_price: ราคาปัจจุบัน
            tolerance_pips: ระยะ tolerance (pips)
            pip_size: ขนาด 1 pip
            
        Returns:
            OrderBlock หรือ None
        """
        tolerance = tolerance_pips * pip_size
        best_ob = None
        best_score = 0
        
        for ob in self._bearish_obs:
            if not ob.is_active:
                continue
            
            if (ob.low - tolerance) <= current_price <= (ob.high + tolerance):
                if ob.strength_score > best_score:
                    best_ob = ob
                    best_score = ob.strength_score
        
        return best_ob

    # =========================================================================
    # 🔗 OB Clustering — รวม OB ที่อยู่ใกล้กันเป็นโซนแข็งแกร่ง
    # =========================================================================

    def get_bullish_cluster_score(
        self, price: float, tolerance_pips: float = 15, pip_size: float = 0.0001
    ) -> Optional[float]:
        """หา cluster ของ Bullish OBs ใกล้ราคา แล้วคืนคะแนนรวม"""
        return self._get_cluster_score(
            self.get_active_bullish_obs(min_score=20), price, tolerance_pips, pip_size
        )

    def get_bearish_cluster_score(
        self, price: float, tolerance_pips: float = 15, pip_size: float = 0.0001
    ) -> Optional[float]:
        """หา cluster ของ Bearish OBs ใกล้ราคา แล้วคืนคะแนนรวม"""
        return self._get_cluster_score(
            self.get_active_bearish_obs(min_score=20), price, tolerance_pips, pip_size
        )

    def _get_cluster_score(
        self, obs: List[OrderBlock], price: float,
        tolerance_pips: float, pip_size: float
    ) -> Optional[float]:
        """
        คืนคะแนน OB ดีที่สุดในรัศมี (ไม่ inflate ด้วย cluster bonus)

        Returns:
            float in [0, 100] — เพราะ ob.strength_score ถูก cap ที่ 100 ใน _score_obs()
            หรือ None ถ้าไม่มี OB ใกล้
        """
        tolerance = tolerance_pips * pip_size
        nearby = [
            ob for ob in obs
            if (ob.low - tolerance) <= price <= (ob.high + tolerance)
        ]
        if not nearby:
            return None
        return max(ob.strength_score for ob in nearby)

    def get_ob_summary(self) -> Dict:
        """สรุปข้อมูล Order Blocks ทั้งหมด"""
        active_bull = [ob for ob in self._bullish_obs if ob.is_active]
        active_bear = [ob for ob in self._bearish_obs if ob.is_active]
        
        return {
            "total_bullish_obs": len(self._bullish_obs),
            "active_bullish_obs": len(active_bull),
            "total_bearish_obs": len(self._bearish_obs),
            "active_bearish_obs": len(active_bear),
            "best_bullish_score": max((ob.strength_score for ob in active_bull), default=0),
            "best_bearish_score": max((ob.strength_score for ob in active_bear), default=0),
            "nearest_bullish_ob": min((ob.low for ob in active_bull), default=None),
            "nearest_bearish_ob": max((ob.high for ob in active_bear), default=None),
        }

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """วิเคราะห์ Order Blocks ทั้งหมดในครั้งเดียว"""
        return self.detect_order_blocks(df)

    def __repr__(self) -> str:
        active_bull = sum(1 for ob in self._bullish_obs if ob.is_active)
        active_bear = sum(1 for ob in self._bearish_obs if ob.is_active)
        return f"OrderBlockDetector(bull_active={active_bull}, bear_active={active_bear})"
