"""
===============================================================================
FTMO Trading Bot — Technical Indicators (ตัวชี้วัดทางเทคนิค)
===============================================================================
โมดูลคำนวณ Indicators ที่ใช้ในกลยุทธ์ SMC:
- ATR (Average True Range) — วัดความผันผวนของตลาด
- EMA (Exponential Moving Average) — วัดแนวโน้ม
- RSI (Relative Strength Index) — วัดแรงซื้อ/ขาย
- Volume Profile — วิเคราะห์ปริมาณการซื้อขาย

คำนวณโดยใช้ pandas/numpy โดยตรง (ไม่พึ่ง ta-lib)
เพื่อให้ควบคุมการคำนวณได้เต็มที่และลด Dependencies
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from config.settings import bot_config


class TechnicalIndicators:
    """
    คลาสคำนวณ Technical Indicators ทั้งหมด
    
    ใช้ DataFrame ที่มีคอลัมน์ ['open', 'high', 'low', 'close', 'volume']
    เป็น Input และเพิ่มคอลัมน์ Indicator เข้าไปใน DataFrame เดิม
    """

    def __init__(self):
        """เริ่มต้น Indicator Calculator — ดึงพารามิเตอร์จาก Config"""
        self._config = bot_config.indicators
        print("📊 [Indicators] เริ่มต้นระบบคำนวณตัวชี้วัดทางเทคนิค")

    # =========================================================================
    # 📈 ATR — Average True Range (วัดความผันผวน)
    # =========================================================================

    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = None,
        column_name: str = "atr"
    ) -> pd.DataFrame:
        """
        คำนวณ ATR (Average True Range) — วัดความผันผวนเฉลี่ย
        
        สูตร True Range:
        TR = max(
            High - Low,                        # ช่วงราคาแท่งปัจจุบัน
            |High - Previous Close|,           # Gap ขึ้นข้างบน
            |Low - Previous Close|             # Gap ลงข้างล่าง
        )
        
        ATR = EMA ของ TR (ใช้ Wilder's Smoothing)
        
        ใช้สำหรับ:
        - กำหนดระยะ Stop Loss แบบ Dynamic
        - กรองสัญญาณเทรด (Volatility Filter)
        - คำนวณขนาด Position
        
        Args:
            df: DataFrame ที่มี ['high', 'low', 'close']
            period: จำนวนแท่งเทียนสำหรับ ATR (ค่าเริ่มต้น: 14)
            column_name: ชื่อคอลัมน์ผลลัพธ์
            
        Returns:
            pd.DataFrame: DataFrame เดิมที่เพิ่มคอลัมน์ ATR
        """
        if period is None:
            period = self._config.atr_period  # 14

        # คำนวณ True Range
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - df['close'].shift(1)).abs()
        low_prev_close = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
        # คำนวณ ATR ด้วย Wilder's Smoothing (EMA พิเศษ)
        # Wilder's Alpha = 1/period (แทนที่ 2/(period+1) ของ EMA ปกติ)
        df[column_name] = true_range.ewm(alpha=1/period, min_periods=period).mean()
        
        return df

    # =========================================================================
    # 📊 EMA — Exponential Moving Average (วัดแนวโน้ม)
    # =========================================================================

    def calculate_ema(
        self,
        df: pd.DataFrame,
        period: int = None,
        source: str = "close",
        column_name: str = None
    ) -> pd.DataFrame:
        """
        คำนวณ EMA (Exponential Moving Average)
        
        สูตร: EMA = Price × α + Previous EMA × (1 - α)
        โดย α = 2 / (period + 1)
        
        ใช้สำหรับ:
        - ระบุแนวโน้ม (Trend Direction)
        - หา Dynamic Support/Resistance
        - กรอง Entry ตามแนวโน้ม
        
        Args:
            df: DataFrame
            period: จำนวนแท่ง (None = ใช้ Config)
            source: คอลัมน์ต้นทาง (ค่าเริ่มต้น: "close")
            column_name: ชื่อคอลัมน์ผลลัพธ์ (None = "ema_{period}")
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ EMA
        """
        if period is None:
            period = self._config.ema_fast  # 21
        if column_name is None:
            column_name = f"ema_{period}"

        df[column_name] = df[source].ewm(span=period, min_periods=period).mean()
        
        return df

    def calculate_all_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        คำนวณ EMA ทั้ง 3 เส้น (Fast, Medium, Slow) ในครั้งเดียว
        
        EMA 21:  แนวโน้มระยะสั้น (สัปดาห์)
        EMA 50:  แนวโน้มระยะกลาง (2 เดือน)
        EMA 200: แนวโน้มระยะยาว (Master Trend)
        
        Args:
            df: DataFrame ที่มีคอลัมน์ 'close'
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่ม 3 คอลัมน์ EMA
        """
        df = self.calculate_ema(df, period=self._config.ema_fast, column_name="ema_fast")     # EMA 21
        df = self.calculate_ema(df, period=self._config.ema_medium, column_name="ema_medium")  # EMA 50
        df = self.calculate_ema(df, period=self._config.ema_slow, column_name="ema_slow")      # EMA 200
        
        return df

    # =========================================================================
    # 💪 RSI — Relative Strength Index (วัดแรงซื้อ/ขาย)
    # =========================================================================

    def calculate_rsi(
        self,
        df: pd.DataFrame,
        period: int = None,
        source: str = "close",
        column_name: str = "rsi"
    ) -> pd.DataFrame:
        """
        คำนวณ RSI (Relative Strength Index)
        
        สูตร:
        1. คำนวณ Price Change = Close - Previous Close
        2. แยก Gain (บวก) และ Loss (ลบ)
        3. Avg Gain = EMA ของ Gain
        4. Avg Loss = EMA ของ Loss
        5. RS = Avg Gain / Avg Loss
        6. RSI = 100 - (100 / (1 + RS))
        
        ใช้สำหรับ:
        - ตรวจจับ Overbought (>70) / Oversold (<30)
        - ยืนยันทิศทาง Momentum
        - หา Divergence (ราคาไปทาง RSI ไปอีกทาง)
        
        Args:
            df: DataFrame ที่มีคอลัมน์ 'close'
            period: จำนวนแท่ง (ค่าเริ่มต้น: 14)
            source: คอลัมน์ต้นทาง
            column_name: ชื่อคอลัมน์ผลลัพธ์
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ RSI
        """
        if period is None:
            period = self._config.rsi_period  # 14

        # คำนวณ Price Change
        delta = df[source].diff()
        
        # แยก Gain และ Loss
        gain = delta.clip(lower=0)          # เก็บเฉพาะค่าบวก (ราคาขึ้น)
        loss = (-delta).clip(lower=0)       # เก็บเฉพาะค่าลบแปลงเป็นบวก (ราคาลง)
        
        # คำนวณค่าเฉลี่ยด้วย Wilder's Smoothing (เหมือน ATR)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        # คำนวณ RS และ RSI
        rs = avg_gain / avg_loss
        df[column_name] = 100 - (100 / (1 + rs))
        
        return df

    # =========================================================================
    # 🔀 Trend Detection (การตรวจจับแนวโน้ม)
    # =========================================================================

    def detect_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ตรวจจับแนวโน้มตลาดจาก EMA ทั้ง 3 เส้น
        
        กฎ:
        - BULLISH (ขาขึ้น): EMA 21 > EMA 50 > EMA 200 และ Close > EMA 200
        - BEARISH (ขาลง): EMA 21 < EMA 50 < EMA 200 และ Close < EMA 200
        - RANGING (ไซด์เวย์): ไม่เข้ากฎข้างต้น
        
        เพิ่มคอลัมน์:
        - 'trend': 1 (Bullish), -1 (Bearish), 0 (Ranging)
        - 'trend_strength': ความแรงของแนวโน้ม (0-100)
        
        Args:
            df: DataFrame ที่ต้องมี EMA ทั้ง 3 เส้น
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ trend
        """
        # ตรวจสอบว่ามี EMA ครบหรือยัง — ถ้ายังไม่มีให้คำนวณ
        if 'ema_fast' not in df.columns:
            df = self.calculate_all_emas(df)

        # === Bullish: EMA 21 > 50 > 200 ===
        bullish_alignment = (
            (df['ema_fast'] > df['ema_medium']) &       # EMA 21 > 50
            (df['ema_medium'] > df['ema_slow']) &       # EMA 50 > 200
            (df['close'] > df['ema_slow'])              # Close > EMA 200
        )

        # === Bearish: EMA 21 < 50 < 200 ===
        bearish_alignment = (
            (df['ema_fast'] < df['ema_medium']) &       # EMA 21 < 50
            (df['ema_medium'] < df['ema_slow']) &       # EMA 50 < 200
            (df['close'] < df['ema_slow'])              # Close < EMA 200
        )

        # กำหนด Trend
        df['trend'] = 0  # ค่าเริ่มต้น: Ranging
        df.loc[bullish_alignment, 'trend'] = 1      # Bullish
        df.loc[bearish_alignment, 'trend'] = -1     # Bearish

        # คำนวณ Trend Strength (ระยะห่างระหว่าง EMA เทียบกับราคา)
        ema_spread = ((df['ema_fast'] - df['ema_slow']).abs() / df['close']) * 10000
        df['trend_strength'] = ema_spread.clip(upper=100)  # จำกัดที่ 100

        return df

    # =========================================================================
    # 📉 Volatility Filter (กรองความผันผวน)
    # =========================================================================

    def calculate_volatility_filter(
        self,
        df: pd.DataFrame,
        atr_min_pips: float = 5.0,
        atr_max_pips: float = 50.0,
        pip_size: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        กรองสภาวะตลาดด้วย ATR — เทรดเฉพาะเมื่อ Volatility อยู่ในช่วงที่เหมาะสม

        กฎ:
        - ATR ต่ำเกินไป (< 5 pips): ตลาดเงียบ → ไม่เทรด
        - ATR สูงเกินไป (> 50 pips): ตลาดผันผวนมาก → ไม่เทรด (เสี่ยงเกินไป)
        - ATR อยู่ในช่วง: เทรดได้

        Args:
            df: DataFrame ที่ต้องมี ATR
            atr_min_pips: ATR ขั้นต่ำ (pips)
            atr_max_pips: ATR สูงสุด (pips)
            pip_size: ขนาด 1 pip (None = autodetect จากระดับราคา)
                      - Major pairs (EURUSD, GBPUSD): 0.0001
                      - JPY pairs (USDJPY, EURJPY): 0.01

        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ 'volatility_ok'
        """
        # ตรวจสอบว่ามี ATR หรือยัง
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)

        # === Autodetect pip_size ถ้าไม่ได้ระบุ ===
        # JPY pairs มีราคาประมาณ 100-200 (e.g. USDJPY ~150)
        # Major pairs มีราคา ~0.5-2.0 (e.g. EURUSD ~1.08)
        if pip_size is None:
            # ใช้ราคาเฉลี่ยของ close ล่าสุดในการจำแนก
            try:
                ref_price = float(df['close'].iloc[-1])
                pip_size = 0.01 if ref_price > 20 else 0.0001
            except Exception:
                pip_size = 0.0001  # ค่าเริ่มต้นสำหรับ Major pairs

        # แปลง ATR (ราคา) → pips
        # EURUSD: ATR=0.0010 / 0.0001 = 10 pips
        # USDJPY: ATR=0.10   / 0.01   = 10 pips
        pip_multiplier = 1.0 / pip_size
        atr_pips = df['atr'] * pip_multiplier

        df['atr_pips'] = atr_pips
        df['volatility_ok'] = (atr_pips >= atr_min_pips) & (atr_pips <= atr_max_pips)

        return df

    # =========================================================================
    # 🔄 คำนวณ Indicator ทั้งหมดในครั้งเดียว
    # =========================================================================

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        คำนวณ Indicator ทั้งหมดที่ Bot ต้องการในครั้งเดียว
        
        เพิ่มคอลัมน์:
        - atr: Average True Range
        - ema_fast, ema_medium, ema_slow: EMA 21, 50, 200
        - rsi: Relative Strength Index
        - trend: ทิศทางแนวโน้ม (1, 0, -1)
        - trend_strength: ความแรงแนวโน้ม
        - volatility_ok: กรอง Volatility
        
        Args:
            df: DataFrame ที่มี ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่ม Indicators ทั้งหมด
        """
        if df is None or len(df) < 200:
            print(f"⚠️ [Indicators] ข้อมูลไม่เพียงพอ (ต้องการ >= 200 แท่ง, มี {len(df) if df is not None else 0})")
            return df

        # คำนวณทุก Indicator
        df = self.calculate_atr(df)
        df = self.calculate_all_emas(df)
        df = self.calculate_rsi(df)
        df = self.detect_trend(df)
        df = self.calculate_volatility_filter(df)
        
        return df

    # =========================================================================
    # 📊 ข้อมูลสรุป
    # =========================================================================

    def get_latest_values(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        ดึงค่า Indicator ล่าสุด (แท่งเทียนล่าสุด)
        
        Args:
            df: DataFrame ที่คำนวณ Indicators แล้ว
            
        Returns:
            Dict: ค่า Indicator ล่าสุด
        """
        if df is None or len(df) == 0:
            return None

        last = df.iloc[-1]
        
        result = {
            "close": last.get("close", 0),
            "atr": last.get("atr", 0),
            "atr_pips": last.get("atr_pips", 0),
            "ema_fast": last.get("ema_fast", 0),
            "ema_medium": last.get("ema_medium", 0),
            "ema_slow": last.get("ema_slow", 0),
            "rsi": last.get("rsi", 50),
            "trend": int(last.get("trend", 0)),
            "trend_strength": last.get("trend_strength", 0),
            "volatility_ok": bool(last.get("volatility_ok", False)),
        }

        # แปลง Trend เป็นข้อความ
        trend_labels = {1: "BULLISH ↑", -1: "BEARISH ↓", 0: "RANGING ↔"}
        result["trend_label"] = trend_labels.get(result["trend"], "UNKNOWN")
        
        return result

    def __repr__(self) -> str:
        return (
            f"TechnicalIndicators("
            f"ATR={self._config.atr_period}, "
            f"EMA={self._config.ema_fast}/{self._config.ema_medium}/{self._config.ema_slow}, "
            f"RSI={self._config.rsi_period})"
        )
