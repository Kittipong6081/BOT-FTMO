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

        # คำนวณ RS และ RSI — ป้องกัน divide-by-zero เมื่อ avg_loss=0 (ตลาดขึ้นล้วน)
        # เคสนี้ rs→∞ ⇒ RSI=100 ใช้ np.where แทนการ fill ค่า epsilon เพื่อได้ค่าถูกต้อง
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss > 0, 100.0)   # avg_loss=0 → ราคาขึ้นล้วน → RSI=100
        rsi = rsi.where(avg_gain > 0, 0.0)     # avg_gain=0 → ราคาลงล้วน → RSI=0
        df[column_name] = rsi

        return df

    # =========================================================================
    # 📉 MACD — Moving Average Convergence Divergence
    # =========================================================================

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        คำนวณ MACD: MACD Line, Signal Line, Histogram

        Histogram > 0 = momentum ขาขึ้น, < 0 = momentum ขาลง
        """
        ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
        ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, min_periods=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    # =========================================================================
    # 📐 ADX — Average Directional Index (วัดว่าตลาด Trend หรือ Range)
    # =========================================================================

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ADX บอกว่าตลาดกำลัง trending หรือ ranging
        > 25 = trending, < 20 = ranging
        ต่างจาก trend_strength (EMA spread) ตรงที่ ADX วัดจาก directional movement โดยตรง
        """
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr_smooth = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_smooth
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_smooth

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=1/period, min_periods=period).mean()
        return df

    # =========================================================================
    # 📊 Stochastic — ตำแหน่งราคาใน Range ล่าสุด
    # =========================================================================

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic %K บอกว่าราคาอยู่ตรงไหนของ high-low range ล่าสุด
        0 = อยู่ที่ low สุด, 100 = อยู่ที่ high สุด
        ช่วย agent เห็นว่า momentum หมดแรงหรือยัง
        """
        lowest = df['low'].rolling(k_period).min()
        highest = df['high'].rolling(k_period).max()
        df['stoch_k'] = 100 * (df['close'] - lowest) / (highest - lowest + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
        return df

    # =========================================================================
    # 📈 Bollinger %B — ตำแหน่งราคาใน Volatility Band
    # =========================================================================

    def calculate_bollinger_pctb(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger %B บอกว่าราคาอยู่ตรงไหนของ Bollinger Bands
        0 = อยู่ที่ lower band, 1 = อยู่ที่ upper band
        < 0 หรือ > 1 = ราคาทะลุออกนอก band
        ช่วย agent ตัดสินว่า entry price ดีหรือไม่
        """
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        df['bb_pctb'] = (df['close'] - lower) / (upper - lower + 1e-10)
        return df

    # =========================================================================
    # 📉 ATR Change Ratio — ความผันผวนกำลังขยายหรือหด
    # =========================================================================

    def calculate_atr_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR(14) / SMA(ATR, 50) - 1
        > 0 = volatility กำลังขยาย (ตลาดเริ่มผันผวน)
        < 0 = volatility กำลังหด (ตลาดเริ่มสงบ)
        ช่วย agent หลีกเลี่ยงช่วง volatility spike
        """
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)
        atr_sma = df['atr'].rolling(50).mean()
        df['atr_change_ratio'] = df['atr'] / (atr_sma + 1e-10) - 1.0
        return df

    # =========================================================================
    # 🌪️ Volatility Regime Classifier (v7.1)
    # =========================================================================

    @staticmethod
    def classify_volatility_regime(
        atr_value: float,
        atr_floor_pips: float,
        pip_size: float,
        atr_zscore: float,
    ) -> str:
        """
        v7.1 — จำแนก regime จาก ATR pips + z-score เทียบ 30-bar mean.

        ใช้ใน SMC pre-filter:
          - 'high' / 'explosive' → block ทุกสัญญาณ (เกินกว่า SL จะรับมือไหว)
          - 'normal' → trade ปกติ
          - 'quiet' → ต้องมี session multiplier ≥ 1.0 (off-overlap quiet = noise)

        Args:
            atr_value: ATR value (price units)
            atr_floor_pips: per-symbol floor (e.g., 8 pips FX, 100 ticks XAU)
            pip_size: 0.0001 / 0.01
            atr_zscore: ATR vs 30-bar mean z-score (จาก compute_atr_zscore_30bars)

        Returns:
            'quiet' | 'normal' | 'high' | 'explosive'
        """
        if pip_size <= 0:
            return "normal"
        atr_pips = atr_value / pip_size
        # quiet: under floor × 1.2 (ตามเดิม Tier F2)
        if atr_pips < atr_floor_pips * 1.2:
            return "quiet"
        # explosive: z-score > 2.0 = ATR สูงกว่า 30-bar mean ≥ 2 sigma (rare event)
        if atr_zscore > 2.0:
            return "explosive"
        # high: z-score > 1.0 OR atr > 3× floor (gold expansion / news shock)
        if atr_zscore > 1.0 or atr_pips > atr_floor_pips * 3.0:
            return "high"
        return "normal"

    @staticmethod
    def compute_atr_zscore_30bars(df: pd.DataFrame) -> float:
        """
        v7.1 — z-score ของ ATR ปัจจุบันเทียบ 30 bars ล่าสุด.

        > 0 = volatility ขยาย, > 1 = expanding regime, > 2 = explosive event.
        ใช้เป็น input ให้ classify_volatility_regime + GBM feature.
        """
        if "atr" not in df.columns or len(df) < 30:
            return 0.0
        recent = df["atr"].tail(30)
        mean = float(recent.mean())
        std = float(recent.std())
        if std <= 1e-10:
            return 0.0
        return (float(df["atr"].iloc[-1]) - mean) / std

    # =========================================================================
    # 🚀 Price ROC — การเปลี่ยนแปลงราคาล่าสุด (Momentum)
    # =========================================================================

    def calculate_price_roc(self, df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        (Close - Close[N]) / ATR
        บอกว่าราคาเคลื่อนที่ไปไกลแค่ไหนใน N แท่งล่าสุด เทียบกับ ATR
        ค่าบวก = ขึ้น, ค่าลบ = ลง, magnitude = ความรุนแรง
        ช่วย agent เห็น short-term momentum ที่ confluence score ไม่บอก
        """
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)
        price_change = df['close'] - df['close'].shift(lookback)
        df['price_roc'] = price_change / (df['atr'] + 1e-10)
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
        atr_min_pips: Optional[float] = None,
        atr_max_pips: Optional[float] = None,
        pip_size: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        กรองสภาวะตลาดด้วย ATR — เทรดเฉพาะเมื่อ Volatility อยู่ในช่วงที่เหมาะสม

        กฎ (Forex):
        - ATR ต่ำเกินไป (< 5 pips): ตลาดเงียบ → ไม่เทรด
        - ATR สูงเกินไป (> 50 pips): ตลาดผันผวนมาก → ไม่เทรด
        - ATR อยู่ในช่วง: เทรดได้

        กฎ (Metals เช่น XAUUSD):
        - ATR <200 ticks ($2): เงียบเกิน
        - ATR >5000 ticks ($50): news event / risk event → skip
        - ค่ากลาง: ATR ปกติของ Gold ~1500-2500 ticks/day ($15-25)

        Args:
            df: DataFrame ที่ต้องมี ATR
            atr_min_pips: ATR ขั้นต่ำ (None = auto by instrument)
            atr_max_pips: ATR สูงสุด (None = auto by instrument)
            pip_size: ขนาด 1 pip (None = autodetect จากระดับราคา)
                      - Major pairs (EURUSD, GBPUSD): 0.0001
                      - JPY pairs (USDJPY, EURJPY): 0.01
                      - Metals (XAUUSD): 0.01

        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มคอลัมน์ 'volatility_ok'
        """
        # ตรวจสอบว่ามี ATR หรือยัง
        if 'atr' not in df.columns:
            df = self.calculate_atr(df)

        # === Detect instrument type จากราคา ===
        # Forex major  : ~0.5-2.0  (EURUSD 1.08)
        # Forex JPY    : ~100-200  (USDJPY 150)
        # Metals (Gold): ~1500-4000 (XAUUSD 3000)
        try:
            ref_price = float(df['close'].iloc[-1])
        except Exception:
            ref_price = 1.0

        is_metal = ref_price > 500   # Gold/Silver/Palladium range
        is_jpy = 20 < ref_price <= 500

        # === Autodetect pip_size ถ้าไม่ได้ระบุ ===
        if pip_size is None:
            if is_metal or is_jpy:
                pip_size = 0.01
            else:
                pip_size = 0.0001  # Major pairs

        # === Auto thresholds by instrument type (backward compat) ===
        # Forex defaults เดิม 5-50 pips ไม่เปลี่ยน
        # Metals ใช้ threshold กว้างกว่าเพราะ ATR ใน ticks (0.01) มีค่าหลักพัน
        if atr_min_pips is None:
            atr_min_pips = 200.0 if is_metal else 5.0
        if atr_max_pips is None:
            atr_max_pips = 5000.0 if is_metal else 50.0

        # แปลง ATR (ราคา) → pips
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

        if 'adx' in df.columns and 'price_roc' in df.columns:
            return df

        # คำนวณทุก Indicator
        df = self.calculate_atr(df)
        df = self.calculate_all_emas(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        df = self.calculate_adx(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_bollinger_pctb(df)
        df = self.calculate_atr_change(df)
        df = self.calculate_price_roc(df)
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
            "macd_histogram": last.get("macd_histogram", 0),
            "adx": last.get("adx", 0),
            "stoch_k": last.get("stoch_k", 50),
            "bb_pctb": last.get("bb_pctb", 0.5),
            "atr_change_ratio": last.get("atr_change_ratio", 0),
            "price_roc": last.get("price_roc", 0),
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
