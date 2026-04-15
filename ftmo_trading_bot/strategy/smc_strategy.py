"""
===============================================================================
FTMO Trading Bot — SMC Strategy Engine (เครื่องยนต์กลยุทธ์ Smart Money Concepts)
===============================================================================
โมดูลหลักที่รวม Indicators, Market Structure, และ Order Blocks เข้าด้วยกัน
เพื่อสร้างสัญญาณเทรดที่มีความน่าเชื่อถือสูง

กลไกหลัก:
1. วิเคราะห์แนวโน้มจาก Higher Timeframe (H4) → กำหนดทิศทาง
2. วิเคราะห์โครงสร้างจาก Structure Timeframe (H1) → ยืนยัน BOS/CHoCH
3. หาจุด Entry จาก Entry Timeframe (M15) → Order Block + Confluence
4. คำนวณ Confluence Score → ยิ่งสูง = ยิ่งน่าเทรด
5. ส่ง Signal เฉพาะที่ผ่านเกณฑ์ขั้นต่ำ

Confluence Scoring (คะแนนความเชื่อมั่น):
- เทรดตามทิศ EMA 200:         +25 คะแนน
- BOS ยืนยันแนวโน้ม:          +20 คะแนน
- ราคาอยู่ที่ Order Block:     +25 คะแนน
- RSI ไม่ Overbought/Oversold: +10 คะแนน
- Volatility อยู่ในช่วง:      +10 คะแนน
- HTF (H4) สนับสนุน:          +10 คะแนน
                    รวม:       100 คะแนน

ต้องได้อย่างน้อย 60 คะแนนจึงจะเทรด
===============================================================================
"""

from datetime import datetime, time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import pytz

from config.settings import bot_config
from config.news_events import is_near_high_impact_news
from core.mt5_connector import MT5Connector
from core.time_manager import TimeManager
from strategy.indicators import TechnicalIndicators
from strategy.market_structure import MarketStructure, StructureType
from strategy.order_blocks import OrderBlockDetector, OBType


class SignalType(Enum):
    """ประเภทสัญญาณเทรด"""
    BUY = "BUY"
    SELL = "SELL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradeSignal:
    """
    สัญญาณเทรดที่สร้างจากกลยุทธ์
    
    ประกอบด้วยข้อมูลทั้งหมดที่ Trade Executor ต้องการ
    """
    signal_type: SignalType         # ประเภท: BUY, SELL, NO_SIGNAL
    symbol: str                     # คู่เงิน
    entry_price: float              # ราคา Entry
    sl_price: float                 # ราคา Stop Loss
    tp_price: float                 # ราคา Take Profit
    sl_distance: float              # ระยะ SL (ราคา)
    tp_distance: float              # ระยะ TP (ราคา)
    rr_ratio: float                 # Risk:Reward Ratio
    confluence_score: float         # คะแนน Confluence (0-100)
    atr_value: float                # ค่า ATR ปัจจุบัน
    timestamp: datetime             # เวลาที่สร้างสัญญาณ
    
    # รายละเอียดเหตุผล
    reasons: List[str]              # รายการเหตุผลที่สนับสนุน
    
    # ข้อมูล Order Block (ถ้ามี)
    ob_high: Optional[float] = None  # ขอบบน OB
    ob_low: Optional[float] = None   # ขอบล่าง OB
    ob_score: float = 0.0            # คะแนน OB
    
    # ข้อมูลโครงสร้างตลาด
    market_bias: int = 0             # 1=Bullish, -1=Bearish, 0=Neutral
    trend: int = 0                   # 1=Uptrend, -1=Downtrend, 0=Ranging
    
    @property
    def is_valid(self) -> bool:
        """สัญญาณถูกต้องหรือไม่ (ต้องไม่ใช่ NO_SIGNAL)"""
        return self.signal_type != SignalType.NO_SIGNAL
    
    def to_dict(self) -> Dict:
        """แปลงเป็น Dictionary สำหรับ Logging"""
        return {
            "signal": self.signal_type.value,
            "symbol": self.symbol,
            "entry": self.entry_price,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "rr_ratio": self.rr_ratio,
            "confluence": self.confluence_score,
            "atr": self.atr_value,
            "timestamp": str(self.timestamp),
            "reasons": "; ".join(self.reasons),
        }


class SMCStrategy:
    """
    กลยุทธ์ Smart Money Concepts (SMC) หลักของ Bot
    
    ขั้นตอนการวิเคราะห์:
    1. [HTF] H4: กำหนดทิศทางหลัก (EMA + Trend)
    2. [MTF] H1: ยืนยันโครงสร้าง (BOS/CHoCH + Market Bias)
    3. [LTF] M15: หาจุด Entry (Order Block + ATR SL/TP)
    4. คำนวณ Confluence Score
    5. ส่ง Signal ถ้าผ่านเกณฑ์
    
    ⚠️ Bot จะเทรดเฉพาะเมื่อ:
    - Confluence Score >= 70 (ยกเกณฑ์จาก 60 เพื่อกรอง over-trading)
    - Risk:Reward >= 1:1.5
    - Volatility อยู่ในช่วง
    - Trading Session ถูกต้อง
    """

    # คะแนน Confluence ขั้นต่ำสำหรับเปิดเทรด (อ่านจาก bot_config.ftmo.MIN_CONFLUENCE_SCORE)
    # ดีฟอลต์ 70 — RL Agent อาจ override ที่ runtime
    MIN_CONFLUENCE_SCORE = float(bot_config.ftmo.MIN_CONFLUENCE_SCORE)

    def __init__(self, connector: MT5Connector):
        """
        เริ่มต้น SMC Strategy
        
        Args:
            connector: ตัวเชื่อมต่อ MT5 สำหรับดึงข้อมูลราคา
        """
        self._connector = connector
        self._indicators = TechnicalIndicators()
        
        # สร้าง Market Structure และ OB Detector สำหรับแต่ละ Timeframe
        self._structure_mtf = MarketStructure()     # โครงสร้าง H1
        self._structure_ltf = MarketStructure()     # โครงสร้าง M15
        self._ob_detector = OrderBlockDetector()    # Order Blocks M15
        
        # Cache ข้อมูลที่วิเคราะห์แล้ว
        self._htf_data: Optional[pd.DataFrame] = None   # H4 data
        self._mtf_data: Optional[pd.DataFrame] = None   # H1 data
        self._ltf_data: Optional[pd.DataFrame] = None   # M15 data
        
        # HTF Bias
        self._htf_bias: int = 0  # 1=Bullish, -1=Bearish, 0=Neutral
        
        print("🎯 [SMC Strategy] เริ่มต้นกลยุทธ์ Smart Money Concepts")

    # =========================================================================
    # 🔄 วิเคราะห์สัญญาณ (Main Analysis)
    # =========================================================================

    def analyze(self, symbol: str) -> TradeSignal:
        """
        วิเคราะห์สัญญาณเทรดสำหรับคู่เงินที่ระบุ
        
        ขั้นตอน:
        1. ดึงข้อมูลราคาทุก Timeframe
        2. คำนวณ Indicators
        3. วิเคราะห์ Market Structure
        4. ตรวจหา Order Blocks
        5. คำนวณ Confluence Score
        6. ตรวจสอบ Trading Session
        7. สร้าง Signal
        
        Args:
            symbol: คู่เงิน เช่น "EURUSD"
            
        Returns:
            TradeSignal: สัญญาณเทรด (อาจเป็น NO_SIGNAL ถ้าไม่มีสัญญาณ)
        """
        no_signal = TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol, entry_price=0, sl_price=0, tp_price=0,
            sl_distance=0, tp_distance=0, rr_ratio=0,
            confluence_score=0, atr_value=0,
            timestamp=datetime.now(), reasons=["ไม่มีสัญญาณ"]
        )

        # === ขั้นตอนที่ 1: ตรวจสอบ Trading Session ===
        if not self._is_trading_session():
            return no_signal

        # === ขั้นตอนที่ 1.5: ตรวจสอบ High-Impact News ===
        now_utc = TimeManager.get_server_time().astimezone(pytz.UTC)
        window_before = getattr(bot_config.sessions, "no_trade_before_news_minutes", 30)
        window_after = getattr(bot_config.sessions, "no_trade_after_news_minutes", 30)
        is_news, news_reason = is_near_high_impact_news(
            symbol, now_utc,
            window_minutes_before=window_before,
            window_minutes_after=window_after,
        )
        if is_news:
            if bot_config.debug_mode:
                print(f"📰 [SMC] {symbol} — {news_reason}")
            return no_signal

        # === ขั้นตอนที่ 2: ตรวจสอบ Spread ===
        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            return no_signal
            
        max_spread = bot_config.symbols.max_spread_points.get(symbol, 20)
        symbol_info = self._connector.get_symbol_info(symbol)
        if symbol_info:
            spread_points = price_info["spread"] / symbol_info["point"]
            if spread_points > max_spread:
                return no_signal

        # === ขั้นตอนที่ 3: ดึงข้อมูลราคาทุก Timeframe ===
        htf_df = self._connector.get_ohlcv(symbol, bot_config.symbols.higher_timeframe, 300)      # H4
        mtf_df = self._connector.get_ohlcv(symbol, bot_config.symbols.structure_timeframe, 500)    # H1
        ltf_df = self._connector.get_ohlcv(symbol, bot_config.symbols.primary_timeframe, 500)      # M15

        if htf_df is None or mtf_df is None or ltf_df is None:
            return no_signal
        if len(htf_df) < 200 or len(mtf_df) < 200 or len(ltf_df) < 200:
            return no_signal

        # === ขั้นตอนที่ 4: คำนวณ Indicators ทุก Timeframe ===
        htf_df = self._indicators.calculate_all(htf_df)
        mtf_df = self._indicators.calculate_all(mtf_df)
        ltf_df = self._indicators.calculate_all(ltf_df)

        # เก็บ Cache
        self._htf_data = htf_df
        self._mtf_data = mtf_df
        self._ltf_data = ltf_df

        # === ขั้นตอนที่ 5: วิเคราะห์ HTF (H4) — ทิศทางหลัก ===
        htf_values = self._indicators.get_latest_values(htf_df)
        self._htf_bias = htf_values["trend"] if htf_values else 0

        # === ขั้นตอนที่ 6: วิเคราะห์ MTF (H1) — Market Structure ===
        mtf_df = self._structure_mtf.analyze(mtf_df)
        mtf_bias = self._structure_mtf.get_current_bias()

        # === ขั้นตอนที่ 7: วิเคราะห์ LTF (M15) — Entry ===
        ltf_df = self._structure_ltf.analyze(ltf_df)
        ltf_df = self._ob_detector.analyze(ltf_df)

        # ดึงค่า Indicator ล่าสุดของ M15
        ltf_values = self._indicators.get_latest_values(ltf_df)
        if ltf_values is None:
            return no_signal

        current_price = ltf_values["close"]
        atr_value = ltf_values["atr"]
        rsi_value = ltf_values["rsi"]
        ltf_trend = ltf_values["trend"]
        volatility_ok = ltf_values["volatility_ok"]

        # === ขั้นตอนที่ 8: ค้นหาสัญญาณ BUY ===
        buy_signal = self._evaluate_buy_signal(
            symbol, current_price, atr_value, rsi_value,
            ltf_trend, mtf_bias, volatility_ok, ltf_df, price_info
        )
        
        # === ขั้นตอนที่ 9: ค้นหาสัญญาณ SELL ===
        sell_signal = self._evaluate_sell_signal(
            symbol, current_price, atr_value, rsi_value,
            ltf_trend, mtf_bias, volatility_ok, ltf_df, price_info
        )

        # === ขั้นตอนที่ 10: เลือกสัญญาณที่ดีที่สุด ===
        if buy_signal.is_valid and sell_signal.is_valid:
            # ถ้ามีทั้ง BUY และ SELL → เลือกที่มี Confluence สูงกว่า
            return buy_signal if buy_signal.confluence_score >= sell_signal.confluence_score else sell_signal
        elif buy_signal.is_valid:
            return buy_signal
        elif sell_signal.is_valid:
            return sell_signal
        
        return no_signal

    # =========================================================================
    # 📈 BUY Signal Evaluation
    # =========================================================================

    def _evaluate_buy_signal(
        self,
        symbol: str,
        current_price: float,
        atr_value: float,
        rsi_value: float,
        ltf_trend: int,
        mtf_bias: int,
        volatility_ok: bool,
        ltf_df: pd.DataFrame,
        price_info: Dict
    ) -> TradeSignal:
        """
        ประเมินสัญญาณ BUY
        
        เงื่อนไข BUY (Confluence Scoring):
        1. HTF (H4) เป็นขาขึ้น → +25
        2. MTF (H1) มี BOS Bullish → +20
        3. ราคาอยู่ที่ Bullish Order Block → +25
        4. RSI ไม่ Overbought (< 70) → +10
        5. Volatility OK → +10
        6. LTF trend เป็นขาขึ้น → +10
        
        ต้องได้ >= 60 จึงจะส่ง Signal
        """
        no_signal = TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol, entry_price=0, sl_price=0, tp_price=0,
            sl_distance=0, tp_distance=0, rr_ratio=0,
            confluence_score=0, atr_value=atr_value,
            timestamp=datetime.now(), reasons=["ไม่ผ่านเกณฑ์ BUY"]
        )

        score = 0.0
        reasons = []

        # === ปัจจัยที่ 1: HTF Trend (25 คะแนน) ===
        if self._htf_bias == 1:
            score += 25
            reasons.append("✅ HTF (H4) ขาขึ้น")
        elif self._htf_bias == 0:
            score += 10  # Neutral ได้บางส่วน
            reasons.append("⚠️ HTF (H4) Neutral")
        else:
            reasons.append("❌ HTF (H4) ขาลง — ขัดกับ BUY")
            # ไม่ return ทันที เพราะ CHoCH อาจเป็นสัญญาณกลับตัว

        # === ปัจจัยที่ 2: MTF Market Bias (20 คะแนน) ===
        if mtf_bias == 1:
            score += 20
            reasons.append("✅ MTF (H1) Bullish Bias")
            
            # เช็คว่ามี BOS ล่าสุดหรือไม่
            latest_event = self._structure_mtf.get_latest_event()
            if latest_event and latest_event.event_type == StructureType.BOS_BULLISH:
                score += 5  # Bonus สำหรับ Fresh BOS
                reasons.append("✅ Fresh BOS Bullish")
        elif mtf_bias == 0:
            score += 5
            reasons.append("⚠️ MTF (H1) Neutral")

        # === ปัจจัยที่ 3: Order Block (25 คะแนน) ===
        # ดึง Pip Size จาก Symbol Info
        symbol_info = self._connector.get_symbol_info(symbol)
        pip_size = 0.0001 if symbol_info and symbol_info["digits"] >= 4 else 0.01
        
        bullish_ob = self._ob_detector.is_price_at_bullish_ob(
            current_price, tolerance_pips=5, pip_size=pip_size
        )
        
        if bullish_ob:
            ob_contribution = min(25, bullish_ob.strength_score * 0.5)
            score += ob_contribution
            reasons.append(f"✅ ราคาอยู่ที่ Bullish OB (score={bullish_ob.strength_score:.0f})")
        else:
            # ยังไม่มี OB → ลดสิทธิ์ แต่ไม่ตัดทั้งหมด
            reasons.append("⚠️ ไม่มี Bullish OB ใกล้ราคา")

        # === ปัจจัยที่ 4: RSI (10 คะแนน) ===
        if rsi_value < bot_config.indicators.rsi_overbought:  # < 70
            score += 10
            reasons.append(f"✅ RSI={rsi_value:.1f} (ไม่ Overbought)")
        else:
            reasons.append(f"❌ RSI={rsi_value:.1f} (Overbought — อันตราย)")

        # === ปัจจัยที่ 5: Volatility (10 คะแนน) ===
        if volatility_ok:
            score += 10
            reasons.append("✅ Volatility อยู่ในช่วง")
        else:
            reasons.append("❌ Volatility ผิดปกติ")

        # === ปัจจัยที่ 6: LTF Trend (10 คะแนน) ===
        if ltf_trend == 1:
            score += 10
            reasons.append("✅ LTF (M15) ขาขึ้น")
        elif ltf_trend == 0:
            score += 3
            reasons.append("⚠️ LTF (M15) Ranging")

        # === ตรวจสอบว่าผ่านเกณฑ์หรือไม่ ===
        if score < self.MIN_CONFLUENCE_SCORE:
            no_signal.confluence_score = score
            no_signal.reasons = reasons
            return no_signal

        # === คำนวณ Entry, SL, TP ===
        entry_price = price_info["ask"]  # BUY ที่ราคา Ask
        
        # SL: ใช้ ATR × Multiplier หรือ ใต้ Order Block
        sl_distance = atr_value * bot_config.indicators.atr_sl_multiplier
        
        # ถ้ามี OB → วาง SL ใต้ OB (ถ้าระยะไม่ไกลเกินไป)
        if bullish_ob:
            ob_sl_distance = entry_price - bullish_ob.low + (2 * pip_size)  # ใต้ OB + 2 pips
            if 0 < ob_sl_distance < sl_distance * 2:  # ไม่ไกลเกิน 2 เท่าของ ATR SL
                sl_distance = ob_sl_distance
        
        sl_price = entry_price - sl_distance
        
        # TP: ใช้ RR ที่ต้องการ
        tp_distance = sl_distance * bot_config.ftmo.PREFERRED_RISK_REWARD_RATIO  # SL × 2
        tp_price = entry_price + tp_distance
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # ตรวจ RR ขั้นต่ำ
        if rr_ratio < bot_config.ftmo.MIN_RISK_REWARD_RATIO:
            no_signal.reasons = reasons + ["❌ RR ต่ำเกินไป"]
            return no_signal

        # ปัดราคา
        digits = symbol_info["digits"] if symbol_info else 5
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        signal = TradeSignal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            entry_price=round(entry_price, digits),
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            rr_ratio=rr_ratio,
            confluence_score=score,
            atr_value=atr_value,
            timestamp=datetime.now(),
            reasons=reasons,
            ob_high=bullish_ob.high if bullish_ob else None,
            ob_low=bullish_ob.low if bullish_ob else None,
            ob_score=bullish_ob.strength_score if bullish_ob else 0,
            market_bias=mtf_bias,
            trend=ltf_trend,
        )

        if bot_config.debug_mode:
            print(f"\n🟢 [SMC] สัญญาณ BUY {symbol} (Confluence: {score:.0f}/100)")
            print(f"   📍 Entry: {entry_price:.{digits}f}")
            print(f"   🔴 SL: {sl_price:.{digits}f} ({sl_distance:.{digits}f})")
            print(f"   🟢 TP: {tp_price:.{digits}f} ({tp_distance:.{digits}f})")
            print(f"   ⚖️ RR: 1:{rr_ratio:.1f}")
            for r in reasons:
                print(f"   {r}")

        return signal

    # =========================================================================
    # 📉 SELL Signal Evaluation
    # =========================================================================

    def _evaluate_sell_signal(
        self,
        symbol: str,
        current_price: float,
        atr_value: float,
        rsi_value: float,
        ltf_trend: int,
        mtf_bias: int,
        volatility_ok: bool,
        ltf_df: pd.DataFrame,
        price_info: Dict
    ) -> TradeSignal:
        """
        ประเมินสัญญาณ SELL (กระจกของ BUY)
        
        เงื่อนไข SELL (Confluence Scoring):
        1. HTF (H4) เป็นขาลง → +25
        2. MTF (H1) มี BOS Bearish → +20
        3. ราคาอยู่ที่ Bearish Order Block → +25
        4. RSI ไม่ Oversold (> 30) → +10
        5. Volatility OK → +10
        6. LTF trend เป็นขาลง → +10
        """
        no_signal = TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol, entry_price=0, sl_price=0, tp_price=0,
            sl_distance=0, tp_distance=0, rr_ratio=0,
            confluence_score=0, atr_value=atr_value,
            timestamp=datetime.now(), reasons=["ไม่ผ่านเกณฑ์ SELL"]
        )

        score = 0.0
        reasons = []

        # === ปัจจัยที่ 1: HTF Trend (25 คะแนน) ===
        if self._htf_bias == -1:
            score += 25
            reasons.append("✅ HTF (H4) ขาลง")
        elif self._htf_bias == 0:
            score += 10
            reasons.append("⚠️ HTF (H4) Neutral")
        else:
            reasons.append("❌ HTF (H4) ขาขึ้น — ขัดกับ SELL")

        # === ปัจจัยที่ 2: MTF Market Bias (20 คะแนน) ===
        if mtf_bias == -1:
            score += 20
            reasons.append("✅ MTF (H1) Bearish Bias")
            
            latest_event = self._structure_mtf.get_latest_event()
            if latest_event and latest_event.event_type == StructureType.BOS_BEARISH:
                score += 5
                reasons.append("✅ Fresh BOS Bearish")
        elif mtf_bias == 0:
            score += 5
            reasons.append("⚠️ MTF (H1) Neutral")

        # === ปัจจัยที่ 3: Order Block (25 คะแนน) ===
        symbol_info = self._connector.get_symbol_info(symbol)
        pip_size = 0.0001 if symbol_info and symbol_info["digits"] >= 4 else 0.01
        
        bearish_ob = self._ob_detector.is_price_at_bearish_ob(
            current_price, tolerance_pips=5, pip_size=pip_size
        )
        
        if bearish_ob:
            ob_contribution = min(25, bearish_ob.strength_score * 0.5)
            score += ob_contribution
            reasons.append(f"✅ ราคาอยู่ที่ Bearish OB (score={bearish_ob.strength_score:.0f})")
        else:
            reasons.append("⚠️ ไม่มี Bearish OB ใกล้ราคา")

        # === ปัจจัยที่ 4: RSI (10 คะแนน) ===
        if rsi_value > bot_config.indicators.rsi_oversold:  # > 30
            score += 10
            reasons.append(f"✅ RSI={rsi_value:.1f} (ไม่ Oversold)")
        else:
            reasons.append(f"❌ RSI={rsi_value:.1f} (Oversold — อันตราย)")

        # === ปัจจัยที่ 5: Volatility (10 คะแนน) ===
        if volatility_ok:
            score += 10
            reasons.append("✅ Volatility อยู่ในช่วง")
        else:
            reasons.append("❌ Volatility ผิดปกติ")

        # === ปัจจัยที่ 6: LTF Trend (10 คะแนน) ===
        if ltf_trend == -1:
            score += 10
            reasons.append("✅ LTF (M15) ขาลง")
        elif ltf_trend == 0:
            score += 3
            reasons.append("⚠️ LTF (M15) Ranging")

        # === ตรวจสอบเกณฑ์ ===
        if score < self.MIN_CONFLUENCE_SCORE:
            no_signal.confluence_score = score
            no_signal.reasons = reasons
            return no_signal

        # === คำนวณ Entry, SL, TP ===
        entry_price = price_info["bid"]  # SELL ที่ราคา Bid
        
        sl_distance = atr_value * bot_config.indicators.atr_sl_multiplier
        
        if bearish_ob:
            ob_sl_distance = bearish_ob.high - entry_price + (2 * pip_size)
            if 0 < ob_sl_distance < sl_distance * 2:
                sl_distance = ob_sl_distance
        
        sl_price = entry_price + sl_distance
        tp_distance = sl_distance * bot_config.ftmo.PREFERRED_RISK_REWARD_RATIO
        tp_price = entry_price - tp_distance
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        if rr_ratio < bot_config.ftmo.MIN_RISK_REWARD_RATIO:
            no_signal.reasons = reasons + ["❌ RR ต่ำเกินไป"]
            return no_signal

        digits = symbol_info["digits"] if symbol_info else 5
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        signal = TradeSignal(
            signal_type=SignalType.SELL,
            symbol=symbol,
            entry_price=round(entry_price, digits),
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            rr_ratio=rr_ratio,
            confluence_score=score,
            atr_value=atr_value,
            timestamp=TimeManager.get_server_time(symbol),
            reasons=reasons,
            ob_high=bearish_ob.high if bearish_ob else None,
            ob_low=bearish_ob.low if bearish_ob else None,
            ob_score=bearish_ob.strength_score if bearish_ob else 0,
            market_bias=mtf_bias,
            trend=ltf_trend,
        )

        if bot_config.debug_mode:
            print(f"\n🔴 [SMC] สัญญาณ SELL {symbol} (Confluence: {score:.0f}/100)")
            print(f"   📍 Entry: {entry_price:.{digits}f}")
            print(f"   🔴 SL: {sl_price:.{digits}f} ({sl_distance:.{digits}f})")
            print(f"   🟢 TP: {tp_price:.{digits}f} ({tp_distance:.{digits}f})")
            print(f"   ⚖️ RR: 1:{rr_ratio:.1f}")
            for r in reasons:
                print(f"   {r}")

        return signal

    # =========================================================================
    # ⏰ Trading Session Check
    # =========================================================================

    def _is_trading_session(self) -> bool:
        """
        ตรวจสอบว่าอยู่ในช่วงเวลาเทรดที่อนุญาตหรือไม่
        
        เทรดเฉพาะช่วง:
        - London: 07:00-12:00 UTC
        - London-NY Overlap: 12:00-17:00 UTC
        
        ไม่เทรด:
        - Asian Session
        - วันศุกร์หลัง 15:00 UTC
        - วันเสาร์-อาทิตย์
        
        Returns:
            bool: True ถ้าอยู่ในช่วงเทรด
        """
        now = TimeManager.get_server_time()  # ใช้เวลาจริงของโบรกเกอร์ (EET) แบบเรียลไทม์
        
        # แปลงเวลาให้เป็น UTC ก่อนนำไปเทียบกับ Config
        now_utc = now.astimezone(pytz.UTC)
        current_time = now_utc.time()
        current_weekday = now_utc.weekday()  # 0=จันทร์, 6=อาทิตย์
        
        session_config = bot_config.sessions
        
        # ตรวจสอบวันเทรด
        if current_weekday not in session_config.trading_days:
            return False
        
        # ตรวจวันศุกร์ — หยุดหลัง 15:00 UTC
        if current_weekday == 4 and current_time >= session_config.friday_cutoff:
            return False
        
        # ตรวจช่วง London Session
        in_london = session_config.london_start <= current_time <= session_config.london_end
        
        # ตรวจช่วง New York Session
        in_newyork = session_config.newyork_start <= current_time <= session_config.newyork_end
        
        return in_london or in_newyork

    # =========================================================================
    # 🔀 Scan ทุกคู่เงิน
    # =========================================================================

    def scan_all_symbols(self) -> List[TradeSignal]:
        """
        สแกนทุกคู่เงินที่ตั้งค่าไว้และคืนสัญญาณที่ Valid ทั้งหมด
        
        Returns:
            List[TradeSignal]: รายการสัญญาณที่ผ่านเกณฑ์ (เรียงตาม Confluence สูง→ต่ำ)
        """
        signals = []
        
        for symbol in bot_config.symbols.symbols:
            try:
                signal = self.analyze(symbol)
                if signal.is_valid:
                    signals.append(signal)
                    print(f"📡 [SMC] พบสัญญาณ {signal.signal_type.value} {symbol} "
                          f"(Confluence: {signal.confluence_score:.0f})")
            except Exception as e:
                print(f"⚠️ [SMC] วิเคราะห์ {symbol} ล้มเหลว: {e}")
                continue

        # เรียงลำดับตาม Confluence Score (สูง → ต่ำ)
        signals.sort(key=lambda s: s.confluence_score, reverse=True)
        
        if not signals:
            print("ℹ️ [SMC] ไม่พบสัญญาณที่ผ่านเกณฑ์")
        
        return signals

    # =========================================================================
    # 📊 ข้อมูลสรุป
    # =========================================================================

    def get_analysis_summary(self, symbol: str) -> Dict:
        """
        สรุปผลการวิเคราะห์ทั้งหมดสำหรับ Symbol ที่ระบุ
        
        Args:
            symbol: คู่เงิน
            
        Returns:
            Dict: ข้อมูลสรุปการวิเคราะห์
        """
        return {
            "symbol": symbol,
            "htf_bias": self._htf_bias,
            "mtf_structure": self._structure_mtf.get_structure_summary(),
            "ltf_structure": self._structure_ltf.get_structure_summary(),
            "order_blocks": self._ob_detector.get_ob_summary(),
            "is_trading_session": self._is_trading_session(),
        }

    def __repr__(self) -> str:
        return f"SMCStrategy(min_confluence={self.MIN_CONFLUENCE_SCORE}, htf_bias={self._htf_bias})"
