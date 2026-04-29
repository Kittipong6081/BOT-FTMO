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

from config.settings import bot_config, get_symbol_config
from config.news_events import is_near_high_impact_news
from core.mt5_connector import MT5Connector
from core.time_manager import TimeManager
from strategy.indicators import TechnicalIndicators
from strategy.market_structure import MarketStructure, StructureType
from strategy.order_blocks import OrderBlockDetector, OBType
from strategy.fair_value_gaps import FVGDetector
from strategy.liquidity_sweeps import LiquiditySweepDetector
from strategy.inducement import InducementDetector


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

    # Indicator values สำหรับ RL Agent observation
    rsi_value: float = 50.0
    trend_strength: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    stoch_k: float = 50.0
    bb_pctb: float = 0.5
    atr_change_ratio: float = 0.0
    price_roc: float = 0.0

    # v6.11 (Tier 2.4) Per-component confluence breakdown (สำหรับ Trades sheet logging)
    # ค่าเหล่านี้คือ contribution ของแต่ละหมวด (หลัง session multiplier ยังไม่ apply)
    htf_score: int = 0          # HTF (H4) trend pts
    mtf_score: int = 0          # MTF (H1) bias pts
    ob_pts: int = 0             # Order Block pts
    fvg_pts: int = 0            # Fair Value Gap pts
    sweep_pts: int = 0          # Liquidity Sweep pts
    sweep_age_bars: int = -1    # อายุของ sweep ที่ใช้ (-1 = ไม่มี)
    htf_bias: str = ""          # "BULLISH" / "BEARISH" / "RANGING" — string สำหรับ Excel
    d1_bias: int = 0            # Daily bias snapshot (-1/0/+1)
    
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
        self._fvg_detector = FVGDetector()          # Fair Value Gaps M15
        self._sweep_detector = LiquiditySweepDetector()  # Liquidity Sweeps M15
        self._idm_detector = InducementDetector(lookback=8)  # v6.11 Tier 3.1 — IDM rejection candles M15

        # Cache ข้อมูลที่วิเคราะห์แล้ว
        self._htf_data: Optional[pd.DataFrame] = None   # H4 data
        self._mtf_data: Optional[pd.DataFrame] = None   # H1 data
        self._ltf_data: Optional[pd.DataFrame] = None   # M15 data
        
        # HTF Bias
        self._htf_bias: int = 0  # 1=Bullish, -1=Bearish, 0=Neutral

        # D1 bias cache (symbol -> {"bias": int, "time": float timestamp})
        # D1 เปลี่ยนช้า → cache ได้ 1 ชั่วโมง เพื่อประหยัด MT5 calls
        self._d1_bias_cache: Dict[str, Dict] = {}

        print("🎯 [SMC Strategy] เริ่มต้นกลยุทธ์ Smart Money Concepts")

    # =========================================================================
    # 🗓️ D1 Bias Helper (Daily Trend Overlay)
    # =========================================================================

    def _get_d1_bias(self, symbol: str) -> int:
        """
        คำนวณ D1 bias จากราคาเทียบ EMA50 บน Daily chart

        ใช้ buffer 0.5% กัน noise → neutral ถ้าราคาใกล้ EMA50 มาก
        Cache 1 ชม. + invalidate เมื่อข้าม UTC day (D1 close ใหม่)

        Returns:
            +1 bullish / -1 bearish / 0 neutral (หรือดึงข้อมูลไม่ได้)
        """
        import time as _t
        now_ts = _t.time()
        now_date = TimeManager.get_server_time().astimezone(pytz.UTC).date()
        cached = self._d1_bias_cache.get(symbol)
        if (
            cached
            and (now_ts - cached.get("time", 0)) < 3600
            and cached.get("date") == now_date
        ):
            return int(cached.get("bias", 0))

        try:
            d1_df = self._connector.get_ohlcv(symbol, "D1", 80)
            if d1_df is None or len(d1_df) < 50:
                return 0
            ema50 = d1_df["close"].ewm(span=50, adjust=False).mean()
            current_close = float(d1_df["close"].iloc[-1])
            ema50_last = float(ema50.iloc[-1])
            if ema50_last <= 0:
                return 0
            if current_close > ema50_last * 1.005:
                bias = 1
            elif current_close < ema50_last * 0.995:
                bias = -1
            else:
                bias = 0
            self._d1_bias_cache[symbol] = {"bias": bias, "time": now_ts, "date": now_date}
            return bias
        except Exception:
            return 0

    # =========================================================================
    # 🔄 วิเคราะห์สัญญาณ (Main Analysis)
    # =========================================================================

    def analyze_with_data(
        self, symbol: str, htf_df, mtf_df, ltf_df, price_info: dict
    ) -> 'TradeSignal':
        """วิเคราะห์สัญญาณจาก DataFrames ที่ส่งมา (สำหรับ Backtesting — ไม่ต้องต่อ MT5)"""
        no_signal = TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol, entry_price=0, sl_price=0, tp_price=0,
            sl_distance=0, tp_distance=0, rr_ratio=0,
            confluence_score=0, atr_value=0,
            timestamp=datetime.now(), reasons=["ไม่มีสัญญาณ"]
        )

        if htf_df is None or mtf_df is None or ltf_df is None:
            return no_signal
        if len(htf_df) < 200 or len(mtf_df) < 200 or len(ltf_df) < 200:
            return no_signal

        htf_df = self._indicators.calculate_all(htf_df)
        mtf_df = self._indicators.calculate_all(mtf_df)
        ltf_df = self._indicators.calculate_all(ltf_df)

        self._htf_data = htf_df
        self._mtf_data = mtf_df
        self._ltf_data = ltf_df

        # HTF bias — ใช้ 5 closed bars (exclude last forming bar) → anti-lookahead
        # (ของเก่า iloc[-5:] รวม bar ปัจจุบันที่ยังไม่ close → backtest เห็นอนาคต)
        if htf_df is not None and len(htf_df) >= 6:
            recent_trends = htf_df["trend"].iloc[-6:-1].tolist()
            bullish_count = sum(1 for t in recent_trends if t == 1)
            bearish_count = sum(1 for t in recent_trends if t == -1)
            # 3/5 threshold + ห้าม mixed strong (ถ้า opposite >=2 → neutral)
            if bullish_count >= 3 and bearish_count < 2:
                self._htf_bias = 1
            elif bearish_count >= 3 and bullish_count < 2:
                self._htf_bias = -1
            else:
                self._htf_bias = 0
        else:
            htf_values = self._indicators.get_latest_values(htf_df)
            self._htf_bias = htf_values["trend"] if htf_values else 0

        mtf_df = self._structure_mtf.analyze(mtf_df)
        mtf_bias = self._structure_mtf.get_current_bias()

        ltf_df = self._structure_ltf.analyze(ltf_df)
        ltf_df = self._ob_detector.analyze(ltf_df)
        ltf_df = self._fvg_detector.analyze(ltf_df)
        self._sweep_detector.analyze(ltf_df, self._structure_ltf)

        ltf_values = self._indicators.get_latest_values(ltf_df)
        if ltf_values is None:
            return no_signal

        current_price = ltf_values["close"]
        atr_value = ltf_values["atr"]
        rsi_value = ltf_values["rsi"]
        ltf_trend = ltf_values["trend"]
        volatility_ok = ltf_values["volatility_ok"]

        buy_signal = self._evaluate_buy_signal(
            symbol, current_price, atr_value, rsi_value,
            ltf_trend, mtf_bias, volatility_ok, ltf_df, price_info
        )
        sell_signal = self._evaluate_sell_signal(
            symbol, current_price, atr_value, rsi_value,
            ltf_trend, mtf_bias, volatility_ok, ltf_df, price_info
        )

        if buy_signal.is_valid and sell_signal.is_valid:
            return buy_signal if buy_signal.confluence_score >= sell_signal.confluence_score else sell_signal
        elif buy_signal.is_valid:
            return buy_signal
        elif sell_signal.is_valid:
            return sell_signal
        return no_signal

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

        # === ขั้นตอนที่ 5: วิเคราะห์ HTF (H4) — 5 closed bars (anti-lookahead) ===
        htf_values = self._indicators.get_latest_values(htf_df)
        if htf_values and len(htf_df) >= 6:
            recent_trends = htf_df["trend"].iloc[-6:-1].tolist()
            bullish_count = sum(1 for t in recent_trends if t == 1)
            bearish_count = sum(1 for t in recent_trends if t == -1)
            if bullish_count >= 3 and bearish_count < 2:
                self._htf_bias = 1
            elif bearish_count >= 3 and bullish_count < 2:
                self._htf_bias = -1
            else:
                self._htf_bias = 0
        else:
            self._htf_bias = htf_values["trend"] if htf_values else 0

        # === ขั้นตอนที่ 6: วิเคราะห์ MTF (H1) — Market Structure ===
        mtf_df = self._structure_mtf.analyze(mtf_df)
        mtf_bias = self._structure_mtf.get_current_bias()

        # === ขั้นตอนที่ 7: วิเคราะห์ LTF (M15) — Entry ===
        ltf_df = self._structure_ltf.analyze(ltf_df)
        ltf_df = self._ob_detector.analyze(ltf_df)
        ltf_df = self._fvg_detector.analyze(ltf_df)
        self._sweep_detector.analyze(ltf_df, self._structure_ltf)

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
    # 🎯 Dynamic RR Helper (ตาม ADX Trend Strength)
    # =========================================================================

    @staticmethod
    def _dynamic_rr(adx_value: float) -> float:
        """
        คำนวณ Risk:Reward ratio ตามความแรง trend (ADX)

        ต้องตรงกับ strategy_backtester.py:671-676 เพื่อ align sim/live:
          ADX >= 30: strong trend  → RR 2.5 (let winners run)
          ADX 20-30: moderate      → RR 2.0 (default)
          ADX < 20:  ranging       → RR 1.5 (tighter TP, more reliable hits)

        Note: Pool signals ถูกสร้างด้วย logic นี้อยู่แล้ว → agent trained to
              see varying rr_ratio. Previously strategy output fixed 2.0 →
              live obs[1] distribution ต่างจาก pool → now matched.
        """
        if adx_value >= 30.0:
            return 2.5
        elif adx_value >= 20.0:
            return 2.0
        return 1.5

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

        # === MANDATORY PRE-FILTERS (2026-04-18) ===
        # Data-driven: WR ของ BUY = 26.3% vs SELL = 34.5% — BUY เป็น bias หลักของขาดทุน
        # Filter เข้มเพื่อเอาเฉพาะ setup ที่ backtest พิสูจน์ว่า +EV
        symbol_info = self._connector.get_symbol_info(symbol)
        pip_size = 0.0001 if symbol_info and symbol_info["digits"] >= 4 else 0.01
        atr_pips = atr_value / pip_size if pip_size > 0 else 0

        # A. Volatility floor — gold-aware + per-symbol override (GBPJPY=20, EURJPY=15)
        is_metal = "XAU" in symbol.upper() or "XAG" in symbol.upper()
        default_floor = 100.0 if is_metal else 8.0
        min_atr_pips = float(get_symbol_config(symbol, "atr_floor_pips", default_floor))
        if atr_pips < min_atr_pips:
            no_signal.confluence_score = 0
            no_signal.reasons = [f"❌ [REJECT] ATR {atr_pips:.1f} pips < {min_atr_pips:.0f} (volatility ต่ำ)"]
            return no_signal

        # B. RSI must be in pullback zone for BUY: RSI >= 60 → WR ตก (23%)
        #    ที่ดี: RSI < 30 (WR 37.7%, EV +0.16R), RSI 30-50 (WR 28%)
        if rsi_value >= 60:
            no_signal.confluence_score = 0
            no_signal.reasons = [f"❌ [REJECT] RSI {rsi_value:.1f} >= 60 (BUY ห้ามใกล้ overbought)"]
            return no_signal

        # C. Trend alignment: ต้องมี HTF หรือ MTF ฝั่ง bullish อย่างน้อย 1
        #    counter-trend BUY (HTF+MTF bearish ทั้งคู่) = losing setup
        if self._htf_bias == -1 and mtf_bias == -1:
            no_signal.confluence_score = 0
            no_signal.reasons = ["❌ [REJECT] HTF+MTF bearish ทั้งคู่ — ห้าม BUY counter-trend"]
            return no_signal

        # D. EMA200 alignment (H1) — hard veto counter-trend เชิง intraday
        #    H1 EMA200 = "fair value line" ระยะกลาง → ซื้อใต้เส้น = กินมีด
        if self._mtf_data is not None and "ema_slow" in self._mtf_data.columns:
            ema200_h1 = float(self._mtf_data["ema_slow"].iloc[-1])
            if pd.notna(ema200_h1) and ema200_h1 > 0 and current_price < ema200_h1:
                no_signal.confluence_score = 0
                no_signal.reasons = [
                    f"❌ [REJECT] BUY below H1 EMA200 "
                    f"(price={current_price:.5f}, EMA200={ema200_h1:.5f})"
                ]
                return no_signal

        # E. ADX(H1) ≥ 20 — skip ranging market (whipsaw บ่อย)
        if self._mtf_data is not None and "adx" in self._mtf_data.columns:
            adx_h1 = float(self._mtf_data["adx"].iloc[-1])
            if adx_h1 < 20.0:
                no_signal.confluence_score = 0
                no_signal.reasons = [f"❌ [REJECT] ADX(H1) {adx_h1:.1f} < 20 — ranging market"]
                return no_signal

        # E2. v6.11 (Tier 2.1) ADX(H4) ≥ 22 — H4 trend confirmation hard gate
        # เหตุผล: live demo 2026-04-28 เจอ trade quiet-H4 ทำตัวเป็น mean-revert
        # H4 ranging → SMC continuation pattern ไม่น่าเชื่อถือ
        if self._htf_data is not None and "adx" in self._htf_data.columns:
            adx_h4 = float(self._htf_data["adx"].iloc[-1])
            if pd.notna(adx_h4) and adx_h4 < 20.0:
                # v6.11.3: lower from 22 → 20 (consistent with ADX H1 floor, +10-15% signal volume)
                no_signal.confluence_score = 0
                no_signal.reasons = [f"❌ [REJECT] ADX(H4) {adx_h4:.1f} < 20 — H4 ranging"]
                return no_signal

        # F. v6.11 (Tier 1.3) D1 bias hard veto — BUY ห้าม counter-D1
        # เก่า: soft +15 confluence bonus → overcome ได้ตอน confluence 95-100
        # ใหม่: D1 == -1 → reject ทันที. Neutral (D1 == 0) ผ่านได้ปกติ
        d1_bias = self._get_d1_bias(symbol)
        if d1_bias == -1:
            no_signal.confluence_score = 0
            no_signal.reasons = ["❌ [REJECT] BUY counter-D1 (D1 bearish) — hard veto"]
            return no_signal

        # F2. v6.11 (Tier 1.4) Quiet-vol × off-overlap session blocker
        # เหตุผล: live demo 2026-04-28 — trades quiet vol นอก London-NY overlap = SMC unreliable
        # quiet = atr_pips < 1.2 × atr_floor; off-overlap = session multiplier < 1.0
        if atr_pips < min_atr_pips * 1.2 and self._get_session_multiplier() < 1.0:
            no_signal.confluence_score = 0
            no_signal.reasons = [
                f"❌ [REJECT] Quiet vol ({atr_pips:.1f} pips < {min_atr_pips * 1.2:.1f}) × off-overlap session"
            ]
            return no_signal

        # v6.11.2 (rollback Tier 2.2 + 2.3 hard gates → soft scoring)
        # — eval ด้วย pool ที่ rebuild ด้วย hard gates ได้ Pass Rate 0.0% เพราะ
        # Sweep + Fresh BOS prereq stack ทำให้ pool หาย 96% (avg 36.9 → 1.3 sigs/ep)
        # Sweep ยังถูก score เป็น bonus ใน factor 3.6 (max +15), Fresh M15 BOS เป็น bonus ใน factor 2.5 ด้านล่าง

        # === ปัจจัยที่ 1: HTF Trend (25 คะแนน) ===
        if self._htf_bias == 1:
            htf_pts = 25
            reasons.append("✅ HTF (H4) ขาขึ้น")
        elif self._htf_bias == 0:
            htf_pts = 10  # Neutral ได้บางส่วน
            reasons.append("⚠️ HTF (H4) Neutral")
        else:
            htf_pts = -15
            reasons.append("❌ HTF (H4) ขาลง — ขัดกับ BUY (-15)")
        score += htf_pts

        # === ปัจจัยที่ 2: MTF Market Bias (20 คะแนน) ===
        mtf_pts = 0
        if mtf_bias == 1:
            mtf_pts = 20
            reasons.append("✅ MTF (H1) Bullish Bias")

            # เช็คว่ามี BOS ล่าสุดหรือไม่
            latest_event = self._structure_mtf.get_latest_event()
            if latest_event and latest_event.event_type == StructureType.BOS_BULLISH:
                mtf_pts += 5  # Bonus สำหรับ Fresh BOS
                reasons.append("✅ Fresh BOS Bullish")
        elif mtf_bias == 0:
            mtf_pts = 5
            reasons.append("⚠️ MTF (H1) Neutral")
        score += mtf_pts

        # === ปัจจัยที่ 2.5: Fresh M15 BOS/CHoCH (soft bonus, v6.11.2) ===
        # — ทดแทน hard gate H ที่ rolled back. ถ้ามี structural shift ใน M15 ภายใน 6 bars → +5
        ltf_event = self._structure_ltf.get_latest_event()
        if ltf_event is not None and ltf_event.is_bullish and ltf_event.index >= len(ltf_df) - 6:
            mtf_pts += 5  # rolled into mtf_pts สำหรับ logging
            score += 5
            reasons.append(f"✅ Fresh M15 BOS/CHoCH Bullish ({len(ltf_df) - 1 - ltf_event.index} bars ago, +5)")

        # === ปัจจัยที่ 3: Order Block (25 คะแนน) ===
        ob_pts_local = 0
        bullish_ob = self._ob_detector.is_price_at_bullish_ob(
            current_price, tolerance_pips=5, pip_size=pip_size
        )

        if bullish_ob:
            cluster_score = self._ob_detector.get_bullish_cluster_score(
                current_price, tolerance_pips=15, pip_size=pip_size
            )
            if cluster_score:
                ob_pts_local = min(25, cluster_score * 0.25)
                reasons.append(f"✅ Bullish OB Cluster (score={cluster_score:.0f})")
            else:
                ob_pts_local = min(25, bullish_ob.strength_score * 0.25)
                reasons.append(f"✅ ราคาอยู่ที่ Bullish OB (score={bullish_ob.strength_score:.0f})")
            score += ob_pts_local
        else:
            reasons.append("⚠️ ไม่มี Bullish OB ใกล้ราคา")

        # === ปัจจัยที่ 3.5: Fair Value Gap (10 คะแนน) ===
        fvg_pts_local = 0
        bullish_fvg = self._fvg_detector.is_price_at_bullish_fvg(
            current_price, tolerance_pips=5, pip_size=pip_size
        )
        if bullish_fvg:
            fvg_pts_local = min(10, bullish_fvg.strength_score * 0.1)
            score += fvg_pts_local
            reasons.append(f"✅ Bullish FVG (score={bullish_fvg.strength_score:.0f})")

        # === ปัจจัยที่ 3.6: Liquidity Sweep (15 คะแนน) ===
        # v6.11 (Tier 2.2): บังคับ sweep within 8 bars แล้วใน pre-filter G — ที่นี่ scoring เพิ่ม
        sweep_pts_local = 0
        sweep_age = -1
        bullish_sweep = self._sweep_detector.get_recent_bullish_sweep(max_bars_ago=8)
        if bullish_sweep:
            sweep_pts_local = min(15, bullish_sweep.strength_score * 0.15)
            sweep_age = bullish_sweep.bars_ago
            score += sweep_pts_local
            reasons.append(f"✅ Bullish Liquidity Sweep (score={bullish_sweep.strength_score:.0f}, {sweep_age} bars ago)")

        # === ปัจจัยที่ 3.7: Inducement (IDM) — v6.11 Tier 3.1 ===
        # หา bullish rejection candle (wick down failed) ภายใน 8 bars
        # IDM = +10 (smart-money trap confirmed), ไม่มี IDM = -5 (อาจเป็น obvious swing)
        idm_event = self._idm_detector.detect_idm(ltf_df, direction=1)
        if idm_event is not None:
            score += 10
            reasons.append(f"✅ IDM Bullish (wick/body={idm_event.wick_to_body_ratio:.2f}, {idm_event.bars_ago} bars ago)")
        else:
            # v6.11.3: penalty 5 → 2 (mild) — กัน over-penalize ใน calm market
            score -= 2
            reasons.append("⚠️ ไม่มี IDM rejection ใน 8 bars — entry อาจเป็น obvious swing (-2)")

        # === ปัจจัยที่ 4: RSI (10 คะแนน) ===
        # Data-driven (2026-04-18): RSI < 30 WR=37.7%, 30-50 WR=28.3%, 50-60 WR=23%
        # ให้ bonus สูงสุดที่ oversold zone (pullback ที่แท้จริง)
        if rsi_value < 30:
            score += 15
            reasons.append(f"✅ RSI={rsi_value:.1f} (Oversold — pullback ที่ดี +15)")
        elif rsi_value < 50:
            score += 8
            reasons.append(f"✅ RSI={rsi_value:.1f} (Below mid +8)")
        else:  # 50-60
            score += 2
            reasons.append(f"⚠️ RSI={rsi_value:.1f} (Above mid — เตือน)")

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

        # === ปัจจัยที่ 7: Bollinger Band %B (2026-04-18) ===
        # Data: BUY + BB<0.2 WR=37%, BB>0.6 WR=27%, BB>0.8 WR=24%
        # BB %B ต่ำ → ราคาอยู่ล่าง = pullback ที่ดีสำหรับ BUY
        bb_pctb = float(ltf_df['bb_pctb'].iloc[-1]) if 'bb_pctb' in ltf_df.columns else 0.5
        if bb_pctb > 0.7:
            # BUY ใน overbought zone = กับดัก → REJECT
            no_signal.confluence_score = score
            no_signal.reasons = reasons + [f"❌ [REJECT] BB %B={bb_pctb:.2f} > 0.7 (BUY ห้ามที่ upper band)"]
            return no_signal
        if bb_pctb < 0.2:
            score += 12
            reasons.append(f"✅ BB %B={bb_pctb:.2f} (ต่ำมาก — pullback แรง +12)")
        elif bb_pctb < 0.4:
            score += 7
            reasons.append(f"✅ BB %B={bb_pctb:.2f} (Lower half +7)")
        else:  # 0.4 - 0.7
            reasons.append(f"⚠️ BB %B={bb_pctb:.2f} (กลาง/บน)")

        # === Session Weighting — ปรับคะแนนตามช่วงเวลา ===
        session_mult = self._get_session_multiplier()
        if session_mult != 1.0:
            score *= session_mult
            reasons.append(f"📊 Session multiplier: ×{session_mult:.2f}")

        # === Cap score ที่ 100 ===
        score = min(score, 100.0)

        # === ตรวจสอบว่าผ่านเกณฑ์หรือไม่ ===
        # v6.11: Counter-D1 ย้ายเป็น hard veto ที่ pre-filter F → ไม่ใช้ soft threshold bump แล้ว
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
            if 0 < ob_sl_distance < sl_distance * 1.5:  # ไม่ไกลเกิน 1.5 เท่าของ ATR SL
                sl_distance = ob_sl_distance

        # v6.2: MIN_SL guard — ป้องกัน SL แคบเกินไป (เช่น EURUSD SL 5 pips ที่เจอ 23 Apr)
        # spread จะกิน % ของ SL สูงเกินไป → RR ไม่คุ้ม
        min_sl_pips = get_symbol_config(symbol, "min_sl_pips", 10)
        min_sl_distance = min_sl_pips * pip_size
        if sl_distance < min_sl_distance:
            if bot_config.debug_mode:
                print(
                    f"⚠️ [SMC {symbol}] BUY SL clamped: "
                    f"{sl_distance/pip_size:.1f} → {min_sl_pips} pips (v6.2 min_sl guard)"
                )
            sl_distance = min_sl_distance

        sl_price = entry_price - sl_distance

        # TP: Dynamic RR ตาม ADX (align กับ backtester rr distribution)
        adx_current = float(ltf_df['adx'].iloc[-1]) if 'adx' in ltf_df.columns else 0.0
        rr_target = self._dynamic_rr(adx_current)
        tp_distance = sl_distance * rr_target
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

        htf_label = {1: "BULLISH", -1: "BEARISH", 0: "RANGING"}.get(self._htf_bias, "RANGING")
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
            timestamp=TimeManager.get_server_time(symbol),
            reasons=reasons,
            ob_high=bullish_ob.high if bullish_ob else None,
            ob_low=bullish_ob.low if bullish_ob else None,
            ob_score=bullish_ob.strength_score if bullish_ob else 0,
            market_bias=mtf_bias,
            trend=ltf_trend,
            rsi_value=rsi_value,
            trend_strength=float(ltf_df['trend_strength'].iloc[-1]) if 'trend_strength' in ltf_df.columns else 0.0,
            macd_histogram=float(ltf_df['macd_histogram'].iloc[-1]) if 'macd_histogram' in ltf_df.columns else 0.0,
            adx=float(ltf_df['adx'].iloc[-1]) if 'adx' in ltf_df.columns else 0.0,
            stoch_k=float(ltf_df['stoch_k'].iloc[-1]) if 'stoch_k' in ltf_df.columns else 50.0,
            bb_pctb=float(ltf_df['bb_pctb'].iloc[-1]) if 'bb_pctb' in ltf_df.columns else 0.5,
            atr_change_ratio=float(ltf_df['atr_change_ratio'].iloc[-1]) if 'atr_change_ratio' in ltf_df.columns else 0.0,
            price_roc=float(ltf_df['price_roc'].iloc[-1]) if 'price_roc' in ltf_df.columns else 0.0,
            # v6.11 (Tier 2.4) per-component breakdown สำหรับ Trades sheet
            htf_score=int(htf_pts),
            mtf_score=int(mtf_pts),
            ob_pts=int(ob_pts_local),
            fvg_pts=int(fvg_pts_local),
            sweep_pts=int(sweep_pts_local),
            sweep_age_bars=sweep_age,
            htf_bias=htf_label,
            d1_bias=d1_bias,
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

        # === MANDATORY PRE-FILTERS (2026-04-18) — mirror ของ BUY ===
        # SELL คือ setup หลักที่ profitable: SELL+RSI>50+align>0+adx>=25 = EV +0.27R
        symbol_info = self._connector.get_symbol_info(symbol)
        pip_size = 0.0001 if symbol_info and symbol_info["digits"] >= 4 else 0.01
        atr_pips = atr_value / pip_size if pip_size > 0 else 0

        # A. Volatility floor — gold-aware + per-symbol override (mirror ของ BUY)
        is_metal = "XAU" in symbol.upper() or "XAG" in symbol.upper()
        default_floor = 100.0 if is_metal else 8.0
        min_atr_pips = float(get_symbol_config(symbol, "atr_floor_pips", default_floor))
        if atr_pips < min_atr_pips:
            no_signal.confluence_score = 0
            no_signal.reasons = [f"❌ [REJECT] ATR {atr_pips:.1f} pips < {min_atr_pips:.0f} (volatility ต่ำ)"]
            return no_signal

        # B. RSI must be in pullback zone for SELL: RSI <= 40 → WR 27%, EV -0.1R
        #    ที่ดี: RSI 50-70 (WR 37%, EV +0.05R)
        if rsi_value <= 40:
            no_signal.confluence_score = 0
            no_signal.reasons = [f"❌ [REJECT] RSI {rsi_value:.1f} <= 40 (SELL ห้ามใกล้ oversold)"]
            return no_signal

        # C. Trend alignment: ห้าม SELL ใน bull market เต็ม (HTF+MTF bullish)
        if self._htf_bias == 1 and mtf_bias == 1:
            no_signal.confluence_score = 0
            no_signal.reasons = ["❌ [REJECT] HTF+MTF bullish ทั้งคู่ — ห้าม SELL counter-trend"]
            return no_signal

        # D. EMA200 alignment (H1) — hard veto: ขายเหนือเส้น = ยืนขวางรถไฟ
        if self._mtf_data is not None and "ema_slow" in self._mtf_data.columns:
            ema200_h1 = float(self._mtf_data["ema_slow"].iloc[-1])
            if pd.notna(ema200_h1) and ema200_h1 > 0 and current_price > ema200_h1:
                no_signal.confluence_score = 0
                no_signal.reasons = [
                    f"❌ [REJECT] SELL above H1 EMA200 "
                    f"(price={current_price:.5f}, EMA200={ema200_h1:.5f})"
                ]
                return no_signal

        # E. ADX(H1) ≥ 20 — skip ranging market
        if self._mtf_data is not None and "adx" in self._mtf_data.columns:
            adx_h1 = float(self._mtf_data["adx"].iloc[-1])
            if adx_h1 < 20.0:
                no_signal.confluence_score = 0
                no_signal.reasons = [f"❌ [REJECT] ADX(H1) {adx_h1:.1f} < 20 — ranging market"]
                return no_signal

        # E2. v6.11 (Tier 2.1) ADX(H4) ≥ 22 — H4 trend confirmation hard gate
        if self._htf_data is not None and "adx" in self._htf_data.columns:
            adx_h4 = float(self._htf_data["adx"].iloc[-1])
            if pd.notna(adx_h4) and adx_h4 < 20.0:
                # v6.11.3: lower from 22 → 20 (consistent with ADX H1 floor, +10-15% signal volume)
                no_signal.confluence_score = 0
                no_signal.reasons = [f"❌ [REJECT] ADX(H4) {adx_h4:.1f} < 20 — H4 ranging"]
                return no_signal

        # F. v6.11 (Tier 1.3) D1 bias hard veto — SELL ห้าม counter-D1
        # เก่า: soft +15 confluence bonus → overcome ได้ตอน confluence 95-100
        # ใหม่: D1 == +1 → reject ทันที. Neutral (D1 == 0) ผ่านได้ปกติ
        d1_bias = self._get_d1_bias(symbol)
        if d1_bias == 1:
            no_signal.confluence_score = 0
            no_signal.reasons = ["❌ [REJECT] SELL counter-D1 (D1 bullish) — hard veto"]
            return no_signal

        # F2. v6.11 (Tier 1.4) Quiet-vol × off-overlap session blocker (mirror ของ BUY)
        if atr_pips < min_atr_pips * 1.2 and self._get_session_multiplier() < 1.0:
            no_signal.confluence_score = 0
            no_signal.reasons = [
                f"❌ [REJECT] Quiet vol ({atr_pips:.1f} pips < {min_atr_pips * 1.2:.1f}) × off-overlap session"
            ]
            return no_signal

        # v6.11.2 (rollback Tier 2.2 + 2.3 hard gates → soft scoring) — mirror ของ BUY

        # === ปัจจัยที่ 1: HTF Trend (25 คะแนน) ===
        if self._htf_bias == -1:
            htf_pts = 25
            reasons.append("✅ HTF (H4) ขาลง")
        elif self._htf_bias == 0:
            htf_pts = 10
            reasons.append("⚠️ HTF (H4) Neutral")
        else:
            htf_pts = -15
            reasons.append("❌ HTF (H4) ขาขึ้น — ขัดกับ SELL (-15)")
        score += htf_pts

        # === ปัจจัยที่ 2: MTF Market Bias (20 คะแนน) ===
        mtf_pts = 0
        if mtf_bias == -1:
            mtf_pts = 20
            reasons.append("✅ MTF (H1) Bearish Bias")

            latest_event = self._structure_mtf.get_latest_event()
            if latest_event and latest_event.event_type == StructureType.BOS_BEARISH:
                mtf_pts += 5
                reasons.append("✅ Fresh BOS Bearish")
        elif mtf_bias == 0:
            mtf_pts = 5
            reasons.append("⚠️ MTF (H1) Neutral")
        score += mtf_pts

        # === ปัจจัยที่ 2.5: Fresh M15 BOS/CHoCH (soft bonus, v6.11.2) — mirror ของ BUY ===
        ltf_event = self._structure_ltf.get_latest_event()
        if ltf_event is not None and not ltf_event.is_bullish and ltf_event.index >= len(ltf_df) - 6:
            mtf_pts += 5
            score += 5
            reasons.append(f"✅ Fresh M15 BOS/CHoCH Bearish ({len(ltf_df) - 1 - ltf_event.index} bars ago, +5)")

        # === ปัจจัยที่ 3: Order Block (25 คะแนน) ===
        ob_pts_local = 0
        bearish_ob = self._ob_detector.is_price_at_bearish_ob(
            current_price, tolerance_pips=5, pip_size=pip_size
        )

        if bearish_ob:
            cluster_score = self._ob_detector.get_bearish_cluster_score(
                current_price, tolerance_pips=15, pip_size=pip_size
            )
            if cluster_score:
                ob_pts_local = min(25, cluster_score * 0.25)
                reasons.append(f"✅ Bearish OB Cluster (score={cluster_score:.0f})")
            else:
                ob_pts_local = min(25, bearish_ob.strength_score * 0.25)
                reasons.append(f"✅ ราคาอยู่ที่ Bearish OB (score={bearish_ob.strength_score:.0f})")
            score += ob_pts_local
        else:
            reasons.append("⚠️ ไม่มี Bearish OB ใกล้ราคา")

        # === ปัจจัยที่ 3.5: Fair Value Gap (10 คะแนน) ===
        fvg_pts_local = 0
        bearish_fvg = self._fvg_detector.is_price_at_bearish_fvg(
            current_price, tolerance_pips=5, pip_size=pip_size
        )
        if bearish_fvg:
            fvg_pts_local = min(10, bearish_fvg.strength_score * 0.1)
            score += fvg_pts_local
            reasons.append(f"✅ Bearish FVG (score={bearish_fvg.strength_score:.0f})")

        # === ปัจจัยที่ 3.6: Liquidity Sweep (15 คะแนน) ===
        sweep_pts_local = 0
        sweep_age = -1
        bearish_sweep = self._sweep_detector.get_recent_bearish_sweep(max_bars_ago=8)
        if bearish_sweep:
            sweep_pts_local = min(15, bearish_sweep.strength_score * 0.15)
            sweep_age = bearish_sweep.bars_ago
            score += sweep_pts_local
            reasons.append(f"✅ Bearish Liquidity Sweep (score={bearish_sweep.strength_score:.0f}, {sweep_age} bars ago)")

        # === ปัจจัยที่ 3.7: Inducement (IDM) — v6.11 Tier 3.1 (mirror ของ BUY) ===
        idm_event = self._idm_detector.detect_idm(ltf_df, direction=-1)
        if idm_event is not None:
            score += 10
            reasons.append(f"✅ IDM Bearish (wick/body={idm_event.wick_to_body_ratio:.2f}, {idm_event.bars_ago} bars ago)")
        else:
            # v6.11.3: penalty 5 → 2 (mild) — mirror ของ BUY
            score -= 2
            reasons.append("⚠️ ไม่มี IDM rejection ใน 8 bars — entry อาจเป็น obvious swing (-2)")

        # === ปัจจัยที่ 4: RSI (10 คะแนน) ===
        # Data-driven (2026-04-18): SELL RSI 50-70 WR=37.3%, 30-50 WR=32.7%, >70 WR=25%
        # ให้ bonus สูงสุดที่ overbought zone (pullback sell)
        if rsi_value > 70:
            score += 15
            reasons.append(f"✅ RSI={rsi_value:.1f} (Overbought — pullback ที่ดี +15)")
        elif rsi_value > 50:
            score += 8
            reasons.append(f"✅ RSI={rsi_value:.1f} (Above mid +8)")
        else:  # 40-50
            score += 2
            reasons.append(f"⚠️ RSI={rsi_value:.1f} (Below mid — เตือน)")

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

        # === ปัจจัยที่ 7: Bollinger Band %B (2026-04-18) — mirror ของ BUY ===
        # SELL: อยู่ที่ upper band (%B > 0.6) = pullback ดี
        bb_pctb = float(ltf_df['bb_pctb'].iloc[-1]) if 'bb_pctb' in ltf_df.columns else 0.5
        if bb_pctb < 0.3:
            # SELL ใน oversold zone = กับดัก → REJECT
            no_signal.confluence_score = score
            no_signal.reasons = reasons + [f"❌ [REJECT] BB %B={bb_pctb:.2f} < 0.3 (SELL ห้ามที่ lower band)"]
            return no_signal
        if bb_pctb > 0.8:
            score += 12
            reasons.append(f"✅ BB %B={bb_pctb:.2f} (สูงมาก — pullback แรง +12)")
        elif bb_pctb > 0.6:
            score += 7
            reasons.append(f"✅ BB %B={bb_pctb:.2f} (Upper half +7)")
        else:  # 0.3 - 0.6
            reasons.append(f"⚠️ BB %B={bb_pctb:.2f} (กลาง/ล่าง)")

        # === Session Weighting — ปรับคะแนนตามช่วงเวลา ===
        session_mult = self._get_session_multiplier()
        if session_mult != 1.0:
            score *= session_mult
            reasons.append(f"📊 Session multiplier: ×{session_mult:.2f}")

        # === Cap score ที่ 100 ===
        score = min(score, 100.0)

        # === ตรวจสอบเกณฑ์ ===
        # v6.11: Counter-D1 ย้ายเป็น hard veto ที่ pre-filter F → ไม่ใช้ soft threshold bump แล้ว
        if score < self.MIN_CONFLUENCE_SCORE:
            no_signal.confluence_score = score
            no_signal.reasons = reasons
            return no_signal

        # === คำนวณ Entry, SL, TP ===
        entry_price = price_info["bid"]  # SELL ที่ราคา Bid

        sl_distance = atr_value * bot_config.indicators.atr_sl_multiplier

        if bearish_ob:
            ob_sl_distance = bearish_ob.high - entry_price + (2 * pip_size)
            if 0 < ob_sl_distance < sl_distance * 1.5:
                sl_distance = ob_sl_distance

        # v6.2: MIN_SL guard — ป้องกัน SL แคบเกินไป (mirror ของ BUY)
        min_sl_pips = get_symbol_config(symbol, "min_sl_pips", 10)
        min_sl_distance = min_sl_pips * pip_size
        if sl_distance < min_sl_distance:
            if bot_config.debug_mode:
                print(
                    f"⚠️ [SMC {symbol}] SELL SL clamped: "
                    f"{sl_distance/pip_size:.1f} → {min_sl_pips} pips (v6.2 min_sl guard)"
                )
            sl_distance = min_sl_distance

        sl_price = entry_price + sl_distance
        # TP: Dynamic RR ตาม ADX (align กับ backtester rr distribution)
        adx_current = float(ltf_df['adx'].iloc[-1]) if 'adx' in ltf_df.columns else 0.0
        rr_target = self._dynamic_rr(adx_current)
        tp_distance = sl_distance * rr_target
        tp_price = entry_price - tp_distance
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        if rr_ratio < bot_config.ftmo.MIN_RISK_REWARD_RATIO:
            no_signal.reasons = reasons + ["❌ RR ต่ำเกินไป"]
            return no_signal

        digits = symbol_info["digits"] if symbol_info else 5
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        htf_label = {1: "BULLISH", -1: "BEARISH", 0: "RANGING"}.get(self._htf_bias, "RANGING")
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
            rsi_value=rsi_value,
            trend_strength=float(ltf_df['trend_strength'].iloc[-1]) if 'trend_strength' in ltf_df.columns else 0.0,
            macd_histogram=float(ltf_df['macd_histogram'].iloc[-1]) if 'macd_histogram' in ltf_df.columns else 0.0,
            adx=float(ltf_df['adx'].iloc[-1]) if 'adx' in ltf_df.columns else 0.0,
            stoch_k=float(ltf_df['stoch_k'].iloc[-1]) if 'stoch_k' in ltf_df.columns else 50.0,
            bb_pctb=float(ltf_df['bb_pctb'].iloc[-1]) if 'bb_pctb' in ltf_df.columns else 0.5,
            atr_change_ratio=float(ltf_df['atr_change_ratio'].iloc[-1]) if 'atr_change_ratio' in ltf_df.columns else 0.0,
            price_roc=float(ltf_df['price_roc'].iloc[-1]) if 'price_roc' in ltf_df.columns else 0.0,
            # v6.11 (Tier 2.4) per-component breakdown สำหรับ Trades sheet
            htf_score=int(htf_pts),
            mtf_score=int(mtf_pts),
            ob_pts=int(ob_pts_local),
            fvg_pts=int(fvg_pts_local),
            sweep_pts=int(sweep_pts_local),
            sweep_age_bars=sweep_age,
            htf_bias=htf_label,
            d1_bias=d1_bias,
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

    def _get_session_multiplier(self) -> float:
        """คืนค่า multiplier ตามช่วงเวลา (Session Weighting)"""
        try:
            now_utc = TimeManager.get_server_time().astimezone(pytz.UTC)
            hour = now_utc.hour
        except Exception:
            return 1.0

        if 12 <= hour < 13:
            return 1.10   # London/NY overlap
        elif 8 <= hour < 12:
            return 1.00   # Core London
        elif 13 <= hour < 16:
            return 1.00   # Core NY
        elif 7 <= hour < 8:
            return 0.90   # Early London
        elif 16 <= hour < 17:
            return 0.90   # Late NY
        return 1.0

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

        ⚠️ CRITICAL Timezone Contract:
            MT5 Server Time (FTMO) = EET (UTC+2 หรือ UTC+3 ตาม DST)
            bot_config.sessions.*_start/*_end/friday_cutoff ต้องเก็บเป็น **UTC** เสมอ
            — ถ้า config เก็บเป็น EET ค่านี้จะผิดทั้ง session window และ Friday cutoff
            ตรวจเช็คไฟล์ config/settings.py ว่ารับค่าเป็น UTC จริง

        Returns:
            bool: True ถ้าอยู่ในช่วงเทรด
        """
        now = TimeManager.get_server_time()  # ดึงเวลา EET จาก MT5 server

        # แปลง EET → UTC เพื่อเทียบกับ session config (UTC)
        now_utc = now.astimezone(pytz.UTC)
        current_time = now_utc.time()
        current_weekday = now_utc.weekday()  # 0=จันทร์, 6=อาทิตย์

        session_config = bot_config.sessions

        # ตรวจสอบวันเทรด
        if current_weekday not in session_config.trading_days:
            return False

        # ตรวจวันศุกร์ — หยุดหลัง friday_cutoff (UTC)
        if current_weekday == 4 and current_time >= session_config.friday_cutoff:
            return False

        # ตรวจช่วง London Session (UTC)
        in_london = session_config.london_start <= current_time <= session_config.london_end

        # ตรวจช่วง New York Session (UTC)
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
