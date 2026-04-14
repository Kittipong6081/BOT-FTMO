"""
===============================================================================
FTMO Trading Bot — การตั้งค่าระบบทั้งหมด (System Configuration)
===============================================================================
ไฟล์นี้เก็บค่าคงที่และการตั้งค่าสำคัญทั้งหมดของระบบเทรด
รวมถึงกฎ FTMO, ค่าเชื่อมต่อ MT5, คู่เงินที่เทรด, และพารามิเตอร์ความเสี่ยง

⚠️  ห้ามเปลี่ยนค่า FTMO_* โดยไม่เข้าใจผลกระทบ — อาจทำให้ฝ่าฝืนกฎ FTMO
===============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import time
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env ที่อยู่ในโฟลเดอร์รันของบอท
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, ".env"))


# =============================================================================
# 📊 การตั้งค่า MetaTrader 5 — ข้อมูลเชื่อมต่อโบรกเกอร์
# =============================================================================
@dataclass
class MT5Config:
    """
    การตั้งค่าสำหรับเชื่อมต่อกับ MetaTrader 5 Terminal
    ผู้ใช้ต้องกรอกข้อมูลบัญชีจริงก่อนใช้งาน
    """
    # เส้นทางไปยังไฟล์ terminal64.exe ของ MT5
    terminal_path: str = os.getenv("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5/terminal64.exe")
    
    # ข้อมูลบัญชี MT5
    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    
    # Timeout สำหรับการเชื่อมต่อ (มิลลิวินาที)
    timeout: int = 10000        # รอเชื่อมต่อสูงสุด 10 วินาที
    
    # โหมดเทรด: True = เทรดจริง, False = เทรดจำลอง (Paper Trading)
    live_trading: bool = False# ⚠️ เริ่มต้นเป็น False เพื่อความปลอดภัย


# =============================================================================
# 🛡️ กฎ FTMO — ค่าความเสี่ยงที่ห้ามเกิน (HARD LIMITS)
# =============================================================================
@dataclass
class FTMOConfig:
    """
    กฎ FTMO ที่ต้องปฏิบัติตามอย่างเคร่งครัด
    ค่าที่ตั้งไว้มี Buffer เพื่อป้องกันการฝ่าฝืนกฎ
    
    กฎ FTMO จริง:
    - ขาดทุนรายวันสูงสุด: 5% ของ Equity เริ่มต้นแต่ละวัน
    - ขาดทุนรวมสูงสุด: 10% ของ Balance เริ่มต้น
    
    ค่าที่ Bot ใช้ (มี Safety Buffer):
    - Daily Hard Stop: 4% (Buffer 1% จากกฎ 5%)
    - Max Drawdown Hard Stop: 8% (Buffer 2% จากกฎ 10%)
    """
    
    # === ขาดทุนรายวันสูงสุด ===
    # กฎ FTMO = 5%, Bot ใช้ 4% เป็น Buffer ป้องกัน
    DAILY_LOSS_HARD_STOP_PCT: float = 0.04  # 4%
    
    # === ขาดทุนรวมสูงสุด (Max Drawdown) ===
    # กฎ FTMO = 10%, Bot ใช้ 8% เป็น Buffer ป้องกัน
    MAX_DRAWDOWN_HARD_STOP_PCT: float = 0.08  # 8%
    
    # === ความเสี่ยงต่อการเทรดแต่ละครั้ง ===
    MIN_RISK_PER_TRADE_PCT: float = 0.005   # 0.5% ขั้นต่ำ
    MAX_RISK_PER_TRADE_PCT: float = 0.01    # 1.0% สูงสุด
    DEFAULT_RISK_PER_TRADE_PCT: float = 0.0075  # 0.75% ค่าเริ่มต้น
    
    # === อัตราส่วน Risk:Reward ขั้นต่ำ ===
    MIN_RISK_REWARD_RATIO: float = 1.5      # ต้องได้อย่างน้อย 1:1.5
    PREFERRED_RISK_REWARD_RATIO: float = 2.0  # เป้าหมาย 1:2
    
    # === จำนวน Position สูงสุดที่เปิดได้พร้อมกัน ===
    MAX_OPEN_POSITIONS: int = 3
    
    # === ระดับกำไรเป้าหมาย (FTMO Challenge) ===
    # FTMO Challenge ต้องทำกำไร 10% ใน 30 วัน
    PROFIT_TARGET_PCT: float = 0.10  # 10%
    
    # === จำนวนวันเทรดขั้นต่ำ ===
    MIN_TRADING_DAYS: int = 4  # ต้องเทรดอย่างน้อย 4 วัน

    # === Anti-Overtrading Guardrails ===
    MAX_TRADES_PER_DAY: int = 5                 # ยิงได้สูงสุด 5 ออเดอร์/วัน
    MAX_CORRELATED_POSITIONS: int = 1           # 1 ตำแหน่งต่อกลุ่ม correlation
    MIN_CONFLUENCE_SCORE: float = 70.0          # เกณฑ์ confluence ขั้นต่ำ (ปรับได้ runtime)

    # === Cooldown / Anti-Revenge-Trading ===
    # หลังโดน SL บนคู่เงินใดคู่เงินหนึ่ง ต้องรอกี่นาทีก่อนเปิดคู่เดิมอีกครั้ง
    COOLDOWN_AFTER_LOSS_MIN: int = 30
    # ถ้าแพ้ติดกันกี่ครั้งให้ pause ทั้งระบบ
    CONSECUTIVE_LOSS_PAUSE_COUNT: int = 2
    CONSECUTIVE_LOSS_PAUSE_MIN: int = 60
    # ถ้าแพ้ติดกันกี่ครั้งให้ halt ทั้งวัน
    CONSECUTIVE_LOSS_HALT_COUNT: int = 3


# =============================================================================
# 💱 การตั้งค่าคู่เงินและ Symbol ที่เทรด
# =============================================================================
@dataclass
class SymbolConfig:
    """
    การตั้งค่าคู่เงินที่ Bot จะเทรด
    เลือกคู่เงินที่มี Spread ต่ำและสภาพคล่องสูง
    """
    
    # คู่เงินหลักที่เทรด (Major Pairs — Spread ต่ำ)
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD",   # Euro / US Dollar
        "GBPUSD",   # British Pound / US Dollar
        "USDJPY",   # US Dollar / Japanese Yen
        "AUDUSD",   # Australian Dollar / US Dollar
	"USDCAD",   # US Dollar / Canadian Dollar
        "USDCHF",   # US Dollar / Swiss Franc
        "NZDUSD",   # New Zealand Dollar / US Dollar
        "EURJPY",   # Euro / Japanese Yen
        "GBPJPY",   # British Pound / Japanese Yen
    ])
    
    # Timeframe หลักสำหรับการวิเคราะห์ (ใช้ค่า MT5 constant)
    # mt5.TIMEFRAME_M15 = 15 นาที
    primary_timeframe: str = "M15"      # Timeframe หลักสำหรับ Entry
    higher_timeframe: str = "H4"        # Timeframe สูงกว่าสำหรับ Trend Confirmation
    structure_timeframe: str = "H1"     # Timeframe สำหรับวิเคราะห์โครงสร้างตลาด
    
    # จำนวนแท่งเทียนที่ดึงมาวิเคราะห์
    candles_to_fetch: int = 500         # ดึง 500 แท่งล่าสุด
    
    # ค่า Spread สูงสุดที่ยอมรับ (เป็น points)
    max_spread_points: Dict[str, int] = field(default_factory=lambda: {
        "EURUSD": 15,   # ไม่เทรดถ้า Spread > 1.5 pips
        "GBPUSD": 20,   # ไม่เทรดถ้า Spread > 2.0 pips
        "USDJPY": 15,   # ไม่เทรดถ้า Spread > 1.5 pips
        "AUDUSD": 18,   # ไม่เทรดถ้า Spread > 1.8 pips
	"USDCAD": 20,   # ไม่เทรดถ้า Spread > 2.0 pips
        "USDCHF": 20,   # ไม่เทรดถ้า Spread > 2.0 pips
        "NZDUSD": 20,   # ไม่เทรดถ้า Spread > 2.0 pips
        "EURJPY": 25,   # ไม่เทรดถ้า Spread > 2.5 pips
        "GBPJPY": 30,   # ไม่เทรดถ้า Spread > 3.0 pips
    })


# =============================================================================
# ⏰ การตั้งค่าช่วงเวลาเทรด (Trading Sessions)
# =============================================================================
@dataclass
class SessionConfig:
    """
    กำหนดช่วงเวลาที่ Bot จะเทรด
    เน้น London และ New York Session ที่มี Volume สูง
    หลีกเลี่ยง Asian Session ที่ Volatility ต่ำ
    """
    
    # === London Session (เวลา UTC) ===
    london_start: time = time(7, 0)     # 07:00 UTC (เปิดก่อน 1 ชม. เพื่อเตรียมตัว)
    london_end: time = time(12, 0)      # 12:00 UTC
    
    # === New York Session (เวลา UTC) ===
    newyork_start: time = time(12, 0)   # 12:00 UTC (ช่วง Overlap กับ London)
    newyork_end: time = time(17, 0)     # 17:00 UTC
    
    # === ห้ามเทรดช่วงนี้ ===
    # ก่อนข่าวสำคัญ (NFP, FOMC, CPI) — ต้อง implement แยก
    no_trade_before_news_minutes: int = 30   # หยุดเทรด 30 นาที ก่อนข่าว
    no_trade_after_news_minutes: int = 15    # หยุดเทรด 15 นาที หลังข่าว
    
    # === FTMO Friday Rule (บังคับปิดออเดอร์วันศุกร์) ===
    # กฎ FTMO: ห้ามถือออเดอร์ข้ามสัปดาห์ในสุดสัปดาห์ 
    # เราจะปิดบังคับวันศุกร์เวลา 20:45 น. ตามเวลา Server (EET)
    friday_force_close: time = time(20, 45)
    
    # === Rollover Protection (เตะตัดขาสเปรดถ่าง) ===
    # บอทจะเข้าสู่โหมดหลับ (Pause) ช่วงรอยต่อวันเพื่อหนี Spread กว้าง
    rollover_start: time = time(23, 55)
    rollover_end: time = time(1, 5)

    # === วันที่ห้ามเทรด ===
    # วันศุกร์ช่วงบ่ายมักมี Volatility ผิดปกติ (หยุดรับออเดอร์ใหม่)
    friday_cutoff: time = time(15, 0)   # หยุดเปิดออเดอร์ใหม่วันศุกร์หลัง 15:00 UTC
    
    # เทรดได้เฉพาะวันจันทร์-ศุกร์
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # จันทร์=0 ถึง ศุกร์=4


# =============================================================================
# 📈 การตั้งค่า Indicator Parameters
# =============================================================================
@dataclass
class IndicatorConfig:
    """
    พารามิเตอร์สำหรับ Technical Indicators ที่ใช้ในกลยุทธ์
    """
    
    # ATR (Average True Range) — วัดความผันผวน
    atr_period: int = 14                # คำนวณ ATR จาก 14 แท่งเทียน
    atr_sl_multiplier: float = 1.5      # Stop Loss = ATR * 1.5
    atr_tp_multiplier: float = 3.0      # Take Profit = ATR * 3.0 (RR 1:2)
    
    # EMA (Exponential Moving Average) — วัดแนวโน้ม
    ema_fast: int = 21                  # EMA เร็ว (แนวโน้มระยะสั้น)
    ema_medium: int = 50                # EMA กลาง (แนวโน้มระยะกลาง)
    ema_slow: int = 200                 # EMA ช้า (แนวโน้มระยะยาว)
    
    # RSI (Relative Strength Index) — วัดแรงซื้อ/ขาย
    rsi_period: int = 14                # คำนวณ RSI จาก 14 แท่งเทียน
    rsi_overbought: float = 70.0        # โซน Overbought (ซื้อมากเกินไป)
    rsi_oversold: float = 30.0          # โซน Oversold (ขายมากเกินไป)
    
    # Order Block Parameters
    ob_lookback: int = 50               # ดูย้อนหลัง 50 แท่งเพื่อหา Order Blocks
    ob_min_body_ratio: float = 0.5      # แท่งเทียนต้องมี Body > 50% ของ Range
    
    # Market Structure Parameters
    swing_lookback: int = 5             # ดูย้อนหลัง 5 แท่งเพื่อหา Swing Points


# =============================================================================
# 📱 การตั้งค่าการแจ้งเตือน (Notifications)
# =============================================================================
@dataclass
class NotificationConfig:
    """
    การตั้งค่าสำหรับการแจ้งเตือนผ่าน Discord Webhook
    """
    enable_notifications: bool = os.getenv("DISCORD_ENABLE", "True").lower() in ("true", "1", "yes")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")


# =============================================================================
# 📁 การตั้งค่าเส้นทางไฟล์ (File Paths)
# =============================================================================
@dataclass
class PathConfig:
    """
    เส้นทางไฟล์สำหรับ Logging, State Management, และ RL Models
    """
    
    # ฐานเส้นทาง (ใช้ Directory ของโปรเจกต์)
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # ไฟล์บันทึกผลเทรด (Excel)
    trade_log_file: str = ""
    
    # ไฟล์เก็บสถานะ Bot (เพื่อ Resume หลัง Restart)
    state_file: str = ""
    
    # โฟลเดอร์เก็บ RL Models
    model_dir: str = ""
    
    # โฟลเดอร์เก็บ Log ทั่วไป
    log_dir: str = ""
    
    def __post_init__(self):
        """สร้างเส้นทางไฟล์จาก base_dir และสร้างโฟลเดอร์ที่จำเป็น"""
        # ใช้ไฟล์เดียวตลอด (ไม่แยกรายเดือน) เพื่อให้ ML อ่านข้อมูลครบเสมอ
        self.trade_log_file = os.path.join(self.base_dir, "logs", "ftmo_trades.xlsx")
        self.state_file = os.path.join(self.base_dir, "logs", "bot_state.json")
        self.model_dir = os.path.join(self.base_dir, "models")
        self.log_dir = os.path.join(self.base_dir, "logs")
        
        # สร้างโฟลเดอร์ที่จำเป็นอัตโนมัติ
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


# =============================================================================
# 🏗️ การตั้งค่าหลักรวม (Master Configuration)
# =============================================================================
@dataclass
class BotConfig:
    """
    คลาสรวมการตั้งค่าทั้งหมดของ Bot
    เรียกใช้จากทุกโมดูลผ่าน instance เดียว
    """
    mt5: MT5Config = field(default_factory=MT5Config)
    ftmo: FTMOConfig = field(default_factory=FTMOConfig)
    symbols: SymbolConfig = field(default_factory=SymbolConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # === โหมดดีบัก ===
    debug_mode: bool = True             # แสดงข้อมูลละเอียดใน Console
    
    # === ช่วงเวลาตรวจสอบ (วินาที) ===
    main_loop_interval: int = 5         # ตรวจสอบทุก 5 วินาที
    risk_check_interval: int = 1        # ตรวจสอบความเสี่ยงทุก 1 วินาที


# =============================================================================
# 🔧 สร้าง Instance สำหรับใช้งานทั่วทั้งระบบ (Singleton Pattern)
# =============================================================================

# สร้าง Config ตัวหลักที่ทุกโมดูลจะ import ไปใช้
# ใช้: from config.settings import bot_config
bot_config = BotConfig()


# =============================================================================
# 🧪 ทดสอบการตั้งค่า (รันไฟล์นี้โดยตรง)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 FTMO Trading Bot — System Configuration")
    print("=" * 60)
    
    # แสดงค่า FTMO Rules
    print(f"\n🛡️ FTMO Rules (พร้อม Safety Buffer):")
    print(f"   Daily Hard Stop:     {bot_config.ftmo.DAILY_LOSS_HARD_STOP_PCT * 100:.1f}%")
    print(f"   Max Drawdown Stop:   {bot_config.ftmo.MAX_DRAWDOWN_HARD_STOP_PCT * 100:.1f}%")
    print(f"   Risk Per Trade:      {bot_config.ftmo.MIN_RISK_PER_TRADE_PCT * 100:.1f}% - {bot_config.ftmo.MAX_RISK_PER_TRADE_PCT * 100:.1f}%")
    print(f"   Min Risk:Reward:     1:{bot_config.ftmo.MIN_RISK_REWARD_RATIO:.1f}")
    print(f"   Max Open Positions:  {bot_config.ftmo.MAX_OPEN_POSITIONS}")
    
    # แสดงคู่เงินที่เทรด
    print(f"\n💱 Symbols: {', '.join(bot_config.symbols.symbols)}")
    
    # แสดงเส้นทางไฟล์
    print(f"\n📁 Trade Log: {bot_config.paths.trade_log_file}")
    print(f"📁 State File: {bot_config.paths.state_file}")
    
    print(f"\n✅ การตั้งค่าถูกต้องทั้งหมด")
