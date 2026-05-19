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
from typing import List, Dict, Optional
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
    # v7.1.9 (2026-05-05): bump 0.7% → 0.99% — sync กับ RL training (--risk_per_trade 0.0099)
    # FTMO 1% rule: strictly < 1% per trade. 0.99% = max allowed under rule.
    # Eval: Pass 6.1%, Profit +2.58%, DD max 5.30%/8% (66% buffer), Daily DD 3.04%/4% (76% buffer).
    # Daily worst-case = 4 losing trades × 0.99% = 3.96% (under FTMO 4% Daily limit).
    MIN_RISK_PER_TRADE_PCT: float = 0.005   # 0.5% ขั้นต่ำ (floor ปลอดภัย)
    MAX_RISK_PER_TRADE_PCT: float = 0.0099  # 0.99% สูงสุด (FTMO < 1% rule)
    DEFAULT_RISK_PER_TRADE_PCT: float = 0.0099  # 0.99% — sync กับ RL training v7.1.9
    
    # === อัตราส่วน Risk:Reward ขั้นต่ำ ===
    # v8.0.9 (2026-05-07): ลด 1.5 → 1.0 เพื่อรองรับ MR strategy ที่ใช้ RR 1:1
    # (quick TP design — เน้น win rate สูงแทน reward ต่อไม้). ค่า 1.5 เดิมเป็น
    # SMC-era ที่ใช้ RR 1:1.5-2.5. ทุก MR signal ถูก reject ใน RiskManager
    # เพราะ check นี้จนกว่าจะแก้
    MIN_RISK_REWARD_RATIO: float = 1.0      # ต้องได้อย่างน้อย 1:1 (MR uses 1:1)
    PREFERRED_RISK_REWARD_RATIO: float = 1.0  # MR fix RR 1:1
    
    # === จำนวน Position สูงสุดที่เปิดได้พร้อมกัน ===
    MAX_OPEN_POSITIONS: int = 3

    # === Min Seconds Between Opens (v8.0.26 — anti bulk-trading) ===
    # The5ers ห้าม "bulk trading" = เปิดหลายไม้พร้อมกันในวินาทีเดียว
    # → flag ว่าเป็นบอท (rationale: "not behind the strategy")
    # บอทเราเปิดทีละไม้ทุก 5s scan อยู่แล้ว แต่ถ้า 2 symbol เข้า signal
    # พร้อมกัน อาจเปิดห่างกัน <1s ได้ (เคยเจอ EURUSD pre-news 12 พ.ค. 2 ไม้ห่าง 0.01s)
    # Gate นี้บังคับช่องว่างขั้นต่ำระหว่างการเปิด — กัน flag จาก The5ers
    MIN_SECONDS_BETWEEN_OPENS_ENABLED: bool = True
    MIN_SECONDS_BETWEEN_OPENS_SEC: int = 60  # 60s = 1 นาที (safe margin)
    
    # === ระดับกำไรเป้าหมาย (FTMO Challenge) ===
    # FTMO Challenge ต้องทำกำไร 10% ใน 30 วัน
    PROFIT_TARGET_PCT: float = 0.10  # 10%

    # === Monday Morning Delay (v8.0.19 — กัน volatile post-weekend) ===
    # ข้อมูลจริง 11 พ.ค.: 3 ไม้แรก (01:06, 03:50, 03:54 EET) เสีย $300+ ก่อน
    # ตลาดสงบลง. หน่วงไม่ให้บอทเทรดในช่วง MONDAY_DELAY_END_HOUR_EET ชม.แรก
    # ของ Monday (EET) — ตามเวลา server FTMO. หลังจากนั้นเทรดปกติ.
    # Tue-Fri ไม่กระทบ.
    MONDAY_DELAY_ENABLED: bool = True
    MONDAY_DELAY_END_HOUR_EET: int = 4   # ห้ามเทรด Monday 00:00-03:59 EET (= 04:00-07:59 ICT)

    # === Weekday Asian Early Delay (v8.0.24, extended v8.0.25) — Mon-Fri non-XAU ===
    # ข้อมูลจริง 12-14 พ.ค. (3 วัน):
    #   Asian early (00-07 EET) non-XAU: -$300 (ขาดทุนหนัก)
    #   Asian late (07-10 EET): +$242 (sweet spot — WR 83%)
    #   XAUUSD Asian early: +$159 (WR 80% — เก่ง!)
    # → ดังนั้น: block trade Mon-Fri 00:00-06:59 EET (= 04:00-10:59 ICT)
    #   ยกเว้น XAUUSD (Gold เก่ง Asian early — anytime Tue-Fri / 04 EET+ Mon)
    # v8.0.25: ขยาย Mon ด้วย — Mon non-XAU เริ่ม 07 EET (= 11 ICT)
    #          Mon XAU: เริ่ม 04 EET (= 08 ICT) — v8.0.19 Monday delay ยังบล็อก 00-04 EET
    WEEKDAY_DELAY_ENABLED: bool = True
    WEEKDAY_DELAY_END_HOUR_EET: int = 7  # ห้าม Mon-Fri 00:00-06:59 EET (= 04:00-10:59 ICT)
    WEEKDAY_DELAY_EXCEPT_SYMBOLS: tuple = ("XAUUSD",)  # ทอง = exception

    # === XAU Tue-Fri Delay (v8.0.36) ===
    # XAU เริ่มเทรดได้ตอน 05:05 ICT หลัง rollover (ก่อน v8.0.36)
    # แต่ตลาดยังไม่มีทิศทางที่แน่นอน — หน่วงไปเริ่ม 05 EET (= 09 ICT) Tue-Fri
    # Mon XAU ยังใช้ MONDAY_DELAY = 04 EET (= 08 ICT)
    XAU_WEEKDAY_DELAY_ENABLED: bool = True
    XAU_WEEKDAY_DELAY_END_HOUR_EET: int = 5  # XAU Tue-Fri เริ่ม 05 EET (= 09 ICT)

    # === Flip-Lock TTL (v8.0.18 — fix lock ค้างข้าม weekend) ===
    # FLIP_LOCK_MIN_MINUTES = MIN wait (เวลาเร็วสุดที่ unlock ได้)
    # FLIP_LOCK_MAX_MINUTES = MAX TTL — หลังจากนี้ unlock อัตโนมัติ (ไม่ต้องรอ retrace)
    # กัน case Friday ปิด SELL XAUUSD @ 4746 → Monday ราคา 4673 (ลงต่อ ไม่ retrace)
    # → ก่อน fix: lock ค้างถาวร; หลัง fix: TTL 4 ชม. ล้าง lock ทิ้งเอง
    FLIP_LOCK_MIN_MINUTES: int = 5      # เวลาขั้นต่ำก่อน unlock ได้
    FLIP_LOCK_MAX_MINUTES: int = 240    # TTL (4 ชม.) — auto-expire ไม่ว่าราคาจะ retrace หรือไม่

    # === Daily Profit Cap (v8.0.17 — Option D Hard Stop) ===
    # หยุดเทรดเมื่อกำไรวันนี้ ถึง DAILY_PROFIT_CAP_PCT × INITIAL balance.
    # นับจาก initial balance (anchor ของ challenge) — ไม่ใช่ daily start
    # → $10k port: cap = $160 ทุกวัน, $100k port: cap = $1600 ทุกวัน
    # Trigger ใช้ closed P/L + floating P/L (Option B)
    # Reset = broker EET midnight (Option A) ผ่าน _on_new_day
    # Action = ปิดทุก open position + block new trades จนกว่าวันใหม่
    # v8.0.17: 1.6% (เพิ่มจาก 1.5% เป็น 0.1pp buffer สำหรับ slippage + spread + commission)
    DAILY_PROFIT_CAP_ENABLED: bool = True
    DAILY_PROFIT_CAP_PCT: float = 0.016     # 1.6% ของ initial balance

    # === Daily Loss Cap (v8.0.22 — Option D mirror) ===
    # ห้ามเทรดเมื่อ daily P/L ≤ -DAILY_LOSS_CAP_PCT × INITIAL balance.
    # Symmetric กับ profit cap → $10k port: -$300 ทุกวัน, $100k port: -$3000
    # Trigger ใช้ closed + floating (เหมือน profit cap)
    # Reset = broker EET midnight ผ่าน _on_new_day
    # Action = ปิดทุก position + block new trades จนกว่าวันใหม่
    # Buffer to FTMO 4% Daily DD: 1% = $100 (กัน slippage + spread)
    DAILY_LOSS_CAP_ENABLED: bool = True
    DAILY_LOSS_CAP_PCT: float = 0.030       # 3% ของ initial balance (= $300/$10k)
    
    # === จำนวนวันเทรดขั้นต่ำ ===
    MIN_TRADING_DAYS: int = 4  # ต้องเทรดอย่างน้อย 4 วัน

    # === Anti-Overtrading Guardrails ===
    MAX_TRADES_PER_DAY: Optional[int] = None    # None = ปิด cap (ใช้ filter ชั้นอื่นคุม); revert: ใส่เลขกลับ (เช่น 5)
    MAX_CORRELATED_POSITIONS: int = 1           # 1 ตำแหน่งต่อกลุ่ม correlation
    # v7.1.10b (2026-05-05): 70 → 65 ทดลองผ่อน filter — eval Pass 4.9% (worse than v7.1.9 6.1%)
    #   borderline signals ทำให้ agent ระมัดระวังเกิน + Daily DD แตะ 4.00% (FTMO limit)
    # v7.1.11 (2026-05-05): 65 → 70 — กลับ filter เข้ม + push reward ดัน TAKE
    MIN_CONFLUENCE_SCORE: float = 70.0          # เกณฑ์ confluence ขั้นต่ำ (ปรับได้ runtime)

    # === FTMO Consistency Rule ===
    # FTMO 2-step Standard: ไม่มีกฎนี้ → ตั้ง 1.0 เพื่อปิด check (ratio จะไม่เกิน 100% ตลอด)
    # FTMO Swing/Pro: มีกฎ max day ≤ 50% ของ total profit → ใช้ 0.45 (buffer)
    # หากย้าย program ในอนาคต — เปลี่ยนค่านี้ให้ตรงกับ program จริง
    CONSISTENCY_RULE_THRESHOLD: float = 1.0     # 2-step Standard ไม่มีกฎนี้
    # เริ่มบังคับใช้เมื่อ total_profit ≥ 2% ของ initial (ต่ำกว่านี้ข้าม เพื่อไม่ block วันแรกๆ)
    CONSISTENCY_MIN_PROFIT_PCT: float = 0.02

    # === ML Quality Filter (live ↔ training distribution sync, v6.12) ===
    # Live ต้องบังคับ ML threshold ให้ตรงกับค่า --ml_threshold ที่ใช้ตอน train
    # v7.1.3 (2026-05-04): tried 0.40 — eval Pass 1.1% (regression). Pool effective ลด → agent ไม่มี data พอ
    # v7.1.6 (2026-05-05): revert 0.40 → 0.36 — combine กับ pool 10k (v7.1.5) + reward soft (v7.1.2):
    #   v7.1.2 (4.5k pool, 0.36, soft reward) = Pass 3.0% best. v7.1.5 (10k, 0.40, hard reward) = Pass 0.8% worst
    #   v7.1.6 (10k, 0.36, soft reward) = expected Pass 4-7% (combine "best of all")
    # หากปรับค่านี้ → ต้อง retrain RL ด้วย --ml_threshold ตรงกัน (training-live sync invariant #10)
    # v8.0.3 (2026-05-06): 0.40 → 0.30 to match MR trainer default (lowered
    # after iter 1+2 showed agent under-trading at 0.40). Parity rule: live
    # must filter signals with the same threshold the agent trained under.
    ML_FILTER_THRESHOLD: float = 0.30

    # === Cooldown / Anti-Revenge-Trading ===
    # หลังโดน SL บนคู่เงินใดคู่เงินหนึ่ง ต้องรอกี่นาทีก่อนเปิดคู่เดิมอีกครั้ง
    # ตั้ง 60 นาที (2026-04-23): กัน "BUY→SL→BUY อีก 1 ชม." pattern ที่เจอใน USDJPY
    COOLDOWN_AFTER_LOSS_MIN: int = 60
    # Per-symbol override — ถ้าต้องการให้บางคู่ cooldown นานกว่า 60 นาที
    COOLDOWN_OVERRIDE_MIN: Dict[str, int] = field(default_factory=lambda: {})
    # ถ้าแพ้ติดกันกี่ครั้งให้ pause ทั้งระบบ
    # v6.13 (2026-04-29): 2 → 3 — risk 0.7 % × 2 = 1.4 % DD trigger ถี่เกิน vs FTMO 4 % limit
    #                     ผ่อนเป็น 3 = 2.1 % DD trigger ยังห่าง 4 % แต่ลด over-pause ที่กิน signals
    CONSECUTIVE_LOSS_PAUSE_COUNT: int = 3
    CONSECUTIVE_LOSS_PAUSE_MIN: int = 60
    # ถ้าแพ้ติดกันกี่ครั้งให้ halt ทั้งวัน
    # v6.13: 3 → 4 — รักษา invariant (PAUSE_COUNT < HALT_COUNT) หลังเพิ่ม pause เป็น 3
    CONSECUTIVE_LOSS_HALT_COUNT: int = 4
    # Loss ขนาดเล็ก (< X% ของ daily start equity) ไม่นับเป็น consecutive loss
    # เหตุผล: tick/spread noise ไม่ใช่ decision error → ไม่ควร trigger anti-revenge
    # ค่า 0.0005 = 0.05% = $5 บน $10K
    MIN_LOSS_TO_COUNT_PCT: float = 0.0005

    # === Unrealized DD Circuit Breaker (v7.1) ===
    # เบรกเกอร์ใหม่: ถ้า floating loss สะสม ≤ UNREALIZED_PAUSE_PCT (% ของ daily_start_balance)
    # และมี open positions ≥ UNREALIZED_PAUSE_MIN_OPEN → block การเปิด trade ใหม่
    # เหตุผล (v7.1 RCA 2026-05-04): ConsecutiveLoss counter นับเฉพาะ closed losses → 4 trades
    # bleed พร้อมกันได้โดย counter ยังเป็น 0. UNREALIZED_PAUSE = gate เพิ่มเติมที่ track real-time risk
    # ค่า −1.5%: 1.5% floating + 1 trade ≈ 2% effective DD = ห่าง FTMO 4% Daily limit ~50%
    UNREALIZED_PAUSE_PCT: float = -1.5  # % ของ daily_start_balance (เป็นลบ)
    UNREALIZED_PAUSE_MIN_OPEN: int = 2  # ต้อง open ≥ 2 ก่อน gate ถึง active

    # === Cross-group USD Theme Guard (v7.1) ===
    # MAX_CORRELATED_POSITIONS เดิมเช็คเฉพาะภายใน group เดียว (USD_STRONG, USD_WEAK ฯลฯ)
    # ปัญหาที่เจอ 2026-05-04: USDCHF SELL (USD_STRONG, effective −1) + NZDUSD BUY (USD_WEAK,
    # effective +1) = ทั้งคู่ short USD แต่อยู่คนละ group → guard ปล่อยผ่าน
    # MAX_USD_THEME_POSITIONS = cap "ฝั่ง USD เดียวกัน" ข้าม group (USD_SHORT/USD_LONG virtual)
    MAX_USD_THEME_POSITIONS: int = 2

    # === Spread/ATR Liquidity Guard (v7.1, relaxed v7.1.1, v7.1.10) ===
    # ถ้า spread > SPREAD_ATR_RATIO_LIMIT × ATR → reject signal (สภาพคล่องแย่ / ตลาดเปิดผันผวน)
    # v7.1.1 (2026-05-04): 0.20 → 0.30 — เดิมตึงเกิน ตัด pool 90% (Trade 3 GBPJPY 91% ratio
    # ยังถูก block ที่ 30%, แต่ normal-spread signals กลับมา)
    # v7.1.10b (2026-05-05): 0.30 → 0.40 — Pass drop 6.1% → 4.9% (borderline signals confuse agent)
    # v7.1.11 (2026-05-05): 0.40 → 0.30 — กลับเข้มเดิม + push reward แทน
    SPREAD_ATR_RATIO_LIMIT: float = 0.30

    # === Post-TP Pullback Lock ===
    # หลัง TP hit บน (symbol, direction) ใดก็ตาม block การเข้าทิศเดิมใน symbol นั้น
    # จนกว่า price เด้งกลับ (เกิน TP + ATR buffer) หรือแตะ EMA20 M15 หรือ TTL หมด
    # เหตุผล: หลัง TP ราคา extended แล้ว เข้าทิศเดิมทันที = ไล่จับก้นเหว
    POST_TP_LOCK_TTL_MIN: int = 60                # TTL 60 นาทีหลัง TP hit
    POST_TP_ATR_BUFFER: float = 0.3               # ต้อง pullback ≥ 0.3×ATR จาก TP price
    POST_TP_EMA_PERIOD: int = 20                  # EMA20 (คำนวณบน M15)
    POST_TP_EMA_TIMEFRAME: str = "M15"
    POST_TP_EMA_LOOKBACK_BARS: int = 3            # เช็ค EMA touch ใน 3 bars ล่าสุด (~45 นาที)


# =============================================================================
# 💱 การตั้งค่าคู่เงินและ Symbol ที่เทรด
# =============================================================================
@dataclass
class SymbolConfig:
    """
    การตั้งค่าคู่เงินที่ Bot จะเทรด
    เลือกคู่เงินที่มี Spread ต่ำและสภาพคล่องสูง
    """
    
    # คู่เงินหลักที่เทรด (Major Pairs — Spread ต่ำ) + Gold
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
        "XAUUSD",   # Gold / US Dollar (metal — digits=2, contract=100 oz)
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
        "XAUUSD": 80,   # Gold spread ~50-80 points (0.5-0.8 USD)
    })

    # === Per-Symbol Overrides (v6.2, 2026-04-23 — ATR floor + MIN_SL guard) ===
    # atr_floor_pips: signal accepted เมื่อ ATR ≥ floor (กันช่วง dead market)
    # min_sl_pips: SL guard — ไม่ให้ SL แคบเกินไปจน spread กิน > 15% ของ SL
    #   (ป้องกัน EURUSD SL 5 pips แบบที่เจอ 23 Apr 2026)
    # spread_atr_ratio_max: skip signal ถ้า spread/ATR > ratio
    symbol_overrides: Dict[str, Dict] = field(default_factory=lambda: {
        # === Major FX (pip = 0.0001) ===
        "EURUSD": {"atr_floor_pips": 4, "min_sl_pips": 10},
        "GBPUSD": {"atr_floor_pips": 4, "min_sl_pips": 12},
        "AUDUSD": {"atr_floor_pips": 4, "min_sl_pips": 10},
        "USDCAD": {"atr_floor_pips": 3, "min_sl_pips": 12},
        "USDCHF": {"atr_floor_pips": 3, "min_sl_pips": 12},
        "NZDUSD": {"atr_floor_pips": 3, "min_sl_pips": 12},
        # === JPY pairs (pip = 0.01) ===
        "USDJPY": {"atr_floor_pips": 5, "min_sl_pips": 10},
        "EURJPY": {
            "atr_floor_pips": 6,
            "min_sl_pips": 15,
            "spread_atr_ratio_max": 0.6,
            "flip_lock_retrace_mult": 0.6,
        },
        "GBPJPY": {
            "atr_floor_pips": 8,             # recent median 9.9 pips → filter 15-20% ของเวลา
            "min_sl_pips": 20,               # spread 5 = 25% of SL 20 (acceptable)
            "min_confidence": 0.65,
            "spread_atr_ratio_max": 0.5,
            "max_trades_per_day": 3,
            "flip_lock_retrace_mult": 0.7,
        },
        # === Metals (pip = 0.01, ticks) ===
        "XAUUSD": {
            "atr_floor_pips": 500,           # recent median 892 ticks → filter ช่วง Gold เงียบ
            # v6.14: 300→1000 ticks ($3→$10) — กัน OB clamp + SL floor ลด SL ต่ำเกินไป
            # (live trade #0 [437211678] โดน SL hit ใน 12s เพราะ guard เก่า $3 ต่ำเกินสำหรับ Gold ATR 8-15 USD)
            "min_sl_pips": 1000,             # 1000 ticks = $10 SL min ≈ 1.0×ATR ปกติ
            "spread_atr_ratio_max": 0.7,
            "flip_lock_retrace_mult": 0.6,
            # v6.13: per-symbol SL/TP multipliers — XAU wick noise สูงกว่า FX
            # 1.5×ATR (global default) แคบเกิน → เคยโดน SL hit by 3 ticks (ticket 436840790)
            # 1.8×ATR + RR คงที่ 1:2 → SL ห่าง wick พอ, lot ↓ แต่ expectancy คล้ายเดิม
            "sl_atr_multiplier": 1.8,
            "tp_atr_multiplier": 3.6,
        },
    })


# =============================================================================
# ⏰ การตั้งค่าช่วงเวลาเทรด (Trading Sessions)
# =============================================================================
@dataclass
class SessionConfig:
    """
    กำหนดช่วงเวลาเทรด + กฎ FTMO compliance.

    v8.0.6 (2026-05-07) — SMC-era "block Asia session" guidance was REMOVED.
    Mean Reversion thrives in ranging markets (= Asia session). Pool eval
    shows Asia has the HIGHEST WR (47.1%) of all sessions. The bot now
    trades 24/5 — only `enforce_daily_close` + `friday_force_close` +
    `rollover_*` + news blackout limit when trades open.

    `_compute_current_session()` in main.py still labels current session
    for Excel logging only — not used as a filter.
    """

    # === ห้ามเทรดช่วงข่าวสำคัญ ===
    # ก่อนข่าวสำคัญ (NFP, FOMC, CPI) — ต้อง implement แยก
    no_trade_before_news_minutes: int = 30   # หยุดเทรด 30 นาที ก่อนข่าว
    no_trade_after_news_minutes: int = 15    # หยุดเทรด 15 นาที หลังข่าว
    
    # === FTMO Friday Rule (บังคับปิดออเดอร์วันศุกร์) ===
    # กฎ FTMO: ห้ามถือออเดอร์ข้ามสัปดาห์ในสุดสัปดาห์
    # เราจะปิดบังคับวันศุกร์เวลา 20:45 น. ตามเวลา Server (EET)
    friday_force_close: time = time(20, 45)

    # === Zero-Overnight Policy (FTMO-compliant) ===
    # ปิดทุก position ก่อนข้ามวัน Mon-Thu (Friday ใช้ friday_force_close)
    # เหตุผล: ไม่มี swap fee, ไม่มี gap risk, สอดคล้องกับ FTMO swing rule
    # กำหนด 23:30 EET = 25 นาทีก่อน rollover → มี buffer ปิดทันเวลา
    # Buffer กับ NY session end: SMC ปิดสัญญาณที่ 19:00 UTC (~21:00-22:00 EET)
    # → late trade มี 1.5-2.5 ชม. ให้ hit TP/SL ก่อนถูก force close
    enforce_daily_close: bool = True
    daily_close_time: time = time(23, 30)   # EET

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
# 🔮 การตั้งค่า ML / Forecaster (v7 — Chronos 2)
# =============================================================================
@dataclass
class MLConfig:
    """
    การตั้งค่าโมเดล ML / Forecaster สำหรับ obs feature engineering

    ⚠️ Chronos model name + prediction_length + context_length
       ต้องเหมือนกันทั้ง training (StrategyBacktester) และ live (main.py)
       มิฉะนั้น obs distribution shift → silent regression
    """

    # Amazon Chronos 2 zero-shot forecaster — obs[27] chronos_alignment, obs[28] chronos_uncertainty_norm
    CHRONOS_MODEL_NAME: str = "amazon/chronos-bolt-small"
    CHRONOS_DEVICE: str = "cpu"
    CHRONOS_PREDICTION_LENGTH: int = 8       # M15 × 8 ≈ 2 hours ahead
    CHRONOS_CONTEXT_LENGTH: int = 512        # M15 × 512 ≈ 5.3 days history
    CHRONOS_ENABLED: bool = True             # set False เพื่อ disable Chronos (fallback obs[27,28] = 0)


# =============================================================================
# 🔄 Mean Reversion Strategy Config (v8.0 pivot)
# =============================================================================
@dataclass
class MeanReversionConfig:
    """Tunables for the v8.0 Mean Reversion strategy.

    The auto-train pipeline can override these via env or per-symbol overrides.
    Live `FTMOTradingBot` keeps SMC by default; flip `strategy_mode` to
    "mean_reversion" only after the MR pipeline meets eval gates.
    """

    # Strategy selection. v8.0 default = "mean_reversion" (full pivot).
    # SMC code is preserved in `strategy/smc_strategy.py` as a deprecated
    # reference module but the live entry no longer routes through it.
    strategy_mode: str = "mean_reversion"

    # BB / RSI entry rules — v8.0.2 relaxed (sync with strategy class defaults)
    # Original spec was tight (0.10/0.90, RSI 30/70, wick 1.2) — produced only
    # ~5 sig/ep. Pilot v4 relaxed to give RL enough variety: ~14 sig/ep median
    # with healthy WR. Class defaults in MeanReversionStrategy match these.
    bb_period: int = 20
    bb_std: float = 2.0
    bb_oversold: float = 0.30        # was 0.10 — wider oversold gate
    bb_overbought: float = 0.70      # was 0.90 — wider overbought gate
    rsi_period: int = 14
    rsi_oversold: float = 40.0       # was 30 — looser
    rsi_overbought: float = 60.0     # was 70 — looser

    # Trend filter (ADX H1) — > threshold blocks MR entries
    adx_trend_block: float = 30.0    # was 25 — only block extreme trends
    # v8.0.27: per-symbol ADX threshold for XAUUSD (tighter than default) — REVERTED v8.0.34
    # v8.0.34 (2026-05-18): 27 → 30 (กลับ default) — v8.0.28 conf floor 70 = safety net ชั้น 2
    # Analysis 14 XAU trades: ไม้แพ้ -$103 (5/15) Conf=48 → ถูก v8.0.28 block อยู่แล้ว.
    # Conf ≥ 70 group: 7W/2L, WR 78%, Net +$245. ADX 27 block ทำให้ XAU = 0 signals 3 วัน.
    adx_trend_block_xau: float = 30.0

    # v8.0.28: Confluence quality floor — LIVE filter (analysis 48 trades 2026-05-12..15)
    # Data showed: < 70 = WR 33-50% loss, ≥ 70 = WR 76% +$425.
    # Block 23/48 trades, net +$425, lose 2 winning XAU trades but gain 3 losses prevented.
    # v8.0.29: split training/live — training keeps wide pool for diverse signal exposure,
    # live narrows down to only A-grade signals (model already learned to like them).
    # v8.0.35 (2026-05-19): 70 → 75 — analysis 6 days / 44 trades (5/12-5/19):
    #   Conf 70-74.99: 24 trades, Net +$12 (avg +$0.51) — break-even with high variance
    #   Conf ≥75: 20 trades, Net +$539 (avg +$26.94) — ALL profit from this zone
    #   Cut 70-74.99 = same Net, half DD, smoother daily returns
    min_confluence_score: float = 75.0          # live (used in main.py path)
    min_confluence_score_training: float = 30.0  # training pool (kept wide for diversity)

    # v8.0.29: ADX threshold split (live tighter than training)
    # Training: default 30/30 (any-symbol, wide pool)
    # Live: 30 FX / 27 XAU (tighter on XAU based on 14-trade analysis 2026-05-15)
    adx_trend_block_training: float = 30.0       # training pool (both XAU + FX)
    adx_trend_block_xau_training: float = 30.0   # training pool — XAU same as FX

    # SL / TP
    sl_atr_mult: float = 1.0     # tight SL — capital preservation
    rr_ratio: float = 1.0        # 1:1 quick-TP target
    min_reversal_wick_ratio: float = 0.4   # was 1.2 — RL learns to filter

    # Reward shaping defaults (env constructor uses these)
    quick_tp_bars: int = 5
    quick_tp_bonus: float = 0.50
    slow_win_bonus: float = 0.20
    prolonged_loss_bars: int = 12
    prolonged_loss_penalty: float = 0.40
    base_loss_penalty: float = 0.10
    duration_fine_coef: float = 0.02


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
    ml: MLConfig = field(default_factory=MLConfig)
    mr: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # === โหมดดีบัก / Verbose ===
    # v8.0.30: console verbose level (Option A — quiet mode)
    # 0 = SILENT  (errors + critical only — สำหรับ VPS หลังบ้าน)
    # 1 = NORMAL  (trade events + warnings + daily summary — default ใช้จริง)
    # 2 = DEBUG   (เห็นทุกอย่าง รวม scan/calc detail — สำหรับ dev)
    verbose_level: int = 1
    debug_mode: bool = True             # legacy flag (true เพื่อ backward compat)
    
    # === ช่วงเวลาตรวจสอบ (วินาที) ===
    main_loop_interval: int = 5         # ตรวจสอบทุก 5 วินาที
    risk_check_interval: int = 1        # ตรวจสอบความเสี่ยงทุก 1 วินาที


# =============================================================================
# 🔧 สร้าง Instance สำหรับใช้งานทั่วทั้งระบบ (Singleton Pattern)
# =============================================================================

# สร้าง Config ตัวหลักที่ทุกโมดูลจะ import ไปใช้
# ใช้: from config.settings import bot_config
bot_config = BotConfig()

# Environment-based quiet mode — ปิด debug prints ตอน training
# Training scripts ตั้ง SMC_QUIET=1 ก่อน import เพื่อ silence strategy logs
# ใน subprocess (SubprocVecEnv) env var inherit จาก parent → propagate อัตโนมัติ
import os as _os
if _os.environ.get("SMC_QUIET", "0") == "1":
    bot_config.debug_mode = False


def get_symbol_config(symbol: str, key: str, default):
    """
    ดึงค่า config ที่ override เฉพาะ symbol

    ใช้ใน strategy / risk_manager สำหรับคู่ volatile (GBPJPY, EURJPY, XAUUSD)
    ที่ต้องมีเงื่อนไขเข้มกว่าคู่ทั่วไป

    Args:
        symbol: ชื่อคู่เงิน เช่น "GBPJPY"
        key: ชื่อ config เช่น "atr_floor_pips", "max_trades_per_day"
        default: ค่า fallback ถ้าไม่มี override

    Returns:
        ค่า override ถ้ามี, ไม่งั้น default
    """
    overrides = bot_config.symbols.symbol_overrides.get(symbol, {})
    return overrides.get(key, default)


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
