"""
===============================================================================
FTMO Trading Bot — ระบบจัดการความเสี่ยง (Risk Manager)
===============================================================================
โมดูลที่สำคัญที่สุดของ Bot — ป้องกันการฝ่าฝืนกฎ FTMO อย่างเด็ดขาด

กลไกป้องกัน:
1. Daily Hard Stop  (4%):  ปิดทุกออเดอร์ + หยุดเทรดทั้งวัน
2. Max Drawdown     (8%):  หยุด Bot ทั้งหมดจนกว่าจะรีเซ็ตด้วยมือ
3. Per-Trade Risk   (0.5-1%): ตรวจสอบก่อนเปิดทุกออเดอร์
4. Position Limit   (3):   จำกัดจำนวน Position เปิดพร้อมกัน
5. Risk:Reward Check (1:1.5): ปฏิเสธเทรดที่ RR ต่ำเกินไป

⚠️  โมดูลนี้มีสิทธิ์สูงสุดในการปิดทุก Position และหยุดการทำงานของ Bot
    กลยุทธ์ (Strategy) ไม่สามารถ Override การตัดสินใจของ Risk Manager ได้
===============================================================================
"""

import json
import os
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple
from enum import Enum

from config.settings import bot_config
from core.mt5_connector import MT5Connector
from core.time_manager import TimeManager


class BotState(Enum):
    """
    สถานะการทำงานของ Bot
    ใช้ควบคุมว่า Bot สามารถเทรดได้หรือไม่
    """
    ACTIVE = "ACTIVE"                           # ทำงานปกติ — เทรดได้
    DAILY_HALT = "DAILY_HALT"                   # หยุดเทรดวันนี้ (Daily Loss เกิน 4%)
    MAX_DRAWDOWN_HALT = "MAX_DRAWDOWN_HALT"     # หยุดถาวร (Max DD เกิน 8%)
    MANUAL_HALT = "MANUAL_HALT"                 # หยุดด้วยมือ (ผู้ใช้สั่ง)
    DISCONNECTED = "DISCONNECTED"               # ไม่ได้เชื่อมต่อ MT5


class RiskManager:
    """
    ระบบจัดการความเสี่ยงหลักของ FTMO Trading Bot
    
    ทำหน้าที่:
    - ตรวจสอบ Daily Loss ทุก Tick (Floating + Closed P/L)
    - ตรวจสอบ Max Drawdown เทียบกับ Starting Balance
    - ตัดสินใจว่าจะอนุญาตให้เทรดหรือไม่
    - ปิดทุก Position ทันทีเมื่อฝ่าฝืนกฎ
    - บันทึกสถานะลงไฟล์เพื่อ Resume หลัง Restart
    
    ⚠️ Risk Manager มีสิทธิ์เหนือทุกโมดูล — สามารถปิด Position ได้ทันที
    """

    def __init__(self, connector: MT5Connector):
        """
        เริ่มต้น Risk Manager
        
        Args:
            connector: ตัวเชื่อมต่อ MT5 สำหรับดึงข้อมูลและส่งคำสั่ง
        """
        self._connector = connector
        self._config = bot_config.ftmo
        
        # === สถานะปัจจุบัน ===
        self._state: BotState = BotState.ACTIVE
        
        # === ค่าอ้างอิงสำหรับคำนวณ Drawdown ===
        self._initial_balance: float = 0.0          # Balance เริ่มต้น (เริ่มใช้ Bot)
        self._daily_start_equity: float = 0.0       # Equity ตอนเริ่มวัน (รีเซ็ตทุกวัน)
        self._daily_start_balance: float = 0.0      # Balance ตอนเริ่มวัน
        self._peak_daily_equity: float = 0.0        # Equity สูงสุดภายในวัน (reset ทุกวัน)
        self._highest_balance: float = 0.0          # Balance สูงสุดที่เคยถึง (High Water Mark)
        self._current_day: date = TimeManager.get_server_time().date()  # วันปัจจุบันตามเวลาโบรกเกอร์
        
        # === สถิติรายวัน ===
        self._daily_closed_pnl: float = 0.0         # P/L ที่ปิดไปแล้ววันนี้
        self._daily_trades_count: int = 0            # จำนวนเทรดวันนี้

        # === Cooldown / Anti-Revenge-Trading State (v2) ===
        # เวลาที่โดน SL ล่าสุดของแต่ละ symbol (ISO string)
        self._last_loss_time_per_symbol: Dict[str, str] = {}
        # จำนวนครั้งที่แพ้ติดกัน (reset ทุกครั้งที่ชนะ)
        self._consecutive_losses: int = 0
        # global halt-until timestamp (ISO) — ใช้ตอนแพ้ติดกันครบ threshold
        self._halt_until: Optional[str] = None

        # === FTMO Consistency Rule (v3) ===
        # เก็บกำไร/ขาดทุนที่ปิดแล้วรายวัน (date_iso → USD)
        # ใช้ตรวจสอบว่า max_day_profit / total_profit ≤ 45%
        self._daily_pnl_history: Dict[str, float] = {}

        # === Challenge Identity (v4) — กัน state หลุดระหว่าง account ===
        self._mt5_login: Optional[int] = None
        self._challenge_start_date: Optional[str] = None

        # === เส้นทางไฟล์สถานะ ===
        self._state_file = bot_config.paths.state_file
        
        print("🛡️ [Risk Manager] เริ่มต้นระบบจัดการความเสี่ยง FTMO")

    # =========================================================================
    # 🔄 การเริ่มต้นและโหลดสถานะ
    # =========================================================================

    def initialize(self) -> bool:
        """
        เริ่มต้น Risk Manager — ดึงข้อมูลบัญชีและตั้งค่าอ้างอิง
        ต้องเรียกหลังจาก MT5 เชื่อมต่อสำเร็จ
        
        ขั้นตอน:
        1. โหลดสถานะจากไฟล์ (ถ้ามี)
        2. ดึง Balance/Equity ปัจจุบัน
        3. ตั้งค่าอ้างอิงสำหรับการคำนวณ
        
        Returns:
            bool: True ถ้าเริ่มต้นสำเร็จ
        """
        # ดึงข้อมูลบัญชี
        account = self._connector.get_account_info()
        if account is None:
            print("❌ [Risk Manager] ไม่สามารถดึงข้อมูลบัญชีได้")
            self._state = BotState.DISCONNECTED
            return False

        current_balance = account["balance"]
        current_equity = account["equity"]
        
        # พยายามโหลดสถานะเก่า
        loaded = self._load_state()

        # === Validation (v4): ตรวจสอบความถูกต้องของ state ก่อนใช้ ===
        if loaded:
            current_login = account.get("login")

            # 1) MT5 login mismatch → account ต่าง → ต้อง re-init
            if self._mt5_login is not None and current_login is not None \
                    and self._mt5_login != current_login:
                print(f"⚠️ [Risk Manager] MT5 Login เปลี่ยน "
                      f"(saved={self._mt5_login}, now={current_login})")
                print("   → บอทคิดว่าเป็น account ใหม่ — reset state")
                loaded = False

            # 2) Balance ห่างจาก saved initial มาก (>20%) → สงสัย account ใหม่ / balance reset
            elif self._initial_balance > 0:
                diff_pct = abs(current_balance - self._initial_balance) / self._initial_balance
                if diff_pct > 0.20:
                    print(f"⚠️ [Risk Manager] Balance ห่างจาก initial มาก "
                          f"(saved=${self._initial_balance:,.2f} vs MT5=${current_balance:,.2f}, "
                          f"diff={diff_pct*100:.1f}%)")
                    print("   → อาจเป็น Challenge ใหม่ / account ใหม่")
                    print("   💡 ถ้าต้องการเริ่ม Challenge ใหม่ ให้ลบไฟล์ logs/bot_state.json")
                    print("   → ใช้ state เดิมต่อ (keep integrity — ห้าม reset อัตโนมัติ)")

        if loaded:
            # ตรวจสอบว่าวันเปลี่ยนหรือยัง (ใช้เวลาโบรกเกอร์ ไม่ใช่เวลาท้องถิ่น)
            broker_today = TimeManager.get_server_time().date()
            if self._current_day != broker_today:
                self._on_new_day(current_balance, current_equity)
            else:
                print(f"📂 [Risk Manager] โหลดสถานะเดิม: สถานะ={self._state.value}")
                if self._challenge_start_date:
                    print(f"   📅 Challenge Start: {self._challenge_start_date}")
                    print(f"   💰 Initial Balance: ${self._initial_balance:,.2f}")

                # ถ้าสถานะเป็น MAX_DRAWDOWN_HALT ต้องหยุดต่อ
                if self._state == BotState.MAX_DRAWDOWN_HALT:
                    print("🚫 [Risk Manager] ⚠️ Max Drawdown ยังเกินอยู่ — Bot หยุดทำงาน")
                    return True

                # ถ้าสถานะเป็น DAILY_HALT แต่วันเปลี่ยนแล้ว → รีเซ็ต
                if self._state == BotState.DAILY_HALT:
                    print("🔒 [Risk Manager] Daily Halt ยังมีผล — รอวันถัดไป")
                    return True
        else:
            # ครั้งแรก — ตั้งค่าใหม่ทั้งหมด
            self._initial_balance = current_balance
            self._highest_balance = current_balance
            self._daily_start_equity = current_equity
            self._daily_start_balance = current_balance
            self._peak_daily_equity = current_equity
            self._current_day = TimeManager.get_server_time().date()
            self._state = BotState.ACTIVE
            self._mt5_login = account.get("login")
            self._challenge_start_date = self._current_day.isoformat()

            print(f"🆕 [Risk Manager] เริ่ม Challenge ใหม่:")
            print(f"   🔐 MT5 Login: {self._mt5_login}")
            print(f"   📅 Start Date: {self._challenge_start_date}")
            print(f"   💰 Initial Balance: ${self._initial_balance:,.2f}")

        # แสดงข้อมูลสรุป
        self._print_risk_status(current_balance, current_equity)
        
        # บันทึกสถานะ
        self._save_state()
        
        return True

    def _on_new_day(self, current_balance: float, current_equity: float):
        """
        รีเซ็ตค่าเมื่อวันใหม่เริ่มต้น
        
        FTMO วัด Daily Loss จาก Equity ตอนเริ่มวัน หรือ Balance (แล้วแต่ค่าไหนมากกว่า)
        
        Args:
            current_balance: Balance ปัจจุบัน
            current_equity: Equity ปัจจุบัน
        """
        broker_today = TimeManager.get_server_time().date()
        print(f"\n🌅 [Risk Manager] === วันใหม่เริ่มต้น: {broker_today} (Broker Time) ===")

        # Finalize วันก่อนลง daily_pnl_history (Consistency Rule)
        yesterday_str = str(self._current_day)
        if self._daily_closed_pnl != 0.0:
            self._daily_pnl_history[yesterday_str] = (
                self._daily_pnl_history.get(yesterday_str, 0.0) + self._daily_closed_pnl
            )

        # FTMO ใช้ค่ามากกว่าระหว่าง Balance กับ Equity ตอนเริ่มวัน
        self._daily_start_equity = max(current_balance, current_equity)
        self._daily_start_balance = current_balance
        self._peak_daily_equity = current_equity
        self._current_day = broker_today
        self._daily_closed_pnl = 0.0
        self._daily_trades_count = 0

        # Reset cooldown / revenge-trading counters ข้ามวัน (รวม weekend rollover)
        # Reason: ขาดทุน 3 ไม้วันศุกร์ไม่ควรทำให้เช้าวันจันทร์ halt ทันที
        if self._consecutive_losses > 0:
            print(f"🧊 [Risk Manager] Reset consecutive_losses ({self._consecutive_losses} → 0) ข้ามวันใหม่")
        self._consecutive_losses = 0
        self._halt_until = None

        # อัพเดท High Water Mark
        if current_balance > self._highest_balance:
            self._highest_balance = current_balance

        # รีเซ็ตสถานะ (ยกเว้น MAX_DRAWDOWN_HALT)
        if self._state == BotState.DAILY_HALT:
            self._state = BotState.ACTIVE
            print("✅ [Risk Manager] รีเซ็ต Daily Halt → กลับมาเทรดได้")
        
        print(f"   📊 Daily Start Equity: ${self._daily_start_equity:,.2f}")
        print(f"   📈 High Water Mark: ${self._highest_balance:,.2f}")
        print(f"   💰 Balance: ${current_balance:,.2f}")

    # =========================================================================
    # 🔍 การตรวจสอบความเสี่ยง (CHECK — เรียกทุก Tick)
    # =========================================================================

    def check_risk(self) -> BotState:
        """
        ตรวจสอบสถานะความเสี่ยงทั้งหมด (เรียกทุก Tick / ทุก Loop)
        
        ลำดับการตรวจสอบ:
        1. ตรวจสอบวันเปลี่ยน
        2. ตรวจสอบ Max Drawdown (ร้ายแรงที่สุด)
        3. ตรวจสอบ Daily Loss
        
        Returns:
            BotState: สถานะปัจจุบันของ Bot
        """
        # ถ้า Bot หยุดถาวร (Max DD) — ไม่ต้องตรวจอะไร
        if self._state == BotState.MAX_DRAWDOWN_HALT:
            return self._state

        # ดึงข้อมูลบัญชีล่าสุด
        account = self._connector.get_account_info()
        if account is None:
            print("⚠️ [Risk Manager] ดึงข้อมูลบัญชีล้มเหลว — หยุดชั่วคราว")
            return BotState.DISCONNECTED

        current_balance = account["balance"]
        current_equity = account["equity"]

        # === ตรวจสอบที่ 1: วันเปลี่ยนหรือยัง? (ใช้เวลาโบรกเกอร์) ===
        if self._current_day != TimeManager.get_server_time().date():
            self._on_new_day(current_balance, current_equity)
            self._save_state()

        # === Track Intraday Peak Equity (สำหรับ warning + dashboard) ===
        # ใช้เตือนการ give-back จากกำไรลอยกลางวัน — ไม่ใช่กฎ FTMO แต่เป็น internal safety
        if current_equity > self._peak_daily_equity:
            self._peak_daily_equity = current_equity
        elif self._peak_daily_equity > 0:
            give_back = self._peak_daily_equity - current_equity
            give_back_pct = give_back / self._peak_daily_equity
            if give_back_pct >= 0.02 and self._state == BotState.ACTIVE:
                print(f"⚠️ [Risk Manager] Give-back จาก peak วันนี้: {give_back_pct:.2%} "
                      f"(peak=${self._peak_daily_equity:,.2f} → now=${current_equity:,.2f})")
            
        # === ตรวจสอบที่ 2: Max Drawdown (ร้ายแรงที่สุด) ===
        max_dd_result = self._check_max_drawdown(current_equity)
        if max_dd_result:
            return self._state

        # === ตรวจสอบที่ 3: Daily Loss ===
        daily_result = self._check_daily_loss(current_equity)
        if daily_result:
            return self._state

        # ผ่านทุกการตรวจสอบ — Bot ทำงานปกติ
        return self._state

    def _check_max_drawdown(self, current_equity: float) -> bool:
        """
        ตรวจสอบ Max Drawdown เทียบกับ Initial Balance
        
        กฎ FTMO: ขาดทุนรวมสูงสุด 10% ของ Balance เริ่มต้น
        Bot ใช้ 8% เป็น Buffer ป้องกัน
        
        หากเกิน → ปิดทุก Position + หยุด Bot ถาวร
        
        Args:
            current_equity: Equity ปัจจุบัน
            
        Returns:
            bool: True ถ้า Max Drawdown เกิน (Bot ต้องหยุด)
        """
        if self._initial_balance <= 0:
            return False

        # คำนวณ Drawdown (เทียบกับ Initial Balance)
        drawdown_amount = self._initial_balance - current_equity
        drawdown_pct = drawdown_amount / self._initial_balance
        
        # ค่าขีดจำกัด
        max_dd_limit = self._config.MAX_DRAWDOWN_HARD_STOP_PCT  # 8%
        
        # แสดงคำเตือนเมื่อเข้าใกล้ขีดจำกัด (เกิน 6%)
        if drawdown_pct > 0.06 and self._state == BotState.ACTIVE:
            print(f"⚠️ [Risk Manager] คำเตือน! Max Drawdown: {drawdown_pct:.2%} (ขีดจำกัด: {max_dd_limit:.0%})")
        
        # ตรวจสอบว่าเกินขีดจำกัดหรือยัง
        if drawdown_pct >= max_dd_limit:
            print("\n" + "=" * 70)
            print("🚨🚨🚨 MAX DRAWDOWN HARD STOP TRIGGERED 🚨🚨🚨")
            print("=" * 70)
            print(f"   💀 Drawdown ปัจจุบัน: {drawdown_pct:.2%}")
            print(f"   🔴 ขีดจำกัด: {max_dd_limit:.0%}")
            print(f"   💰 Initial Balance: ${self._initial_balance:,.2f}")
            print(f"   📉 Equity ปัจจุบัน: ${current_equity:,.2f}")
            print(f"   💸 ขาดทุนรวม: ${drawdown_amount:,.2f}")
            print("=" * 70)
            print("🛑 กำลังปิดทุก Position และหยุด Bot ถาวร...")
            print("=" * 70)
            
            # ⚡ ปิดทุก Position ทันที
            self._emergency_close_all()
            
            # เปลี่ยนสถานะเป็นหยุดถาวร
            self._state = BotState.MAX_DRAWDOWN_HALT
            self._save_state()
            
            return True
            
        return False

    def _check_daily_loss(self, current_equity: float) -> bool:
        """
        ตรวจสอบ Daily Loss (ขาดทุนรายวัน)
        
        กฎ FTMO: ขาดทุนรายวันสูงสุด 5% ของ Equity เริ่มวัน (หรือ Balance แล้วแต่ค่ามากกว่า)
        Bot ใช้ 4% เป็น Buffer ป้องกัน
        
        การคำนวณ:
        Daily Loss = (Daily Start Equity - Current Equity)
        รวมทั้ง Closed P/L และ Floating P/L
        
        Args:
            current_equity: Equity ปัจจุบัน
            
        Returns:
            bool: True ถ้า Daily Loss เกิน (ต้องหยุดเทรดวันนี้)
        """
        if self._state == BotState.DAILY_HALT:
            return True  # หยุดแล้ว — ไม่ต้องตรวจซ้ำ
            
        if self._daily_start_equity <= 0:
            return False

        # คำนวณ Daily Loss (รวมทั้ง Closed + Floating)
        daily_loss_amount = self._daily_start_equity - current_equity
        daily_loss_pct = daily_loss_amount / self._daily_start_equity
        
        # ค่าขีดจำกัด
        daily_limit = self._config.DAILY_LOSS_HARD_STOP_PCT  # 4%
        
        # แสดงคำเตือนเมื่อเข้าใกล้ขีดจำกัด (เกิน 3%)
        if daily_loss_pct > 0.03 and self._state == BotState.ACTIVE:
            print(f"⚠️ [Risk Manager] คำเตือน! Daily Loss: {daily_loss_pct:.2%} (ขีดจำกัด: {daily_limit:.0%})")
        
        # ตรวจสอบว่าเกินขีดจำกัดหรือยัง
        if daily_loss_pct >= daily_limit:
            print("\n" + "=" * 70)
            print("🚨🚨🚨 DAILY LOSS HARD STOP TRIGGERED 🚨🚨🚨")
            print("=" * 70)
            print(f"   📅 วันที่: {TimeManager.get_server_time().date()}")
            print(f"   💀 Daily Loss ปัจจุบัน: {daily_loss_pct:.2%}")
            print(f"   🔴 ขีดจำกัด: {daily_limit:.0%}")
            print(f"   📊 Daily Start Equity: ${self._daily_start_equity:,.2f}")
            print(f"   📉 Equity ปัจจุบัน: ${current_equity:,.2f}")
            print(f"   💸 ขาดทุนวันนี้: ${daily_loss_amount:,.2f}")
            print("=" * 70)
            print("🛑 กำลังปิดทุก Position และหยุดเทรดจนวันพรุ่งนี้...")
            print("=" * 70)
            
            # ⚡ ปิดทุก Position ทันที
            self._emergency_close_all()
            
            # เปลี่ยนสถานะเป็นหยุดรายวัน
            self._state = BotState.DAILY_HALT
            self._save_state()
            
            return True
            
        return False

    # =========================================================================
    # ✅ การตรวจสอบก่อนเปิดออเดอร์ใหม่ (Pre-Trade Checks)
    # =========================================================================

    def can_open_trade(
        self,
        symbol: str,
        risk_amount: float,
        sl_distance_pips: float,
        rr_ratio: float
    ) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าสามารถเปิดออเดอร์ใหม่ได้หรือไม่
        
        ตรวจสอบทุกกฎ FTMO ก่อนอนุญาตให้เทรด:
        1. Bot State ต้องเป็น ACTIVE
        2. จำนวน Position ต้องไม่เกินขีดจำกัด
        3. ความเสี่ยงต่อเทรดต้องอยู่ในช่วง 0.5-1%
        4. Risk:Reward ต้อง >= 1:1.5
        5. Daily Remaining Loss ต้องเพียงพอ
        
        Args:
            symbol: คู่เงินที่จะเทรด
            risk_amount: จำนวนเงินที่เสี่ยง (USD)
            sl_distance_pips: ระยะ SL (pips)
            rr_ratio: อัตราส่วน Risk:Reward
            
        Returns:
            Tuple[bool, str]: (อนุญาตหรือไม่, เหตุผล)
        """
        # === ตรวจสอบที่ 1: สถานะ Bot ===
        if self._state != BotState.ACTIVE:
            return (False, f"❌ Bot ไม่พร้อมเทรด: สถานะ={self._state.value}")

        # === ตรวจสอบที่ 1.1: Global Pause (จาก consecutive losses) ===
        halted, hmsg = self.is_global_halted()
        if halted:
            return (False, f"⏸️ {hmsg}")

        # === ตรวจสอบที่ 1.2: Cooldown ต่อ symbol (Anti-Revenge-Trading) ===
        in_cd, cd_msg = self.is_symbol_in_cooldown(symbol)
        if in_cd:
            return (False, f"🧊 {cd_msg}")

        # === ตรวจสอบที่ 1.3: Max Trades Per Day (Anti-Overtrading) ===
        max_per_day = getattr(self._config, "MAX_TRADES_PER_DAY", 5)
        if self._daily_trades_count >= max_per_day:
            return (False, f"🚫 เทรดครบ {max_per_day} ครั้งวันนี้แล้ว — หยุดเพื่อไม่ over-trade")

        # === ตรวจสอบที่ 2: จำนวน Position ===
        current_positions = self._connector.get_positions_count()
        if current_positions >= self._config.MAX_OPEN_POSITIONS:
            return (False, f"❌ เปิด Position ครบ {self._config.MAX_OPEN_POSITIONS} ตำแหน่งแล้ว (ปัจจุบัน: {current_positions})")

        # === ตรวจสอบที่ 3: Risk:Reward Ratio ===
        if rr_ratio < self._config.MIN_RISK_REWARD_RATIO:
            return (False, f"❌ Risk:Reward ({rr_ratio:.2f}) ต่ำกว่าขั้นต่ำ ({self._config.MIN_RISK_REWARD_RATIO})")

        # === ตรวจสอบที่ 4: ความเสี่ยงต่อเทรด ===
        account = self._connector.get_account_info()
        if account is None:
            return (False, "❌ ไม่สามารถดึงข้อมูลบัญชีได้")

        balance = account["balance"]
        risk_pct = risk_amount / balance if balance > 0 else 1.0
        
        if risk_pct > self._config.MAX_RISK_PER_TRADE_PCT:
            return (False, f"❌ ความเสี่ยงต่อเทรด ({risk_pct:.2%}) เกิน {self._config.MAX_RISK_PER_TRADE_PCT:.1%}")

        if risk_pct < self._config.MIN_RISK_PER_TRADE_PCT:
            return (False, f"⚠️ ความเสี่ยงต่อเทรด ({risk_pct:.2%}) ต่ำกว่าขั้นต่ำ {self._config.MIN_RISK_PER_TRADE_PCT:.1%}")

        # === ตรวจสอบที่ 5: Daily Remaining Loss Budget ===
        equity = account["equity"]
        daily_loss_so_far = self._daily_start_equity - equity
        daily_loss_limit_amount = self._daily_start_equity * self._config.DAILY_LOSS_HARD_STOP_PCT
        remaining_budget = daily_loss_limit_amount - daily_loss_so_far
        
        if risk_amount > remaining_budget:
            return (False, f"❌ งบเสี่ยงรายวันไม่พอ (เหลือ: ${remaining_budget:,.2f}, ต้องการ: ${risk_amount:,.2f})")

        # === ตรวจสอบที่ 6: SL ต้องมีค่า (ห้ามเทรดโดยไม่มี Stop Loss) ===
        if sl_distance_pips <= 0:
            return (False, "❌ ห้ามเทรดโดยไม่มี Stop Loss! (SL distance = 0)")

        # === ตรวจสอบที่ 7: FTMO Consistency Rule ===
        # สมมติ best case: trade ชนะเต็ม RR → วันนี้ได้เพิ่มเท่าไร
        potential_best_profit = risk_amount * rr_ratio
        consistency_ok, consistency_reason = self.check_consistency_rule(potential_best_profit)
        if not consistency_ok:
            return (False, consistency_reason)

        # ✅ ผ่านทุกการตรวจสอบ
        return (True, f"✅ อนุญาตให้เทรด {symbol} (Risk: {risk_pct:.2%}, RR: 1:{rr_ratio:.1f})")

    def check_consistency_rule(self, potential_profit: float = 0.0) -> Tuple[bool, str]:
        """
        ตรวจ FTMO Consistency Rule: วันที่กำไรสูงสุดห้ามเกิน threshold ของ total profit

        Args:
            potential_profit: กำไรสมมติที่อาจได้ถ้า trade นี้ชนะ (ใช้คัดกรอง pre-trade)

        Returns:
            (ok, reason) — ok=True ถ้ายังไม่ละเมิด
        """
        threshold = getattr(self._config, "CONSISTENCY_RULE_THRESHOLD", 0.45)
        min_profit_pct = getattr(self._config, "CONSISTENCY_MIN_PROFIT_PCT", 0.02)
        initial = self._initial_balance or 100_000.0

        # รวม PnL ของวันนี้ (ปิดแล้ว + potential)
        today_str = str(self._current_day)
        today_pnl = self._daily_pnl_history.get(today_str, 0.0) + self._daily_closed_pnl + potential_profit

        # รวม total profit จากทุกวัน + วันนี้
        total_profit = sum(self._daily_pnl_history.values()) + self._daily_closed_pnl + potential_profit
        # ลบ today ที่อยู่ใน history ออก (เพราะรวมซ้ำกับ _daily_closed_pnl)
        total_profit -= self._daily_pnl_history.get(today_str, 0.0)

        # ข้ามการตรวจถ้า total profit ยังต่ำ (ช่วงเริ่มต้น challenge)
        if total_profit <= initial * min_profit_pct:
            return (True, "")

        # ถ้า total ≤ 0 ไม่ต้องตรวจ (ไม่มีกำไรรวม → consistency ไม่เกี่ยว)
        if total_profit <= 0:
            return (True, "")

        # หาวันที่กำไรสูงสุด (รวม today ด้วย)
        all_days = dict(self._daily_pnl_history)
        all_days[today_str] = today_pnl  # override today ด้วยค่าล่าสุด

        max_day_profit = max(all_days.values())
        if max_day_profit <= 0:
            return (True, "")

        ratio = max_day_profit / total_profit
        if ratio > threshold:
            return (
                False,
                f"📏 Consistency Rule: วันที่กำไรสูงสุด ${max_day_profit:,.0f} = "
                f"{ratio:.0%} ของ total ${total_profit:,.0f} (limit: {threshold:.0%})"
            )
        return (True, "")

    def get_remaining_daily_budget(self) -> float:
        """
        คำนวณงบเสี่ยงที่เหลือสำหรับวันนี้
        
        Returns:
            float: จำนวนเงิน (USD) ที่ยังเสี่ยงได้อีกวันนี้
        """
        equity = self._connector.get_equity()
        daily_loss_so_far = self._daily_start_equity - equity
        daily_loss_limit = self._daily_start_equity * self._config.DAILY_LOSS_HARD_STOP_PCT
        return max(0, daily_loss_limit - daily_loss_so_far)

    # =========================================================================
    # 🚨 การปิด Position ฉุกเฉิน
    # =========================================================================

    def _emergency_close_all(self):
        """
        ปิดทุก Position ทันที — ใช้เมื่อฝ่าฝืนกฎ FTMO
        
        ⚠️ ฟังก์ชันนี้ไม่มี Confirmation — ปิดทันที
        ถ้าปิดไม่สำเร็จ จะพยายามปิดซ้ำ 3 ครั้ง
        """
        print("🚨 [Risk Manager] === EMERGENCY: ปิดทุก Position ทันที ===")
        
        success, failed = self._connector.close_all_positions()
        
        if failed > 0:
            print(f"⚠️ [Risk Manager] ปิดไม่สำเร็จ {failed} Position — พยายามใหม่...")
            # ลองอีกครั้ง
            for retry in range(3):
                remaining_positions = self._connector.get_positions_count()
                if remaining_positions == 0:
                    break
                print(f"🔄 [Risk Manager] ลองปิดซ้ำครั้งที่ {retry + 1}...")
                self._connector.close_all_positions()
        
        remaining = self._connector.get_positions_count()
        if remaining > 0:
            print(f"❌ [Risk Manager] ⚠️ ยังมี {remaining} Position เปิดอยู่! ต้องปิดด้วยมือ!")
        else:
            print("✅ [Risk Manager] ปิดทุก Position สำเร็จ")

    def force_halt(self, reason: str = "Manual halt"):
        """
        บังคับหยุด Bot ด้วยมือ (Manual Halt)
        
        Args:
            reason: เหตุผลที่หยุด
        """
        print(f"\n🛑 [Risk Manager] === MANUAL HALT: {reason} ===")
        self._emergency_close_all()
        self._state = BotState.MANUAL_HALT
        self._save_state()

    def resume(self) -> bool:
        """
        กลับมาทำงานต่อหลังจากหยุดด้วยมือ (Manual Halt)
        
        ⚠️ ไม่สามารถ Resume ได้ถ้า Max Drawdown เกิน
        
        Returns:
            bool: True ถ้า Resume สำเร็จ
        """
        if self._state == BotState.MAX_DRAWDOWN_HALT:
            print("❌ [Risk Manager] ไม่สามารถ Resume — Max Drawdown เกินแล้ว")
            return False
            
        self._state = BotState.ACTIVE
        self._save_state()
        print("✅ [Risk Manager] กลับมาทำงานต่อ")
        return True

    # =========================================================================
    # 📊 ข้อมูลสถานะ
    # =========================================================================

    @property
    def state(self) -> BotState:
        """สถานะปัจจุบันของ Bot"""
        return self._state
    
    @property
    def is_trading_allowed(self) -> bool:
        """ตรวจสอบว่า Bot อนุญาตให้เทรดหรือไม่"""
        return self._state == BotState.ACTIVE

    @property
    def initial_balance(self) -> float:
        """Balance เริ่มต้น"""
        return self._initial_balance

    @property
    def highest_balance(self) -> float:
        """Balance สูงสุดที่เคยถึง (High Water Mark)"""
        return self._highest_balance

    def get_risk_status(self) -> Dict:
        """
        ดึงข้อมูลสถานะความเสี่ยงทั้งหมดในรูปแบบ Dictionary
        ใช้สำหรับแสดงผลและบันทึก
        
        Returns:
            Dict: ข้อมูลสถานะความเสี่ยง
        """
        account = self._connector.get_account_info()
        balance = account["balance"] if account else 0
        equity = account["equity"] if account else 0
        
        # คำนวณ Drawdown
        overall_dd_pct = (self._initial_balance - equity) / self._initial_balance if self._initial_balance > 0 else 0
        daily_loss_pct = (self._daily_start_equity - equity) / self._daily_start_equity if self._daily_start_equity > 0 else 0
        
        return {
            "state": self._state.value,
            "initial_balance": self._initial_balance,
            "current_balance": balance,
            "current_equity": equity,
            "highest_balance": self._highest_balance,
            "daily_start_equity": self._daily_start_equity,
            "overall_drawdown_pct": overall_dd_pct,
            "daily_loss_pct": daily_loss_pct,
            "remaining_daily_budget": self.get_remaining_daily_budget(),
            "max_dd_limit": self._config.MAX_DRAWDOWN_HARD_STOP_PCT,
            "daily_limit": self._config.DAILY_LOSS_HARD_STOP_PCT,
            "open_positions": self._connector.get_positions_count(),
            "max_positions": self._config.MAX_OPEN_POSITIONS,
            "current_date": str(TimeManager.get_server_time().date()),
            # Consistency Rule
            "consistency_ok": self.check_consistency_rule()[0],
            "daily_pnl_today": self._daily_closed_pnl,
        }

    def _print_risk_status(self, balance: float, equity: float):
        """แสดงสถานะความเสี่ยงบน Console"""
        overall_dd = (self._initial_balance - equity) / self._initial_balance if self._initial_balance > 0 else 0
        daily_loss = (self._daily_start_equity - equity) / self._daily_start_equity if self._daily_start_equity > 0 else 0
        remaining = self.get_remaining_daily_budget()
        
        print(f"\n{'=' * 60}")
        print(f"🛡️ FTMO Risk Dashboard — {TimeManager.get_server_time().date()}")
        print(f"{'=' * 60}")
        print(f"   📊 สถานะ Bot:          {self._state.value}")
        print(f"   💰 Initial Balance:     ${self._initial_balance:,.2f}")
        print(f"   💎 Current Equity:      ${equity:,.2f}")
        print(f"   📈 High Water Mark:     ${self._highest_balance:,.2f}")
        print(f"   📅 Daily Start Equity:  ${self._daily_start_equity:,.2f}")
        print(f"   {'─' * 56}")
        print(f"   📉 Overall Drawdown:    {overall_dd:.2%} / {self._config.MAX_DRAWDOWN_HARD_STOP_PCT:.0%}")
        print(f"   📉 Daily Loss:          {daily_loss:.2%} / {self._config.DAILY_LOSS_HARD_STOP_PCT:.0%}")
        print(f"   💵 Remaining Budget:    ${remaining:,.2f}")
        print(f"   📋 Open Positions:      {self._connector.get_positions_count()} / {self._config.MAX_OPEN_POSITIONS}")
        print(f"{'=' * 60}\n")

    # =========================================================================
    # 💾 บันทึก/โหลดสถานะ (State Persistence)
    # =========================================================================

    def _save_state(self):
        """
        บันทึกสถานะ Bot ลงไฟล์ JSON
        ใช้เพื่อ Resume หลังจาก Bot Restart
        
        ข้อมูลที่บันทึก:
        - สถานะ Bot (ACTIVE/HALT)
        - ค่า Balance อ้างอิง
        - วันที่ปัจจุบัน
        """
        state_data = {
            "state": self._state.value,
            "initial_balance": self._initial_balance,
            "highest_balance": self._highest_balance,
            "daily_start_equity": self._daily_start_equity,
            "daily_start_balance": self._daily_start_balance,
            "peak_daily_equity": self._peak_daily_equity,
            "current_day": str(self._current_day),
            "daily_closed_pnl": self._daily_closed_pnl,
            "daily_trades_count": self._daily_trades_count,
            # --- v2 fields ---
            "last_loss_time_per_symbol": self._last_loss_time_per_symbol,
            "consecutive_losses": self._consecutive_losses,
            "halt_until": self._halt_until,
            # --- v3: Consistency Rule ---
            "daily_pnl_history": self._daily_pnl_history,
            # --- v4: Challenge Identity ---
            "mt5_login": self._mt5_login,
            "challenge_start_date": self._challenge_start_date,
            "schema_version": 4,
            "last_updated": datetime.now().isoformat(),
        }

        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)

            # Atomic write: tmp file + os.replace → ป้องกันไฟล์พังถ้า crash ระหว่างเขียน
            tmp_path = self._state_file + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._state_file)

        except Exception as e:
            print(f"⚠️ [Risk Manager] บันทึกสถานะล้มเหลว: {e}")

    def _load_state(self) -> bool:
        """
        โหลดสถานะ Bot จากไฟล์ JSON
        
        Returns:
            bool: True ถ้าโหลดสำเร็จ
        """
        if not os.path.exists(self._state_file):
            print("ℹ️ [Risk Manager] ไม่พบไฟล์สถานะ — เริ่มต้นใหม่")
            return False

        try:
            with open(self._state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._state = BotState(data.get("state", "ACTIVE"))
            self._initial_balance = data.get("initial_balance", 0.0)
            self._highest_balance = data.get("highest_balance", 0.0)
            self._daily_start_equity = data.get("daily_start_equity", 0.0)
            self._daily_start_balance = data.get("daily_start_balance", 0.0)
            self._peak_daily_equity = data.get("peak_daily_equity", self._daily_start_equity)
            self._daily_closed_pnl = data.get("daily_closed_pnl", 0.0)
            self._daily_trades_count = data.get("daily_trades_count", 0)

            # --- v2 fields (fallback ถ้าเป็นไฟล์เก่า) ---
            self._last_loss_time_per_symbol = data.get("last_loss_time_per_symbol", {}) or {}
            self._consecutive_losses = int(data.get("consecutive_losses", 0) or 0)
            self._halt_until = data.get("halt_until") or None
            # --- v3: Consistency Rule ---
            self._daily_pnl_history = data.get("daily_pnl_history", {}) or {}

            # --- v4: Challenge Identity (fallback สำหรับไฟล์เก่า) ---
            self._mt5_login = data.get("mt5_login")
            self._challenge_start_date = data.get("challenge_start_date")

            # แปลงวันที่
            day_str = data.get("current_day", str(TimeManager.get_server_time().date()))
            self._current_day = date.fromisoformat(day_str)

            schema_v = data.get("schema_version", 1)
            print(f"📂 [Risk Manager] โหลดสถานะจากไฟล์สำเร็จ "
                  f"(schema v{schema_v}, อัพเดทล่าสุด: {data.get('last_updated', 'N/A')})")
            return True
            
        except Exception as e:
            print(f"⚠️ [Risk Manager] โหลดสถานะล้มเหลว: {e}")
            return False

    def update_daily_pnl(self, trade_pnl: float, symbol: Optional[str] = None):
        """
        อัพเดท P/L ที่ปิดไปแล้ววันนี้ (เรียกหลังจาก Position ปิด)

        Args:
            trade_pnl: กำไร/ขาดทุนของเทรดที่ปิด (USD)
            symbol: คู่เงิน (ใช้ track cooldown)
        """
        self._daily_closed_pnl += trade_pnl
        self._daily_trades_count += 1

        # อัพเดท High Water Mark ถ้า Balance สูงขึ้น
        current_balance = self._connector.get_balance()
        if current_balance > self._highest_balance:
            self._highest_balance = current_balance

        # === Cooldown / Revenge-Trading Tracking ===
        self._record_trade_outcome(symbol, trade_pnl)

        self._save_state()

        print(f"📊 [Risk Manager] อัพเดท Daily P/L: ${self._daily_closed_pnl:,.2f} "
              f"(เทรดที่ {self._daily_trades_count} วันนี้, แพ้ติด={self._consecutive_losses})")

    # =========================================================================
    # 🧊 Cooldown / Anti-Revenge-Trading
    # =========================================================================

    def _record_trade_outcome(self, symbol: Optional[str], pnl: float):
        """
        บันทึกผลเทรดเพื่อคำนวณ cooldown

        Logic:
        - ชนะ (pnl > 0): reset consecutive_losses
        - แพ้ (pnl <= 0): เพิ่ม consecutive_losses, บันทึก last_loss_time_per_symbol
        - แพ้ติด N ครั้ง → pause global halt_until
        - แพ้ติด M ครั้ง → DAILY_HALT
        """
        now_iso = TimeManager.get_server_time().isoformat()

        if pnl > 0:
            self._consecutive_losses = 0
            self._halt_until = None  # clear pause เมื่อ win reset counter
            return

        # pnl <= 0 — ข้าม noise-level loss (เช่น tick/spread noise ที่ไม่ใช่ decision error)
        if self._daily_start_equity > 0:
            min_pct = getattr(self._config, "MIN_LOSS_TO_COUNT_PCT", 0.0005)
            loss_pct = abs(pnl) / self._daily_start_equity
            if loss_pct < min_pct:
                return

        self._consecutive_losses += 1
        if symbol:
            self._last_loss_time_per_symbol[symbol] = now_iso

        halt_cnt = getattr(self._config, "CONSECUTIVE_LOSS_HALT_COUNT", 3)
        pause_cnt = getattr(self._config, "CONSECUTIVE_LOSS_PAUSE_COUNT", 2)
        pause_min = getattr(self._config, "CONSECUTIVE_LOSS_PAUSE_MIN", 60)

        if self._consecutive_losses >= halt_cnt:
            print(f"🛑 [Risk Manager] แพ้ติดกัน {self._consecutive_losses} ครั้ง → DAILY HALT")
            self._state = BotState.DAILY_HALT
            self._halt_until = None  # จะรอจนวันเปลี่ยน
        elif self._consecutive_losses >= pause_cnt:
            pause_until = TimeManager.get_server_time() + timedelta(minutes=pause_min)
            self._halt_until = pause_until.isoformat()
            print(f"⏸️ [Risk Manager] แพ้ติด {self._consecutive_losses} → pause {pause_min}min "
                  f"ถึง {pause_until.strftime('%H:%M:%S')}")

    def is_symbol_in_cooldown(self, symbol: str) -> Tuple[bool, str]:
        """
        ตรวจว่า symbol ยังอยู่ใน cooldown หลังโดน SL ล่าสุดหรือไม่

        Returns:
            (is_in_cooldown, reason_str)
        """
        last_iso = self._last_loss_time_per_symbol.get(symbol)
        if not last_iso:
            return (False, "")

        try:
            last_loss = datetime.fromisoformat(last_iso)
        except Exception:
            return (False, "")

        cd_min = getattr(self._config, "COOLDOWN_AFTER_LOSS_MIN", 30)
        now = TimeManager.get_server_time()
        # Legacy state (pre-fix) เก็บเป็น naive → ถือว่าเป็น server time
        if last_loss.tzinfo is None:
            last_loss = last_loss.replace(tzinfo=now.tzinfo)
        elapsed = (now - last_loss).total_seconds() / 60.0
        if elapsed < cd_min:
            remaining = cd_min - elapsed
            return (True, f"Cooldown {symbol}: เหลือ {remaining:.1f} นาที")
        return (False, "")

    def is_global_halted(self) -> Tuple[bool, str]:
        """ตรวจ global halt_until (จาก consecutive losses pause)"""
        if not self._halt_until:
            return (False, "")
        try:
            until = datetime.fromisoformat(self._halt_until)
        except Exception:
            self._halt_until = None
            return (False, "")

        now = TimeManager.get_server_time()
        if until.tzinfo is None:
            until = until.replace(tzinfo=now.tzinfo)
        if now < until:
            remaining = (until - now).total_seconds() / 60.0
            return (True, f"Global pause: เหลือ {remaining:.1f} นาที")

        # หมดเวลา pause แล้ว → clear
        self._halt_until = None
        self._save_state()
        return (False, "")

    def save(self):
        """เรียกภายนอกเพื่อ flush state ลง disk (ใช้ตอน shutdown)"""
        self._save_state()

    # =========================================================================
    # 🧪 ทดสอบ
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"RiskManager(state={self._state.value}, "
            f"init_bal=${self._initial_balance:,.0f}, "
            f"dd_limit={self._config.MAX_DRAWDOWN_HARD_STOP_PCT:.0%}, "
            f"daily_limit={self._config.DAILY_LOSS_HARD_STOP_PCT:.0%})"
        )
