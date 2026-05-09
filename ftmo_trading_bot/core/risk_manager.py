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

from config.settings import bot_config, get_symbol_config
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
        # v8.0.16: throttle "Give-back from peak" alert — print เฉพาะข้าม 1% milestone ใหม่
        # (เดิม print ทุก 5s = spam; reset วันใหม่ + ทุกครั้งที่ peak ใหม่)
        self._last_give_back_alert_pct: float = 0.0
        # v8.0.17: Daily profit cap (Option D Hard Stop) — เมื่อ trigger จะปิดทุก
        # position + block new trades จนกว่าวันใหม่. Anchor ใช้ initial_balance
        # ของ challenge (ไม่ใช่ daily start) → cap คงที่ตลอด challenge.
        self._daily_profit_locked: bool = False

        # === Cooldown / Anti-Revenge-Trading State (v2) ===
        # เวลาที่โดน SL ล่าสุดของแต่ละ symbol (ISO string)
        self._last_loss_time_per_symbol: Dict[str, str] = {}
        # จำนวนครั้งที่แพ้ติดกัน (reset ทุกครั้งที่ชนะ)
        self._consecutive_losses: int = 0
        # global halt-until timestamp (ISO) — ใช้ตอนแพ้ติดกันครบ threshold
        self._halt_until: Optional[str] = None

        # === Post-TP Pullback Lock State (v5) ===
        # key = "SYMBOL|DIRECTION" (เช่น "XAUUSD|SELL"), value = {"tp_price": float, "tp_time": iso}
        # หลัง TP hit lock การเข้าทิศเดิม symbol เดิม จนกว่า pullback จะผ่าน
        self._last_tp_per_symbol_dir: Dict[str, Dict] = {}

        # === Directional Flip-Lock State (v6) — กัน whipsaw BUY↔SELL flip ===
        # key = symbol, value = {closed_direction, close_price, atr_at_close,
        #                         unlock_threshold, min_unlock_time (iso)}
        # หลังปิดไม้ใดๆ (TP/SL) → lock opposite direction จนกว่าราคาจะ retrace ≥ K×ATR
        self._flip_lock: Dict[str, Dict] = {}

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
        self._last_give_back_alert_pct = 0.0   # v8.0.16: clean slate วันใหม่
        self._daily_profit_locked = False      # v8.0.17: reset daily profit cap

        # Reset cooldown / revenge-trading counters ข้ามวัน (รวม weekend rollover)
        # Reason: ขาดทุน 3 ไม้วันศุกร์ไม่ควรทำให้เช้าวันจันทร์ halt ทันที
        if self._consecutive_losses > 0:
            print(f"🧊 [Risk Manager] Reset consecutive_losses ({self._consecutive_losses} → 0) ข้ามวันใหม่")
        self._consecutive_losses = 0
        self._halt_until = None

        # Reset post-TP pullback locks ข้ามวัน — structure ของวันใหม่ไม่เกี่ยวกับ TP วันก่อน
        if self._last_tp_per_symbol_dir:
            print(f"🔓 [Risk Manager] Reset post-TP locks ({len(self._last_tp_per_symbol_dir)} entries) ข้ามวันใหม่")
        self._last_tp_per_symbol_dir = {}

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
        # v8.0.16: throttle ให้ print เฉพาะข้าม 1% milestone ใหม่ (กัน spam ทุก 5s)
        if current_equity > self._peak_daily_equity:
            self._peak_daily_equity = current_equity
            self._last_give_back_alert_pct = 0.0  # reset เมื่อ peak ใหม่
        elif self._peak_daily_equity > 0:
            give_back = self._peak_daily_equity - current_equity
            give_back_pct = give_back / self._peak_daily_equity
            current_milestone = int(give_back_pct * 100)        # 2.5% → 2
            last_milestone = int(self._last_give_back_alert_pct * 100)
            if (give_back_pct >= 0.02
                    and current_milestone > last_milestone
                    and self._state == BotState.ACTIVE):
                print(f"⚠️ [Risk Manager] Give-back จาก peak วันนี้: {give_back_pct:.2%} "
                      f"(peak=${self._peak_daily_equity:,.2f} → now=${current_equity:,.2f})")
                self._last_give_back_alert_pct = give_back_pct


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

    # =========================================================================
    # 🎯 Daily Profit Cap (v8.0.17 — Option D Hard Stop)
    # =========================================================================

    def check_daily_profit_cap(self, current_equity: float, floating_pnl: float) -> bool:
        """
        เช็คว่ากำไรวันนี้ (closed + floating) ถึง cap แล้วหรือยัง.

        ถ้าเพิ่งถึง: ตั้ง _daily_profit_locked = True + คืน True เพื่อให้ caller
        ปิด open positions ทั้งหมด.

        Args:
            current_equity: equity ปัจจุบัน (จาก MT5)
            floating_pnl: P/L ลอย (sum ของ open positions)

        Returns:
            True = เพิ่ง trigger รอบนี้ (close all + block new)
            False = ไม่ถึง cap หรือ locked อยู่แล้ว (no-op)
        """
        if not self._config.DAILY_PROFIT_CAP_ENABLED:
            return False
        if self._daily_profit_locked:
            return False  # locked แล้ว ไม่ต้อง trigger ซ้ำ

        cap_amount = self._initial_balance * self._config.DAILY_PROFIT_CAP_PCT
        # daily P/L = closed P/L วันนี้ + floating ของ open positions
        # closed_pnl วันนี้ = current_equity - daily_start_equity - floating
        # → total_pnl_today = current_equity - daily_start_equity (รวม floating อยู่แล้ว)
        total_pnl_today = current_equity - self._daily_start_equity

        if total_pnl_today >= cap_amount:
            self._daily_profit_locked = True
            self._save_state()
            closed_today = total_pnl_today - floating_pnl
            print(f"🎯 [Risk Manager] Daily profit cap REACHED: "
                  f"+${total_pnl_today:.2f} = "
                  f"closed +${closed_today:.2f} + floating +${floating_pnl:.2f} "
                  f"(cap +${cap_amount:.2f} = {self._config.DAILY_PROFIT_CAP_PCT:.1%} ของ "
                  f"initial ${self._initial_balance:,.2f}) — ปิดทุก position + หยุดเทรดวันนี้")
            return True
        return False

    def is_daily_locked(self) -> bool:
        """ตอบว่าวันนี้ล็อกแล้วหรือไม่ (จาก daily profit cap)"""
        return self._daily_profit_locked

    def can_open_trade(
        self,
        symbol: str,
        risk_amount: float,
        sl_distance_pips: float,
        rr_ratio: float,
        direction: Optional[str] = None,
        atr: Optional[float] = None,
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
        # === ตรวจสอบที่ 0: Daily Profit Cap (v8.0.17 — Option D) ===
        # ถ้า cap ถึงแล้ววันนี้ → block เปิด trade ใหม่จนกว่าวันใหม่
        if self._daily_profit_locked:
            cap_pct = self._config.DAILY_PROFIT_CAP_PCT
            return (False, f"🎯 Daily profit cap ถึง {cap_pct:.1%} แล้ววันนี้ — รอวันใหม่")

        # === ตรวจสอบที่ 1: สถานะ Bot ===
        if self._state != BotState.ACTIVE:
            return (False, f"❌ Bot ไม่พร้อมเทรด: สถานะ={self._state.value}")

        # === ตรวจสอบที่ 1.1: Global Pause (จาก consecutive losses) ===
        halted, hmsg = self.is_global_halted()
        if halted:
            return (False, f"⏸️ {hmsg}")

        # === ตรวจสอบที่ 1.1b: Unrealized DD Circuit Breaker (v7.1) ===
        # ConsecLoss counter นับเฉพาะ closed losses → ปัญหา 2026-05-04: 4 positions bleed
        # พร้อมกันโดย counter ยัง 0. Breaker นี้เช็ค floating drawdown realtime
        breached, br_msg = self.check_unrealized_circuit_breaker()
        if breached:
            return (False, f"⛑️ {br_msg}")

        # === ตรวจสอบที่ 1.2: Cooldown ต่อ symbol (Anti-Revenge-Trading) ===
        in_cd, cd_msg = self.is_symbol_in_cooldown(symbol)
        if in_cd:
            return (False, f"🧊 {cd_msg}")

        # === ตรวจสอบที่ 1.3: Post-TP Pullback Lock (ห้ามเข้าทิศเดิมหลัง TP ทันที) ===
        # ป้องกัน "ไล่จับก้นเหว" — หลัง TP ราคา extended แล้ว entry ใหม่มักโดน SL
        # ปลดล็อกเมื่อ: (a) price เด้ง ≥ ATR buffer จาก TP หรือ (b) EMA20 M15 touched หรือ (c) TTL หมด
        if direction:
            locked, lock_msg = self._is_post_tp_locked(symbol, direction, atr)
            if locked:
                return (False, f"🔒 {lock_msg}")

        # === ตรวจสอบที่ 1.4: Flip-Lock (กัน whipsaw BUY↔SELL) ===
        # ห้ามเปิดทิศตรงข้ามทันทีหลังปิดไม้ล่าสุด จนกว่าราคาจะ retrace ≥ K×ATR
        if direction:
            flip_locked, flip_msg = self.is_flip_locked(symbol, direction)
            if flip_locked:
                return (False, f"🔄 {flip_msg}")

        # === ตรวจสอบที่ 1.3: Max Trades Per Day (Anti-Overtrading) ===
        # None = ปิด cap (filter ชั้นอื่น: cooldown, post-TP lock, consecutive-loss halt, DD halt ยังคุมอยู่)
        # Per-symbol override (เช่น GBPJPY=3) สำคัญสำหรับคู่ volatile — แต่ track เป็น total count
        # ใช้ symbol override ถ้ามี (เพราะคู่ volatile กินงบเทรดทั้งวันเอง)
        default_max = getattr(self._config, "MAX_TRADES_PER_DAY", 5)
        max_per_day = get_symbol_config(symbol, "max_trades_per_day", default_max)
        if max_per_day is not None and self._daily_trades_count >= max_per_day:
            return (False, f"🚫 {symbol}: เทรดครบ {max_per_day} ครั้งวันนี้แล้ว — หยุดเพื่อไม่ over-trade")

        # === ตรวจสอบที่ 2: จำนวน Position ===
        current_positions = self._connector.get_positions_count()
        if current_positions >= self._config.MAX_OPEN_POSITIONS:
            return (False, f"❌ เปิด Position ครบ {self._config.MAX_OPEN_POSITIONS} ตำแหน่งแล้ว (ปัจจุบัน: {current_positions})")

        # === ตรวจสอบที่ 3: Risk:Reward Ratio ===
        # v8.0.11 fix: epsilon tolerance for FP precision. MR strategy stores
        # sl_distance directly but derives tp_distance from rounded prices, so
        # rr_ratio can drift below 1.0 even when designed as 1.0.
        # v8.0.15 fix: tolerance 1e-4 ไม่พอสำหรับ XAUUSD (digits=2 → drift ~0.4%).
        # ใช้ 1% relative tolerance ครอบคลุมทุก symbol (FX 0.001%, JPY 0.04%, XAU 0.4%).
        if rr_ratio < self._config.MIN_RISK_REWARD_RATIO * (1.0 - 0.01):
            return (False, f"❌ Risk:Reward ({rr_ratio:.4f}) ต่ำกว่าขั้นต่ำ ({self._config.MIN_RISK_REWARD_RATIO})")

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
            # --- v5: Post-TP Pullback Lock ---
            "last_tp_per_symbol_dir": self._last_tp_per_symbol_dir,
            # --- v6: Directional Flip-Lock (anti-whipsaw) ---
            "flip_lock": self._flip_lock,
            # --- v7 (v8.0.17): Daily Profit Cap (Option D) ---
            "daily_profit_locked": self._daily_profit_locked,
            "schema_version": 7,
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

            # --- v5: Post-TP Pullback Lock (fallback สำหรับไฟล์เก่า) ---
            self._last_tp_per_symbol_dir = data.get("last_tp_per_symbol_dir", {}) or {}

            # --- v6: Directional Flip-Lock (fallback สำหรับไฟล์เก่า) ---
            self._flip_lock = data.get("flip_lock", {}) or {}

            # --- v7 (v8.0.17): Daily Profit Cap state ---
            self._daily_profit_locked = bool(data.get("daily_profit_locked", False))

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

    def update_daily_pnl(
        self,
        trade_pnl: float,
        symbol: Optional[str] = None,
        reason_code: int = -1,
        direction: Optional[str] = None,
        close_price: Optional[float] = None,
    ):
        """
        อัพเดท P/L ที่ปิดไปแล้ววันนี้ (เรียกหลังจาก Position ปิด)

        Args:
            trade_pnl: กำไร/ขาดทุนของเทรดที่ปิด (USD)
            symbol: คู่เงิน (ใช้ track cooldown)
            reason_code: MT5 DEAL_REASON code (4=SL, 5=TP, 6=SO, 3=EXPERT, 0-2=Manual).
                         -1 = ไม่ทราบ → fallback ใช้ pnl sign
            direction: "BUY" | "SELL" — ใช้ track post-TP lock per symbol+direction
            close_price: ราคาปิดจริง — ใช้เก็บเป็น reference TP price สำหรับ pullback lock
        """
        self._daily_closed_pnl += trade_pnl
        self._daily_trades_count += 1

        # อัพเดท High Water Mark ถ้า Balance สูงขึ้น
        current_balance = self._connector.get_balance()
        if current_balance > self._highest_balance:
            self._highest_balance = current_balance

        # === Cooldown / Revenge-Trading Tracking + Post-TP Lock ===
        self._record_trade_outcome(symbol, trade_pnl, reason_code, direction, close_price)

        self._save_state()

        print(f"📊 [Risk Manager] อัพเดท Daily P/L: ${self._daily_closed_pnl:,.2f} "
              f"(เทรดที่ {self._daily_trades_count} วันนี้, แพ้ติด={self._consecutive_losses})")

    # =========================================================================
    # 🧊 Cooldown / Anti-Revenge-Trading
    # =========================================================================

    # === MT5 DEAL_REASON codes (ต้องตรงกับ trade_executor._DEAL_REASON_MAP) ===
    _DR_CLIENT = 0
    _DR_MOBILE = 1
    _DR_WEB = 2
    _DR_EXPERT = 3
    _DR_SL = 4
    _DR_TP = 5
    _DR_SO = 6  # Stop-out (margin call)

    def _record_trade_outcome(
        self,
        symbol: Optional[str],
        pnl: float,
        reason_code: int = -1,
        direction: Optional[str] = None,
        close_price: Optional[float] = None,
    ):
        """
        บันทึกผลเทรดเพื่อคำนวณ cooldown (ใช้ MT5 DEAL_REASON ตัดสิน)

        Policy:
        - Stop-out (DR_SO) → DAILY_HALT ทันที (disaster recovery)
        - Manual close (DR_CLIENT/MOBILE/WEB) → no-op (user intervention ไม่นับ)
        - TP hit (DR_TP) หรือ ชนะ (pnl>0) → clear cooldown ของ symbol นั้น + reset consecutive
                                             + set post-TP lock (symbol+direction)
        - SL hit / ขาดทุนอื่น → set cooldown, เพิ่ม consecutive, อาจ pause/halt
        - Loss ขนาดเล็ก (< MIN_LOSS_TO_COUNT_PCT) → ไม่นับเป็น decision error (noise)

        reason_code = -1 (ไม่ทราบ) → fallback ใช้ pnl sign เหมือนเดิม
        """
        now_iso = TimeManager.get_server_time().isoformat()

        # === Stop-out = DAILY_HALT ทันที (ข้ามชั้น consecutive) ===
        if reason_code == self._DR_SO:
            print(f"🆘 [Risk Manager] Stop-out detected บน {symbol} (P/L=${pnl:,.2f}) → DAILY_HALT")
            self._state = BotState.DAILY_HALT
            self._halt_until = None
            return

        # === Manual close = no-op (user ปิดเอง ไม่ใช่ decision error ของ bot) ===
        if reason_code in (self._DR_CLIENT, self._DR_MOBILE, self._DR_WEB):
            return

        # === TP hit หรือ ชนะด้วยวิธีอื่น → clear cooldown + reset + set post-TP lock ===
        if reason_code == self._DR_TP or pnl > 0:
            self._consecutive_losses = 0
            self._halt_until = None  # clear global pause
            if symbol:
                self._last_loss_time_per_symbol.pop(symbol, None)  # clear symbol cooldown
            # Set post-TP lock (ป้องกันเข้าทิศเดิมซ้ำทันที)
            # เงื่อนไข: มี symbol + direction + close_price (ไม่ใช่ partial close หรือ manual)
            if symbol and direction and close_price and close_price > 0:
                key = f"{symbol}|{direction}"
                self._last_tp_per_symbol_dir[key] = {
                    "tp_price": close_price,
                    "tp_time": now_iso,
                }
                print(f"🔒 [Risk Manager] Post-TP lock: {symbol} {direction} "
                      f"@ {close_price} (รอ pullback)")
                # Flip-lock (กัน opposite direction whipsaw) — ใช้ ATR ล่าสุดถ้ามี
                atr_hint = self._get_recent_atr(symbol)
                self.register_flip_lock(symbol, direction, close_price, atr_hint, self._DR_TP)
            return

        # === pnl <= 0 (SL hit หรือ bot-close ขาดทุน) ===
        # ข้าม noise-level loss (tick/spread noise ที่ไม่ใช่ decision error)
        if self._daily_start_equity > 0:
            min_pct = getattr(self._config, "MIN_LOSS_TO_COUNT_PCT", 0.0005)
            loss_pct = abs(pnl) / self._daily_start_equity
            if loss_pct < min_pct:
                return

        self._consecutive_losses += 1
        if symbol:
            self._last_loss_time_per_symbol[symbol] = now_iso

        # Flip-lock หลัง SL (กันเปิด opposite direction ทันที)
        if symbol and direction and close_price and close_price > 0:
            atr_hint = self._get_recent_atr(symbol)
            self.register_flip_lock(symbol, direction, close_price, atr_hint, self._DR_SL)

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

    def get_unrealized_drawdown_pct(self) -> float:
        """
        คำนวณ floating drawdown ของทุก open positions เป็น % ของ daily_start_balance.

        Returns:
            float: เป็นบวกถ้า floating profit, เป็นลบถ้า floating loss.
                   ใช้ daily_start_balance เป็น base (ไม่ใช่ initial_balance) เพื่อให้
                   threshold สอดคล้องกับ daily DD limit.
        """
        if self._daily_start_balance <= 0:
            return 0.0
        try:
            floating = self._connector.get_total_floating_pnl()
        except Exception:
            return 0.0
        return (float(floating) / self._daily_start_balance) * 100.0

    def check_unrealized_circuit_breaker(self) -> Tuple[bool, str]:
        """
        v7.1 — Unrealized DD circuit breaker.

        Block การเปิด trade ใหม่เมื่อ:
          - floating drawdown ≤ UNREALIZED_PAUSE_PCT (เช่น −1.5%)
          - AND open positions count ≥ UNREALIZED_PAUSE_MIN_OPEN (เช่น 2)

        เหตุผล: กัน scenario "4 positions bleeding พร้อมกัน" ที่ ConsecLoss counter
        ไม่ทันเพราะมัน count เฉพาะ closed losses.

        Returns:
            (breached, reason)
        """
        threshold_pct = getattr(self._config, "UNREALIZED_PAUSE_PCT", -1.5)
        min_open = getattr(self._config, "UNREALIZED_PAUSE_MIN_OPEN", 2)

        try:
            open_count = self._connector.get_positions_count()
        except Exception:
            open_count = 0

        if open_count < min_open:
            return (False, "")

        fdd_pct = self.get_unrealized_drawdown_pct()
        if fdd_pct <= threshold_pct:
            return (
                True,
                f"Unrealized DD breaker: floating {fdd_pct:.2f}% "
                f"(open={open_count}, threshold={threshold_pct:.2f}%) — pause new entries"
            )
        return (False, "")

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
        # Per-symbol override (เช่น XAUUSD ใช้ cooldown นานกว่า)
        overrides = getattr(self._config, "COOLDOWN_OVERRIDE_MIN", {})
        cd_min = overrides.get(symbol, cd_min)
        now = TimeManager.get_server_time()
        # Legacy state (pre-fix) เก็บเป็น naive → ถือว่าเป็น server time
        if last_loss.tzinfo is None:
            last_loss = last_loss.replace(tzinfo=now.tzinfo)
        elapsed = (now - last_loss).total_seconds() / 60.0
        if elapsed < cd_min:
            remaining = cd_min - elapsed
            return (True, f"Cooldown {symbol}: เหลือ {remaining:.1f} นาที")
        return (False, "")

    def _is_post_tp_locked(
        self,
        symbol: str,
        direction: str,
        atr: Optional[float],
    ) -> Tuple[bool, str]:
        """
        ตรวจ post-TP lock สำหรับ (symbol, direction)

        ปลดล็อกเมื่อ OR condition ข้อใดข้อหนึ่ง:
        (A) price_pullback: สำหรับ SELL lock — current ask > tp + ATR_buffer × ATR
                           สำหรับ BUY lock  — current bid < tp − ATR_buffer × ATR
        (B) ema_touched:    EMA20 M15 อยู่ในช่วง (low, high) ของ bar ใดบาร์หนึ่งในช่วง lookback
        (C) ttl_expired:    elapsed > POST_TP_LOCK_TTL_MIN → ลบ entry และปลดล็อก
        ถ้าปลดล็อกด้วย A หรือ B ให้ลบ entry ออกเพื่อไม่เช็คซ้ำ

        Returns:
            (is_locked, reason_str)
        """
        key = f"{symbol}|{direction}"
        entry = self._last_tp_per_symbol_dir.get(key)
        if not entry:
            return (False, "")

        tp_price = entry.get("tp_price", 0)
        tp_time_iso = entry.get("tp_time", "")
        if tp_price <= 0 or not tp_time_iso:
            self._last_tp_per_symbol_dir.pop(key, None)
            return (False, "")

        # === (C) TTL expired ===
        try:
            tp_time = datetime.fromisoformat(tp_time_iso)
        except Exception:
            self._last_tp_per_symbol_dir.pop(key, None)
            return (False, "")

        now = TimeManager.get_server_time()
        if tp_time.tzinfo is None:
            tp_time = tp_time.replace(tzinfo=now.tzinfo)
        ttl_min = getattr(self._config, "POST_TP_LOCK_TTL_MIN", 60)
        elapsed_min = (now - tp_time).total_seconds() / 60.0
        if elapsed_min >= ttl_min:
            self._last_tp_per_symbol_dir.pop(key, None)
            return (False, "")

        # === (A) Price pullback with ATR buffer ===
        atr_buffer_mult = getattr(self._config, "POST_TP_ATR_BUFFER", 0.3)
        atr_buf = (atr or 0.0) * atr_buffer_mult  # 0 ถ้าไม่รู้ ATR → เท่ากับ no buffer

        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            # ดึงราคาไม่ได้ → fail-safe คงล็อกไว้ (ดีกว่ายอมเปิด)
            return (True, f"Post-TP Lock {symbol} {direction}: ดึงราคาไม่ได้")

        bid = price_info.get("bid", 0)
        ask = price_info.get("ask", 0)

        price_ok = False
        if direction == "SELL":
            # ต้องการ pullback ขึ้น: ask > tp + buffer
            price_ok = ask > (tp_price + atr_buf)
        elif direction == "BUY":
            # ต้องการ pullback ลง: bid < tp − buffer
            price_ok = bid < (tp_price - atr_buf)

        if price_ok:
            self._last_tp_per_symbol_dir.pop(key, None)
            return (False, "")

        # === (B) EMA20 M15 touched in last N bars ===
        if self._ema_touched_recently(symbol):
            self._last_tp_per_symbol_dir.pop(key, None)
            return (False, "")

        # ยังล็อกอยู่ — สร้างข้อความอธิบายให้ user
        remaining_min = ttl_min - elapsed_min
        if direction == "SELL":
            need = f"ask > {tp_price + atr_buf:.5f} (ปัจจุบัน {ask:.5f})"
        else:
            need = f"bid < {tp_price - atr_buf:.5f} (ปัจจุบัน {bid:.5f})"
        return (
            True,
            f"Post-TP Lock {symbol} {direction}: รอ {need} หรือ EMA20 touch "
            f"(TTL เหลือ {remaining_min:.1f} นาที)"
        )

    def _get_recent_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        ดึง ATR ล่าสุดของ symbol บน M15 (ใช้สำหรับ flip-lock threshold)

        Returns:
            ATR value (float) หรือ None ถ้าดึงไม่ได้
        """
        try:
            count = max(period * 3, 60)
            df = self._connector.get_historical_data(symbol, "M15", count)
            if df is None or len(df) < period + 1:
                return None
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.shift(1)
            tr = (high - low).combine((high - prev_close).abs(), max).combine(
                (low - prev_close).abs(), max
            )
            atr = tr.ewm(alpha=1 / period, adjust=False).mean()
            return float(atr.iloc[-1])
        except Exception:
            return None

    def _ema_touched_recently(self, symbol: str) -> bool:
        """
        เช็คว่า EMA20 (config: POST_TP_EMA_PERIOD) บน timeframe M15 ถูก "สัมผัส"
        ใน N bars ล่าสุด (lookback จาก config) หรือไม่

        นิยาม "touched": EMA value อยู่ในช่วง [low, high] ของ bar นั้น
        — ใช้ได้ทั้ง SELL lock (price below EMA, rally ขึ้นแตะ) และ BUY lock (price above EMA, ย่อลงแตะ)

        Returns:
            True = EMA ถูกแตะในช่วง lookback (ปลดล็อก), False = ยังไม่ถูกแตะหรือดึงข้อมูลไม่ได้
        """
        tf = getattr(self._config, "POST_TP_EMA_TIMEFRAME", "M15")
        period = getattr(self._config, "POST_TP_EMA_PERIOD", 20)
        lookback = getattr(self._config, "POST_TP_EMA_LOOKBACK_BARS", 3)
        # ต้องการ bars พอสำหรับ EMA warm-up + lookback
        count = max(period * 3, 60)

        df = self._connector.get_historical_data(symbol, tf, count)
        if df is None or len(df) < period + lookback:
            return False  # ดึงไม่ได้ → ไม่ปลดล็อก (fail-safe)

        try:
            ema = df["close"].ewm(span=period, adjust=False).mean()
            recent_ema = ema.tail(lookback).values
            recent_low = df["low"].tail(lookback).values
            recent_high = df["high"].tail(lookback).values
            for i in range(len(recent_ema)):
                if recent_low[i] <= recent_ema[i] <= recent_high[i]:
                    return True
            return False
        except Exception:
            return False

    # =========================================================================
    # 🔄 Directional Flip-Lock (v6) — กัน whipsaw BUY↔SELL
    # =========================================================================

    def register_flip_lock(
        self,
        symbol: str,
        closed_direction: str,
        close_price: float,
        atr: Optional[float],
        reason_code: int,
    ):
        """
        ลงทะเบียน flip-lock หลังไม้ของ symbol ถูกปิด (TP/SL/bot-close)

        หลังปิด BUY → lock ห้ามเปิด SELL จนกว่าราคาจะ retrace ลง K×ATR
        หลังปิด SELL → lock ห้ามเปิด BUY จนกว่าราคาจะ retrace ขึ้น K×ATR
        K default = 0.5 (TP) / 0.7 (SL) — per-symbol override ได้ผ่าน settings
        """
        if not symbol or not closed_direction or not close_price or close_price <= 0:
            return

        # K multiplier — SL ให้ retrace มากกว่า TP (SL = overshoot, TP = profit taken ตามแผน)
        if reason_code == self._DR_SL:
            retrace_mult = 0.7
        else:
            retrace_mult = 0.5

        # Per-symbol override (เช่น GBPJPY = 0.7)
        retrace_mult = float(get_symbol_config(symbol, "flip_lock_retrace_mult", retrace_mult))

        atr_val = atr if atr and atr > 0 else 0.0
        if closed_direction == "BUY":
            # ปิด BUY → จะเปิด SELL ได้ต่อเมื่อราคาลง K×ATR
            threshold = close_price - retrace_mult * atr_val
        else:  # SELL
            threshold = close_price + retrace_mult * atr_val

        min_unlock_min = getattr(self._config, "FLIP_LOCK_MIN_MINUTES", 5)
        now = TimeManager.get_server_time()
        min_unlock_time = (now + timedelta(minutes=min_unlock_min)).isoformat()

        self._flip_lock[symbol] = {
            "closed_direction": closed_direction,
            "close_price": float(close_price),
            "atr_at_close": float(atr_val),
            "unlock_threshold": float(threshold),
            "min_unlock_time": min_unlock_time,
        }
        print(f"🔄 [Risk Manager] Flip-lock set: {symbol} {closed_direction} closed @ {close_price} "
              f"→ opposite dir lock until price retraces {retrace_mult}×ATR "
              f"(threshold={threshold:.5f}, min_time={min_unlock_min}m)")

    def is_flip_locked(
        self,
        symbol: str,
        proposed_direction: str,
    ) -> Tuple[bool, str]:
        """
        ตรวจว่าทิศ `proposed_direction` ถูก lock จาก flip ล่าสุดหรือไม่

        ปลดล็อกเมื่อ:
        (A) ราคา retrace ผ่าน unlock_threshold และ
        (B) เลย min_unlock_time แล้ว

        Returns:
            (is_locked, reason_str)
        """
        lock = self._flip_lock.get(symbol)
        if not lock:
            return (False, "")

        closed_dir = lock.get("closed_direction", "")
        if closed_dir == proposed_direction:
            # same direction → ใช้ _is_post_tp_locked แทน, flip-lock ไม่เกี่ยวข้อง
            return (False, "")

        # === Min unlock time (safety floor) ===
        try:
            min_unlock = datetime.fromisoformat(lock.get("min_unlock_time", ""))
        except Exception:
            # broken state — clear and pass
            self._flip_lock.pop(symbol, None)
            return (False, "")

        now = TimeManager.get_server_time()
        if min_unlock.tzinfo is None:
            min_unlock = min_unlock.replace(tzinfo=now.tzinfo)
        if now < min_unlock:
            remaining_s = int((min_unlock - now).total_seconds())
            return (True, f"Flip-lock {symbol}: เพิ่งปิด {closed_dir} — รอ {remaining_s}s")

        # === Price retrace check ===
        threshold = lock.get("unlock_threshold", 0.0)
        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            # ดึงราคาไม่ได้ → fail-safe คงล็อก
            return (True, f"Flip-lock {symbol}: ดึงราคาไม่ได้ (fail-safe)")

        bid = price_info.get("bid", 0)
        ask = price_info.get("ask", 0)

        if proposed_direction == "SELL":
            # ปิด BUY แล้วจะ SELL → ต้องรอราคา "ลงต่ำกว่า" threshold
            if bid > threshold:
                return (
                    True,
                    f"Flip-lock {symbol}: เพิ่งปิด BUY @ {lock['close_price']:.5f} "
                    f"— รอ bid < {threshold:.5f} (ปัจจุบัน {bid:.5f})"
                )
        else:  # BUY
            # ปิด SELL แล้วจะ BUY → ต้องรอราคา "ขึ้นสูงกว่า" threshold
            if ask < threshold:
                return (
                    True,
                    f"Flip-lock {symbol}: เพิ่งปิด SELL @ {lock['close_price']:.5f} "
                    f"— รอ ask > {threshold:.5f} (ปัจจุบัน {ask:.5f})"
                )

        # Retrace ผ่านแล้ว — ปลดล็อก
        self._flip_lock.pop(symbol, None)
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
