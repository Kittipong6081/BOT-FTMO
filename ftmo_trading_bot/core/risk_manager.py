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
        self._highest_balance: float = 0.0          # Balance สูงสุดที่เคยถึง (High Water Mark)
        self._current_day: date = TimeManager.get_server_time().date()  # วันปัจจุบันตามเวลาโบรกเกอร์
        
        # === สถิติรายวัน ===
        self._daily_closed_pnl: float = 0.0         # P/L ที่ปิดไปแล้ววันนี้
        self._daily_trades_count: int = 0            # จำนวนเทรดวันนี้
        
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
        
        if loaded:
            # ตรวจสอบว่าวันเปลี่ยนหรือยัง (ใช้เวลาโบรกเกอร์ ไม่ใช่เวลาท้องถิ่น)
            broker_today = TimeManager.get_server_time().date()
            if self._current_day != broker_today:
                self._on_new_day(current_balance, current_equity)
            else:
                print(f"📂 [Risk Manager] โหลดสถานะเดิม: สถานะ={self._state.value}")
                
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
            self._current_day = TimeManager.get_server_time().date()
            self._state = BotState.ACTIVE
            
            print(f"🆕 [Risk Manager] เริ่มต้นใหม่:")

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

        # FTMO ใช้ค่ามากกว่าระหว่าง Balance กับ Equity ตอนเริ่มวัน
        self._daily_start_equity = max(current_balance, current_equity)
        self._daily_start_balance = current_balance
        self._current_day = broker_today
        self._daily_closed_pnl = 0.0
        self._daily_trades_count = 0
        
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

        # ✅ ผ่านทุกการตรวจสอบ
        return (True, f"✅ อนุญาตให้เทรด {symbol} (Risk: {risk_pct:.2%}, RR: 1:{rr_ratio:.1f})")

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
            "current_day": str(self._current_day),
            "daily_closed_pnl": self._daily_closed_pnl,
            "daily_trades_count": self._daily_trades_count,
            "last_updated": datetime.now().isoformat(),
        }

        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                
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
            self._daily_closed_pnl = data.get("daily_closed_pnl", 0.0)
            self._daily_trades_count = data.get("daily_trades_count", 0)
            
            # แปลงวันที่
            day_str = data.get("current_day", str(TimeManager.get_server_time().date()))
            self._current_day = date.fromisoformat(day_str)
            
            print(f"📂 [Risk Manager] โหลดสถานะจากไฟล์สำเร็จ (อัพเดทล่าสุด: {data.get('last_updated', 'N/A')})")
            return True
            
        except Exception as e:
            print(f"⚠️ [Risk Manager] โหลดสถานะล้มเหลว: {e}")
            return False

    def update_daily_pnl(self, trade_pnl: float):
        """
        อัพเดท P/L ที่ปิดไปแล้ววันนี้ (เรียกหลังจาก Position ปิด)
        
        Args:
            trade_pnl: กำไร/ขาดทุนของเทรดที่ปิด (USD)
        """
        self._daily_closed_pnl += trade_pnl
        self._daily_trades_count += 1
        
        # อัพเดท High Water Mark ถ้า Balance สูงขึ้น
        current_balance = self._connector.get_balance()
        if current_balance > self._highest_balance:
            self._highest_balance = current_balance
            
        self._save_state()
        
        print(f"📊 [Risk Manager] อัพเดท Daily P/L: ${self._daily_closed_pnl:,.2f} (เทรดที่ {self._daily_trades_count} วันนี้)")

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
