"""
===============================================================================
FTMO Trading Bot — Trade Manager (ระบบจัดการเทรดที่เปิดอยู่)
===============================================================================
จัดการ Position ที่เปิดอยู่แบบ Real-time:
1. Trailing Stop (ATR-based) — เลื่อน SL ตามราคาเพื่อล็อคกำไร
2. Break-Even Move — เลื่อน SL มาที่จุดคุ้มทุนเมื่อกำไรถึง 1:1
3. Partial Take Profit — ปิดบางส่วนที่ TP1 (1:1) เหลือวิ่งถึง TP2
4. Session Close — ปิดเทรดก่อนหมด Session

⚠️ Trailing Stop ทำงานเฉพาะเมื่อเทรดกำไรเท่านั้น — ไม่เลื่อน SL ให้ขาดทุนเพิ่ม
===============================================================================
"""

import time as time_module
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

from config.settings import bot_config
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor, ExecutedTrade

# พยายาม Import MT5 สำหรับ Modify Order
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


@dataclass
class TrailingState:
    """สถานะ Trailing Stop ของแต่ละ Position"""
    ticket: int
    initial_sl: float           # SL เริ่มต้น
    current_sl: float           # SL ปัจจุบัน
    best_price: float           # ราคาดีที่สุดที่เคยถึง (สำหรับ Trailing)
    breakeven_moved: bool       # เลื่อน SL มาจุดคุ้มทุนแล้วหรือยัง
    partial_closed: bool        # ปิดบางส่วนแล้วหรือยัง
    trailing_active: bool       # Trailing Stop กำลังทำงานหรือไม่
    trail_distance: float       # ระยะ Trailing (ราคา)


class TradeManager:
    """
    จัดการ Position ที่เปิดอยู่ — ปรับ SL/TP, Trailing, Partial Close

    ลำดับการจัดการ (ทุก Tick):
    1. ซิงค์กับ MT5 (ตรวจ SL/TP Hit)
    2. ตรวจสอบ Break-Even Move (กำไร >= 1:1 → SL → Entry + 1 pip)
    3. ตรวจสอบ Partial Close (กำไร >= 1:1 → ปิด 50%)
    4. ตรวจสอบ Trailing Stop (ราคาใหม่ดีกว่าเดิม → เลื่อน SL)
    5. ตรวจสอบ Session End (ใกล้หมด Session → ปิดทุก Position)
    """

    # === ค่าคงที่สำหรับการจัดการ ===
    BE_TRIGGER_RR: float = 1.0          # เลื่อน SL มา Entry เมื่อกำไรถึง RR 1:1
    BE_OFFSET_PIPS: float = 1.0         # เลื่อน SL เลย Entry + 1 pip (ไม่ใช่แค่ Entry)
    PARTIAL_CLOSE_PCT: float = 0.5      # ปิด 50% เมื่อ TP1 Hit
    TRAIL_ACTIVATION_RR: float = 1.0    # เริ่ม Trail เมื่อกำไรถึง RR 1:1
    TRAIL_ATR_MULTIPLIER: float = 1.0   # Trail SL ห่างจาก Best Price = ATR × 1.0

    def __init__(
        self,
        connector: MT5Connector,
        risk_manager: RiskManager,
        executor: TradeExecutor,
    ):
        """
        เริ่มต้น Trade Manager

        Args:
            connector: ตัวเชื่อมต่อ MT5
            risk_manager: ระบบจัดการความเสี่ยง
            executor: ระบบส่งคำสั่ง (สำหรับปิดเทรด)
        """
        self._connector = connector
        self._risk_manager = risk_manager
        self._executor = executor

        # สถานะ Trailing ของแต่ละ Position
        self._trail_states: Dict[int, TrailingState] = {}

        print("📋 [Trade Manager] เริ่มต้นระบบจัดการเทรด")

    # =========================================================================
    # 🔄 Main Management Loop (เรียกทุก Tick)
    # =========================================================================

    def manage_all_positions(self):
        """
        จัดการ Position ทั้งหมดที่เปิดอยู่ — เรียกทุก Loop ของ Main

        ทำงานตามลำดับ:
        1. ซิงค์กับ MT5
        2. จัดการแต่ละ Position
        3. ลบ Trail State ของ Position ที่ปิดแล้ว
        """
        # ซิงค์กับ MT5 — ตรวจ SL/TP Hit
        self._executor.sync_with_mt5()

        # จัดการแต่ละ Position ที่ยังเปิดอยู่
        active_trades = self._executor.active_trades

        for ticket, trade in list(active_trades.items()):
            try:
                self._manage_single_position(trade)
            except Exception as e:
                print(f"⚠️ [Trade Manager] จัดการ Ticket {ticket} ล้มเหลว: {e}")

        # ลบ Trail State ของ Position ที่ปิดแล้ว
        closed_tickets = [t for t in self._trail_states if t not in active_trades]
        for ticket in closed_tickets:
            del self._trail_states[ticket]

    def _manage_single_position(self, trade: ExecutedTrade):
        """
        จัดการ Position เดียว

        ลำดับ:
        1. ดึงราคาปัจจุบัน
        2. สร้าง/ดึง Trailing State
        3. ตรวจ Break-Even Move
        4. ตรวจ Partial Close
        5. ตรวจ Trailing Stop

        Args:
            trade: ข้อมูลเทรดที่จัดการ
        """
        # ดึงราคาปัจจุบัน
        price_info = self._connector.get_current_price(trade.symbol)
        if price_info is None:
            return

        current_price = price_info["bid"] if trade.trade_type == "BUY" else price_info["ask"]

        # สร้าง Trailing State ถ้ายังไม่มี
        if trade.ticket not in self._trail_states:
            self._trail_states[trade.ticket] = TrailingState(
                ticket=trade.ticket,
                initial_sl=trade.sl_price,
                current_sl=trade.sl_price,
                best_price=trade.entry_price,
                breakeven_moved=False,
                partial_closed=False,
                trailing_active=False,
                trail_distance=trade.atr_value * self.TRAIL_ATR_MULTIPLIER,
            )

        state = self._trail_states[trade.ticket]

        # คำนวณกำไรปัจจุบัน (pips)
        if trade.trade_type == "BUY":
            profit_distance = current_price - trade.entry_price
        else:
            profit_distance = trade.entry_price - current_price

        # คำนวณ RR ปัจจุบัน
        sl_distance = abs(trade.entry_price - state.initial_sl)
        current_rr = profit_distance / sl_distance if sl_distance > 0 else 0

        # === ขั้นตอนที่ 1: Break-Even Move ===
        if not state.breakeven_moved and current_rr >= self.BE_TRIGGER_RR:
            self._move_to_breakeven(trade, state, price_info)

        # === ขั้นตอนที่ 2: Partial Close ===
        if not state.partial_closed and current_rr >= self.BE_TRIGGER_RR:
            self._partial_close(trade, state)

        # === ขั้นตอนที่ 3: Trailing Stop ===
        if current_rr >= self.TRAIL_ACTIVATION_RR:
            state.trailing_active = True
            self._update_trailing_stop(trade, state, current_price)

    # =========================================================================
    # 🔒 Break-Even Move (เลื่อน SL มาจุดคุ้มทุน)
    # =========================================================================

    def _move_to_breakeven(
        self,
        trade: ExecutedTrade,
        state: TrailingState,
        price_info: Dict
    ):
        """
        เลื่อน SL มาที่ Entry Price + offset เมื่อกำไรถึง RR 1:1

        ผลลัพธ์:
        - BUY: SL = Entry + 1 pip (ไม่ขาดทุนแล้ว)
        - SELL: SL = Entry - 1 pip

        Args:
            trade: ข้อมูลเทรด
            state: สถานะ Trailing
            price_info: ราคาปัจจุบัน
        """
        symbol_info = self._connector.get_symbol_info(trade.symbol)
        pip_size = 0.0001 if symbol_info and symbol_info["digits"] >= 4 else 0.01
        offset = self.BE_OFFSET_PIPS * pip_size

        if trade.trade_type == "BUY":
            new_sl = trade.entry_price + offset
        else:
            new_sl = trade.entry_price - offset

        # ตรวจสอบว่า SL ใหม่ดีกว่าเดิม
        if trade.trade_type == "BUY" and new_sl <= state.current_sl:
            return  # SL ใหม่ไม่ดีกว่า — ไม่เลื่อน
        if trade.trade_type == "SELL" and new_sl >= state.current_sl:
            return

        # เลื่อน SL
        if self._modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
            state.current_sl = new_sl
            state.breakeven_moved = True
            digits = symbol_info["digits"] if symbol_info else 5
            print(f"🔒 [Trade Manager] เลื่อน SL มา Break-Even: Ticket {trade.ticket} "
                  f"SL={new_sl:.{digits}f} (Entry+{self.BE_OFFSET_PIPS} pip)")

    # =========================================================================
    # ✂️ Partial Close (ปิดบางส่วน)
    # =========================================================================

    def _partial_close(self, trade: ExecutedTrade, state: TrailingState):
        """
        ปิดบางส่วนของ Position เมื่อกำไรถึง TP1 (RR 1:1)

        ผลลัพธ์:
        - ปิด 50% ของ Volume
        - เหลืออีก 50% วิ่งต่อไปถึง TP2 (RR 2:1)

        Args:
            trade: ข้อมูลเทรด
            state: สถานะ Trailing
        """
        # คำนวณ Volume ที่จะปิด
        symbol_info = self._connector.get_symbol_info(trade.symbol)
        lot_step = symbol_info["lot_step"] if symbol_info else 0.01
        lot_min = symbol_info["lot_min"] if symbol_info else 0.01

        close_volume = trade.lot_size * self.PARTIAL_CLOSE_PCT
        # ปัดลงตาม Lot Step
        import math
        close_volume = math.floor(close_volume / lot_step) * lot_step
        close_volume = round(close_volume, 8)

        # ตรวจว่า Volume ที่เหลือจะไม่ต่ำกว่า Lot Min
        remaining = trade.lot_size - close_volume
        if remaining < lot_min or close_volume < lot_min:
            # Volume น้อยเกินไป — ไม่ปิดบางส่วน
            state.partial_closed = True  # ถือว่าผ่านไปแล้ว
            return

        # ส่งคำสั่งปิดบางส่วน
        if self._partial_close_position(trade.ticket, trade.symbol, trade.trade_type, close_volume):
            state.partial_closed = True
            trade.lot_size = remaining  # อัพเดท Volume ที่เหลือ
            print(f"✂️ [Trade Manager] ปิดบางส่วน Ticket {trade.ticket}: "
                  f"ปิด {close_volume:.2f} lot, เหลือ {remaining:.2f} lot")
        else:
            print(f"⚠️ [Trade Manager] ปิดบางส่วน Ticket {trade.ticket} ล้มเหลว")

    def _partial_close_position(
        self,
        ticket: int,
        symbol: str,
        trade_type: str,
        volume: float
    ) -> bool:
        """
        ส่งคำสั่งปิดบางส่วนของ Position

        Args:
            ticket: หมายเลข Ticket
            symbol: คู่เงิน
            trade_type: "BUY" หรือ "SELL"
            volume: Volume ที่ต้องการปิด

        Returns:
            bool: สำเร็จหรือไม่
        """
        if not MT5_AVAILABLE:
            print(f"📝 [MOCK] จำลองปิดบางส่วน Ticket {ticket}: {volume:.2f} lot")
            return True

        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            return False

        if trade_type == "BUY":
            close_type = mt5.ORDER_TYPE_SELL
            price = price_info["bid"]
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = price_info["ask"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "FTMO_PARTIAL_TP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    # =========================================================================
    # 📏 Trailing Stop (เลื่อน SL ตามราคา)
    # =========================================================================

    def _update_trailing_stop(
        self,
        trade: ExecutedTrade,
        state: TrailingState,
        current_price: float
    ):
        """
        อัพเดท Trailing Stop — เลื่อน SL ตามราคาที่ดีขึ้น

        หลักการ:
        - BUY: ถ้าราคาขึ้น → SL ขึ้นตาม (ห่างจาก Best Price = ATR × multiplier)
        - SELL: ถ้าราคาลง → SL ลงตาม (ห่างจาก Best Price = ATR × multiplier)
        - SL จะเลื่อนทิศทางเดียวเท่านั้น (ไม่เลื่อนถอยหลัง)

        Args:
            trade: ข้อมูลเทรด
            state: สถานะ Trailing
            current_price: ราคาปัจจุบัน
        """
        trail_dist = state.trail_distance

        if trade.trade_type == "BUY":
            # อัพเดท Best Price (สูงสุด)
            if current_price > state.best_price:
                state.best_price = current_price

            # คำนวณ SL ใหม่
            new_sl = state.best_price - trail_dist

            # SL ต้องสูงกว่าเดิมเท่านั้น (ไม่เลื่อนลง)
            if new_sl > state.current_sl:
                if self._modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
                    state.current_sl = new_sl
                    if bot_config.debug_mode:
                        print(f"📏 [Trail] Ticket {trade.ticket} BUY: SL↑ {new_sl:.5f} "
                              f"(Best={state.best_price:.5f})")

        elif trade.trade_type == "SELL":
            # อัพเดท Best Price (ต่ำสุด)
            if current_price < state.best_price:
                state.best_price = current_price

            new_sl = state.best_price + trail_dist

            # SL ต้องต่ำกว่าเดิมเท่านั้น (ไม่เลื่อนขึ้น)
            if new_sl < state.current_sl:
                if self._modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
                    state.current_sl = new_sl
                    if bot_config.debug_mode:
                        print(f"📏 [Trail] Ticket {trade.ticket} SELL: SL↓ {new_sl:.5f} "
                              f"(Best={state.best_price:.5f})")

    # =========================================================================
    # 🔧 Modify SL ใน MT5
    # =========================================================================

    def _modify_sl(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        tp: float
    ) -> bool:
        """
        แก้ไข Stop Loss ของ Position ที่เปิดอยู่

        Args:
            ticket: หมายเลข Ticket
            symbol: คู่เงิน
            new_sl: SL ใหม่
            tp: TP (ไม่เปลี่ยน)

        Returns:
            bool: สำเร็จหรือไม่
        """
        if not MT5_AVAILABLE:
            return True  # จำลองสำเร็จ

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return False
        return True

    # =========================================================================
    # ⏰ Session Close (ปิดก่อนหมด Session)
    # =========================================================================

    def check_session_close(self) -> int:
        """
        ตรวจสอบว่าใกล้หมด Trading Session หรือไม่
        ถ้าใกล้หมด → ปิดทุก Position ที่กำไรเพื่อล็อคกำไร

        กฎ:
        - ก่อนหมด Session 15 นาที → ปิดเทรดที่กำไร
        - ก่อนวันศุกร์หมด → ปิดทุกเทรด (ไม่ถือข้ามสุดสัปดาห์)

        Returns:
            int: จำนวนเทรดที่ปิด
        """
        from core.time_manager import TimeManager
        server_time_now = TimeManager.get_server_time()
        current_time = server_time_now.time()
        current_weekday = server_time_now.weekday()
        sessions = bot_config.sessions
        closed_count = 0

        # === ความปลอดภัยระดับสูดสุด: ตรวจสอบ Friday Force Close ตามกฎ FTMO ===
        # ถ้าเป็นวันศุกร์และเกินเวลา 20:45 Server Time (EET) ให้บังคับปิดทุกออเดอร์ทันที 
        if TimeManager.is_friday_close_time(server_time_now):
            print("🚨 [Trade Manager] ⚠️ FRIDAY FORCE CLOSE ⚠️ — ปิดทุก Position ทันทีเพื่อรักษากฎ FTMO!")
            for ticket in list(self._executor.active_trades.keys()):
                if self._executor.close_trade(ticket, reason="FTMO Friday Force Close"):
                    closed_count += 1
            return closed_count

        # === ตรวจวันศุกร์ (แจ้งเตือน) — ปิดทุกเทรดก่อน cutoff เดิม ===
        from datetime import time as dt_time
        friday_warning = dt_time(
            sessions.friday_cutoff.hour,
            max(0, sessions.friday_cutoff.minute - 15)
        )

        if current_weekday == 4 and current_time >= friday_warning and current_time < sessions.friday_force_close:
            print("⏰ [Trade Manager] ใกล้หมดวันศุกร์ช่วงเย็น — พยายามเคลียร์ Position...")
            for ticket in list(self._executor.active_trades.keys()):
                if self._executor.close_trade(ticket, reason="Friday Session Close"):
                    closed_count += 1
            return closed_count

        # === ตรวจ NY Session End — ปิดเทรดที่กำไร ===
        ny_warning = dt_time(
            sessions.newyork_end.hour,
            max(0, sessions.newyork_end.minute - 15)
        )

        if current_time >= ny_warning:
            for ticket, trade in list(self._executor.active_trades.items()):
                price_info = self._connector.get_current_price(trade.symbol)
                if price_info:
                    if trade.trade_type == "BUY":
                        pnl = price_info["bid"] - trade.entry_price
                    else:
                        pnl = trade.entry_price - price_info["ask"]

                    if pnl > 0:
                        print(f"⏰ [Trade Manager] Session End — ปิดเทรดกำไร Ticket {ticket}")
                        if self._executor.close_trade(ticket, reason="Session End (Profit Lock)"):
                            closed_count += 1

        return closed_count

    # =========================================================================
    # 📊 สรุปข้อมูล
    # =========================================================================

    def get_management_summary(self) -> Dict:
        """สรุปสถานะการจัดการ Position"""
        active_count = self._executor.active_count
        be_count = sum(1 for s in self._trail_states.values() if s.breakeven_moved)
        trailing_count = sum(1 for s in self._trail_states.values() if s.trailing_active)
        partial_count = sum(1 for s in self._trail_states.values() if s.partial_closed)

        return {
            "active_positions": active_count,
            "breakeven_moved": be_count,
            "trailing_active": trailing_count,
            "partial_closed": partial_count,
            "trail_states_count": len(self._trail_states),
        }

    def __repr__(self) -> str:
        summary = self.get_management_summary()
        return (
            f"TradeManager(active={summary['active_positions']}, "
            f"trailing={summary['trailing_active']}, "
            f"BE={summary['breakeven_moved']})"
        )
