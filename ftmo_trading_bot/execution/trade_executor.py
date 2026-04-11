"""
===============================================================================
FTMO Trading Bot — Trade Executor (ระบบส่งคำสั่งซื้อขาย)
===============================================================================
จัดการการส่งคำสั่งเทรดจากสัญญาณ SMC ไปยัง MT5

ขั้นตอนการทำงาน:
1. รับ TradeSignal จาก SMC Strategy
2. ตรวจสอบ Risk final time (Double-check)
3. คำนวณ Lot Size จาก Position Sizer
4. ส่ง Market Order พร้อม SL/TP
5. ตรวจสอบผลลัพธ์ + Retry ถ้าล้มเหลว
6. บันทึกข้อมูลเทรดสำหรับ Logger (Phase 4)

⚠️ ทุกคำสั่งต้องมี Stop Loss — ห้ามส่งคำสั่งโดยไม่มี SL เด็ดขาด
===============================================================================
"""

import time as time_module
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from config.settings import bot_config
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from core.position_sizer import PositionSizer
from strategy.smc_strategy import TradeSignal, SignalType

try:
    from analytics.trade_logger import TradeLogger
    from analytics.performance import PerformanceAnalyzer
except ImportError:
    pass


@dataclass
class ExecutedTrade:
    """
    ข้อมูลเทรดที่ส่งคำสั่งสำเร็จแล้ว
    ใช้ส่งต่อไปยัง Trade Logger (Phase 4) และ Trade Manager
    """
    ticket: int                     # หมายเลข Ticket จาก MT5
    symbol: str                     # คู่เงิน
    trade_type: str                 # "BUY" หรือ "SELL"
    entry_price: float              # ราคาที่เปิดจริง (อาจต่างจาก Signal เล็กน้อย)
    sl_price: float                 # Stop Loss
    tp_price: float                 # Take Profit
    lot_size: float                 # ขนาด Lot ที่เปิด
    risk_amount: float              # จำนวนเงินที่เสี่ยง (USD)
    risk_pct: float                 # เปอร์เซ็นต์ความเสี่ยง
    rr_ratio: float                 # Risk:Reward Ratio
    confluence_score: float         # คะแนน Confluence จาก Strategy
    atr_value: float                # ATR ตอนเปิดเทรด (ใช้สำหรับ Trailing)
    open_time: datetime             # เวลาเปิดเทรด
    signal_reasons: List[str]       # เหตุผลจาก Strategy
    magic_number: int = 123456      # Magic Number ของ Bot
    
    # สถานะ
    is_open: bool = True            # ยังเปิดอยู่หรือไม่
    close_price: float = 0.0        # ราคาปิด (เมื่อปิดแล้ว)
    close_time: Optional[datetime] = None  # เวลาปิด
    profit: float = 0.0             # กำไร/ขาดทุน (USD)
    close_reason: str = ""          # เหตุผลที่ปิด

    def to_dict(self) -> Dict:
        """แปลงเป็น Dictionary สำหรับ Excel Logging"""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "type": self.trade_type,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "lot_size": self.lot_size,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "rr_ratio": self.rr_ratio,
            "confluence": self.confluence_score,
            "atr": self.atr_value,
            "open_time": str(self.open_time),
            "close_price": self.close_price,
            "close_time": str(self.close_time) if self.close_time else "",
            "profit": self.profit,
            "close_reason": self.close_reason,
            "reasons": "; ".join(self.signal_reasons),
        }


class TradeExecutor:
    """
    ระบบส่งคำสั่งเทรดหลัก

    ทำหน้าที่:
    - รับสัญญาณจาก SMC Strategy → แปลงเป็นคำสั่ง MT5
    - ตรวจสอบ Risk ซ้ำอีกครั้งก่อนส่ง (Double Safety)
    - คำนวณ Lot Size ที่เหมาะสม
    - ส่ง Market Order พร้อม SL/TP
    - จัดการ Retry เมื่อคำสั่งล้มเหลว
    - เก็บประวัติเทรดที่เปิดได้สำเร็จ

    ⚠️ ทุกคำสั่งต้องผ่าน 3 ด่านตรวจสอบ:
    1. Strategy Confluence >= 60
    2. Risk Manager approve
    3. Position Sizer validate
    """

    # จำนวนครั้ง Retry สูงสุดเมื่อคำสั่งล้มเหลว
    MAX_RETRY = 3
    RETRY_DELAY_SEC = 2

    # Magic Number สำหรับระบุว่าเป็นคำสั่งจาก Bot
    MAGIC_NUMBER = 123456

    def __init__(
        self,
        connector: MT5Connector,
        risk_manager: RiskManager,
        position_sizer: PositionSizer,
        logger: Optional['TradeLogger'] = None,
        analyzer: Optional['PerformanceAnalyzer'] = None
    ):
        """
        เริ่มต้น Trade Executor

        Args:
            connector: ตัวเชื่อมต่อ MT5
            risk_manager: ระบบจัดการความเสี่ยง
            position_sizer: ระบบคำนวณขนาด Position
        """
        self._connector = connector
        self._risk_manager = risk_manager
        self._position_sizer = position_sizer
        self._logger = logger
        self._analyzer = analyzer

        # เก็บประวัติเทรดที่เปิดอยู่ (key = ticket)
        self._active_trades: Dict[int, ExecutedTrade] = {}

        # เก็บประวัติเทรดที่ปิดแล้ว
        self._closed_trades: List[ExecutedTrade] = []

        # สถิติ
        self._total_executed = 0
        self._total_rejected = 0

        print("⚡ [Trade Executor] เริ่มต้นระบบส่งคำสั่งเทรด")

    # =========================================================================
    # 🚀 ส่งคำสั่งเทรดจากสัญญาณ
    # =========================================================================

    def execute_signal(self, signal: TradeSignal) -> Optional[ExecutedTrade]:
        """
        ดำเนินการส่งคำสั่งเทรดจาก TradeSignal

        ขั้นตอน:
        1. ตรวจสอบสัญญาณ Valid
        2. คำนวณ Lot Size
        3. ตรวจสอบ Risk ซ้ำ (Double-check)
        4. ตรวจสอบ Spread ปัจจุบัน
        5. ส่ง Market Order
        6. บันทึกผลลัพธ์

        Args:
            signal: สัญญาณเทรดจาก SMC Strategy

        Returns:
            ExecutedTrade หรือ None: ข้อมูลเทรดที่เปิดสำเร็จ
        """
        # === ด่านที่ 1: ตรวจสอบสัญญาณ ===
        if not signal.is_valid:
            print("❌ [Executor] สัญญาณไม่ Valid — ยกเลิก")
            return None

        symbol = signal.symbol
        print(f"\n{'━' * 60}")
        print(f"⚡ [Executor] ดำเนินการ {signal.signal_type.value} {symbol}")
        print(f"   Confluence: {signal.confluence_score:.0f}/100 | RR: 1:{signal.rr_ratio:.1f}")
        print(f"{'━' * 60}")

        # === ด่านที่ 2: คำนวณ Lot Size ===
        lot_result = self._position_sizer.calculate_lot_size(
            symbol=symbol,
            sl_distance_price=signal.sl_distance,
        )

        if lot_result is None:
            print("❌ [Executor] คำนวณ Lot Size ล้มเหลว — ยกเลิก")
            self._total_rejected += 1
            return None

        lot_size = lot_result["lot_size"]
        risk_amount = lot_result["risk_amount"]
        risk_pct = lot_result["risk_pct"]

        # === ด่านที่ 3: ตรวจสอบ Risk ซ้ำ (Double Safety) ===
        allowed, reason = self._risk_manager.can_open_trade(
            symbol=symbol,
            risk_amount=risk_amount,
            sl_distance_pips=lot_result["sl_pips"],
            rr_ratio=signal.rr_ratio,
        )

        if not allowed:
            print(f"🚫 [Executor] Risk Manager ปฏิเสธ: {reason}")
            self._total_rejected += 1
            return None

        # === ด่านที่ 4: ตรวจสอบ Spread ===
        price_info = self._connector.get_current_price(symbol)
        if price_info is None:
            print("❌ [Executor] ดึงราคาปัจจุบันล้มเหลว — ยกเลิก")
            self._total_rejected += 1
            return None

        symbol_info = self._connector.get_symbol_info(symbol)
        max_spread = bot_config.symbols.max_spread_points.get(symbol, 20)

        if symbol_info:
            current_spread_pts = price_info["spread"] / symbol_info["point"]
            if current_spread_pts > max_spread:
                print(f"🚫 [Executor] Spread สูงเกินไป ({current_spread_pts:.0f} > {max_spread}) — รอ Spread ลด")
                self._total_rejected += 1
                return None

        # === ด่านที่ 5: Final Validation ===
        remaining_budget = self._risk_manager.get_remaining_daily_budget()
        valid, val_reason = self._position_sizer.validate_trade_risk(
            symbol=symbol,
            lot_size=lot_size,
            sl_distance_price=signal.sl_distance,
            remaining_daily_budget=remaining_budget,
        )

        if not valid:
            print(f"🚫 [Executor] Final Validation ล้มเหลว: {val_reason}")
            self._total_rejected += 1
            return None

        # === ด่านที่ 6: ส่ง Market Order (พร้อม Retry) ===
        order_result = self._send_order_with_retry(
            symbol=symbol,
            order_type=signal.signal_type.value,
            volume=lot_size,
            sl=signal.sl_price,
            tp=signal.tp_price,
        )

        if order_result is None:
            print("❌ [Executor] ส่งคำสั่งล้มเหลวหลังจาก Retry ทุกครั้ง")
            self._total_rejected += 1
            return None

        # === ด่านที่ 7: บันทึกผลลัพธ์ ===
        executed = ExecutedTrade(
            ticket=order_result["ticket"],
            symbol=symbol,
            trade_type=signal.signal_type.value,
            entry_price=order_result["price"],
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            lot_size=lot_size,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            rr_ratio=signal.rr_ratio,
            confluence_score=signal.confluence_score,
            atr_value=signal.atr_value,
            open_time=datetime.now(),
            signal_reasons=signal.reasons,
            magic_number=self.MAGIC_NUMBER,
        )

        # เก็บใน Active Trades
        self._active_trades[executed.ticket] = executed
        self._total_executed += 1

        if self._logger:
            self._logger.log_trade_opened(executed.to_dict())

        # อัพเดท Risk Manager
        print(f"\n✅ [Executor] เปิดเทรดสำเร็จ!")
        print(f"   🎫 Ticket:   {executed.ticket}")
        print(f"   💱 Symbol:   {executed.symbol} {executed.trade_type}")
        print(f"   📍 Entry:    {executed.entry_price}")
        print(f"   🔴 SL:       {executed.sl_price}")
        print(f"   🟢 TP:       {executed.tp_price}")
        print(f"   📦 Lot:      {executed.lot_size}")
        print(f"   🎯 Risk:     {executed.risk_pct:.2%} (${executed.risk_amount:,.2f})")
        print(f"   ⚖️ RR:       1:{executed.rr_ratio:.1f}")
        print(f"   🏆 Confluence: {executed.confluence_score:.0f}")

        return executed

    # =========================================================================
    # 🔄 ส่งคำสั่งพร้อม Retry
    # =========================================================================

    def _send_order_with_retry(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float,
        tp: float,
    ) -> Optional[Dict]:
        """
        ส่ง Market Order พร้อม Retry Logic

        ถ้าล้มเหลวจะลองใหม่สูงสุด MAX_RETRY ครั้ง
        แต่ละครั้งจะรอ RETRY_DELAY_SEC วินาที

        Args:
            symbol: คู่เงิน
            order_type: "BUY" หรือ "SELL"
            volume: ขนาด Lot
            sl: ราคา Stop Loss
            tp: ราคา Take Profit

        Returns:
            Dict หรือ None: ผลลัพธ์การส่งคำสั่ง
        """
        for attempt in range(1, self.MAX_RETRY + 1):
            print(f"📤 [Executor] ส่งคำสั่ง {order_type} {symbol} (ครั้งที่ {attempt}/{self.MAX_RETRY})")

            result = self._connector.send_market_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                sl=sl,
                tp=tp,
                comment=f"FTMO_SMC_{order_type}",
                magic=self.MAGIC_NUMBER,
            )

            if result is not None:
                print(f"✅ [Executor] คำสั่งสำเร็จในครั้งที่ {attempt}")
                return result

            if attempt < self.MAX_RETRY:
                print(f"⚠️ [Executor] ล้มเหลว — รอ {self.RETRY_DELAY_SEC}s ก่อนลองใหม่...")
                time_module.sleep(self.RETRY_DELAY_SEC)

        return None

    # =========================================================================
    # 📋 จัดการเทรดที่เปิดอยู่
    # =========================================================================

    def close_trade(
        self,
        ticket: int,
        reason: str = "Manual Close"
    ) -> bool:
        """
        ปิดเทรดที่ระบุด้วย Ticket Number

        Args:
            ticket: หมายเลข Ticket
            reason: เหตุผลที่ปิด

        Returns:
            bool: True ถ้าปิดสำเร็จ
        """
        if ticket not in self._active_trades:
            print(f"⚠️ [Executor] ไม่พบ Ticket {ticket} ใน Active Trades")
            # ลองปิดจาก MT5 โดยตรง
            return self._connector.close_position(ticket)

        trade = self._active_trades[ticket]

        if self._connector.close_position(ticket):
            # อัพเดทข้อมูลเทรด
            price_info = self._connector.get_current_price(trade.symbol)
            if price_info:
                if trade.trade_type == "BUY":
                    trade.close_price = price_info["bid"]
                else:
                    trade.close_price = price_info["ask"]

            trade.is_open = False
            trade.close_time = datetime.now()
            trade.close_reason = reason

            # คำนวณ P/L (ประมาณ)
            if trade.trade_type == "BUY":
                trade.profit = (trade.close_price - trade.entry_price) * trade.lot_size * 100000
            else:
                trade.profit = (trade.entry_price - trade.close_price) * trade.lot_size * 100000

            # ย้ายจาก Active → Closed
            self._closed_trades.append(trade)
            del self._active_trades[ticket]

            # แจ้ง Risk Manager
            self._risk_manager.update_daily_pnl(trade.profit)

            if self._logger:
                self._logger.log_trade_closed(trade.to_dict())
            if self._analyzer:
                self._analyzer.add_trade(trade.to_dict())

            print(f"✅ [Executor] ปิดเทรด Ticket {ticket}: P/L=${trade.profit:,.2f} ({reason})")
            return True
        else:
            print(f"❌ [Executor] ปิดเทรด Ticket {ticket} ล้มเหลว")
            return False

    def record_external_close(
        self,
        ticket: int,
        close_price: float,
        profit: float,
        reason: str = "TP/SL Hit"
    ):
        """
        บันทึกเทรดที่ปิดจากภายนอก (SL/TP Hit โดย MT5)

        เมื่อ SL หรือ TP ถูก Hit โดย MT5 โดยตรง Bot ต้องอัพเดทข้อมูล

        Args:
            ticket: หมายเลข Ticket
            close_price: ราคาปิด
            profit: กำไร/ขาดทุน
            reason: เหตุผล
        """
        if ticket in self._active_trades:
            trade = self._active_trades[ticket]
            trade.is_open = False
            trade.close_price = close_price
            trade.close_time = datetime.now()
            trade.profit = profit
            trade.close_reason = reason

            self._closed_trades.append(trade)
            del self._active_trades[ticket]

            self._risk_manager.update_daily_pnl(profit)

            if self._logger:
                self._logger.log_trade_closed(trade.to_dict())
            if self._analyzer:
                self._analyzer.add_trade(trade.to_dict())

            pnl_emoji = "🟢" if profit >= 0 else "🔴"
            print(f"{pnl_emoji} [Executor] เทรด Ticket {ticket} ปิดจากภายนอก: "
                  f"P/L=${profit:,.2f} ({reason})")

    # =========================================================================
    # 🔍 ตรวจสอบสถานะเทรด
    # =========================================================================

    def sync_with_mt5(self):
        """
        ซิงค์สถานะเทรดกับ MT5

        ตรวจสอบ:
        1. เทรดใน Active Trades ที่ปิดไปแล้ว (SL/TP Hit)
        2. เทรดที่เปิดใน MT5 แต่ไม่อยู่ใน Active Trades

        ⚠️ ข้ามในโหมดจำลอง (Mock) เพราะ get_open_positions คืน []
        ซึ่งจะทำให้ระบบเข้าใจผิดว่าทุกเทรดถูกปิดไปแล้ว
        """
        # ข้ามถ้าอยู่ในโหมดจำลอง — ไม่มีข้อมูล Position จริงจาก MT5
        try:
            from core.mt5_connector import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                return
        except ImportError:
            return

        mt5_positions = self._connector.get_open_positions()
        mt5_tickets = {pos["ticket"] for pos in mt5_positions}

        # ตรวจหาเทรดที่ปิดไปแล้ว (อยู่ใน Active แต่ไม่อยู่ใน MT5)
        closed_tickets = []
        for ticket, trade in self._active_trades.items():
            if ticket not in mt5_tickets:
                closed_tickets.append(ticket)

        for ticket in closed_tickets:
            trade = self._active_trades[ticket]
            # ดึง History เพื่อหาราคาปิดและ P/L จริง
            history = self._connector.get_trade_history(days=1)
            actual_profit = 0.0
            actual_close_price = 0.0

            for h in history:
                if h.get("order") == ticket or h.get("ticket") == ticket:
                    actual_profit = h.get("profit", 0) + h.get("swap", 0) + h.get("commission", 0)
                    actual_close_price = h.get("price", 0)
                    break

            # ถ้าหาไม่เจอใน History → ประมาณจากราคาปัจจุบัน
            if actual_close_price == 0:
                price_info = self._connector.get_current_price(trade.symbol)
                if price_info:
                    actual_close_price = price_info["bid"] if trade.trade_type == "BUY" else price_info["ask"]

            self.record_external_close(
                ticket=ticket,
                close_price=actual_close_price,
                profit=actual_profit,
                reason="SL/TP Hit (synced)"
            )

    # =========================================================================
    # 📊 ข้อมูลสรุป
    # =========================================================================

    @property
    def active_trades(self) -> Dict[int, ExecutedTrade]:
        """เทรดที่เปิดอยู่ทั้งหมด"""
        return self._active_trades

    @property
    def closed_trades(self) -> List[ExecutedTrade]:
        """เทรดที่ปิดแล้วทั้งหมด"""
        return self._closed_trades

    @property
    def active_count(self) -> int:
        """จำนวนเทรดที่เปิดอยู่"""
        return len(self._active_trades)

    def get_stats(self) -> Dict:
        """สรุปสถิติการทำงาน"""
        wins = sum(1 for t in self._closed_trades if t.profit > 0)
        losses = sum(1 for t in self._closed_trades if t.profit <= 0)
        total_pnl = sum(t.profit for t in self._closed_trades)
        win_rate = (wins / len(self._closed_trades) * 100) if self._closed_trades else 0

        return {
            "total_executed": self._total_executed,
            "total_rejected": self._total_rejected,
            "active_trades": self.active_count,
            "closed_trades": len(self._closed_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        }

    def __repr__(self) -> str:
        return (
            f"TradeExecutor(active={self.active_count}, "
            f"executed={self._total_executed}, rejected={self._total_rejected})"
        )
