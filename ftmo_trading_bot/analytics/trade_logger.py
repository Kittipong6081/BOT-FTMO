"""
===============================================================================
FTMO Trading Bot — Trade Logger (ระบบบันทึกเทรดอัตโนมัติ)
===============================================================================
บันทึกทุกเทรดลงไฟล์ Excel (.xlsx) อัตโนมัติ

ความสามารถ:
1. บันทึกเทรดที่เปิด/ปิด ลง Sheet "Trades"
2. บันทึก Daily Summary ลง Sheet "Daily"
3. สร้างไฟล์ใหม่ทุกเดือน (เพื่อไม่ให้ไฟล์ใหญ่เกินไป)
4. Auto-format: สี, ความกว้างคอลัมน์, ตัวเลข

⚠️ ใช้ openpyxl — ไม่ต้องติดตั้ง Excel
===============================================================================
"""

import os
from datetime import datetime, date
from typing import Optional, Dict, List
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("⚠️ [คำเตือน] ไม่พบ openpyxl — ระบบ Logger จะบันทึกเฉพาะ Console")

from config.settings import bot_config


class TradeLogger:
    """
    ระบบบันทึกเทรดลง Excel อัตโนมัติ

    โครงสร้างไฟล์ .xlsx:
    ├── Sheet "Trades"   — รายการเทรดทั้งหมด (1 แถว = 1 เทรด)
    ├── Sheet "Daily"    — สรุปรายวัน (1 แถว = 1 วัน)
    └── Sheet "Stats"    — สถิติรวม (Win Rate, Sharpe, etc.)

    คอลัมน์ใน Trades:
    Ticket | Symbol | Type | Entry | SL | TP | Lot | Risk% | RR |
    Confluence | Open Time | Close Price | Close Time | P/L | Reason | Close Reason
    """

    # === คอลัมน์สำหรับ Sheet "Trades" ===
    TRADE_HEADERS = [
        "Ticket", "Symbol", "Type", "Entry", "SL", "TP", "Lot",
        "Risk%", "Risk$", "RR", "Confluence", "ATR",
        "Open Time", "Close Price", "Close Time",
        "P/L ($)", "P/L (%)", "Close Reason", "Signal Reasons"
    ]

    # === คอลัมน์สำหรับ Sheet "Daily" ===
    DAILY_HEADERS = [
        "Date", "Trades", "Wins", "Losses", "Win Rate%",
        "Gross Profit", "Gross Loss", "Net P/L",
        "Max DD%", "Daily DD%", "Balance EOD"
    ]

    def __init__(self, log_dir: str = None):
        """
        เริ่มต้น Trade Logger

        Args:
            log_dir: โฟลเดอร์สำหรับเก็บไฟล์ Log (ค่าเริ่มต้น: ./logs/)
        """
        self._log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        self._current_file: Optional[str] = None
        self._trade_count = 0
        self._daily_trades: List[Dict] = []  # เก็บเทรดของวันนี้

        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(self._log_dir, exist_ok=True)

        print(f"📝 [Trade Logger] เริ่มต้นระบบบันทึกเทรด")
        print(f"   📂 โฟลเดอร์: {self._log_dir}")

    # =========================================================================
    # 📝 บันทึกเทรดที่เปิด
    # =========================================================================

    def log_trade_opened(self, trade_data: Dict):
        """
        บันทึกเทรดที่เปิดใหม่

        Args:
            trade_data: ข้อมูลเทรดจาก ExecutedTrade.to_dict()
        """
        self._trade_count += 1
        print(f"📝 [Logger] บันทึกเทรดเปิด #{self._trade_count}: "
              f"{trade_data.get('symbol', '?')} {trade_data.get('type', '?')} "
              f"Lot={trade_data.get('lot_size', 0):.2f}")

        if not OPENPYXL_AVAILABLE:
            return

        try:
            wb, filepath = self._get_or_create_workbook()
            ws = wb["Trades"]

            # เพิ่มแถวใหม่
            row = [
                trade_data.get("ticket", 0),
                trade_data.get("symbol", ""),
                trade_data.get("type", ""),
                trade_data.get("entry_price", 0),
                trade_data.get("sl_price", 0),
                trade_data.get("tp_price", 0),
                trade_data.get("lot_size", 0),
                trade_data.get("risk_pct", 0),
                trade_data.get("risk_amount", 0),
                trade_data.get("rr_ratio", 0),
                trade_data.get("confluence", 0),
                trade_data.get("atr", 0),
                trade_data.get("open_time", ""),
                "",   # close_price (ยังไม่ปิด)
                "",   # close_time
                "",   # P/L $
                "",   # P/L %
                "",   # close_reason
                trade_data.get("reasons", ""),
            ]
            ws.append(row)

            # ใส่สี BUY=เขียว, SELL=แดง
            last_row = ws.max_row
            trade_type = trade_data.get("type", "")
            if trade_type == "BUY":
                ws.cell(row=last_row, column=3).fill = PatternFill("solid", fgColor="C6EFCE")
            elif trade_type == "SELL":
                ws.cell(row=last_row, column=3).fill = PatternFill("solid", fgColor="FFC7CE")

            wb.save(filepath)

        except Exception as e:
            print(f"⚠️ [Logger] บันทึกเทรดเปิดล้มเหลว: {e}")

    # =========================================================================
    # 📝 บันทึกเทรดที่ปิด
    # =========================================================================

    def log_trade_closed(self, trade_data: Dict):
        """
        อัพเดทเทรดที่ปิดแล้ว — เพิ่มราคาปิด, เวลาปิด, P/L

        Args:
            trade_data: ข้อมูลเทรดที่ปิดแล้ว
        """
        profit = trade_data.get("profit", 0)
        pnl_emoji = "🟢" if profit >= 0 else "🔴"
        print(f"{pnl_emoji} [Logger] บันทึกเทรดปิด: "
              f"{trade_data.get('symbol', '?')} P/L=${profit:,.2f}")

        # เก็บไว้สำหรับ Daily Summary
        self._daily_trades.append(trade_data)

        if not OPENPYXL_AVAILABLE:
            return

        try:
            wb, filepath = self._get_or_create_workbook()
            ws = wb["Trades"]

            ticket = trade_data.get("ticket", 0)

            # ค้นหาแถวที่ Ticket ตรงกัน
            target_row = None
            for row_idx in range(2, ws.max_row + 1):
                cell_val = ws.cell(row=row_idx, column=1).value
                if cell_val == ticket:
                    target_row = row_idx
                    break

            if target_row:
                # อัพเดทคอลัมน์ที่ยังว่าง
                ws.cell(row=target_row, column=14, value=trade_data.get("close_price", 0))
                ws.cell(row=target_row, column=15, value=trade_data.get("close_time", ""))
                ws.cell(row=target_row, column=16, value=profit)

                # คำนวณ P/L %
                risk_amount = trade_data.get("risk_amount", 1)
                pnl_pct = (profit / risk_amount * 100) if risk_amount > 0 else 0
                ws.cell(row=target_row, column=17, value=round(pnl_pct, 2))

                ws.cell(row=target_row, column=18, value=trade_data.get("close_reason", ""))

                # ใส่สี P/L
                pnl_fill = PatternFill("solid", fgColor="C6EFCE") if profit >= 0 else PatternFill("solid", fgColor="FFC7CE")
                ws.cell(row=target_row, column=16).fill = pnl_fill
            else:
                # ไม่เจอ Ticket — เพิ่มแถวใหม่พร้อมข้อมูลครบ
                row = [
                    ticket,
                    trade_data.get("symbol", ""),
                    trade_data.get("type", ""),
                    trade_data.get("entry_price", 0),
                    trade_data.get("sl_price", 0),
                    trade_data.get("tp_price", 0),
                    trade_data.get("lot_size", 0),
                    trade_data.get("risk_pct", 0),
                    trade_data.get("risk_amount", 0),
                    trade_data.get("rr_ratio", 0),
                    trade_data.get("confluence", 0),
                    trade_data.get("atr", 0),
                    trade_data.get("open_time", ""),
                    trade_data.get("close_price", 0),
                    trade_data.get("close_time", ""),
                    profit,
                    0,
                    trade_data.get("close_reason", ""),
                    trade_data.get("reasons", ""),
                ]
                ws.append(row)

            wb.save(filepath)

        except Exception as e:
            print(f"⚠️ [Logger] บันทึกเทรดปิดล้มเหลว: {e}")

    # =========================================================================
    # 📊 บันทึก Daily Summary
    # =========================================================================

    def log_daily_summary(
        self,
        balance: float,
        daily_dd_pct: float = 0.0,
        max_dd_pct: float = 0.0,
    ):
        """
        บันทึกสรุปรายวัน — เรียกตอนสิ้นวัน

        Args:
            balance: Balance สิ้นวัน
            daily_dd_pct: Daily Drawdown %
            max_dd_pct: Max Drawdown %
        """
        today = date.today().isoformat()
        trades = self._daily_trades
        total = len(trades)
        wins = sum(1 for t in trades if t.get("profit", 0) > 0)
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0

        gross_profit = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
        gross_loss = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) <= 0)
        net_pnl = gross_profit + gross_loss

        print(f"\n📊 [Logger] === สรุปรายวัน {today} ===")
        print(f"   📈 เทรดทั้งหมด: {total} (ชนะ {wins}, แพ้ {losses})")
        print(f"   🏆 Win Rate: {win_rate:.0f}%")
        print(f"   💰 Net P/L: ${net_pnl:,.2f}")
        print(f"   💎 Balance: ${balance:,.2f}")

        if not OPENPYXL_AVAILABLE:
            self._daily_trades.clear()
            return

        try:
            wb, filepath = self._get_or_create_workbook()

            if "Daily" not in wb.sheetnames:
                self._create_daily_sheet(wb)

            ws = wb["Daily"]
            row = [
                today, total, wins, losses, round(win_rate, 1),
                round(gross_profit, 2), round(gross_loss, 2), round(net_pnl, 2),
                round(max_dd_pct * 100, 2), round(daily_dd_pct * 100, 2),
                round(balance, 2)
            ]
            ws.append(row)

            # ใส่สี Net P/L
            last_row = ws.max_row
            pnl_fill = PatternFill("solid", fgColor="C6EFCE") if net_pnl >= 0 else PatternFill("solid", fgColor="FFC7CE")
            ws.cell(row=last_row, column=8).fill = pnl_fill

            wb.save(filepath)
            print(f"   📂 บันทึกลง: {filepath}")

        except Exception as e:
            print(f"⚠️ [Logger] บันทึก Daily Summary ล้มเหลว: {e}")

        # เคลียร์เทรดของวันนี้
        self._daily_trades.clear()

    # =========================================================================
    # 📂 จัดการไฟล์ Excel
    # =========================================================================

    def _get_or_create_workbook(self):
        """
        ดึง Workbook ปัจจุบัน หรือสร้างใหม่ถ้ายังไม่มี

        สร้างไฟล์ใหม่ทุกเดือน: ftmo_trades_2026_04.xlsx

        Returns:
            Tuple[Workbook, str]: (Workbook, filepath)
        """
        now = datetime.now()
        filename = f"ftmo_trades_{now.strftime('%Y_%m')}.xlsx"
        filepath = os.path.join(self._log_dir, filename)

        if os.path.exists(filepath):
            wb = openpyxl.load_workbook(filepath)
        else:
            wb = self._create_new_workbook()
            wb.save(filepath)
            print(f"📂 [Logger] สร้างไฟล์ใหม่: {filepath}")

        self._current_file = filepath
        return wb, filepath

    def _create_new_workbook(self):
        """สร้าง Workbook ใหม่พร้อม Headers และ Formatting"""
        wb = openpyxl.Workbook()

        # === Sheet "Trades" ===
        ws_trades = wb.active
        ws_trades.title = "Trades"
        self._setup_trade_sheet(ws_trades)

        # === Sheet "Daily" ===
        ws_daily = wb.create_sheet("Daily")
        self._create_daily_sheet(wb)

        # === Sheet "Stats" ===
        ws_stats = wb.create_sheet("Stats")
        self._setup_stats_sheet(ws_stats)

        return wb

    def _setup_trade_sheet(self, ws):
        """ตั้งค่า Sheet Trades — Headers + Formatting"""
        # Header Style
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill("solid", fgColor="2F5496")
        header_align = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # เพิ่ม Headers
        ws.append(self.TRADE_HEADERS)
        for col_idx, header in enumerate(self.TRADE_HEADERS, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = thin_border

        # กำหนดความกว้างคอลัมน์
        col_widths = [12, 10, 6, 10, 10, 10, 7, 7, 10, 6, 10, 10, 20, 10, 20, 12, 8, 20, 40]
        for idx, width in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(idx)].width = width

        # Freeze Header
        ws.freeze_panes = "A2"

    def _create_daily_sheet(self, wb):
        """ตั้งค่า Sheet Daily"""
        if "Daily" not in wb.sheetnames:
            wb.create_sheet("Daily")

        ws = wb["Daily"]

        # ถ้ายังไม่มี Header
        if ws.max_row <= 1 and ws.cell(row=1, column=1).value is None:
            header_font = Font(bold=True, color="FFFFFF", size=11)
            header_fill = PatternFill("solid", fgColor="548235")

            ws.append(self.DAILY_HEADERS)
            for col_idx, header in enumerate(self.DAILY_HEADERS, 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

            col_widths = [12, 8, 8, 8, 10, 12, 12, 12, 8, 10, 14]
            for idx, width in enumerate(col_widths, 1):
                ws.column_dimensions[get_column_letter(idx)].width = width

            ws.freeze_panes = "A2"

    def _setup_stats_sheet(self, ws):
        """ตั้งค่า Sheet Stats — สถิติรวม"""
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill("solid", fgColor="BF8F00")

        stats_labels = [
            "Metric", "Value"
        ]
        ws.append(stats_labels)
        for col_idx, header in enumerate(stats_labels, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        ws.column_dimensions["A"].width = 25
        ws.column_dimensions["B"].width = 20
        ws.freeze_panes = "A2"

    # =========================================================================
    # 📊 อัพเดท Sheet Stats
    # =========================================================================

    def update_stats_sheet(self, stats: Dict):
        """
        อัพเดท Sheet "Stats" ด้วยสถิติล่าสุด

        Args:
            stats: Dict จาก PerformanceAnalyzer.get_full_report()
        """
        if not OPENPYXL_AVAILABLE:
            return

        try:
            wb, filepath = self._get_or_create_workbook()

            if "Stats" not in wb.sheetnames:
                ws = wb.create_sheet("Stats")
                self._setup_stats_sheet(ws)
            else:
                ws = wb["Stats"]

            # เคลียร์ข้อมูลเดิม (เก็บ Header)
            for row_idx in range(ws.max_row, 1, -1):
                ws.delete_rows(row_idx)

            # เพิ่มข้อมูลใหม่
            value_font = Font(size=11)
            label_font = Font(bold=True, size=11)

            for key, value in stats.items():
                ws.append([key, value])
                last_row = ws.max_row
                ws.cell(row=last_row, column=1).font = label_font
                ws.cell(row=last_row, column=2).font = value_font
                ws.cell(row=last_row, column=2).alignment = Alignment(horizontal="right")

            wb.save(filepath)

        except Exception as e:
            print(f"⚠️ [Logger] อัพเดท Stats ล้มเหลว: {e}")

    # =========================================================================
    # 📊 ข้อมูลสรุป
    # =========================================================================

    @property
    def current_file(self) -> Optional[str]:
        """ไฟล์ Log ปัจจุบัน"""
        return self._current_file

    @property
    def trade_count(self) -> int:
        """จำนวนเทรดที่บันทึก"""
        return self._trade_count

    def get_daily_trades(self) -> List[Dict]:
        """เทรดของวันนี้"""
        return self._daily_trades.copy()

    def __repr__(self) -> str:
        return f"TradeLogger(trades={self._trade_count}, file={self._current_file})"
