"""
===============================================================================
FTMO Trading Bot — Discord Notifier (ระบบแจ้งเตือนผ่าน Discord Webhook)
===============================================================================
ทำหน้าที่ส่งข้อความและ Embed Info ไปยัง Discord เพื่อแจ้งเตือนสถานะต่างๆ 
เช่น การเปิด/ปิดออเดอร์, แจ้งเตือนความเสี่ยง, และรายงานสถานะรายชั่วโมง
===============================================================================
"""

import requests
import json
import time
import threading
from collections import deque
from datetime import datetime, timezone
from config.settings import bot_config


class DiscordNotifier:
    """
    ระบบส่งการแจ้งเตือนผ่าน Discord Webhook พร้อม Rate Limiting
    Discord จำกัดที่ ~30 requests/minute ต่อ webhook — เราจำกัดที่ 20/min เพื่อความปลอดภัย
    """

    # Discord webhook rate limit (ใช้ sliding window)
    MAX_REQUESTS_PER_MINUTE: int = 20
    WINDOW_SECONDS: float = 60.0
    # เว้นขั้นต่ำระหว่าง request (วินาที) เพื่อไม่ให้ spam
    MIN_INTERVAL_SECONDS: float = 1.0

    def __init__(self):
        self.config = bot_config.notifications
        self.webhook_url = self.config.discord_webhook_url
        self.enabled = self.config.enable_notifications

        # Rate limiting state
        self._request_timestamps: deque = deque()
        self._last_request_time: float = 0.0
        self._rate_limit_lock = threading.Lock()
        self._dropped_count: int = 0  # counter สำหรับ messages ที่ถูก drop

    def _check_rate_limit(self) -> bool:
        """
        ตรวจสอบ rate limit — คืน True ถ้าส่งได้, False ถ้าต้อง drop message

        ใช้ sliding window 60 วินาที + minimum interval
        """
        with self._rate_limit_lock:
            now = time.monotonic()

            # ล้าง timestamp เก่าที่ออกจาก window แล้ว
            while self._request_timestamps and (now - self._request_timestamps[0]) > self.WINDOW_SECONDS:
                self._request_timestamps.popleft()

            # ตรวจ min interval (กัน burst)
            if (now - self._last_request_time) < self.MIN_INTERVAL_SECONDS:
                self._dropped_count += 1
                return False

            # ตรวจ max req/minute
            if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
                self._dropped_count += 1
                return False

            # อนุญาต — บันทึก timestamp
            self._request_timestamps.append(now)
            self._last_request_time = now
            return True

    def _send_payload(self, payload: dict):
        if not self.enabled or not self.webhook_url:
            return

        # ตรวจ rate limit ก่อนส่ง
        if not self._check_rate_limit():
            # Log dropped message (ทุกๆ 10 ครั้งเพื่อไม่ spam)
            if self._dropped_count % 10 == 1:
                print(f"⏸️  [Notifier] Rate limited — drop Discord message "
                      f"(dropped {self._dropped_count} total)")
            return

        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=5,
            )
            # Discord อาจตอบ 429 พร้อม Retry-After ถ้าโดน rate limit จริง
            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get("Retry-After", 1))
                except Exception:
                    retry_after = 1.0
                # เลื่อน window ออกไปตาม Retry-After เพื่อหลีกเลี่ยงการยิงต่อ
                with self._rate_limit_lock:
                    penalty_until = time.monotonic() + retry_after
                    # Push ghost timestamps เพื่อ block การส่งจนกว่า penalty_until
                    while len(self._request_timestamps) < self.MAX_REQUESTS_PER_MINUTE:
                        self._request_timestamps.append(penalty_until)
                print(f"⚠️ [Notifier] Discord 429 — backing off {retry_after:.1f}s")
                return
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️ [Notifier] ไม่สามารถส่งแจ้งเตือน Discord ได้: {e}")

    def send_startup(self):
        """แจ้งเตือนเมื่อบอทเริ่มทำงาน"""
        payload = {
            "username": "FTMO Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
            "embeds": [{
                "title": "🚀 [System Output] Bot Started",
                "description": "ระบบ FTMO Trading Bot เริ่มทำงานแล้ว และพร้อมทำการแสวงหากำไร",
                "color": 3066993,  # Green
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "System Online"}
            }]
        }
        self._send_payload(payload)

    def send_shutdown(self):
        """แจ้งเตือนเมื่อบอทหยุดทำงาน"""
        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": "🛑 [System Output] Bot Shutdown",
                "description": "ระบบ FTMO Trading Bot หยุดทำงานแล้วเรียบร้อย",
                "color": 15158332,  # Red
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "System Offline"}
            }]
        }
        self._send_payload(payload)

    @staticmethod
    def _fmt_price(v) -> str:
        """Format ราคาให้ปลอดภัย — ถ้า 0/None แสดง 'N/A' ไม่ให้โชว์ 0"""
        try:
            x = float(v or 0)
        except Exception:
            return "N/A"
        if x == 0:
            return "N/A"
        # 5 decimals for FX, auto-trim trailing zeros
        s = f"{x:.5f}".rstrip("0").rstrip(".")
        return s or "N/A"

    @staticmethod
    def _fmt_num(v, fmt: str = ",.2f") -> str:
        try:
            return format(float(v or 0), fmt)
        except Exception:
            return "0"

    def send_trade_open(self, trade_dict: dict):
        """แจ้งเตือนเมื่อเปิดเทรด (มีสีต่างกันตามประเภท)"""
        is_buy = (trade_dict.get("type") or "").upper() == "BUY"
        color = 3447003 if is_buy else 15158332  # Blue for BUY, Red for SELL
        emoji = "🔵" if is_buy else "🔴"

        entry_str = self._fmt_price(trade_dict.get("entry_price"))
        sl_str = self._fmt_price(trade_dict.get("sl_price"))
        tp_str = self._fmt_price(trade_dict.get("tp_price"))
        # ถ้า entry ยังเป็น 0 แปลว่า MT5 ไม่ได้คืน price → แสดง warning
        entry_warn = " ⚠️" if entry_str == "N/A" else ""

        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": f"{emoji} OPENED: {trade_dict.get('type')} {trade_dict.get('symbol')}",
                "color": color,
                "fields": [
                    {"name": "Ticket", "value": str(trade_dict.get("ticket") or "N/A"), "inline": True},
                    {"name": "Lot Size", "value": self._fmt_num(trade_dict.get("lot_size"), ",.2f"), "inline": True},
                    {"name": "Entry", "value": entry_str + entry_warn, "inline": True},
                    {"name": "SL", "value": sl_str, "inline": True},
                    {"name": "TP", "value": tp_str, "inline": True},
                    {"name": "Risk", "value": f"{(trade_dict.get('risk_pct') or 0):.2%} (${self._fmt_num(trade_dict.get('risk_amount'), ',.2f')})", "inline": True},
                    {"name": "Confluence", "value": self._fmt_num(trade_dict.get("confluence"), ".0f"), "inline": True},
                    {"name": "Session", "value": str(trade_dict.get("session") or "-"), "inline": True},
                    {"name": "RR", "value": f"1:{(trade_dict.get('rr_ratio') or 0):.1f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)

    def send_trade_close(self, trade_dict: dict):
        """แจ้งเตือนเมื่อเทรดถูกปิด พร้อม P/L (แสดงค่า 0 ไม่เงียบ)"""
        profit = float(trade_dict.get("profit") or 0)
        risk_amt = float(trade_dict.get("risk_amount") or 0)
        pnl_pct = (profit / risk_amt * 100) if risk_amt > 0 else 0.0
        is_win = profit > 0
        color = 3066993 if is_win else 15158332
        emoji = "✅" if is_win else "❌"

        # แสดงเตือนถ้า profit เป็น 0 ทั้งที่ไม่ควร (เช่น sync ไม่เจอ deal)
        pnl_line = f"**P/L: ${profit:,.2f}**  ({pnl_pct:+.1f}R)"
        reason = trade_dict.get("close_reason") or trade_dict.get("exit_path") or "-"

        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": f"{emoji} CLOSED: {trade_dict.get('type')} {trade_dict.get('symbol')}",
                "description": f"{pnl_line}\nReason: {reason}",
                "color": color,
                "fields": [
                    {"name": "Ticket", "value": str(trade_dict.get("ticket") or "N/A"), "inline": True},
                    {"name": "Entry", "value": self._fmt_price(trade_dict.get("entry_price")), "inline": True},
                    {"name": "Close", "value": self._fmt_price(trade_dict.get("close_price")), "inline": True},
                    {"name": "SL", "value": self._fmt_price(trade_dict.get("sl_price")), "inline": True},
                    {"name": "TP", "value": self._fmt_price(trade_dict.get("tp_price")), "inline": True},
                    {"name": "Time-in-Trade", "value": f"{int(trade_dict.get('time_in_trade') or 0)}s", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)

    def send_trade_partial_close(
        self,
        trade_dict: dict,
        closed_volume: float,
        remaining_volume: float,
        fill_price: float,
        realized_pnl: float,
    ):
        """
        แจ้งเตือนเมื่อบอทปิด Position บางส่วน (partial close)

        Args:
            trade_dict: trade ที่แปลงเป็น dict (มี symbol, type, ticket, entry_price, ...)
            closed_volume: lot ที่ปิด (partial)
            remaining_volume: lot ที่เหลืออยู่
            fill_price: ราคาที่ปิดจริง
            realized_pnl: P/L realized จาก partial นี้ (USD, ประมาณ)
        """
        pnl_sign = "+" if realized_pnl >= 0 else ""
        original_lot = float(trade_dict.get("lot_size") or 0) + closed_volume
        pct_closed = (closed_volume / original_lot * 100) if original_lot > 0 else 0.0
        color = 3447003  # Blue — partial ไม่ใช่เขียว/แดงเต็ม

        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": f"✂️ PARTIAL CLOSE: {trade_dict.get('type')} {trade_dict.get('symbol')}",
                "description": (
                    f"**Realized P/L: {pnl_sign}${realized_pnl:,.2f}** "
                    f"(ปิด {pct_closed:.0f}% ของ position)"
                ),
                "color": color,
                "fields": [
                    {"name": "Ticket", "value": str(trade_dict.get("ticket") or "N/A"), "inline": True},
                    {"name": "Entry", "value": self._fmt_price(trade_dict.get("entry_price")), "inline": True},
                    {"name": "Partial Fill", "value": self._fmt_price(fill_price), "inline": True},
                    {"name": "Closed Lot", "value": f"{closed_volume:.2f}", "inline": True},
                    {"name": "Remaining Lot", "value": f"{remaining_volume:.2f}", "inline": True},
                    {"name": "TP (final)", "value": self._fmt_price(trade_dict.get("tp_price")), "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)

    def send_risk_alert(self, title: str, message: str):
        """แจ้งเตือนความเสี่ยงฉุกเฉิน (เช่น Daily Limit)"""
        payload = {
            "username": "FTMO Bot",
            "content": "@here ⚠️ **EMERGENCY RISK ALERT**",
            "embeds": [{
                "title": title,
                "description": message,
                "color": 16753920,  # Orange
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)

    def send_periodic_status(self, risk_status: dict, loop_count: int, uptime_str: str):
        """รายงานภาพรวมสถานะตามช่วงเวลา (เช่น รายชั่วโมง)"""
        payload = {
            "username": "FTMO Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/9334/9334415.png",
            "embeds": [{
                "title": "📊 Hourly Status Report",
                "description": "ภาพรวมพอร์ตและการทำงานของบอทปัจจุบัน",
                "color": 9807270,  # Grey/Purple
                "fields": [
                    {"name": "Balance", "value": f"${risk_status.get('current_balance', 0):,.2f}", "inline": True},
                    {"name": "Equity", "value": f"${risk_status.get('current_equity', 0):,.2f}", "inline": True},
                    {"name": "Daily DD", "value": f"{risk_status.get('daily_loss_pct', 0):.2%}", "inline": True},
                    {"name": "Max DD", "value": f"{risk_status.get('overall_drawdown_pct', 0):.2%}", "inline": True},
                    {"name": "Positions", "value": f"{risk_status.get('open_positions', 0)} / {risk_status.get('max_positions', 0)}", "inline": True},
                    {"name": "Uptime", "value": uptime_str, "inline": True},
                ],
                "footer": {"text": f"Loop #{loop_count}"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)

    def send_ai_tuning(self, params: dict):
        """แจ้งเตือนเมื่อ AI จูนระบบเสร็จ"""
        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": "🧠 AI Parameters Tuned",
                "description": "Reinforcement Learning Agent ปรับตั้งค่าเสร็จสิ้น",
                "color": 10181046,  # Purple for AI
                "fields": [
                    {"name": "Risk per trade", "value": f"{params.get('risk_per_trade_pct',0)*100:.2f}%", "inline": True},
                    {"name": "Confluence Min", "value": str(params.get('min_confluence_score')), "inline": True},
                    {"name": "Target R:R", "value": f"1:{params.get('preferred_risk_reward_ratio',0):.1f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }
        self._send_payload(payload)
