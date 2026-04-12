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
from datetime import datetime
from config.settings import bot_config


class DiscordNotifier:
    def __init__(self):
        self.config = bot_config.notifications
        self.webhook_url = self.config.discord_webhook_url
        self.enabled = self.config.enable_notifications

    def _send_payload(self, payload: dict):
        if not self.enabled or not self.webhook_url:
            return
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers, timeout=5)
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
                "timestamp": datetime.utcnow().isoformat(),
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
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "System Offline"}
            }]
        }
        self._send_payload(payload)

    def send_trade_open(self, trade_dict: dict):
        """แจ้งเตือนเมื่อเปิดเทรด (มีสีต่างกันตามประเภท)"""
        is_buy = trade_dict.get("type", "").upper() == "BUY"
        color = 3447003 if is_buy else 15158332  # Blue for BUY, Red for SELL
        emoji = "🔵" if is_buy else "🔴"
        
        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": f"{emoji} OPENED: {trade_dict.get('type')} {trade_dict.get('symbol')}",
                "color": color,
                "fields": [
                    {"name": "Ticket", "value": str(trade_dict.get("ticket")), "inline": True},
                    {"name": "Lot Size", "value": f"{trade_dict.get('lot_size'):.2f}", "inline": True},
                    {"name": "Entry", "value": str(trade_dict.get("entry_price")), "inline": True},
                    {"name": "SL", "value": str(trade_dict.get("sl_price")), "inline": True},
                    {"name": "TP", "value": str(trade_dict.get("tp_price")), "inline": True},
                    {"name": "Risk", "value": f"{trade_dict.get('risk_pct', 0):.2%} (${trade_dict.get('risk_amount', 0):.2f})", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        self._send_payload(payload)

    def send_trade_close(self, trade_dict: dict):
        """แจ้งเตือนเมื่อเทรดถูกปิด พร้อม P/L"""
        profit = float(trade_dict.get("profit", 0))
        is_win = profit > 0
        color = 3066993 if is_win else 15158332  # Green if win, Red if loss
        emoji = "✅" if is_win else "❌"
        
        payload = {
            "username": "FTMO Bot",
            "embeds": [{
                "title": f"{emoji} CLOSED: {trade_dict.get('type')} {trade_dict.get('symbol')}",
                "description": f"**P/L: ${profit:,.2f}**\nReason: {trade_dict.get('close_reason')}",
                "color": color,
                "fields": [
                    {"name": "Ticket", "value": str(trade_dict.get("ticket")), "inline": True},
                    {"name": "Entry", "value": str(trade_dict.get("entry_price")), "inline": True},
                    {"name": "Close Price", "value": str(trade_dict.get("close_price")), "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat()
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
                "timestamp": datetime.utcnow().isoformat()
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
                "timestamp": datetime.utcnow().isoformat()
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
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        self._send_payload(payload)
