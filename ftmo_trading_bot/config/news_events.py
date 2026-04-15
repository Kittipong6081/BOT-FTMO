"""
===============================================================================
FTMO Trading Bot — High-Impact News Filter
===============================================================================
Hardcoded economic events ที่ส่งผลต่อ spread/slippage อย่างรุนแรง
Block signal ในช่วงก่อน/หลังข่าวตาม config.no_trade_before_news_minutes

กติกา:
- เก็บ events เป็น recurring rules (ไม่ต้องพึ่ง external API)
- เวลาทุกอย่างเป็น **UTC** (เทียบกับ server time ที่แปลงเป็น UTC แล้ว)
- Symbol filter: ข่าว USD กระทบเกือบทุกคู่, ข่าวเฉพาะประเทศกระทบเฉพาะคู่ที่เกี่ยวข้อง

Sources: economic calendar ForexFactory/Investing (high-impact recurring events)
===============================================================================
"""

from datetime import datetime, time, timedelta
from typing import List, Optional, Set
from dataclasses import dataclass, field
import pytz


@dataclass
class RecurringNewsEvent:
    """
    ข่าวที่เกิดซ้ำตามรอบ (weekly/monthly)
    """
    name: str                       # ชื่อข่าว เช่น "NFP"
    currencies: Set[str]            # ประเทศ/สกุลที่กระทบ เช่น {"USD"}
    # Recurrence pattern
    weekday: Optional[int] = None   # 0=Mon ... 6=Sun (None = ไม่จำกัด)
    nth_weekday_of_month: Optional[int] = None  # 1=first, 2=second ...; None = ทุกครั้งของ weekday
    day_of_month: Optional[int] = None  # วันที่คงที่ (e.g. 1=ต้นเดือน) — ถ้ากำหนดจะ override weekday
    time_utc: time = time(13, 30)   # เวลาปล่อยข่าว (UTC)

    def occurs_on(self, dt_utc: datetime) -> bool:
        """ตรวจว่า dt_utc (วันที่) ตรงกับรอบของข่าวนี้หรือไม่"""
        if self.day_of_month is not None:
            return dt_utc.day == self.day_of_month
        if self.weekday is None:
            return False
        if dt_utc.weekday() != self.weekday:
            return False
        if self.nth_weekday_of_month is None:
            return True
        # คำนวณว่าเป็น weekday ครั้งที่เท่าไรของเดือน
        nth = (dt_utc.day - 1) // 7 + 1
        return nth == self.nth_weekday_of_month


# =============================================================================
# High-impact recurring events (UTC)
# =============================================================================
# หมายเหตุ: เวลาเหล่านี้อาจเลื่อน ±30 นาทีตาม DST — window filter 30 min ก่อน/หลัง
# ครอบคลุมความเสี่ยงได้ดีพอสำหรับการสอบ 30 วัน
HIGH_IMPACT_EVENTS: List[RecurringNewsEvent] = [
    # === USD ===
    RecurringNewsEvent(
        name="NFP (Non-Farm Payrolls)",
        currencies={"USD"},
        weekday=4,  # Friday
        nth_weekday_of_month=1,  # First Friday
        time_utc=time(13, 30),
    ),
    RecurringNewsEvent(
        name="US CPI",
        currencies={"USD"},
        # CPI ไม่มี pattern weekday ชัด — ใช้ day_of_month ~10-15 (ประมาณการ)
        day_of_month=12,
        time_utc=time(13, 30),
    ),
    RecurringNewsEvent(
        name="FOMC Rate Decision (approx)",
        currencies={"USD"},
        # FOMC เดือนละครั้งช่วงกลางเดือน — ประมาณ Wednesday ที่ 3 ของบางเดือน
        weekday=2,  # Wednesday
        nth_weekday_of_month=3,
        time_utc=time(18, 0),
    ),
    # === EUR ===
    RecurringNewsEvent(
        name="ECB Rate Decision (approx)",
        currencies={"EUR"},
        weekday=3,  # Thursday
        nth_weekday_of_month=2,
        time_utc=time(12, 45),
    ),
    # === GBP ===
    RecurringNewsEvent(
        name="BoE Rate Decision (approx)",
        currencies={"GBP"},
        weekday=3,  # Thursday
        nth_weekday_of_month=1,
        time_utc=time(11, 0),
    ),
    # === JPY ===
    # BoJ ไม่มี pattern ที่ regular พอจะ hardcode — ข้าม (ใช้ general USD events ก็กัน JPY cross ได้มาก)
]


# Currency → symbols ที่ถูกกระทบ
_CURRENCY_TO_SYMBOLS = {
    "USD": {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"},
    "EUR": {"EURUSD", "EURJPY"},
    "GBP": {"GBPUSD", "GBPJPY"},
    "JPY": {"USDJPY", "EURJPY", "GBPJPY"},
    "AUD": {"AUDUSD"},
    "CAD": {"USDCAD"},
    "CHF": {"USDCHF"},
    "NZD": {"NZDUSD"},
}


def is_near_high_impact_news(
    symbol: str,
    now_utc: datetime,
    window_minutes_before: int = 30,
    window_minutes_after: int = 30,
) -> tuple[bool, str]:
    """
    ตรวจว่า symbol กำลังจะชน/เพิ่งชนข่าวแรงหรือไม่

    Args:
        symbol: คู่เงิน เช่น "EURUSD"
        now_utc: เวลาปัจจุบัน (tz-aware UTC หรือ naive UTC)
        window_minutes_before: กี่นาทีก่อนข่าวจึงจะ block
        window_minutes_after: กี่นาทีหลังข่าวจึงจะยังคง block

    Returns:
        (is_near, reason_str) — is_near=True ถ้าอยู่ใน window
    """
    # Normalize to naive UTC for comparison
    if now_utc.tzinfo is not None:
        now_utc = now_utc.astimezone(pytz.UTC).replace(tzinfo=None)

    # วันที่สำหรับตรวจ: วันนี้ + พรุ่งนี้ (เผื่อข่าวเที่ยงคืน UTC)
    dates_to_check = [now_utc.date(), (now_utc + timedelta(days=1)).date()]

    for event in HIGH_IMPACT_EVENTS:
        # Symbol ไม่ได้รับผลกระทบจากข่าวนี้
        impacted = set()
        for ccy in event.currencies:
            impacted.update(_CURRENCY_TO_SYMBOLS.get(ccy, set()))
        if symbol not in impacted:
            continue

        for d in dates_to_check:
            event_dt = datetime.combine(d, event.time_utc)
            if not event.occurs_on(event_dt):
                continue
            start = event_dt - timedelta(minutes=window_minutes_before)
            end = event_dt + timedelta(minutes=window_minutes_after)
            if start <= now_utc <= end:
                delta_min = (event_dt - now_utc).total_seconds() / 60.0
                when = f"{abs(delta_min):.0f} นาที{'ก่อน' if delta_min > 0 else 'หลัง'}"
                return (True, f"⚠️ {event.name} ({when}) — block signal {symbol}")

    return (False, "")
