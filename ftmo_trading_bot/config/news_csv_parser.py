"""
===============================================================================
News CSV Parser — แปลง ForexFactory CSV → JSON events
===============================================================================
ใช้ร่วมกันโดย:
- scripts/import_forexfactory_csv.py (CLI manual)
- core/news_scheduler.py (auto scheduler)
===============================================================================
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


# แปลง country code ของ ForexFactory → ISO currency code
_COUNTRY_TO_CURRENCY = {
    "USD": "USD", "EUR": "EUR", "GBP": "GBP", "JPY": "JPY",
    "AUD": "AUD", "CAD": "CAD", "CHF": "CHF", "NZD": "NZD",
    # ForexFactory บางครั้งใช้ชื่อเต็ม
    "United States": "USD", "Euro Area": "EUR", "United Kingdom": "GBP",
    "Japan": "JPY", "Australia": "AUD", "Canada": "CAD",
    "Switzerland": "CHF", "New Zealand": "NZD",
}


def parse_forexfactory_csv(
    csv_path: Path,
    source_tz_offset_hours: float = 0.0,
    verbose: bool = True,
) -> List[Dict]:
    """
    Parse ForexFactory CSV → list of event dicts

    Args:
        csv_path: Path to CSV file
        source_tz_offset_hours: timezone offset ของ CSV (เทียบ UTC)
            0 = UTC (ตั้ง ForexFactory เป็น GMT/UTC ก่อน export)
            -5 = EST, +7 = ICT
        verbose: print สรุปผลไหม (False สำหรับ scheduler ที่เรียกจาก bot loop)

    Returns:
        List of dicts: {datetime_utc, currency, name, impact}
        กรองเฉพาะ High-impact, currencies ที่รู้จัก, มีเวลาแน่นอน
    """
    events: List[Dict] = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter high-impact only
            impact = (row.get("Impact") or "").strip().lower()
            if impact not in ("high", "holiday"):
                skipped += 1
                continue

            # Currency mapping
            country = (row.get("Country") or row.get("Currency") or "").strip()
            currency = _COUNTRY_TO_CURRENCY.get(country) or _COUNTRY_TO_CURRENCY.get(country.upper())
            if not currency:
                skipped += 1
                continue

            # Date + Time parsing
            date_str = (row.get("Date") or "").strip()
            time_str = (row.get("Time") or "").strip()

            # All Day / Tentative events — ข้าม (ไม่มีเวลาแน่นอน)
            if not time_str or time_str.lower() in ("all day", "tentative", ""):
                skipped += 1
                continue

            # Try multiple date formats
            dt_local = None
            for date_fmt in ("%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                for time_fmt in ("%I:%M%p", "%H:%M", "%I:%M %p"):
                    try:
                        dt_local = datetime.strptime(
                            f"{date_str} {time_str}", f"{date_fmt} {time_fmt}"
                        )
                        break
                    except ValueError:
                        continue
                if dt_local:
                    break

            if dt_local is None:
                skipped += 1
                continue

            # Convert to UTC
            dt_utc = dt_local - timedelta(hours=source_tz_offset_hours)

            events.append({
                "datetime_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "currency": currency,
                "name": (row.get("Title") or row.get("Event") or "Unknown").strip(),
                "impact": "high",
            })

    if verbose:
        print(f"✅ แปลง {len(events)} events (ข้าม {skipped} low-impact/invalid)")
    return events
