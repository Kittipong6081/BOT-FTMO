"""
===============================================================================
News Calendar Auto-Scheduler
===============================================================================
Auto-import ForexFactory CSV → news_calendar.json ทุกวันอาทิตย์ 23:30 EET

วิธีใช้ (user workflow):
1. User วาง CSV (export จาก ForexFactory) ใน config/news_inbox/
2. Bot รันปกติ — loop hook เรียก check_and_run() ทุก tick
3. ถึงอาทิตย์ 23:30 EET → scheduler หา CSV ล่าสุด → import → move ไป processed/

Safety:
- รันครั้งเดียวต่อวัน (state file ป้องกัน)
- Catch exception ทั้งหมด → ไม่ให้ block main loop
- ถ้าไม่มี CSV → skip เงียบ (บันทึก last_run กัน log spam)
===============================================================================
"""

import json
import shutil
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

from core.time_manager import TimeManager
from config.news_csv_parser import parse_forexfactory_csv


class NewsCalendarScheduler:
    """
    Auto-import news CSV ทุกวันอาทิตย์ 23:30 EET

    Args:
        inbox_dir: folder ที่ user วาง CSV
        output_json: path news_calendar.json
        state_file: path สำหรับ track last_run_date
        run_weekday: 0=Mon ... 6=Sun (default 6 = Sunday)
        run_time: เวลาเริ่มทำงาน (default 23:30 EET)
    """

    def __init__(
        self,
        inbox_dir: Path,
        output_json: Path,
        state_file: Path,
        run_weekday: int = 6,
        run_time: dtime = dtime(23, 30),
    ):
        self._inbox_dir = Path(inbox_dir)
        self._processed_dir = self._inbox_dir / "processed"
        self._output_json = Path(output_json)
        self._state_file = Path(state_file)
        self._run_weekday = run_weekday
        self._run_time = run_time

        # สร้าง folder ถ้ายังไม่มี
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"📅 [NewsScheduler] เริ่มต้น — inbox={self._inbox_dir.name}, "
              f"schedule={['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][run_weekday]} "
              f"{run_time.strftime('%H:%M')} EET")

    def check_and_run(self) -> bool:
        """
        เรียกทุก loop tick — ตรวจและรัน import ถ้าถึงเวลา

        Returns:
            bool: True ถ้ารัน import สำเร็จ (event update), False ถ้าไม่ถึงเวลา/skip
        """
        try:
            now_server = TimeManager.get_server_time()  # EET

            if not self._is_scheduled_window(now_server):
                return False

            # กันรันซ้ำวันเดียวกัน
            today_str = now_server.date().isoformat()
            if self._get_last_run_date() == today_str:
                return False

            return self._run_import(now_server)
        except Exception as e:
            print(f"⚠️ [NewsScheduler] check_and_run error: {e}")
            return False

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _is_scheduled_window(self, now: datetime) -> bool:
        """
        ตรวจว่าอยู่ใน window ที่ควรรัน (อาทิตย์ 23:30-23:59 EET)
        Window 30 นาที เผื่อ bot sleep 60s ช่วง weekend — มีเวลาให้ catch
        """
        if now.weekday() != self._run_weekday:
            return False
        # หน้าต่าง run_time → 23:59:59 ของวันนั้น
        return now.time() >= self._run_time

    def _get_last_run_date(self) -> Optional[str]:
        if not self._state_file.exists():
            return None
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f).get("last_run_date")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_last_run(self, date_str: str):
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump({"last_run_date": date_str}, f)
        except OSError as e:
            print(f"⚠️ [NewsScheduler] บันทึก state ไม่สำเร็จ: {e}")

    def _run_import(self, now: datetime) -> bool:
        """รัน import — รันได้แค่ครั้งเดียวต่อวัน (protected by caller)"""
        today_str = now.date().isoformat()

        csv_files = sorted(self._inbox_dir.glob("*.csv"))
        if not csv_files:
            print(f"📅 [NewsScheduler] ถึงเวลา import แต่ไม่พบ CSV ใน "
                  f"{self._inbox_dir.name}/ — ข้าม (ใช้ไฟล์เดิมหรือ fallback hardcoded)")
            self._save_last_run(today_str)  # กัน log ซ้ำทุก loop
            return False

        # ใช้ CSV ล่าสุด (sort by name — ถ้า user ใส่หลายไฟล์)
        csv_path = csv_files[-1]
        print(f"📅 [NewsScheduler] พบ {csv_path.name} — เริ่ม import...")

        try:
            events = parse_forexfactory_csv(csv_path, source_tz_offset_hours=0.0, verbose=False)
        except Exception as e:
            print(f"❌ [NewsScheduler] Parse CSV ล้มเหลว: {e}")
            self._save_last_run(today_str)
            return False

        if not events:
            print(f"⚠️ [NewsScheduler] {csv_path.name} ไม่มี high-impact events — ข้าม")
            self._save_last_run(today_str)
            return False

        # Write JSON
        now_utc = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
        valid_until = now_utc + timedelta(days=7)
        out = {
            "_comment": f"Auto-imported by NewsScheduler from {csv_path.name} at {now.isoformat()}",
            "updated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_until": valid_until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "events": sorted(events, key=lambda e: e["datetime_utc"]),
        }

        try:
            with open(self._output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"❌ [NewsScheduler] เขียน {self._output_json.name} ล้มเหลว: {e}")
            return False

        # Move CSV → processed/
        dest = self._processed_dir / f"{today_str}_{csv_path.name}"
        try:
            shutil.move(str(csv_path), str(dest))
        except OSError as e:
            print(f"⚠️ [NewsScheduler] Move CSV ไม่สำเร็จ ({e}) — JSON updated แล้ว")

        self._save_last_run(today_str)
        print(f"✅ [NewsScheduler] Import เสร็จ: {len(events)} events → {self._output_json.name}")
        print(f"   📦 Move CSV → processed/{dest.name}")
        return True
