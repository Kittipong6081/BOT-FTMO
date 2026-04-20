"""
===============================================================================
FTMO Trading Bot — Time Manager (ระบบจัดการเวลาสำหรับ Market Session)
===============================================================================
โมดูลนี้รับหน้าที่รับเวลา Server Time (EET Timezone) โดยตรงจาก MT5
เพื่อป้องกันความคลาดเคลื่อนหากเวลาของเครื่อง VPS เดินไม่ตรงกับ Broker

กฎความปลอดภัย (FTMO Compliance):
1. ตรวจสอบเวลาเทรดข้ามวัน (Rollover Time)
2. สั่งหยุดและปิดรวบยอดวันศุกร์ (Friday Force Close)
===============================================================================
"""

import pytz
from datetime import datetime, time
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config.settings import bot_config


class TimeManager:
    """
    คลาสจัดการการตรวจสอบ Market Session และตรวจสอบกฎเวลาตามเซิร์ฟเวอร์
    ใช้ Timezone เป็นพาราเมนเตอร์เสริม (ค่าดั้งเดิมโบรกเกอร์ MT5 คือ EET - Eastern European Time)
    """

    # ใช้ Timezone Bucharest เพื่อปรับชั่วโมงตามเวลา EET/EEST อัตโนมัติ (Daylight Saving)
    BROKER_TIMEZONE = pytz.timezone("Europe/Bucharest")

    @classmethod
    def get_server_time(cls, symbol: str = "EURUSD") -> datetime:  # noqa: ARG003
        """
        คืนเวลาของโบรกเกอร์ (EET/EEST) จาก VPS local clock ที่ sync ผ่าน NTP

        ทำไมไม่ใช้ mt5.symbol_info_tick().time?
            MT5 brokers (รวม FTMO-Demo) มี quirk: ส่ง tick.time เป็น broker-local
            epoch (EEST/EET) แทน UTC epoch มาตรฐาน → datetime.fromtimestamp(tick.time,
            tz=Bucharest) จะ double-add timezone offset → display ล่วงหน้า 2-3 ชั่วโมง
            (ผล: Friday/Daily close trigger ผิดเวลา)

        Solution:
            ใช้ datetime.now(Europe/Bucharest) ซึ่งอิงกับ VPS system clock
            (sync NTP → แม่นยำระดับวินาที) + pytz แปลง EET/EEST + DST อัตโนมัติ
            UTC moment ตรงกับ FTMO server เสมอ (ทั้งคู่ sync NTP เดียวกัน)

        Precondition:
            VPS ต้อง sync NTP (ดู readme.md: "FAQ: NTP Time Sync")

        Args:
            symbol: ไม่ใช้แล้ว — รักษาไว้เพื่อ backward compatibility

        Returns:
            datetime: เวลาโบรกเกอร์ (Europe/Bucharest — EET/EEST)
        """
        return datetime.now(cls.BROKER_TIMEZONE)

    @classmethod
    def is_tick_fresh(cls, symbol: str = "EURUSD", max_age_s: int = 600) -> bool:
        """
        ตรวจว่า broker tick ล่าสุดยังสดอยู่ไหม (market open + broker online)

        ใช้สำหรับ: market activity detection, dashboard status
        *ไม่* ใช้เป็น source of truth สำหรับ wall-clock time

        Handles MT5 tick.time quirk: broker อาจส่งเป็น UTC หรือ broker-local epoch
        → ลอง interpret ทั้ง 3 offsets (UTC / +2h EET / +3h EEST) เลือกที่ใกล้ now ที่สุด

        Args:
            symbol: คู่เงินตัวแทน
            max_age_s: ถือว่า stale ถ้า tick เก่ากว่านี้ (default 10 นาที)

        Returns:
            bool: True ถ้า tick สด (market active)
        """
        if not MT5_AVAILABLE:
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time <= 0:
            return False

        import time as _t
        now_utc = _t.time()
        # Try 3 interpretations of tick.time
        age_as_utc = abs(now_utc - tick.time)
        age_as_eest = abs(now_utc - (tick.time - 10800))  # subtract +3h
        age_as_eet = abs(now_utc - (tick.time - 7200))    # subtract +2h
        return min(age_as_utc, age_as_eest, age_as_eet) <= max_age_s

    @classmethod
    def is_rollover_period(cls, current_server_time: datetime) -> bool:
        """
        ตรวจสอบช่วงเวลาการหนีสเปรดถ่างตอนตี 5 ไทย (Rollover Spread Expansion)
        
        Returns:
            bool: True ถ้ายืนยันว่าอยู่ในช่วงพักการเทรดข้ามวัน
        """
        current_t = current_server_time.time()
        start = bot_config.sessions.rollover_start
        end = bot_config.sessions.rollover_end

        # ตัวอย่าง 23:55 น. จนถึง 01:05 น. คร่อมเที่ยงคืน
        if start > end:
            return current_t >= start or current_t < end
        else:
            return start <= current_t < end

    @classmethod
    def is_friday_close_time(cls, current_server_time: datetime) -> bool:
        """
        ตรวจสอบว่าถึงเวลาบังคับตัดออเดอร์วันศุกร์ตามกฎ FTMO หรือไม่
        (ป้องกันการถือออเดอร์ข้ามสัปดาห์ สำหรับบางประเภทบัญชี)

        Returns:
            bool: True ถ้าอยู่หลังเวลา Friday Force Close วันศุกร์
        """
        # 0 = Monday, ..., 4 = Friday
        if current_server_time.weekday() == 4:
            if current_server_time.time() >= bot_config.sessions.friday_force_close:
                return True
        return False

    @classmethod
    def is_daily_close_time(cls, current_server_time: datetime) -> bool:
        """
        ตรวจสอบว่าถึงเวลาบังคับปิด position ประจำวัน (Mon-Thu) — FTMO Zero-Overnight
        วันศุกร์ใช้ is_friday_close_time (เวลา 20:45 ที่เข้มกว่า)

        Window: daily_close_time (23:30) → rollover_start (23:55)
                = 25 นาที สำหรับปิดออเดอร์ก่อน spread ถ่าง

        Returns:
            bool: True ถ้า Mon-Thu + ในช่วง daily close window
        """
        # ต้องเปิดใช้ผ่าน config
        if not getattr(bot_config.sessions, 'enforce_daily_close', False):
            return False

        # Mon-Thu เท่านั้น (Friday มี friday_force_close ที่เข้มกว่า)
        if current_server_time.weekday() not in (0, 1, 2, 3):
            return False

        close_t = bot_config.sessions.daily_close_time
        rollover_start = bot_config.sessions.rollover_start
        current_t = current_server_time.time()

        return close_t <= current_t < rollover_start

    @classmethod
    def is_weekend(cls, current_server_time: datetime) -> bool:
        """ตรวจสอบว่าเป็นวันสุดสัปดาห์ (เสาร์อาทิตย์) หรือไม่"""
        # 5 = Saturday, 6 = Sunday
        return current_server_time.weekday() in (5, 6)
