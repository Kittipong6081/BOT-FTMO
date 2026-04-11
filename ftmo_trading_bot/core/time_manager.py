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
    def get_server_time(cls, symbol: str = "EURUSD") -> datetime:
        """
        ดึงเวลาปัจจุบันของโบรกเกอร์จาก Tick ล่าสุดของ MT5
        
        หาก MT5 มีปัญหาไม่ได้เชื่อมต่อ จะยอมใช้เวลาปัจจุบันของ Local (VPS) 
        และแปลงเป็น Broker Timezone แทนชั่วคราว
        
        Args:
            symbol (str): คู่เงินตัวแทนที่ใช้ดึง Tick
            
        Returns:
            datetime: เวลาของโบรกเกอร์ (Server Time)
        """
        if MT5_AVAILABLE:
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None and tick.time > 0:
                # MT5 ให้เวลาในรูปแบบ Timestamp (UTC)
                # เราใช้ datetime.fromtimestamp ซึ่งต้องจัดการ timezone
                server_time = datetime.fromtimestamp(tick.time, tz=cls.BROKER_TIMEZONE)
                return server_time

        # กรณีหาค่าไม่ได้ หรือ Mock Mode (นำเวลา Local มาบิดเป็นเวลา Broker)
        return datetime.now(cls.BROKER_TIMEZONE)

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
    def is_weekend(cls, current_server_time: datetime) -> bool:
        """ตรวจสอบว่าเป็นวันสุดสัปดาห์ (เสาร์อาทิตย์) หรือไม่"""
        # 5 = Saturday, 6 = Sunday
        return current_server_time.weekday() in (5, 6)
