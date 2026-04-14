"""
===============================================================================
FTMO Trading Bot — โมดูลเชื่อมต่อ MetaTrader 5
===============================================================================
จัดการการเชื่อมต่อ, ดึงข้อมูลราคา, ดึงข้อมูลบัญชี, และส่งคำสั่งเทรด
ผ่าน MetaTrader 5 Python API

การใช้งาน:
    from core.mt5_connector import MT5Connector
    connector = MT5Connector()
    connector.connect()
===============================================================================
"""

import sys
import time as time_module
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# พยายาม Import MetaTrader5 — ถ้าไม่มีจะใช้ Mock สำหรับทดสอบ
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ [คำเตือน] ไม่พบไลบรารี MetaTrader5 — ใช้โหมดจำลอง (Mock Mode)")

from config.settings import bot_config


class MT5Connector:
    """
    คลาสจัดการการเชื่อมต่อกับ MetaTrader 5
    
    ความรับผิดชอบ:
    - เชื่อมต่อ/ตัดการเชื่อมต่อกับ MT5 Terminal
    - ดึงข้อมูลบัญชี (Balance, Equity, Margin)
    - ดึงข้อมูลราคา (OHLCV) สำหรับวิเคราะห์
    - ดึงข้อมูล Position ที่เปิดอยู่
    - ส่งคำสั่ง Market Order, Pending Order
    - ปิด Position ทั้งหมด (Emergency Close)
    """

    def __init__(self):
        """เริ่มต้นตัวเชื่อมต่อ MT5 — ยังไม่เชื่อมต่อจนกว่าจะเรียก connect()"""
        self._connected = False
        self._config = bot_config.mt5
        self._last_connection_check = datetime.now()
        
        # เก็บข้อมูล Symbol Info เพื่อลดการเรียก API ซ้ำ
        self._symbol_cache: Dict[str, object] = {}
        
        print("🔧 [MT5 Connector] เตรียมระบบเชื่อมต่อ MetaTrader 5")

    # =========================================================================
    # 🔌 การเชื่อมต่อและตัดการเชื่อมต่อ
    # =========================================================================

    def connect(self) -> bool:
        """
        เชื่อมต่อกับ MT5 Terminal
        
        ขั้นตอน:
        1. เริ่มต้น MT5 Terminal
        2. Login ด้วยข้อมูลบัญชี
        3. ตรวจสอบการเชื่อมต่อสำเร็จ
        
        Returns:
            bool: True ถ้าเชื่อมต่อสำเร็จ, False ถ้าล้มเหลว
        """
        if not MT5_AVAILABLE:
            print("⚠️ [MT5] โหมดจำลอง — ไม่ได้เชื่อมต่อจริง")
            self._connected = True  # จำลองว่าเชื่อมต่อสำเร็จ
            return True

        # ขั้นตอนที่ 1: เริ่มต้น MT5 Terminal
        print("🔄 [MT5] กำลังเริ่มต้น MetaTrader 5 Terminal...")
        
        init_params = {}
        if self._config.terminal_path:
            init_params["path"] = self._config.terminal_path
        if self._config.timeout:
            init_params["timeout"] = self._config.timeout
            
        if not mt5.initialize(**init_params):
            error = mt5.last_error()
            print(f"❌ [MT5] เริ่มต้น Terminal ล้มเหลว: {error}")
            return False

        # ขั้นตอนที่ 2: Login เข้าบัญชี
        if self._config.login and self._config.password and self._config.server:
            print(f"🔐 [MT5] กำลัง Login บัญชี {self._config.login}...")
            
            authorized = mt5.login(
                login=self._config.login,
                password=self._config.password,
                server=self._config.server
            )
            
            if not authorized:
                error = mt5.last_error()
                print(f"❌ [MT5] Login ล้มเหลว: {error}")
                mt5.shutdown()
                return False

        # ขั้นตอนที่ 3: ตรวจสอบการเชื่อมต่อ
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("❌ [MT5] ไม่สามารถดึงข้อมูล Terminal ได้")
            mt5.shutdown()
            return False

        self._connected = True
        account_info = mt5.account_info()
        
        print(f"✅ [MT5] เชื่อมต่อสำเร็จ!")
        print(f"   📊 บัญชี: {account_info.login}")
        print(f"   💰 Balance: ${account_info.balance:,.2f}")
        print(f"   💎 Equity: ${account_info.equity:,.2f}")
        print(f"   🏢 Server: {account_info.server}")
        print(f"   📈 Leverage: 1:{account_info.leverage}")
        
        return True

    def disconnect(self) -> None:
        """ตัดการเชื่อมต่อจาก MT5 Terminal อย่างปลอดภัย พร้อมล้าง Symbol Cache"""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            print("🔌 [MT5] ตัดการเชื่อมต่อเรียบร้อย")
        self._connected = False

        # ล้าง Symbol Cache — โบรกเกอร์อาจเปลี่ยน spec หลัง reconnect
        # (เช่น contract size, pip size, lot step) ถ้าไม่ล้างจะใช้ข้อมูลเก่าผิด
        self._symbol_cache.clear()

    def is_connected(self) -> bool:
        """
        ตรวจสอบว่ายังเชื่อมต่อกับ MT5 อยู่หรือไม่
        ตรวจสอบทุก 30 วินาที เพื่อไม่ให้เรียก API บ่อยเกินไป
        
        Returns:
            bool: True ถ้ายังเชื่อมต่ออยู่
        """
        if not self._connected:
            return False
            
        if not MT5_AVAILABLE:
            return True  # โหมดจำลอง — ถือว่าเชื่อมต่อเสมอ
            
        # ตรวจสอบจริงทุก 30 วินาที
        now = datetime.now()
        if (now - self._last_connection_check).seconds < 30:
            return self._connected
            
        self._last_connection_check = now
        terminal_info = mt5.terminal_info()
        self._connected = terminal_info is not None and terminal_info.connected
        
        if not self._connected:
            print("⚠️ [MT5] การเชื่อมต่อขาดหาย! พยายามเชื่อมต่อใหม่...")
            self.connect()
            
        return self._connected

    def reconnect(self, max_attempts: int = 3, delay: int = 5) -> bool:
        """
        พยายามเชื่อมต่อใหม่หลายครั้งถ้าการเชื่อมต่อขาด
        
        Args:
            max_attempts: จำนวนครั้งที่พยายามเชื่อมต่อใหม่
            delay: รอกี่วินาทีระหว่างแต่ละครั้ง
            
        Returns:
            bool: True ถ้าเชื่อมต่อสำเร็จ
        """
        for attempt in range(1, max_attempts + 1):
            print(f"🔄 [MT5] พยายามเชื่อมต่อใหม่ครั้งที่ {attempt}/{max_attempts}...")
            self.disconnect()
            time_module.sleep(delay)
            
            if self.connect():
                print(f"✅ [MT5] เชื่อมต่อใหม่สำเร็จในครั้งที่ {attempt}")
                return True
                
        print(f"❌ [MT5] เชื่อมต่อใหม่ไม่สำเร็จหลังจากพยายาม {max_attempts} ครั้ง")
        return False

    # =========================================================================
    # 💰 ข้อมูลบัญชี
    # =========================================================================

    def get_account_info(self) -> Optional[Dict]:
        """
        ดึงข้อมูลบัญชีปัจจุบัน
        
        Returns:
            Dict หรือ None: ข้อมูลบัญชี {balance, equity, margin, free_margin, ...}
        """
        if not MT5_AVAILABLE:
            # โหมดจำลอง — คืนค่าตัวอย่าง
            return {
                "login": 12345678,
                "balance": 100000.0,     # เงินในบัญชี
                "equity": 100000.0,      # มูลค่าสุทธิ (รวม Floating P/L)
                "margin": 0.0,           # Margin ที่ใช้อยู่
                "free_margin": 100000.0, # Margin ที่เหลือ
                "margin_level": 0.0,     # ระดับ Margin (%)
                "profit": 0.0,           # กำไร/ขาดทุนรวมของ Position ที่เปิดอยู่
                "leverage": 100,         # Leverage
                "currency": "USD",       # สกุลเงินบัญชี
                "server": "Demo",        # ชื่อเซิร์ฟเวอร์
            }

        if not self.is_connected():
            print("❌ [MT5] ไม่ได้เชื่อมต่อ — ไม่สามารถดึงข้อมูลบัญชีได้")
            return None

        account = mt5.account_info()
        if account is None:
            print("❌ [MT5] ดึงข้อมูลบัญชีล้มเหลว")
            return None

        return {
            "login": account.login,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "free_margin": account.margin_free,
            "margin_level": account.margin_level if account.margin_level else 0.0,
            "profit": account.profit,
            "leverage": account.leverage,
            "currency": account.currency,
            "server": account.server,
        }

    def get_balance(self) -> float:
        """ดึง Balance ปัจจุบัน (เงินในบัญชี ไม่รวม Floating P/L)"""
        info = self.get_account_info()
        return info["balance"] if info else 0.0

    def get_equity(self) -> float:
        """ดึง Equity ปัจจุบัน (มูลค่าสุทธิ รวม Floating P/L)"""
        info = self.get_account_info()
        return info["equity"] if info else 0.0

    # =========================================================================
    # 📊 ข้อมูลราคา (OHLCV)
    # =========================================================================

    def get_ohlcv(
        self,
        symbol: str,
        timeframe_str: str = "M15",
        count: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        ดึงข้อมูลราคา OHLCV (Open, High, Low, Close, Volume)
        
        Args:
            symbol: คู่เงิน เช่น "EURUSD"
            timeframe_str: Timeframe เช่น "M1", "M5", "M15", "H1", "H4", "D1"
            count: จำนวนแท่งเทียนที่ต้องการ
            
        Returns:
            pd.DataFrame หรือ None: DataFrame ที่มีคอลัมน์ [time, open, high, low, close, volume]
        """
        # แปลง string เป็น MT5 timeframe constant
        timeframe_map = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769,
        }
        
        if not MT5_AVAILABLE:
            # โหมดจำลอง — สร้างข้อมูลจำลอง
            return self._generate_mock_ohlcv(symbol, count)
        
        if not self.is_connected():
            print(f"❌ [MT5] ไม่ได้เชื่อมต่อ — ไม่สามารถดึงข้อมูล {symbol} ได้")
            return None

        # แปลง timeframe string เป็น MT5 constant
        tf_constant = timeframe_map.get(timeframe_str)
        if tf_constant is None:
            print(f"❌ [MT5] Timeframe ไม่ถูกต้อง: {timeframe_str}")
            return None
        
        # ใช้ค่า mt5.TIMEFRAME_* จริงถ้ามี
        mt5_tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        }
        tf = mt5_tf_map.get(timeframe_str, tf_constant)

        # ดึงข้อมูลราคาจาก MT5
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            print(f"❌ [MT5] ดึงข้อมูลราคา {symbol} ล้มเหลว: {error}")
            return None

        # แปลงเป็น DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # เปลี่ยนชื่อคอลัมน์ให้ชัดเจน
        df.rename(columns={
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        ดึงราคาปัจจุบัน (Bid/Ask) ของคู่เงินที่ระบุ
        
        Args:
            symbol: คู่เงิน เช่น "EURUSD"
            
        Returns:
            Dict: {"bid": float, "ask": float, "spread": float}
        """
        if not MT5_AVAILABLE:
            # โหมดจำลอง — spread เป็นค่าราคาจริง (ask - bid)
            # ครอบคลุมทุก symbol ใน SymbolConfig (9 คู่)
            mock_prices = {
                "EURUSD": {"bid": 1.09500, "ask": 1.09510, "spread": 0.00010},
                "GBPUSD": {"bid": 1.26800, "ask": 1.26815, "spread": 0.00015},
                "USDJPY": {"bid": 149.500, "ask": 149.510, "spread": 0.010},
                "AUDUSD": {"bid": 0.66200, "ask": 0.66215, "spread": 0.00015},
                "USDCAD": {"bid": 1.36500, "ask": 1.36520, "spread": 0.00020},
                "USDCHF": {"bid": 0.88300, "ask": 0.88320, "spread": 0.00020},
                "NZDUSD": {"bid": 0.60500, "ask": 0.60520, "spread": 0.00020},
                "EURJPY": {"bid": 163.700, "ask": 163.725, "spread": 0.025},
                "GBPJPY": {"bid": 189.600, "ask": 189.630, "spread": 0.030},
            }
            return mock_prices.get(symbol, {"bid": 1.0, "ask": 1.00010, "spread": 0.00010})

        if not self.is_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ [MT5] ดึงราคา {symbol} ล้มเหลว")
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": (tick.ask - tick.bid),
        }

    # =========================================================================
    # 📋 ข้อมูล Symbol
    # =========================================================================

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        ดึงข้อมูลรายละเอียดของ Symbol (ใช้ Cache เพื่อประสิทธิภาพ)
        
        Args:
            symbol: คู่เงิน เช่น "EURUSD"
            
        Returns:
            Dict: ข้อมูล Symbol {point, digits, lot_min, lot_max, lot_step, ...}
        """
        # ตรวจสอบ Cache ก่อน
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if not MT5_AVAILABLE:
            # ข้อมูลจำลองครอบคลุมทุก Symbol ใน SymbolConfig (9 คู่)
            # 5-digit สำหรับ Major pairs, 3-digit สำหรับ JPY pairs
            common_5d = {
                "point": 0.00001, "digits": 5, "lot_min": 0.01, "lot_max": 100.0,
                "lot_step": 0.01, "trade_contract_size": 100000, "volume_min": 0.01,
            }
            common_3d = {
                "point": 0.001, "digits": 3, "lot_min": 0.01, "lot_max": 100.0,
                "lot_step": 0.01, "trade_contract_size": 100000, "volume_min": 0.01,
            }
            mock_info = {
                "EURUSD": common_5d, "GBPUSD": common_5d, "AUDUSD": common_5d,
                "USDCAD": common_5d, "USDCHF": common_5d, "NZDUSD": common_5d,
                "USDJPY": common_3d, "EURJPY": common_3d, "GBPJPY": common_3d,
            }
            info = mock_info.get(symbol, common_5d)
            self._symbol_cache[symbol] = info
            return info

        if not self.is_connected():
            return None

        # ต้องเปิด Symbol ในตลาดก่อน
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            print(f"❌ [MT5] ไม่สามารถเลือก Symbol {symbol} ได้")
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"❌ [MT5] ดึงข้อมูล Symbol {symbol} ล้มเหลว")
            return None

        result = {
            "point": info.point,                          # ค่า 1 Point
            "digits": info.digits,                        # จำนวนทศนิยม
            "lot_min": info.volume_min,                   # Lot ขั้นต่ำ
            "lot_max": info.volume_max,                   # Lot สูงสุด
            "lot_step": info.volume_step,                 # ขั้นบันได Lot
            "trade_contract_size": info.trade_contract_size,  # ขนาดสัญญา (เช่น 100,000)
            "volume_min": info.volume_min,                # Volume ขั้นต่ำ
        }
        
        # เก็บใน Cache
        self._symbol_cache[symbol] = result
        return result

    # =========================================================================
    # 📊 ข้อมูล Position ที่เปิดอยู่
    # =========================================================================

    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """
        ดึงรายการ Position ที่เปิดอยู่ทั้งหมด หรือเฉพาะ Symbol ที่ระบุ
        
        Args:
            symbol: (ไม่บังคับ) กรองเฉพาะ Symbol นี้
            
        Returns:
            List[Dict]: รายการ Position ที่เปิดอยู่
        """
        if not MT5_AVAILABLE:
            return []  # โหมดจำลอง — ไม่มี Position

        if not self.is_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None or len(positions) == 0:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,                    # หมายเลข Ticket
                "symbol": pos.symbol,                    # คู่เงิน
                "type": "BUY" if pos.type == 0 else "SELL",  # ประเภท (BUY/SELL)
                "volume": pos.volume,                    # ขนาด Lot
                "price_open": pos.price_open,            # ราคาเปิด
                "price_current": pos.price_current,      # ราคาปัจจุบัน
                "sl": pos.sl,                            # Stop Loss
                "tp": pos.tp,                            # Take Profit
                "profit": pos.profit,                    # กำไร/ขาดทุน (USD)
                "swap": pos.swap,                        # ค่า Swap
                "time": datetime.fromtimestamp(pos.time), # เวลาเปิด
                "magic": pos.magic,                      # Magic Number
                "comment": pos.comment,                  # หมายเหตุ
            })

        return result

    def get_total_floating_pnl(self) -> float:
        """
        คำนวณกำไร/ขาดทุนลอยตัว (Floating P/L) รวมทุก Position
        
        Returns:
            float: ค่า Floating P/L รวม (USD)
        """
        positions = self.get_open_positions()
        return sum(pos["profit"] + pos.get("swap", 0) for pos in positions)

    def get_positions_count(self) -> int:
        """นับจำนวน Position ที่เปิดอยู่"""
        return len(self.get_open_positions())

    # =========================================================================
    # 💼 ส่งคำสั่งเทรด
    # =========================================================================

    def send_market_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" หรือ "SELL"
        volume: float,
        sl: float,
        tp: float,
        comment: str = "FTMO_BOT",
        magic: int = 123456
    ) -> Optional[Dict]:
        """
        ส่งคำสั่ง Market Order (ซื้อ/ขายทันทีที่ราคาตลาด)
        
        Args:
            symbol: คู่เงิน
            order_type: "BUY" หรือ "SELL"
            volume: ขนาด Lot
            sl: ราคา Stop Loss
            tp: ราคา Take Profit
            comment: หมายเหตุ (ใช้ระบุว่ามาจาก Bot)
            magic: Magic Number สำหรับระบุ Bot
            
        Returns:
            Dict หรือ None: ผลลัพธ์การส่งคำสั่ง {ticket, price, volume, ...}
        """
        if not MT5_AVAILABLE:
            # โหมดจำลอง — สร้างผลลัพธ์จำลอง
            mock_ticket = int(datetime.now().timestamp())
            print(f"📝 [MOCK] จำลองคำสั่ง {order_type} {symbol} Volume={volume} SL={sl} TP={tp}")
            return {
                "ticket": mock_ticket,
                "price": self.get_current_price(symbol)["ask" if order_type == "BUY" else "bid"],
                "volume": volume,
                "comment": comment,
                "retcode": 10009,  # TRADE_RETCODE_DONE
            }

        if not self.is_connected():
            print("❌ [MT5] ไม่สามารถส่งคำสั่งได้ — ไม่ได้เชื่อมต่อ")
            return None

        # ดึงราคาปัจจุบัน
        price_info = self.get_current_price(symbol)
        if price_info is None:
            return None

        # กำหนดประเภทคำสั่งและราคา
        if order_type.upper() == "BUY":
            mt5_type = mt5.ORDER_TYPE_BUY
            price = price_info["ask"]    # ซื้อที่ราคา Ask
        elif order_type.upper() == "SELL":
            mt5_type = mt5.ORDER_TYPE_SELL
            price = price_info["bid"]    # ขายที่ราคา Bid
        else:
            print(f"❌ [MT5] ประเภทคำสั่งไม่ถูกต้อง: {order_type}")
            return None

        # สร้างโครงสร้างคำสั่ง
        request = {
            "action": mt5.TRADE_ACTION_DEAL,     # Market Order
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,                     # Slippage สูงสุด 2 pips
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,      # Good Till Cancel
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        # ส่งคำสั่ง
        print(f"📤 [MT5] กำลังส่งคำสั่ง {order_type} {symbol}...")
        print(f"   💲 ราคา: {price} | Volume: {volume} | SL: {sl} | TP: {tp}")
        
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            print(f"❌ [MT5] ส่งคำสั่งล้มเหลว: {error}")
            return None

        # ตรวจสอบผลลัพธ์
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ [MT5] คำสั่งถูกปฏิเสธ: retcode={result.retcode}, comment={result.comment}")
            return None

        print(f"✅ [MT5] คำสั่งสำเร็จ! Ticket: {result.order}")
        return {
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "comment": result.comment,
            "retcode": result.retcode,
        }

    def close_position(self, ticket: int) -> bool:
        """
        ปิด Position ที่ระบุด้วย Ticket Number
        
        Args:
            ticket: หมายเลข Ticket ของ Position ที่ต้องการปิด
            
        Returns:
            bool: True ถ้าปิดสำเร็จ
        """
        if not MT5_AVAILABLE:
            print(f"📝 [MOCK] จำลองการปิด Position Ticket: {ticket}")
            return True

        if not self.is_connected():
            return False

        # ค้นหา Position จาก Ticket
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"⚠️ [MT5] ไม่พบ Position Ticket: {ticket}")
            return False

        pos = position[0]
        
        # กำหนดคำสั่งปิด (ตรงข้ามกับคำสั่งเปิด)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price_info = self.get_current_price(pos.symbol)
        if price_info is None:
            return False
            
        close_price = price_info["bid"] if pos.type == 0 else price_info["ask"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "FTMO_BOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            print(f"❌ [MT5] ปิด Position {ticket} ล้มเหลว: {error}")
            return False

        print(f"✅ [MT5] ปิด Position {ticket} สำเร็จ ({pos.symbol})")
        return True

    def close_all_positions(self) -> Tuple[int, int]:
        """
        ⚠️ ปิดทุก Position ที่เปิดอยู่ทันที (Emergency Close)
        ใช้เมื่อฝ่าฝืนกฎ FTMO หรือเกิดเหตุฉุกเฉิน
        
        Returns:
            Tuple[int, int]: (จำนวนที่ปิดสำเร็จ, จำนวนที่ปิดล้มเหลว)
        """
        print("🚨 [MT5] === กำลังปิดทุก Position (EMERGENCY CLOSE) ===")
        
        positions = self.get_open_positions()
        if not positions:
            print("ℹ️ [MT5] ไม่มี Position ที่เปิดอยู่")
            return (0, 0)

        success_count = 0
        fail_count = 0

        for pos in positions:
            if self.close_position(pos["ticket"]):
                success_count += 1
            else:
                fail_count += 1
                # ลองปิดอีกครั้ง
                time_module.sleep(1)
                if self.close_position(pos["ticket"]):
                    success_count += 1
                    fail_count -= 1

        print(f"🏁 [MT5] ผลการปิด Position: สำเร็จ {success_count}, ล้มเหลว {fail_count}")
        return (success_count, fail_count)

    # =========================================================================
    # 📜 ประวัติการเทรด
    # =========================================================================

    def get_trade_history(self, days: int = 30) -> List[Dict]:
        """
        ดึงประวัติการเทรดย้อนหลัง
        
        Args:
            days: จำนวนวันย้อนหลัง (ค่าเริ่มต้น 30 วัน)
            
        Returns:
            List[Dict]: รายการประวัติการเทรด
        """
        if not MT5_AVAILABLE:
            return []

        if not self.is_connected():
            return []

        # กำหนดช่วงเวลา
        date_from = datetime.now() - timedelta(days=days)
        date_to = datetime.now()

        # ดึงประวัติ Deals
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None or len(deals) == 0:
            return []

        result = []
        for deal in deals:
            if deal.entry == 1:  # เฉพาะ Deal ที่เป็นการปิด Position (entry out)
                result.append({
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "time": datetime.fromtimestamp(deal.time),
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "swap": deal.swap,
                    "commission": deal.commission,
                    "comment": deal.comment,
                    "magic": deal.magic,
                })

        return result

    def get_today_closed_pnl(self) -> float:
        """
        คำนวณ P/L ของ Position ที่ปิดไปแล้ววันนี้
        ใช้สำหรับตรวจสอบ Daily Loss Limit
        
        Returns:
            float: รวม P/L ที่ปิดไปแล้ววันนี้ (USD)
        """
        if not MT5_AVAILABLE:
            return 0.0

        if not self.is_connected():
            return 0.0

        # ดึง Deals ของวันนี้
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(today_start, datetime.now())
        
        if deals is None or len(deals) == 0:
            return 0.0

        # รวม P/L เฉพาะ Deal ที่เป็นการปิด Position
        total_pnl = 0.0
        for deal in deals:
            if deal.entry == 1:  # entry out = ปิด Position
                total_pnl += deal.profit + deal.swap + deal.commission

        return total_pnl

    # =========================================================================
    # 🛠️ ฟังก์ชันช่วย (Utility)
    # =========================================================================

    def _generate_mock_ohlcv(self, symbol: str, count: int) -> pd.DataFrame:
        """
        สร้างข้อมูลราคาจำลองสำหรับทดสอบ (ใช้เมื่อไม่มี MT5)
        สร้าง Random Walk ที่มีลักษณะคล้ายราคาจริง
        
        Args:
            symbol: คู่เงิน (ใช้กำหนดราคาเริ่มต้น)
            count: จำนวนแท่งเทียน
            
        Returns:
            pd.DataFrame: ข้อมูลราคาจำลอง
        """
        np.random.seed(42)
        
        # ราคาเริ่มต้นตาม Symbol (ครอบคลุมทุกคู่ใน config)
        base_prices = {
            "EURUSD": 1.0950, "GBPUSD": 1.2680, "AUDUSD": 0.6620,
            "USDCAD": 1.3650, "USDCHF": 0.8830, "NZDUSD": 0.6050,
            "USDJPY": 149.50, "EURJPY": 163.70, "GBPJPY": 189.60,
        }
        base_price = base_prices.get(symbol, 1.0)

        # Volatility ต่อแท่งเทียน — JPY pairs ผันผวนกว่า Major pairs (ในหน่วย ราคา)
        volatility = 0.0005 if base_price < 50 else 0.05

        # สร้าง Random Walk
        returns = np.random.normal(0, volatility / base_price, count)  # ผลตอบแทนรายแท่ง
        prices = base_price * np.cumprod(1 + returns)
        
        # คำนวณ OHLCV
        dates = pd.date_range(end=datetime.now(), periods=count, freq='15min')
        
        data = {
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0003, count))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0003, count))),
            'close': prices * (1 + np.random.normal(0, 0.0002, count)),
            'volume': np.random.randint(100, 5000, count),
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'time'
        
        return df

    def __del__(self):
        """ตัดการเชื่อมต่ออัตโนมัติเมื่อ Object ถูกทำลาย"""
        self.disconnect()

    def __repr__(self) -> str:
        status = "เชื่อมต่อแล้ว" if self._connected else "ยังไม่เชื่อมต่อ"
        return f"MT5Connector(สถานะ={status})"
