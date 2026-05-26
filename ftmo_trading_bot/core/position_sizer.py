"""
===============================================================================
FTMO Trading Bot — ระบบคำนวณขนาด Position (Position Sizer)
===============================================================================
คำนวณขนาด Lot อย่างแม่นยำโดยพิจารณา:
1. ความเสี่ยงต่อเทรด (0.5-1% ของ Balance)
2. ระยะ Stop Loss (ATR-based)
3. ข้อมูล Symbol (Contract Size, Point Value)
4. งบเสี่ยงรายวันที่เหลือ

⚠️ ห้ามเปลี่ยนวิธีคำนวณโดยไม่ทดสอบ — ขนาด Lot ผิดอาจทำให้ฝ่าฝืนกฎ FTMO
===============================================================================
"""

import math
from typing import Optional, Dict, Tuple

from config.settings import bot_config
from core.mt5_connector import MT5Connector


class PositionSizer:
    """
    คำนวณขนาด Lot (Position Size) อย่างปลอดภัยตามกฎ FTMO
    
    หลักการ:
    - Risk Amount = Balance × Risk % (เช่น $100,000 × 0.75% = $750)
    - Lot Size = Risk Amount / (SL Distance in Pips × Pip Value per Lot)
    - ปัดขนาด Lot ลงเสมอ (ไม่ปัดขึ้น) เพื่อความปลอดภัย
    - ตรวจสอบ Lot Min/Max ของโบรกเกอร์
    """

    def __init__(self, connector: MT5Connector):
        """
        เริ่มต้น Position Sizer
        
        Args:
            connector: ตัวเชื่อมต่อ MT5 สำหรับดึงข้อมูล Symbol และ Balance
        """
        self._connector = connector
        self._config = bot_config.ftmo
        
        print("📐 [Position Sizer] เริ่มต้นระบบคำนวณขนาด Position")

    def calculate_lot_size(
        self,
        symbol: str,
        sl_distance_price: float,
        risk_pct: float = None,
        max_risk_usd: float = None
    ) -> Optional[Dict]:
        """
        คำนวณขนาด Lot ที่เหมาะสมสำหรับเทรดหนึ่งครั้ง
        
        วิธีคำนวณ:
        1. คำนวณ Risk Amount (จำนวนเงินที่ยอมขาดทุนได้)
        2. คำนวณ Pip Value ต่อ Lot
        3. คำนวณ Lot Size = Risk Amount / (SL * Pip Value)
        4. ปัดลงตาม Lot Step ของโบรกเกอร์
        5. ตรวจสอบ Lot Min/Max
        
        Args:
            symbol: คู่เงิน เช่น "EURUSD"
            sl_distance_price: ระยะ Stop Loss เป็นราคา (เช่น 0.00150 = 15 pips)
            risk_pct: เปอร์เซ็นต์ความเสี่ยง (None = ใช้ค่า Default 0.75%)
            max_risk_usd: จำกัดความเสี่ยงสูงสุดเป็น USD (ถ้ากำหนด)
            
        Returns:
            Dict: {
                "lot_size": float,       # ขนาด Lot ที่คำนวณได้
                "risk_amount": float,    # จำนวนเงินที่เสี่ยง (USD)
                "risk_pct": float,       # เปอร์เซ็นต์ความเสี่ยงจริง
                "sl_pips": float,        # ระยะ SL เป็น pips
                "pip_value": float,      # มูลค่า 1 pip ต่อ 1 lot
            }
            None: ถ้าคำนวณไม่ได้
        """
        # === ขั้นตอนที่ 1: กำหนด Risk Percentage ===
        if risk_pct is None:
            risk_pct = self._config.DEFAULT_RISK_PER_TRADE_PCT  # 0.75%
        
        # บังคับให้อยู่ในช่วง 0.5% - 1.0%
        risk_pct = max(
            self._config.MIN_RISK_PER_TRADE_PCT,
            min(risk_pct, self._config.MAX_RISK_PER_TRADE_PCT)
        )

        # === ขั้นตอนที่ 2: ดึงข้อมูลบัญชี ===
        account = self._connector.get_account_info()
        if account is None:
            print("❌ [Position Sizer] ดึงข้อมูลบัญชีล้มเหลว")
            return None

        balance = account["balance"]
        account_currency = account.get("currency", "USD")

        # === ขั้นตอนที่ 3: คำนวณ Risk Amount ===
        risk_amount = balance * risk_pct
        
        # จำกัดถ้ามี max_risk_usd
        if max_risk_usd is not None and max_risk_usd > 0:
            risk_amount = min(risk_amount, max_risk_usd)

        # === ขั้นตอนที่ 4: ดึงข้อมูล Symbol ===
        symbol_info = self._connector.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ [Position Sizer] ดึงข้อมูล Symbol {symbol} ล้มเหลว")
            return None

        point = symbol_info["point"]           # ค่า 1 point (เช่น 0.00001 สำหรับ EURUSD)
        digits = symbol_info["digits"]          # จำนวนทศนิยม
        contract_size = symbol_info["trade_contract_size"]  # ขนาดสัญญา (เช่น 100,000)
        lot_min = symbol_info["lot_min"]        # Lot ขั้นต่ำ (เช่น 0.01)
        lot_max = symbol_info["lot_max"]        # Lot สูงสุด
        lot_step = symbol_info["lot_step"]      # ขั้นบันได Lot (เช่น 0.01)

        # === ขั้นตอนที่ 5: ตรวจสอบระยะ SL ===
        if sl_distance_price <= 0:
            print("❌ [Position Sizer] ระยะ SL ต้องมากกว่า 0")
            return None

        # แปลง SL distance จากราคาเป็น pips
        # สำหรับคู่เงินที่มี 5 ทศนิยม: 1 pip = 0.0001 = 10 points
        # สำหรับคู่เงิน JPY ที่มี 3 ทศนิยม: 1 pip = 0.01 = 10 points
        if digits == 5 or digits == 4:
            pip_size = 0.0001       # 1 pip สำหรับคู่เงินทั่วไป
        elif digits == 3 or digits == 2:
            pip_size = 0.01         # 1 pip สำหรับคู่เงิน JPY
        else:
            pip_size = point * 10   # ค่าเริ่มต้น

        sl_pips = sl_distance_price / pip_size

        # === ขั้นตอนที่ 6: คำนวณ Pip Value ===
        # Pip Value = Contract Size × Pip Size
        # สำหรับคู่เงินที่ USD เป็นสกุลเงินหลัง (เช่น EURUSD): Pip Value = 100,000 × 0.0001 = $10
        # สำหรับคู่เงินที่ USD เป็นสกุลเงินหน้า (เช่น USDJPY): ต้องแปลง
        pip_value_per_lot = self._calculate_pip_value(
            symbol, pip_size, contract_size, account_currency
        )

        if pip_value_per_lot <= 0:
            print(f"❌ [Position Sizer] คำนวณ Pip Value ล้มเหลว")
            return None

        # === ขั้นตอนที่ 7: คำนวณ Lot Size ===
        # Lot Size = Risk Amount / (SL Pips × Pip Value per Lot)
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # === ขั้นตอนที่ 8: ปัดลงตาม Lot Step (เพื่อความปลอดภัย) ===
        lot_size = self._round_down_lot(lot_size, lot_step)

        # === ขั้นตอนที่ 9: ตรวจสอบ Lot Min/Max ===
        if lot_size < lot_min:
            print(f"⚠️ [Position Sizer] Lot คำนวณได้ ({lot_size:.2f}) ต่ำกว่าขั้นต่ำ ({lot_min}) — ใช้ Lot ขั้นต่ำ")
            lot_size = lot_min
            # คำนวณความเสี่ยงจริงเมื่อใช้ Lot ขั้นต่ำ
            actual_risk = lot_size * sl_pips * pip_value_per_lot
            actual_risk_pct = actual_risk / balance if balance > 0 else 0
            
            # ถ้าความเสี่ยงจริงเกิน 1% ของ Balance — ปฏิเสธ
            if actual_risk_pct > self._config.MAX_RISK_PER_TRADE_PCT * 1.5:
                print(f"❌ [Position Sizer] แม้ Lot ขั้นต่ำก็เสี่ยงเกินไป ({actual_risk_pct:.2%})")
                return None

        if lot_size > lot_max:
            lot_size = lot_max
            print(f"⚠️ [Position Sizer] จำกัด Lot ที่ {lot_max} (Lot สูงสุดของโบรกเกอร์)")

        # === ขั้นตอนที่ 10: คำนวณความเสี่ยงจริง ===
        actual_risk_amount = lot_size * sl_pips * pip_value_per_lot
        actual_risk_pct = actual_risk_amount / balance if balance > 0 else 0

        result = {
            "lot_size": lot_size,
            "risk_amount": actual_risk_amount,
            "risk_pct": actual_risk_pct,
            "sl_pips": sl_pips,
            "sl_distance_price": sl_distance_price,
            "pip_value": pip_value_per_lot,
            "balance": balance,
        }

        # แสดงผลการคำนวณ (v8.0.30: gated by verbose_level >= 2)
        if getattr(bot_config, 'verbose_level', 1) >= 2:
            print(f"\n📐 [Position Sizer] ผลการคำนวณ {symbol}:")
            print(f"   💰 Balance:       ${balance:,.2f}")
            print(f"   🎯 Risk:          {actual_risk_pct:.2%} (${actual_risk_amount:,.2f})")
            print(f"   📏 SL Distance:   {sl_pips:.1f} pips ({sl_distance_price:.5f})")
            print(f"   💵 Pip Value:     ${pip_value_per_lot:.2f}/lot")
            print(f"   📦 Lot Size:      {lot_size:.2f}")

        return result

    def _calculate_pip_value(
        self,
        symbol: str,
        pip_size: float,
        contract_size: float,
        account_currency: str
    ) -> float:
        """
        คำนวณมูลค่า 1 Pip ต่อ 1 Lot Standard (คืนเป็น Account Currency เช่น USD)

        กรณีต่างๆ:
        - xxxUSD (EURUSD, GBPUSD): Pip Value = Contract × Pip Size = $10
        - USDxxx (USDJPY, USDCHF): Pip Value = (Contract × Pip Size) / USDxxx rate
        - xxxJPY Cross (EURJPY, GBPJPY):
            raw = Contract × Pip Size (ในหน่วย JPY)
            USD value = raw / (USDJPY rate)   ← ต้องใช้ USDJPY rate ไม่ใช่ EURJPY

        Args:
            symbol: คู่เงิน
            pip_size: ขนาด 1 pip
            contract_size: ขนาดสัญญา
            account_currency: สกุลเงินของบัญชี

        Returns:
            float: มูลค่า 1 pip ต่อ 1 lot (ในสกุลบัญชี)
        """
        if len(symbol) < 6:
            return contract_size * pip_size

        base_currency = symbol[0:3].upper()
        quote_currency = symbol[3:6].upper()
        acc_ccy = (account_currency or "USD").upper()

        raw_pip_value = contract_size * pip_size  # ใน Quote Currency

        # === Case 1: Quote = Account Currency (เช่น EURUSD เมื่อ account=USD) ===
        if quote_currency == acc_ccy:
            return raw_pip_value  # = $10 สำหรับ Standard Lot

        # === Case 2: Base = Account Currency (เช่น USDJPY, USDCHF เมื่อ account=USD) ===
        if base_currency == acc_ccy:
            # Pip Value (USD) = raw / symbol_price
            price = self._connector.get_current_price(symbol)
            if price and price.get("bid", 0) > 0:
                return raw_pip_value / price["bid"]
            return raw_pip_value  # fallback

        # === Case 3: Cross Pair (เช่น EURJPY, GBPJPY เมื่อ account=USD) ===
        # raw_pip_value อยู่ใน quote_currency (JPY) → ต้องแปลงเป็น USD
        # ใช้ rate ของ {QuoteCurrency}{AccountCurrency} หรือผกผัน
        conversion_symbol_direct = f"{quote_currency}{acc_ccy}"    # JPYUSD (มักไม่มี)
        conversion_symbol_inverse = f"{acc_ccy}{quote_currency}"   # USDJPY (มี)

        # v8.0.72: ตรวจ existence แบบเงียบก่อน probe direct — กัน MT5 print ❌ ทุก order
        # ที่ผ่านมา FTMO/most brokers ไม่มี JPYUSD/CHFUSD/CADUSD → direct probe ล้มเหลวเสมอ
        # แล้ว fall through ไป inverse path ปกติอยู่แล้ว แค่ noisy log
        if self._connector.symbol_exists(conversion_symbol_direct):
            # ลอง direct ก่อน (QUOTE/ACCOUNT)
            price_direct = self._connector.get_current_price(conversion_symbol_direct)
            if price_direct and price_direct.get("bid", 0) > 0:
                # Pip Value (Account) = raw × direct_rate
                return raw_pip_value * price_direct["bid"]

        # Fallback: ใช้ inverse (ACCOUNT/QUOTE เช่น USDJPY)
        price_inverse = self._connector.get_current_price(conversion_symbol_inverse)
        if price_inverse and price_inverse.get("bid", 0) > 0:
            # Pip Value (Account) = raw / inverse_rate
            return raw_pip_value / price_inverse["bid"]

        # Fallback สุดท้าย: ใช้ราคาของ symbol เอง (เก่า, อาจคลาดเคลื่อน)
        price_self = self._connector.get_current_price(symbol)
        if price_self and price_self.get("bid", 0) > 0:
            print(f"⚠️ [Position Sizer] {symbol}: ไม่พบ conversion rate — ใช้ราคา self (อาจคลาดเคลื่อน)")
            return raw_pip_value / price_self["bid"]

        print(f"⚠️ [Position Sizer] {symbol}: คำนวณ Pip Value ไม่ได้ — ใช้ค่าประมาณ")
        return raw_pip_value

    def _round_down_lot(self, lot_size: float, lot_step: float) -> float:
        """
        ปัดขนาด Lot ลง (Floor) ตาม Lot Step ของโบรกเกอร์
        
        ⚠️ ต้องปัดลงเสมอ ห้ามปัดขึ้น — เพื่อป้องกันความเสี่ยงเกิน
        
        Args:
            lot_size: ขนาด Lot ที่คำนวณได้
            lot_step: ขั้นบันได Lot (เช่น 0.01)
            
        Returns:
            float: ขนาด Lot ที่ปัดลงแล้ว
        """
        if lot_step <= 0:
            lot_step = 0.01  # ค่าเริ่มต้น
            
        # ปัดลง
        steps = math.floor(lot_size / lot_step)
        return round(steps * lot_step, 8)  # ปัดทศนิยมเพื่อป้องกัน Floating Point Error

    def calculate_sl_tp_prices(
        self,
        symbol: str,
        order_type: str,  # "BUY" หรือ "SELL"
        entry_price: float,
        atr_value: float,
        sl_multiplier: float = None,
        tp_multiplier: float = None
    ) -> Optional[Dict]:
        """
        คำนวณราคา Stop Loss และ Take Profit จากค่า ATR
        
        สูตร:
        - BUY:  SL = Entry - (ATR × SL Multiplier)
                TP = Entry + (ATR × TP Multiplier)
        - SELL: SL = Entry + (ATR × SL Multiplier)
                TP = Entry - (ATR × TP Multiplier)
        
        Args:
            symbol: คู่เงิน
            order_type: "BUY" หรือ "SELL"
            entry_price: ราคา Entry
            atr_value: ค่า ATR ปัจจุบัน
            sl_multiplier: ตัวคูณ ATR สำหรับ SL (None = ใช้ Config)
            tp_multiplier: ตัวคูณ ATR สำหรับ TP (None = ใช้ Config)
            
        Returns:
            Dict: {"sl": float, "tp": float, "sl_distance": float, "tp_distance": float, "rr_ratio": float}
        """
        # v6.13: per-symbol override → fallback global
        # XAUUSD ใช้ 1.8×/3.6× (wick noise สูง) — FX ใช้ 1.5×/3.0× (default)
        sym_overrides = bot_config.symbols.symbol_overrides.get(symbol, {})
        if sl_multiplier is None:
            sl_multiplier = sym_overrides.get(
                "sl_atr_multiplier", bot_config.indicators.atr_sl_multiplier
            )
        if tp_multiplier is None:
            tp_multiplier = sym_overrides.get(
                "tp_atr_multiplier", bot_config.indicators.atr_tp_multiplier
            )

        # คำนวณระยะ SL/TP
        sl_distance = atr_value * sl_multiplier
        tp_distance = atr_value * tp_multiplier

        # คำนวณ Risk:Reward Ratio
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # ตรวจสอบ RR ขั้นต่ำ (v8.0.15: 1% relative tolerance — รองรับทุก symbol)
        if rr_ratio < self._config.MIN_RISK_REWARD_RATIO * (1.0 - 0.01):
            print(f"⚠️ [Position Sizer] RR Ratio ({rr_ratio:.4f}) ต่ำกว่าขั้นต่ำ ({self._config.MIN_RISK_REWARD_RATIO})")
            # ปรับ TP ให้ได้ RR ขั้นต่ำ
            tp_distance = sl_distance * self._config.MIN_RISK_REWARD_RATIO
            rr_ratio = self._config.MIN_RISK_REWARD_RATIO
            print(f"   📏 ปรับ TP เป็น {tp_distance:.5f} (RR: 1:{rr_ratio:.1f})")

        # ปัดราคาตาม Digits + บังคับ spread buffer ไม่ให้ SL/TP ชน bid/ask
        symbol_info = self._connector.get_symbol_info(symbol)
        digits = symbol_info["digits"] if symbol_info else 5
        point = symbol_info.get("point", 10 ** -digits) if symbol_info else 10 ** -digits
        stops_level_points = symbol_info.get("trade_stops_level", 0) if symbol_info else 0
        spread_points = symbol_info.get("spread", 0) if symbol_info else 0

        # ระยะ SL ขั้นต่ำ = max(stops_level ของโบรก, 1.5 × spread, 3 points)
        min_sl_distance = max(
            stops_level_points * point,
            1.5 * spread_points * point,
            3 * point,
        )
        if sl_distance < min_sl_distance:
            print(f"⚠️ [Position Sizer] SL distance {sl_distance:.5f} < ขั้นต่ำ {min_sl_distance:.5f} "
                  f"(spread={spread_points}pts, stops_level={stops_level_points}pts) → ขยาย SL")
            sl_distance = min_sl_distance
            # ขยาย TP ตามสัดส่วน RR เดิม
            tp_distance = sl_distance * rr_ratio

        # คำนวณราคา SL/TP ตามทิศทาง
        if order_type.upper() == "BUY":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        elif order_type.upper() == "SELL":
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        else:
            print(f"❌ [Position Sizer] ประเภทคำสั่งไม่ถูกต้อง: {order_type}")
            return None

        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        result = {
            "sl": sl_price,
            "tp": tp_price,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "rr_ratio": rr_ratio,
        }

        if bot_config.debug_mode:
            print(f"\n📏 [Position Sizer] SL/TP สำหรับ {order_type} {symbol}:")
            print(f"   📍 Entry:    {entry_price:.{digits}f}")
            print(f"   🔴 SL:      {sl_price:.{digits}f} (ห่าง {sl_distance:.{digits}f})")
            print(f"   🟢 TP:      {tp_price:.{digits}f} (ห่าง {tp_distance:.{digits}f})")
            print(f"   ⚖️  RR:     1:{rr_ratio:.1f}")

        return result

    def validate_trade_risk(
        self,
        symbol: str,
        lot_size: float,
        sl_distance_price: float,
        remaining_daily_budget: float
    ) -> Tuple[bool, str]:
        """
        ตรวจสอบความเสี่ยงของเทรดก่อนส่งคำสั่ง (Final Validation)
        
        ตรวจสอบ:
        1. Lot Size > 0
        2. SL Distance > 0
        3. Risk Amount ไม่เกิน 1% ของ Balance
        4. Risk Amount ไม่เกิน Daily Budget ที่เหลือ
        
        Args:
            symbol: คู่เงิน
            lot_size: ขนาด Lot
            sl_distance_price: ระยะ SL เป็นราคา
            remaining_daily_budget: งบเสี่ยงรายวันที่เหลือ
            
        Returns:
            Tuple[bool, str]: (ผ่านหรือไม่, เหตุผล)
        """
        # ตรวจสอบขั้นพื้นฐาน
        if lot_size <= 0:
            return (False, "❌ Lot Size ต้องมากกว่า 0")
        if sl_distance_price <= 0:
            return (False, "❌ ห้ามเทรดโดยไม่มี Stop Loss")

        # คำนวณความเสี่ยงจริง
        symbol_info = self._connector.get_symbol_info(symbol)
        if symbol_info is None:
            return (False, "❌ ดึงข้อมูล Symbol ล้มเหลว")

        digits = symbol_info["digits"]
        contract_size = symbol_info["trade_contract_size"]
        
        # คำนวณ Pip Size
        pip_size = 0.0001 if (digits == 5 or digits == 4) else 0.01
        sl_pips = sl_distance_price / pip_size
        pip_value = self._calculate_pip_value(symbol, pip_size, contract_size, "USD")
        
        risk_amount = lot_size * sl_pips * pip_value
        
        # ดึง Balance
        balance = self._connector.get_balance()
        risk_pct = risk_amount / balance if balance > 0 else 1.0

        # ตรวจสอบ Risk %
        if risk_pct > self._config.MAX_RISK_PER_TRADE_PCT * 1.1:  # 10% buffer
            return (False, f"❌ ความเสี่ยง {risk_pct:.2%} เกิน {self._config.MAX_RISK_PER_TRADE_PCT:.1%}")

        # ตรวจสอบ Daily Budget
        if risk_amount > remaining_daily_budget * 1.05:  # 5% buffer
            return (False, f"❌ ความเสี่ยง ${risk_amount:,.2f} เกินงบวันนี้ ${remaining_daily_budget:,.2f}")

        return (True, f"✅ ผ่านการตรวจสอบ (Risk: {risk_pct:.2%}, ${risk_amount:,.2f})")

    def __repr__(self) -> str:
        return (
            f"PositionSizer(risk={self._config.DEFAULT_RISK_PER_TRADE_PCT:.2%}, "
            f"range={self._config.MIN_RISK_PER_TRADE_PCT:.1%}-{self._config.MAX_RISK_PER_TRADE_PCT:.1%})"
        )
