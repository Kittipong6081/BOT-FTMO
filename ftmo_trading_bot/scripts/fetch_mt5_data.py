"""
===============================================================================
Fetch OHLCV Data from MT5 → CSV (สำหรับ RL Training)
===============================================================================
ดึงข้อมูลแท่งเทียน M1 + M15 + H1 + H4 ของ 10 symbols จาก MetaTrader 5 broker
แล้วบันทึกเป็น CSV ที่ ftmo_trading_bot/data/ohlcv/{SYMBOL}_{TF}.csv

Usage:
    python scripts/fetch_mt5_data.py                              # ดึงทุก symbol, ทุก TF, 3 ปี
    python scripts/fetch_mt5_data.py --years 5                    # ดึง 5 ปี
    python scripts/fetch_mt5_data.py --symbols EURUSD,GBPUSD      # เฉพาะบาง symbol
    python scripts/fetch_mt5_data.py --timeframe M15              # เฉพาะ M15 อย่างเดียว
    python scripts/fetch_mt5_data.py --timeframe M1               # เฉพาะ M1 (v8.0.55 — train-live parity)
    python scripts/fetch_mt5_data.py --timeframe M1 --years 1     # M1 1 ปี (~80MB/symbol)

หมายเหตุ M1 (v8.0.55):
    - M1 มี 1440 bars/วัน (15x ของ M15) → ขนาดไฟล์ใหญ่ ~80-100 MB/symbol/ปี
    - ใช้ใน MeanReversionBacktester proxy entry_confirm (mirror live M1 check)
    - แนะนำ --years 1 หรือ 2 (เพียงพอสำหรับ MR training pool 5000 eps)
    - Broker บางที่มี max bars limit ต่ำ — script จะ cap อัตโนมัติ

Prerequisites:
    - MT5 terminal เปิดอยู่ + login broker แล้ว
    - pip install MetaTrader5 pandas
===============================================================================
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
import time

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ ต้องติดตั้ง MetaTrader5: pip install MetaTrader5")
    sys.exit(1)


DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    "XAUUSD",  # Gold (metal — broker อาจใช้ชื่อ XAUUSD, GOLD, XAUUSD.raw, XAUUSDm)
]

TIMEFRAMES = {
    # v8.0.55: M1 added for train-live parity (MeanReversionBacktester entry_confirm)
    "M1": mt5.TIMEFRAME_M1,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


def resolve_symbol(base: str) -> str:
    """หา symbol จริงใน broker (อาจมี suffix เช่น EURUSD.raw, EURUSDm, EURUSD.a)"""
    # ลอง exact match ก่อน
    info = mt5.symbol_info(base)
    if info is not None:
        return base

    # ค้นหาจาก symbols_get ทั้งหมด
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return base

    candidates = [s.name for s in all_symbols if s.name.upper().startswith(base.upper())]

    # Fallback aliases — broker อาจใช้ชื่อต่างจาก DEFAULT_SYMBOLS
    # Gold: XAUUSD ↔ GOLD / XAU_USD / XAUUSD# etc.
    if not candidates and base.upper() == "XAUUSD":
        gold_aliases = ["GOLD", "XAU/USD", "XAU_USD", "XAUUSD#"]
        for alias in gold_aliases:
            candidates = [s.name for s in all_symbols if alias in s.name.upper()]
            if candidates:
                break

    # จัดลำดับตามความยาว (สั้นที่สุดก่อน = มักเป็น main symbol)
    candidates.sort(key=len)
    return candidates[0] if candidates else base


def fetch_symbol(symbol: str, timeframe_key: str, years: int) -> pd.DataFrame:
    """ดึง OHLCV แบบแบ่ง Chunk (Progressive Chunking) เพื่อหลีกเลี่ยง Invalid params"""
    tf = TIMEFRAMES[timeframe_key]

    resolved = resolve_symbol(symbol)
    if resolved != symbol:
        print(f"  🔍 {symbol} → ใช้ชื่อจริงในโบรก: {resolved}")
    symbol = resolved

    if not mt5.symbol_select(symbol, True):
        print(f"  ⚠️  {symbol}: select failed ({mt5.last_error()})")
        return pd.DataFrame()

    # คำนวณจำนวนแท่งที่ต้องการ
    # v8.0.55: M1 = 1440 bars/day (Forex market: 24/5 ≈ 5 days/week, ใช้ 365 × bars_per_day
    #         approximation พอเพียงสำหรับ chunked fetch — broker max bars เป็น cap จริง)
    bars_per_day = {"M1": 1440, "M15": 96, "H1": 24, "H4": 6}[timeframe_key]
    requested_count = years * 365 * bars_per_day

    # ตรวจสอบขีดจำกัด Terminal (Max bars)
    term = mt5.terminal_info()
    max_bars = getattr(term, "maxbars", 100000) if term else 100000
    # เผื่อ buffer ไว้เล็กน้อย ไม่ให้ชนขอบ limit พอดีเป๊ะ
    limit = min(requested_count, max_bars - 500)

    if limit < requested_count:
        # v8.0.55: M1 มักชนขอบ max_bars เพราะ requested ใหญ่มาก (3 ปี ≈ 1.5M bars)
        # → cap ที่ max_bars - 500 พอใช้สำหรับ training pool 5000 eps
        print(f"  ℹ️  {symbol}: จำกัดการดึงที่ {limit:,} bars (ตาม Max bars ใน MT5)")
        if timeframe_key == "M1" and limit < bars_per_day * 180:
            print(f"  ⚠️  {symbol}: M1 limit ต่ำกว่า 6 เดือน — ปรับ Max bars ใน MT5 Tools→Options→Charts")

    all_rates = []
    # v8.0.55: M1 ใช้ chunk ใหญ่กว่าเพื่อลดจำนวน round-trip (M1 = 15x ของ M15)
    chunk_size = 50000 if timeframe_key == "M1" else 20000
    pos = 0
    
    print(f"  📥 {symbol}: เริ่มดึงข้อมูลแบบ chunk (เป้าหมาย {limit:,} bars)...", end="", flush=True)

    while pos < limit:
        # คำนวณจำนวนที่ต้องดึงใน chunk นี้
        current_chunk = min(chunk_size, limit - pos)
        
        # ลองดึงข้อมูล (Retry 2 ครั้งต่อ chunk กันเหนียว)
        rates = None
        for attempt in range(2):
            rates = mt5.copy_rates_from_pos(symbol, tf, pos, current_chunk)
            if rates is not None and len(rates) > 0:
                break
            time.sleep(2) # รอ sync
        
        if rates is None or len(rates) == 0:
            # ถ้าดึง chunk แรกไม่ได้เลย อาจจะไม่มีข้อมูลจุดนี้
            if pos == 0:
                print(f"\n  ⚠️  {symbol}: ดึง chunk แรกไม่ได้ ({mt5.last_error()})")
                return pd.DataFrame()
            else:
                print(f"\n  ℹ️  {symbol}: สิ้นสุดการดึงที่ {pos:,} bars (Broker ไม่มีข้อมูลเก่ากว่านี้)")
                break
        
        all_rates.append(pd.DataFrame(rates))
        pos += len(rates)
        print(".", end="", flush=True)
        
        # ถ้าดึงได้น้อยกว่าที่ขอ แสดงว่าสุดสายส่งแล้ว
        if len(rates) < current_chunk:
            break

    if not all_rates:
        print(" Failed.")
        return pd.DataFrame()

    print(" Done!")
    df = pd.concat(all_rates, ignore_index=True)
    
    # จัดการข้อมูลและการเรียงลำดับ
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df.drop_duplicates(subset=['time']).sort_values('time')
    
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="คอมม่า-คั่น เช่น EURUSD,GBPUSD")
    parser.add_argument("--timeframe", default="ALL",
                        choices=list(TIMEFRAMES.keys()) + ["ALL"],
                        help="timeframe ที่ต้องการ (default: ALL = M15+H1+H4)")
    parser.add_argument("--years", type=int, default=3, help="ดึงย้อนหลังกี่ปี")
    parser.add_argument("--out_dir", default=None,
                        help="โฟลเดอร์ปลายทาง (default: ftmo_trading_bot/data/ohlcv)")
    args = parser.parse_args()

    if args.out_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.join(os.path.dirname(here), "data", "ohlcv")
    os.makedirs(args.out_dir, exist_ok=True)

    if not mt5.initialize():
        print(f"❌ MT5 initialize ล้มเหลว: {mt5.last_error()}")
        print("   ตรวจสอบว่า MT5 terminal เปิดอยู่และ login แล้ว")
        sys.exit(1)

    # === Diagnostic info ===
    term = mt5.terminal_info()
    acc = mt5.account_info()
    ver = mt5.version()
    print(f"🔎 MT5 version: {ver}")
    if term:
        print(f"   Terminal connected: {term.connected}, trade_allowed: {term.trade_allowed}")
        print(f"   Path: {term.path}")
    if acc:
        print(f"   Account: #{acc.login} | {acc.server} | balance={acc.balance}")
    else:
        print("   ⚠️  ไม่มี account — ยังไม่ได้ login เข้า broker!")

    # ทดสอบ minimal call บน symbol แรก
    test_sym = "EURUSD"
    mt5.symbol_select(test_sym, True)
    test_bars = mt5.copy_rates_from_pos(test_sym, mt5.TIMEFRAME_M15, 0, 10)
    if test_bars is None or len(test_bars) == 0:
        print(f"\n❌ Sanity check fail: ขอ {test_sym} 10 แท่งล่าสุดก็ไม่ได้")
        print(f"   last_error = {mt5.last_error()}")
        print(f"   → ปัญหา: MT5 terminal อาจยังไม่ได้ login broker สำเร็จ")
        print(f"   → เปิด MT5 GUI, ตรวจว่า log in แล้ว + เห็น chart EURUSD ขยับ")
        mt5.shutdown()
        sys.exit(1)
    else:
        print(f"   ✅ Sanity OK: ได้ {len(test_bars)} bars ของ {test_sym}\n")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.timeframe == "ALL":
        tf_list = list(TIMEFRAMES.keys())
    else:
        tf_list = [args.timeframe]

    total_files = len(symbols) * len(tf_list)
    print(f"📥 ดึงข้อมูล {', '.join(tf_list)} ของ {len(symbols)} symbols "
          f"ย้อนหลัง ~{args.years} ปี ({total_files} ไฟล์)")

    success = 0
    for sym in symbols:
        for tf_key in tf_list:
            df = fetch_symbol(sym, tf_key, args.years)
            if df.empty:
                continue
            out_path = os.path.join(args.out_dir, f"{sym}_{tf_key}.csv")
            df.to_csv(out_path, index=False)
            print(f"  ✅ {sym}_{tf_key}: {len(df):,} bars → {out_path}")
            success += 1

    mt5.shutdown()
    print(f"\n🎯 เสร็จสิ้น: {success}/{total_files} ไฟล์")
    if success < total_files:
        print("   หมายเหตุ: บาง symbol อาจไม่อยู่ใน Market Watch — ลอง enable ใน MT5 ก่อน")


if __name__ == "__main__":
    main()
