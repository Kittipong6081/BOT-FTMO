"""
===============================================================================
Fetch OHLCV Data from MT5 → CSV (สำหรับ RL Training)
===============================================================================
ดึงข้อมูลแท่งเทียน M15 ของ 9 symbols จาก MetaTrader 5 broker
แล้วบันทึกเป็น CSV ที่ ftmo_trading_bot/data/ohlcv/{SYMBOL}_M15.csv

Usage:
    python scripts/fetch_mt5_data.py --years 3
    python scripts/fetch_mt5_data.py --symbols EURUSD,GBPUSD --years 5

Prerequisites:
    - MT5 terminal เปิดอยู่ + login broker แล้ว
    - pip install MetaTrader5 pandas
===============================================================================
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ ต้องติดตั้ง MetaTrader5: pip install MetaTrader5")
    sys.exit(1)


DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
]

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


def fetch_symbol(symbol: str, timeframe_key: str, start: datetime, end: datetime) -> pd.DataFrame:
    """ดึง OHLCV ของ 1 symbol — return DataFrame หรือ empty"""
    tf = TIMEFRAMES[timeframe_key]

    if not mt5.symbol_select(symbol, True):
        print(f"  ⚠️  {symbol}: select failed ({mt5.last_error()})")
        return pd.DataFrame()

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        print(f"  ⚠️  {symbol}: no data ({mt5.last_error()})")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'tick_volume': 'volume'})
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="คอมม่า-คั่น เช่น EURUSD,GBPUSD")
    parser.add_argument("--timeframe", default="M15", choices=list(TIMEFRAMES.keys()))
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

    end = datetime.now()
    start = end - timedelta(days=args.years * 365)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"📥 ดึงข้อมูล {args.timeframe} ของ {len(symbols)} symbols "
          f"ตั้งแต่ {start.date()} ถึง {end.date()}")

    success = 0
    for sym in symbols:
        df = fetch_symbol(sym, args.timeframe, start, end)
        if df.empty:
            continue
        out_path = os.path.join(args.out_dir, f"{sym}_{args.timeframe}.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✅ {sym}: {len(df):,} bars → {out_path}")
        success += 1

    mt5.shutdown()
    print(f"\n🎯 เสร็จสิ้น: {success}/{len(symbols)} symbols")
    if success < len(symbols):
        print("   หมายเหตุ: บาง symbol อาจไม่อยู่ใน Market Watch — ลอง enable ใน MT5 ก่อน")


if __name__ == "__main__":
    main()
