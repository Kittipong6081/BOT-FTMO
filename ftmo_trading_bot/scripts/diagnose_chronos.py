"""
Chronos Diagnostic — ตรวจสอบว่า Chronos forecast ใช้งานได้ปกติบน VPS ไหม
รัน: .venv\\Scripts\\python ftmo_trading_bot\\scripts\\diagnose_chronos.py
"""
from __future__ import annotations
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

print("=" * 70)
print(" 🔬 Chronos Diagnostic")
print("=" * 70)

# 1) Load test
print("\n[1] Loading model...")
try:
    from ml.chronos_forecaster import ChronosForecaster
    fc = ChronosForecaster(verbose=True)
    if not fc.is_available:
        print("❌ Chronos load FAILED — bot จะ fallback obs[27,28]=0.0")
        sys.exit(1)
    print("✅ Load OK")
except Exception as e:
    print(f"❌ Import/load crash: {e}")
    sys.exit(1)

# 2) Forecast test - try 3 symbols ที่ราคาต่างกันมาก
import pandas as pd
import numpy as np

print("\n[2] Forecast test (3 synthetic symbols)...")
test_cases = [
    ("NZDUSD-low", 0.59700, 0.00053),   # low-priced FX (suspected bug)
    ("EURUSD-mid", 1.08500, 0.00060),   # mid-priced FX
    ("XAUUSD-high", 2300.00, 1.50),     # high-priced (gold)
]

for name, base_price, atr in test_cases:
    # synthetic 60-bar M15 with mild drift + noise
    np.random.seed(42)
    rng_walk = np.cumsum(np.random.randn(60) * atr * 0.3)
    closes = base_price + rng_walk
    df = pd.DataFrame({
        "time": pd.date_range("2026-05-07", periods=60, freq="15min"),
        "close": closes,
    })

    print(f"\n  📊 {name}: base={base_price}, atr={atr}")
    print(f"     last_close={closes[-1]:.5f}, range={closes.min():.5f}-{closes.max():.5f}")

    f = fc.forecast(name.split("-")[0], df)
    print(f"     median_h8={f['median_h8']:.5f}")
    print(f"     q10_h8   ={f['q10_h8']:.5f}")
    print(f"     q90_h8   ={f['q90_h8']:.5f}")
    spread = f['q90_h8'] - f['q10_h8']
    print(f"     spread (q90-q10)={spread:.5f}  ({spread/base_price*100:.2f}% of price)")

    # compute_features for both directions
    align_buy, unc = ChronosForecaster.compute_features(f, signal_direction=+1.0, atr_value=atr)
    align_sell, _ = ChronosForecaster.compute_features(f, signal_direction=-1.0, atr_value=atr)
    print(f"     → BUY  align={align_buy:+.0f}, unc={unc:.3f}")
    print(f"     → SELL align={align_sell:+.0f}")

    if unc >= 2.9:
        print(f"     ⚠️  Unc saturated ({unc:.2f}) — quantile spread กว้างเกินจริง")
        ratio = spread / (atr * 2.83)
        print(f"     raw_ratio = {ratio:.1f} (ปกติควร < 5)")
    elif unc >= 0.1:
        print(f"     ✅ Unc {unc:.3f} = healthy range")
    else:
        print(f"     🟡 Unc {unc:.3f} = very confident (rare for 8-bar forecast)")

print("\n" + "=" * 70)
print(" สรุป:")
print(" - ถ้าทุก symbol unc ≈ 3.0 → Chronos quantile output มี bug บน chronos-bolt-small")
print(" - ถ้า low-price symbol unc=3 แต่ high-price unc<1 → tokenizer issue ที่ราคาต่ำ")
print(" - ถ้าทุกอย่างปกติ (unc 0.1-1.5) → Chronos ทำงานได้ดี")
print("=" * 70)
