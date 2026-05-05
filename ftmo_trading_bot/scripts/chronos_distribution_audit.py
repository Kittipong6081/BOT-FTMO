"""
v7.1 — Chronos uncertainty distribution audit.

อ่าน signal pool → คำนวณ raw ratio (q90-q10)/(atr*√8) ของทุก signal →
plot histogram + percentiles → ใช้ตัดสินใจว่า log1p formula ของ v7.1 ใช้งานได้
หรือต้อง re-tune.

Usage:
    python scripts/chronos_distribution_audit.py --pool data/signal_pool_3000.pkl

Output:
    - histogram percentiles (10/25/50/75/90/99)
    - saturation rate (% of signals ที่จะ saturate ที่ 3.0 ภายใต้ formula เดิม v7.0.2)
    - new (log1p) median ที่ปลอดภัยจาก saturate
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from typing import List

import numpy as np

# Make ftmo_trading_bot/ importable เมื่อเรียกจาก script root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronos uncertainty distribution audit")
    parser.add_argument(
        "--pool", default="data/signal_pool_3000.pkl",
        help="Path to signal pool pickle (relative to ftmo_trading_bot/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pool):
        print(f"❌ Pool not found: {args.pool}")
        return 1

    print(f"→ Loading pool: {args.pool}")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    # Pool = list[list[signal_dict]] (episodes ของ signals)
    sigs: List[dict] = []
    for ep in pool:
        sigs.extend(ep)
    print(f"   {len(pool):,} episodes, {len(sigs):,} signals")

    # signal dict v7+ มี 'chronos_uncertainty_norm' ที่ saved (post-formula)
    # ผมต้องการ raw ratio ก่อน clip — ถ้าไม่มีต้อง compute reverse
    norm_unc = np.array(
        [float(s.get("chronos_uncertainty_norm", 0.0)) for s in sigs],
        dtype=np.float64,
    )

    # ปัจจุบัน formula = clip((q90-q10)/(atr*√8), 0, 3)
    # ถ้าค่าตรงนี้ใน pool = saved value แล้ว → ใช้ตรง ๆ
    print("\n=== Saved chronos_uncertainty_norm distribution ===")
    print(f"  count:       {len(norm_unc):,}")
    print(f"  mean:        {norm_unc.mean():.4f}")
    print(f"  std:         {norm_unc.std():.4f}")
    print(f"  min / max:   {norm_unc.min():.4f} / {norm_unc.max():.4f}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        v = float(np.percentile(norm_unc, p))
        print(f"  p{p:>2}:         {v:.4f}")

    # Saturation rate (= % ของ signals ที่ติด clip ที่ 3.0)
    sat_count = int(np.sum(norm_unc >= 2.99))
    print(f"\n  Saturated (≥ 2.99): {sat_count:,} ({sat_count/len(norm_unc)*100:.2f}%)")

    if sat_count / len(norm_unc) > 0.3:
        print("\n⚠️  Saturation > 30% — feature ไม่มี gradient ใน most signals")
        print("    v7.1 log1p formula จะลด saturation อย่างมาก")
    else:
        print("\n✅ Saturation < 30% — formula ปัจจุบันยังพอใช้ได้")

    # Simulate v7.1 log1p formula ผ่าน inverse
    # ปัจจุบัน: norm = clip(raw_ratio, 0, 3)  → raw_ratio approx norm (ยกเว้นที่ saturate)
    # New v7.1:  log1p_norm = clip(log1p(raw_ratio) / 2, 0, 3)
    #           ที่ raw_ratio = 3 (current saturate point) → log1p(3)/2 = 0.693
    #           ที่ raw_ratio = 10 → log1p(10)/2 = 1.199
    print("\n=== Simulated v7.1 log1p distribution (assume saturated raws ≈ 5) ===")
    raw_proxy = norm_unc.copy()
    raw_proxy[raw_proxy >= 2.99] = 5.0  # assume saturated were ~ratio 5+
    new_unc = np.clip(np.log1p(raw_proxy) / 2.0, 0.0, 3.0)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        v = float(np.percentile(new_unc, p))
        print(f"  p{p:>2}:         {v:.4f}")
    new_sat = int(np.sum(new_unc >= 2.99))
    print(f"  Saturated (≥ 2.99): {new_sat:,} ({new_sat/len(new_unc)*100:.2f}%)")

    print("\n→ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
