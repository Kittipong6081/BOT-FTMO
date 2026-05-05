"""
v7.1 — Walk-forward eval (12 monthly windows).

ใช้ตรวจ regime stability ของ GBM AUC + signal pool win rate ระหว่าง months.
ถ้า variance ของ AUC across windows > 0.04 → regime sensitive → ต้อง retrain ถี่กว่า.

Usage:
    python scripts/walk_forward_eval.py --pool data/signal_pool_3000.pkl

Output:
    - per-window: window_idx, n_signals, baseline_wr, gbm_auc
    - summary: AUC mean / std / min / max
    - verdict: STABLE (std ≤ 0.04) / DRIFTING (std > 0.04)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward eval (12 monthly windows)")
    parser.add_argument(
        "--pool", default="data/signal_pool_3000.pkl",
        help="Path to signal pool pickle",
    )
    parser.add_argument(
        "--n_windows", type=int, default=12,
        help="Number of equal-size windows to split pool by",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pool):
        print(f"❌ Pool not found: {args.pool}")
        return 1

    print(f"→ Loading pool: {args.pool}")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    sigs: List[dict] = []
    for ep in pool:
        sigs.extend(ep)
    print(f"   {len(pool):,} episodes, {len(sigs):,} signals")

    if len(sigs) < args.n_windows * 100:
        print(f"❌ Pool too small for {args.n_windows} windows (need ≥ {args.n_windows * 100})")
        return 1

    # GBM AUC ต้องใช้ same features ทุก window — load model + features
    try:
        from sklearn.metrics import roc_auc_score
        from ml.signal_quality import SignalQualityModel
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1

    model_path = "data/signal_quality_model.pkl"
    if not os.path.exists(model_path):
        print(f"⚠️  GBM model not found: {model_path} — skip AUC, report WR only")
        model = None
    else:
        try:
            model = SignalQualityModel(model_path)
        except Exception as e:
            print(f"⚠️  GBM load failed: {e} — skip AUC")
            model = None

    # Split into equal windows (chronological assumed by signal order)
    window_size = len(sigs) // args.n_windows
    wins_per_window = []
    aucs_per_window = []

    print(f"\n=== Walk-forward eval ({args.n_windows} windows × ~{window_size:,} signals) ===")
    print(f"{'idx':>3} {'n':>6} {'baseline_wr':>12} {'gbm_auc':>10}")
    print("-" * 40)

    for i in range(args.n_windows):
        start = i * window_size
        end = start + window_size if i < args.n_windows - 1 else len(sigs)
        win_sigs = sigs[start:end]

        outcomes = np.array([s.get("outcome_pnl_ratio", 0.0) for s in win_sigs])
        wins = (outcomes > 0).astype(int)
        baseline_wr = wins.mean() * 100
        wins_per_window.append(baseline_wr)

        auc = float("nan")
        if model is not None:
            try:
                scores = model.score_batch(win_sigs)
                # ต้องมี both class
                if 0 in wins and 1 in wins:
                    auc = roc_auc_score(wins, scores)
                    aucs_per_window.append(auc)
            except Exception as e:
                print(f"   AUC fail @ window {i}: {e}")

        print(f"{i:>3} {len(win_sigs):>6,} {baseline_wr:>11.2f}% {auc:>9.4f}")

    # Summary
    print("\n=== Summary ===")
    wr_arr = np.array(wins_per_window)
    print(f"  Baseline WR:  mean={wr_arr.mean():.2f}%  std={wr_arr.std():.2f}  range=[{wr_arr.min():.2f}, {wr_arr.max():.2f}]")
    if aucs_per_window:
        auc_arr = np.array(aucs_per_window)
        auc_std = float(auc_arr.std())
        print(f"  GBM AUC:      mean={auc_arr.mean():.4f}  std={auc_std:.4f}  range=[{auc_arr.min():.4f}, {auc_arr.max():.4f}]")
        verdict = "✅ STABLE" if auc_std <= 0.04 else "⚠️  DRIFTING"
        print(f"\n  Verdict: {verdict} (threshold std ≤ 0.04)")
    else:
        print("  GBM AUC: n/a (model unavailable)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
