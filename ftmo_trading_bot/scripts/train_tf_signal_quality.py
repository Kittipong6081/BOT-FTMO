"""
===============================================================================
TF Signal Quality Trainer (v8.1 Phase 3) — GBM + Isotonic Calibrator on TF pool
===============================================================================
Clone of `train_mr_signal_quality.py` pointed at the TF pool. Uses a TF-tuned
feature list (base + temporal + TF extras: trend_age_bars, pullback_depth_atr,
adx_at_entry). MR-only features (bb_extreme, etc.) are dropped — they are 0 in
the TF pool.

Usage:
    python scripts/train_tf_signal_quality.py
    python scripts/train_tf_signal_quality.py --pool data/tf_signal_pool_5000.pkl

Output: data/tf_signal_quality_model.pkl
===============================================================================
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss


# TF feature list: base 17 + temporal 7 + TF extras 3 (= 27)
FEATURE_KEYS = [
    "confluence_score", "rr_ratio", "atr_pips", "ob_score",
    "market_bias", "bias_alignment", "sl_distance_atr",
    "rsi_value", "trend_strength", "macd_histogram", "ob_size_atr",
    "adx", "stoch_k", "bb_pctb", "atr_change_ratio", "price_roc",
    "direction",
    # temporal / regime
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week",
    "minutes_since_session_start", "is_post_weekend_first_hour",
    "volatility_regime_score", "atr_zscore_30bars",
    # TF-specific extras
    "trend_age_bars", "pullback_depth_atr", "adx_at_entry",
]


def load_pool_features(pool_path: str):
    print(f"→ Loading pool: {pool_path}")
    with open(pool_path, "rb") as f:
        pool = pickle.load(f)
    all_sigs = [sig for ep in pool for sig in ep]
    groups = np.array([ep_idx for ep_idx, ep in enumerate(pool) for _ in ep], dtype=np.int64)
    print(f"   {len(pool):,} episodes, {len(all_sigs):,} signals")
    X = np.array([[sig.get(k, 0.0) for k in FEATURE_KEYS] for sig in all_sigs], dtype=np.float64)
    outcomes = np.array([sig["outcome_pnl_ratio"] for sig in all_sigs])
    wins = (outcomes > 0).astype(int)
    print(f"   Baseline win rate: {wins.mean()*100:.2f}%   Mean outcome: {outcomes.mean():+.4f}")
    return pool, all_sigs, X, outcomes, wins, groups


def train_gbm(X, y, groups, random_state=42, n_splits=5):
    print(f"\n→ Training GBM with GroupKFold OOF (n_splits={n_splits})...")
    t0 = time.time()
    tpl = GradientBoostingClassifier(max_depth=4, n_estimators=300,
                                     learning_rate=0.03, random_state=random_state)
    cv = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(tpl, X, y, groups=groups, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    print(f"   OOF time: {time.time()-t0:.1f}s")
    auc = roc_auc_score(y, oof)
    mark = "🟢 strong" if auc > 0.58 else "🟡 moderate" if auc > 0.55 else "🔴 weak"
    print(f"   OOF AUC: {auc:.4f}  ({mark})")
    brier_u = brier_score_loss(y, oof)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y)
    oof_cal = cal.transform(oof)
    brier_c = brier_score_loss(y, oof_cal)
    print(f"   Brier {brier_u:.4f} → {brier_c:.4f}")
    print("\n→ Training final base GBM on full data...")
    final = GradientBoostingClassifier(max_depth=4, n_estimators=300,
                                       learning_rate=0.03, random_state=random_state)
    final.fit(X, y)
    return final, cal, auc, oof_cal, brier_u, brier_c


def main():
    p = argparse.ArgumentParser(description="Train TF Signal Quality (GBM)")
    p.add_argument("--pool", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--no_rescore", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.pool is None:
        args.pool = os.path.join(ROOT, "data", "tf_signal_pool_5000.pkl")
    if args.save is None:
        args.save = os.path.join(ROOT, "data", "tf_signal_quality_model.pkl")
    if not os.path.exists(args.pool):
        print(f"❌ TF pool not found: {args.pool}")
        print(f"   Run: python scripts/build_tf_signal_pool.py --pool_size 5000")
        sys.exit(1)

    print("=" * 72)
    print(" TF Signal Quality Model Trainer (v8.1 Phase 3)")
    print(f"   Pool: {args.pool}   Save: {args.save}")
    print("=" * 72)

    pool, sigs, X, outs, y, groups = load_pool_features(args.pool)
    model, cal, auc, oof_cal, brier_u, brier_c = train_gbm(X, y, groups, random_state=args.seed)

    rng = np.random.default_rng(args.seed)
    sample_n = min(5000, X.shape[0])
    idx = rng.choice(X.shape[0], size=sample_n, replace=False)
    train_dist = {feat: X[idx, i].astype(np.float32) for i, feat in enumerate(FEATURE_KEYS)}

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "wb") as f:
        pickle.dump({"model": model, "calibrator": cal, "keys": FEATURE_KEYS,
                     "train_dist": train_dist, "strategy": "trend_following"}, f, protocol=4)
    print(f"\n✓ Saved: {args.save} ({os.path.getsize(args.save)/(1024*1024):.1f} MB)")

    if not args.no_rescore:
        for sig, pr in zip(sigs, oof_cal):
            sig["ml_score"] = float(pr)
        with open(args.pool, "wb") as f:
            pickle.dump(pool, f, protocol=4)
        print(f"   ✓ Pool re-scored ({len(sigs):,} ml_scores) + saved")

    print("\n" + "=" * 72)
    print(f" Done — OOF AUC={auc:.4f}, Brier {brier_u:.4f} → {brier_c:.4f}")
    print("=" * 72)
    print("\n🎯 Next: train_tf_signal_filter.py")


if __name__ == "__main__":
    main()
