"""
===============================================================================
MR Signal Quality Trainer (v8.0) — GBM + Isotonic Calibrator on MR pool
===============================================================================
Same logic as `scripts/train_signal_quality.py` but pointed at the MR pool.
Reuses the SMC feature list (the MR pool dict is filled with the same key
names; MR-only fields like `bb_extreme` are appended to the feature set
so the GBM can learn the MR-specific signal).

Usage:
    python scripts/train_mr_signal_quality.py
    python scripts/train_mr_signal_quality.py --pool data/mr_signal_pool_5000.pkl

Output:
    data/mr_signal_quality_model.pkl  (model + calibrator + drift baseline)
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


# Feature list: SMC compat (so existing helpers work) + MR-specific extras.
FEATURE_KEYS = [
    "confluence_score", "rr_ratio", "atr_pips", "ob_score",
    "market_bias", "bias_alignment", "sl_distance_atr",
    "rsi_value", "trend_strength", "macd_histogram", "ob_size_atr",
    "adx", "stoch_k", "bb_pctb", "atr_change_ratio", "price_roc",
    "direction",
    # v7.1 temporal/regime
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week",
    "minutes_since_session_start", "is_post_weekend_first_hour",
    "volatility_regime_score", "atr_zscore_30bars",
    # v8.0 MR extras (zero in SMC pools — fine, just constant features there)
    "mr_setup_score", "bb_extreme", "bb_band_width_atr",
    "reversal_wick_ratio",
]


def load_pool_features(pool_path: str):
    print(f"→ Loading pool: {pool_path}")
    with open(pool_path, "rb") as f:
        pool = pickle.load(f)
    all_sigs = [sig for ep in pool for sig in ep]
    groups = np.array(
        [ep_idx for ep_idx, ep in enumerate(pool) for _ in ep],
        dtype=np.int64,
    )
    print(f"   {len(pool):,} episodes, {len(all_sigs):,} signals")

    X = np.array(
        [[sig.get(k, 0.0) for k in FEATURE_KEYS] for sig in all_sigs],
        dtype=np.float64,
    )
    outcomes = np.array([sig["outcome_pnl_ratio"] for sig in all_sigs])
    wins = (outcomes > 0).astype(int)

    print(f"   Baseline win rate: {wins.mean()*100:.2f}%")
    print(f"   Mean outcome:       {outcomes.mean():+.4f}")
    return pool, all_sigs, X, outcomes, wins, groups


def train_gbm(X, y, outcomes, groups, random_state=42, n_splits=5):
    print(f"\n→ Training GBM with GroupKFold OOF (n_splits={n_splits})...")
    t0 = time.time()
    gbm_template = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    cv = GroupKFold(n_splits=n_splits)
    oof_probs = cross_val_predict(
        gbm_template, X, y,
        groups=groups, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    print(f"   OOF time: {time.time()-t0:.1f}s")

    auc_oof = roc_auc_score(y, oof_probs)
    mark = "🟢 strong" if auc_oof > 0.58 else "🟡 moderate" if auc_oof > 0.55 else "🔴 weak"
    print(f"   OOF AUC: {auc_oof:.4f}  ({mark})")

    brier_uncal = brier_score_loss(y, oof_probs)
    print(f"   Brier (uncalibrated): {brier_uncal:.4f}")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, y)
    oof_probs_cal = calibrator.transform(oof_probs)
    brier_cal = brier_score_loss(y, oof_probs_cal)
    delta = (brier_cal / brier_uncal - 1) * 100
    print(f"   Brier (calibrated):   {brier_cal:.4f}  ({delta:+.1f}%)")

    print("\n→ Training final base GBM on full data...")
    t0 = time.time()
    gbm_final = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    gbm_final.fit(X, y)
    print(f"   Train time: {time.time()-t0:.1f}s")

    return gbm_final, calibrator, auc_oof, oof_probs_cal, brier_uncal, brier_cal


def rescore_pool(sigs, oof_probs):
    print("\n→ Re-scoring pool with calibrated OOF predictions...")
    for sig, p in zip(sigs, oof_probs):
        sig["ml_score"] = float(p)
    print(f"   Updated {len(sigs):,} ml_scores")
    print(f"   mean={oof_probs.mean():.3f}, std={oof_probs.std():.3f}, "
          f"min={oof_probs.min():.3f}, max={oof_probs.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train MR Signal Quality (GBM)")
    parser.add_argument("--pool", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no_rescore", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pool is None:
        args.pool = os.path.join(ROOT, "data", "mr_signal_pool_5000.pkl")
    if args.save is None:
        args.save = os.path.join(ROOT, "data", "mr_signal_quality_model.pkl")

    if not os.path.exists(args.pool):
        print(f"❌ MR pool not found: {args.pool}")
        print(f"   Run: python scripts/build_mr_signal_pool.py --pool_size 5000")
        sys.exit(1)

    print("=" * 72)
    print(" MR Signal Quality Model Trainer (v8.0)")
    print("=" * 72)
    print(f"   Pool: {args.pool}")
    print(f"   Save: {args.save}")
    print(f"   Re-score: {'No' if args.no_rescore else 'Yes (in-place pool update)'}")
    print("=" * 72)

    pool, sigs, X, outs, y, groups = load_pool_features(args.pool)
    model, calibrator, auc, oof_cal, brier_u, brier_c = train_gbm(
        X, y, outs, groups, random_state=args.seed
    )

    rng = np.random.default_rng(args.seed)
    sample_n = min(5000, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=sample_n, replace=False)
    train_dist = {feat: X[sample_idx, i].astype(np.float32)
                  for i, feat in enumerate(FEATURE_KEYS)}

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "wb") as f:
        pickle.dump({
            "model": model,
            "calibrator": calibrator,
            "keys": FEATURE_KEYS,
            "train_dist": train_dist,
            "strategy": "mean_reversion",
        }, f, protocol=4)
    size_mb = os.path.getsize(args.save) / (1024 * 1024)
    print(f"\n✓ Saved: {args.save} ({size_mb:.1f} MB)")

    if not args.no_rescore:
        rescore_pool(sigs, oof_cal)
        print(f"\n→ Saving updated pool back to {args.pool}...")
        with open(args.pool, "wb") as f:
            pickle.dump(pool, f, protocol=4)
        print("   ✓ Pool updated")

    print("\n" + "=" * 72)
    print(f" Done — OOF AUC={auc:.4f}, Brier {brier_u:.4f} → {brier_c:.4f}")
    print("=" * 72)
    print("\n🎯 Next: train_mr_signal_filter.py")


if __name__ == "__main__":
    main()
