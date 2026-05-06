"""
===============================================================================
Train Signal Quality Model (GBM) + Re-score Pool
===============================================================================
1. โหลด pool (มี outcome_pnl_ratio)
2. Train GradientBoostingClassifier → P(win)
3. Save model ที่ data/signal_quality_model.pkl
4. Re-score ทุก signal ใน pool ด้วย model ใหม่ → update ml_score field
5. Save pool กลับ

Usage:
    python scripts/train_signal_quality.py
    python scripts/train_signal_quality.py --pool data/signal_pool_10000.pkl
    python scripts/train_signal_quality.py --no_rescore   # train อย่างเดียว ไม่ touch pool

หลัง run เสร็จ: พร้อมใช้ train_signal_filter.py (RL) ต่อได้ทันที
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


# Features ต้อง match กับ SignalQualityModel.FEATURES
# v7.1 — เพิ่ม 7 temporal/regime features (17 → 24)
FEATURE_KEYS = [
    'confluence_score', 'rr_ratio', 'atr_pips', 'ob_score',
    'market_bias', 'bias_alignment', 'sl_distance_atr',
    'rsi_value', 'trend_strength', 'macd_histogram', 'ob_size_atr',
    'adx', 'stoch_k', 'bb_pctb', 'atr_change_ratio', 'price_roc',
    'direction',
    # v7.1 temporal/regime
    'hour_of_day_sin', 'hour_of_day_cos',
    'day_of_week', 'minutes_since_session_start',
    'is_post_weekend_first_hour',
    'volatility_regime_score', 'atr_zscore_30bars',
]


def load_pool_features(pool_path: str):
    """Load pool → feature matrix + outcome + win labels + episode groups"""
    print(f"→ Loading pool: {pool_path}")
    with open(pool_path, 'rb') as f:
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
    outcomes = np.array([sig['outcome_pnl_ratio'] for sig in all_sigs])
    wins = (outcomes > 0).astype(int)

    print(f"   Baseline win rate: {wins.mean()*100:.2f}%")
    print(f"   Mean outcome:       {outcomes.mean():+.4f}")

    return pool, all_sigs, X, outcomes, wins, groups


def train_gbm(X, y, outcomes, groups, random_state=42, n_splits=5):
    """
    Train GBM + GroupKFold OOF + Isotonic calibration (Phase E1).

    Workflow:
      1. OOF probabilities via GroupKFold cross_val_predict (anti-leakage)
      2. Compute Brier (uncalibrated) — discrimination metric: AUC
      3. Fit IsotonicRegression on (oof_probs, y) — calibrated mapping
      4. Apply calibrator → calibrated OOF probs
      5. Brier (calibrated) — should drop 5-15 % per Niculescu-Mizil 2005
      6. Final base GBM trained on full X
      7. Return base + calibrator → save together

    Returns:
        gbm_final: base GBM (raw predict_proba) — full-data trained
        calibrator: fitted IsotonicRegression
        auc_oof: discrimination AUC on OOF (unchanged by calibration)
        oof_probs_calibrated: per-signal calibrated P(win) — for pool re-score
        brier_uncal, brier_cal: Brier scores (calibration improvement metric)
    """
    print(f"\n→ Training GBM with GroupKFold OOF (n_splits={n_splits}, groups=episode)...")
    t0 = time.time()

    gbm_template = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    cv = GroupKFold(n_splits=n_splits)
    oof_probs = cross_val_predict(
        gbm_template, X, y,
        groups=groups, cv=cv,
        method='predict_proba', n_jobs=-1,
    )[:, 1]
    print(f"   OOF time: {time.time()-t0:.1f}s")

    auc_oof = roc_auc_score(y, oof_probs)
    mark = '🟢 strong edge' if auc_oof > 0.58 else '🟡 moderate' if auc_oof > 0.55 else '🔴 weak'
    print(f"   OOF AUC: {auc_oof:.4f}  ({mark})")

    # ─── Phase E1: Isotonic calibration ───────────────────
    # Fit on OOF probs (group-aware) → no leakage
    # Reference: Niculescu-Mizil & Caruana 2005 — isotonic > Platt for tree models when n > 1000
    brier_uncal = brier_score_loss(y, oof_probs)
    print(f"   Brier (uncalibrated): {brier_uncal:.4f}")

    print(f"\n→ Fitting isotonic calibrator on OOF probs...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_probs, y)
    oof_probs_cal = calibrator.transform(oof_probs)
    brier_cal = brier_score_loss(y, oof_probs_cal)
    delta_pct = (brier_cal / brier_uncal - 1) * 100
    print(f"   Brier (calibrated):   {brier_cal:.4f}  ({delta_pct:+.1f}% vs uncal)")

    # Reliability diagram (5-bin)
    print(f"\n   Reliability bins (calibrated):")
    print(f"   {'bin':>10}  {'pred avg':>9}  {'true avg':>9}  {'n':>7}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]:
        mask = (oof_probs_cal >= lo) & (oof_probs_cal < hi)
        n = int(mask.sum())
        if n < 20:
            continue
        pred_avg = oof_probs_cal[mask].mean()
        true_avg = y[mask].mean()
        match = '✅' if abs(pred_avg - true_avg) < 0.03 else '🟡' if abs(pred_avg - true_avg) < 0.06 else '🔴'
        print(f"   {match} [{lo:.1f},{hi:.1f})  {pred_avg:>9.3f}  {true_avg:>9.3f}  {n:>7,}")

    # Threshold analysis on CALIBRATED OOF (now thresholds map to true probabilities)
    print(f"\n   Threshold analysis (CALIBRATED OOF, n={len(y):,}):")
    print(f"   {'threshold':>10}  {'% kept':>8}  {'n':>7}  {'win rate':>9}  {'EV':>8}")
    for thresh in [0.30, 0.33, 0.36, 0.40, 0.45, 0.50]:
        keep = oof_probs_cal >= thresh
        n = int(keep.sum())
        if n < 20:
            continue
        wr = y[keep].mean() * 100
        ev = outcomes[keep].mean()
        mark = '🟢' if ev > 0.1 else '🟡' if ev > 0 else '🔴'
        print(f"   {mark} >{thresh:.2f}    {keep.mean()*100:>6.1f}%  {n:>7}  {wr:>7.1f}%   {ev:+7.3f}")

    # Final production model trained on FULL data
    print("\n→ Training final base GBM on full data (for live inference)...")
    t0 = time.time()
    gbm_final = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    gbm_final.fit(X, y)
    print(f"   Train time: {time.time()-t0:.1f}s")
    return gbm_final, calibrator, auc_oof, oof_probs_cal, brier_uncal, brier_cal


def rescore_pool(sigs, oof_probs):
    """Update ml_score ใน pool ด้วย OOF probabilities (anti-leakage)"""
    print("\n→ Re-scoring pool with OOF predictions (not in-sample)...")
    for sig, p in zip(sigs, oof_probs):
        sig['ml_score'] = float(p)
    print(f"   Updated {len(sigs):,} signal ml_scores")
    print(f"   Distribution: mean={oof_probs.mean():.3f}, std={oof_probs.std():.3f}, "
          f"min={oof_probs.min():.3f}, max={oof_probs.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train Signal Quality Model")
    parser.add_argument("--pool", default=None,
                        help="Pool file (default: data/signal_pool_10000.pkl)")
    parser.add_argument("--save", default=None,
                        help="Save path (default: data/signal_quality_model.pkl)")
    parser.add_argument("--no_rescore", action="store_true",
                        help="ไม่ re-score pool หลัง train (train อย่างเดียว)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pool is None:
        args.pool = os.path.join(ROOT, "data", "signal_pool_10000.pkl")
    if args.save is None:
        args.save = os.path.join(ROOT, "data", "signal_quality_model.pkl")

    if not os.path.exists(args.pool):
        print(f"❌ Pool ไม่พบ: {args.pool}")
        print(f"   รัน: python scripts/build_signal_pool.py --pool_size 10000")
        sys.exit(1)

    print("=" * 72)
    print(" Signal Quality Model Trainer")
    print("=" * 72)
    print(f"   Pool: {args.pool}")
    print(f"   Save: {args.save}")
    print(f"   Re-score: {'No' if args.no_rescore else 'Yes (update pool in-place)'}")
    print("=" * 72)

    # Load
    pool, sigs, X, outs, y, groups = load_pool_features(args.pool)

    # Train
    model, calibrator, auc, oof_probs_cal, brier_uncal, brier_cal = train_gbm(
        X, y, outs, groups, random_state=args.seed,
    )

    # v7.1 — Snapshot training distribution per feature (sample 5000 rows) สำหรับ live drift monitor
    # KS test ใน SignalQualityModel.detect_drift จะเทียบ live signals กับ snapshot นี้
    rng_drift = np.random.default_rng(args.seed)
    sample_n = min(5000, X.shape[0])
    sample_idx = rng_drift.choice(X.shape[0], size=sample_n, replace=False)
    train_dist = {
        feat: X[sample_idx, i].astype(np.float32)
        for i, feat in enumerate(FEATURE_KEYS)
    }

    # Save model + calibrator (Phase E1) + drift baseline (v7.1)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'wb') as f:
        pickle.dump({
            'model': model,
            'calibrator': calibrator,
            'keys': FEATURE_KEYS,
            'train_dist': train_dist,  # v7.1 drift baseline
        }, f, protocol=4)
    size_mb = os.path.getsize(args.save) / (1024 * 1024)
    print(f"\n✓ Saved model + calibrator + drift baseline: {args.save} ({size_mb:.1f} MB)")

    # Re-score pool with CALIBRATED OOF predictions (anti-leakage + calibrated)
    if not args.no_rescore:
        rescore_pool(sigs, oof_probs_cal)
        print(f"\n→ Saving updated pool back to {args.pool}...")
        with open(args.pool, 'wb') as f:
            pickle.dump(pool, f, protocol=4)
        print(f"   ✓ Pool updated")

    print("\n" + "=" * 72)
    print(f" Done — OOF AUC={auc:.4f}, Brier {brier_uncal:.4f} → {brier_cal:.4f} ({(brier_cal/brier_uncal-1)*100:+.1f}%)")
    print("=" * 72)
    print("\n🎯 พร้อม train RL agent แล้ว:")
    print("   python scripts/train_signal_filter.py --fresh \\")
    print("       --timesteps_p1 10000000 --timesteps_p2 5000000 \\")
    print("       --n_envs 8 --pool_size 10000 --outcome_noise 0.05 --ml_threshold 0.36 --risk_per_trade 0.0099")


if __name__ == "__main__":
    main()
