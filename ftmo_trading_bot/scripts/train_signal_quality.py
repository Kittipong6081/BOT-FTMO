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
    python scripts/train_signal_quality.py --pool data/signal_pool_3000.pkl
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Features ต้อง match กับ SignalQualityModel.FEATURES
FEATURE_KEYS = [
    'confluence_score', 'rr_ratio', 'atr_pips', 'ob_score',
    'market_bias', 'bias_alignment', 'sl_distance_atr',
    'rsi_value', 'trend_strength', 'macd_histogram', 'ob_size_atr',
    'adx', 'stoch_k', 'bb_pctb', 'atr_change_ratio', 'price_roc',
    'direction',
]


def load_pool_features(pool_path: str):
    """Load pool → feature matrix + outcome + win labels"""
    print(f"→ Loading pool: {pool_path}")
    with open(pool_path, 'rb') as f:
        pool = pickle.load(f)

    all_sigs = [sig for ep in pool for sig in ep]
    print(f"   {len(pool):,} episodes, {len(all_sigs):,} signals")

    X = np.array(
        [[sig.get(k, 0.0) for k in FEATURE_KEYS] for sig in all_sigs],
        dtype=np.float64,
    )
    outcomes = np.array([sig['outcome_pnl_ratio'] for sig in all_sigs])
    wins = (outcomes > 0).astype(int)

    print(f"   Baseline win rate: {wins.mean()*100:.2f}%")
    print(f"   Mean outcome:       {outcomes.mean():+.4f}")

    return pool, all_sigs, X, outcomes, wins


def train_gbm(X, y, outcomes, random_state=42):
    """Train GBM + return trained model + CV AUC stats"""
    print("\n→ Training GBM (with held-out test for AUC)...")
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=random_state)

    # Train/test split
    gbm_eval = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    t0 = time.time()
    gbm_eval.fit(X[idx_tr], y[idx_tr])
    print(f"   Train time: {time.time()-t0:.1f}s")

    probs_te = gbm_eval.predict_proba(X[idx_te])[:, 1]
    auc = roc_auc_score(y[idx_te], probs_te)
    print(f"   Test AUC: {auc:.4f}  ({'🟢 strong edge' if auc > 0.58 else '🟡 moderate' if auc > 0.55 else '🔴 weak'})")

    # Show threshold analysis on test set
    o_te = outcomes[idx_te]
    y_te = y[idx_te]
    print(f"\n   Threshold analysis (test set n={len(y_te):,}):")
    print(f"   {'threshold':>10}  {'% kept':>8}  {'n':>6}  {'win rate':>9}  {'EV':>8}")
    for thresh in [0.30, 0.33, 0.36, 0.40, 0.45, 0.50]:
        keep = probs_te >= thresh
        n = keep.sum()
        if n < 20:
            continue
        wr = y_te[keep].mean() * 100
        ev = o_te[keep].mean()
        mark = '🟢' if ev > 0.1 else '🟡' if ev > 0 else '🔴'
        print(f"   {mark} >{thresh:.2f}    {keep.mean()*100:>6.1f}%  {n:>6}  {wr:>7.1f}%   {ev:+7.3f}")

    # Train final on FULL data for production
    print("\n→ Training final model on full data...")
    gbm_final = GradientBoostingClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        random_state=random_state,
    )
    gbm_final.fit(X, y)
    return gbm_final, auc


def rescore_pool(pool, sigs, model, X):
    """Update ml_score field ของทุก signal ใน pool (in-place)"""
    print("\n→ Re-scoring pool with new ML model...")
    probs = model.predict_proba(X)[:, 1]
    for sig, p in zip(sigs, probs):
        sig['ml_score'] = float(p)
    print(f"   Updated {len(sigs):,} signal ml_scores")
    print(f"   Distribution: mean={probs.mean():.3f}, std={probs.std():.3f}, "
          f"min={probs.min():.3f}, max={probs.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train Signal Quality Model")
    parser.add_argument("--pool", default=None,
                        help="Pool file (default: data/signal_pool_3000.pkl)")
    parser.add_argument("--save", default=None,
                        help="Save path (default: data/signal_quality_model.pkl)")
    parser.add_argument("--no_rescore", action="store_true",
                        help="ไม่ re-score pool หลัง train (train อย่างเดียว)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.pool is None:
        args.pool = os.path.join(ROOT, "data", "signal_pool_3000.pkl")
    if args.save is None:
        args.save = os.path.join(ROOT, "data", "signal_quality_model.pkl")

    if not os.path.exists(args.pool):
        print(f"❌ Pool ไม่พบ: {args.pool}")
        print(f"   รัน: python scripts/build_signal_pool.py --pool_size 3000")
        sys.exit(1)

    print("=" * 72)
    print(" Signal Quality Model Trainer")
    print("=" * 72)
    print(f"   Pool: {args.pool}")
    print(f"   Save: {args.save}")
    print(f"   Re-score: {'No' if args.no_rescore else 'Yes (update pool in-place)'}")
    print("=" * 72)

    # Load
    pool, sigs, X, outs, y = load_pool_features(args.pool)

    # Train
    model, auc = train_gbm(X, y, outs, random_state=args.seed)

    # Save model
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, 'wb') as f:
        pickle.dump({'model': model, 'keys': FEATURE_KEYS}, f, protocol=4)
    size_mb = os.path.getsize(args.save) / (1024 * 1024)
    print(f"\n✓ Saved model: {args.save} ({size_mb:.1f} MB)")

    # Re-score pool
    if not args.no_rescore:
        rescore_pool(pool, sigs, model, X)
        print(f"\n→ Saving updated pool back to {args.pool}...")
        with open(args.pool, 'wb') as f:
            pickle.dump(pool, f, protocol=4)
        print(f"   ✓ Pool updated")

    print("\n" + "=" * 72)
    print(f" Done — AUC={auc:.4f}")
    print("=" * 72)
    print("\n🎯 พร้อม train RL agent แล้ว:")
    print("   python scripts/train_signal_filter.py --fresh \\")
    print("       --timesteps_p1 10000000 --timesteps_p2 5000000 \\")
    print("       --n_envs 8 --pool_size 3000 --outcome_noise 0.02")


if __name__ == "__main__":
    main()
