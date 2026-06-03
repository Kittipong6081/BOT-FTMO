"""
===============================================================================
TF Auto-Train Pipeline (v8.1 Phase 3) — slim orchestrator
===============================================================================
Chains the TF training pipeline end-to-end:
    build_tf_signal_pool → train_tf_signal_quality → train_tf_signal_filter → eval

This is a SLIM version of `auto_train_pipeline.py` (no auto-tune search loop yet
— that arrives in Phase 4 once TF has a baseline). It runs one full pass and
checks the result against the eval gates, snapshotting to models/tf/best/ if the
gates pass.

Usage:
    python scripts/auto_train_pipeline_tf.py \
        --pool_size 5000 --timesteps_p1 5000000 --timesteps_p2 2000000 \
        --target_pass_rate 0.08 --target_dd_max 0.06 --target_profitable 0.50
===============================================================================
"""

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("SMC_QUIET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _log(msg):
    print(f"[auto_tf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="TF auto-train pipeline (slim)")
    p.add_argument("--pool_size", type=int, default=5000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--timesteps_p1", type=int, default=5_000_000)
    p.add_argument("--timesteps_p2", type=int, default=2_000_000)
    p.add_argument("--outcome_noise", type=float, default=0.05)
    p.add_argument("--ml_threshold", type=float, default=0.30)
    p.add_argument("--risk_per_trade", type=float, default=0.0060)
    p.add_argument("--rebuild_pool", action="store_true")
    p.add_argument("--rebuild_gbm", action="store_true")
    # gates
    p.add_argument("--target_pass_rate", type=float, default=0.08)
    p.add_argument("--target_dd_max", type=float, default=0.06)
    p.add_argument("--target_daily_dd_max", type=float, default=0.035)
    p.add_argument("--target_profitable", type=float, default=0.50)
    p.add_argument("--breach_rate_max", type=float, default=0.05)
    args = p.parse_args()

    data_dir = os.path.join(ROOT, "data", "ohlcv")
    pool_path = os.path.join(ROOT, "data", f"tf_signal_pool_{args.pool_size}.pkl")
    gbm_path = os.path.join(ROOT, "data", "tf_signal_quality_model.pkl")

    t_start = time.time()

    # ── Step A: pool ──
    if args.rebuild_pool or not os.path.exists(pool_path):
        _log(f"Building TF pool (size={args.pool_size})...")
        from scripts.build_tf_signal_pool import build_pool
        build_pool(data_dir, args.pool_size, 45, pool_path, args.workers)
    else:
        _log(f"Reusing TF pool: {pool_path}")

    # ── Step B: GBM quality ──
    if args.rebuild_gbm or not os.path.exists(gbm_path):
        _log("Training TF GBM quality model...")
        import scripts.train_tf_signal_quality as q
        sys.argv = ["train_tf_signal_quality", "--pool", pool_path, "--save", gbm_path]
        q.main()
    else:
        _log(f"Reusing TF GBM: {gbm_path}")

    # ── Step C: RL filter (2-phase) + eval ──
    _log("Training TF RL filter (2-phase)...")
    import scripts.train_tf_signal_filter as f
    sys.argv = [
        "train_tf_signal_filter", "--fresh",
        "--timesteps_p1", str(args.timesteps_p1),
        "--timesteps_p2", str(args.timesteps_p2),
        "--n_envs", str(args.n_envs),
        "--pool_size", str(args.pool_size),
        "--outcome_noise", str(args.outcome_noise),
        "--ml_threshold", str(args.ml_threshold),
        "--risk_per_trade", str(args.risk_per_trade),
    ]
    metrics = f.main()

    # ── Step D: gate check ──
    gates = {
        "pass_rate>=": (metrics["pass_rate"], args.target_pass_rate),
        "dd_max<=": (metrics["total_dd_max"], args.target_dd_max),
        "daily_dd_max<=": (metrics["daily_dd_max"], args.target_daily_dd_max),
        "profitable>=": (metrics["profitable_rate"], args.target_profitable),
        "breach<=": (metrics["breach_rate"], args.breach_rate_max),
    }
    passed = (
        metrics["pass_rate"] >= args.target_pass_rate
        and metrics["total_dd_max"] <= args.target_dd_max
        and metrics["daily_dd_max"] <= args.target_daily_dd_max
        and metrics["profitable_rate"] >= args.target_profitable
        and metrics["breach_rate"] <= args.breach_rate_max
    )

    _log("=" * 60)
    _log(f"TF eval: pass={metrics['pass_rate']:.1%} dd={metrics['total_dd_max']:.2%} "
         f"daily_dd={metrics['daily_dd_max']:.2%} profitable={metrics['profitable_rate']:.1%} "
         f"breach={metrics['breach_rate']:.1%} → {'✅ PASS' if passed else '❌ FAIL gates'}")
    _log("=" * 60)

    if passed:
        best_dir = os.path.join(ROOT, "models", "tf", "best")
        os.makedirs(best_dir, exist_ok=True)
        md = os.path.join(ROOT, "models", "tf")
        for name in ("ppo_tf_filter.zip", "vec_normalize_tf.pkl"):
            src = os.path.join(md, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(best_dir, name))
        _log(f"Snapshotted best → {best_dir}")
        _log("Next: load models/tf/ in main, set bot_config.tf.paper_mode=False (canary).")
    else:
        _log("Gates not met — tune TF reward params / thresholds (Phase 4) and re-run.")

    _log(f"Total time: {(time.time()-t_start)/60:.1f} min")
    return metrics


if __name__ == "__main__":
    main()
