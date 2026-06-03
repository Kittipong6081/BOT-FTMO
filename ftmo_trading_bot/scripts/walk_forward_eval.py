"""
===============================================================================
walk_forward_eval.py — Out-of-TIME generalization eval (rebuild/regime-aware)
===============================================================================
WHY THIS EXISTS
---------------
The existing `holdout_eval.py` compares the model on a train pool vs a holdout
pool — but BOTH pools are generated from the SAME OHLCV history (`data/ohlcv`),
only on different episode-SEED grids. That is the same *distribution*, not a
different *time period*. It cannot detect regime shift, which is exactly how the
2026-06 FTMO challenge died (model showed sim Pass ~88% yet bled live in a
trending-down regime). See `~/.claude/plans/hedge-fund-quant-flickering-pudding.md`.

This harness splits the OHLCV history by TIME into chronological folds and
evaluates the model on each. M15 bars are time-ordered, so a bar-index fraction
== a time slice (build_mr_signal_pool now accepts start_frac/end_frac).

MODE 1 (default, fast — no retrain): evaluate the CURRENTLY DEPLOYED model on
each time-fold pool. Reveals performance DISPERSION across regimes. If the most
recent fold craters relative to older folds, the edge is regime-dependent and
the "88% pass" is an artifact of the dominant historical regime.

MODE 2 (true walk-forward, expensive — TODO): retrain on the older folds only,
then test on the most recent fold. That is the gold-standard OOS test; it needs
a full training run (~hours) and is wired separately. The fold definitions and
pool-building here are designed to feed it.

USAGE
-----
    .venv/bin/python ftmo_trading_bot/scripts/walk_forward_eval.py
    .venv/bin/python ftmo_trading_bot/scripts/walk_forward_eval.py \\
        --pool_size 600 --n_episodes 800 --reuse
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

os.environ.setdefault("SMC_QUIET", "1")
os.environ.setdefault("BOT_DISABLE_CHRONOS", "1")


# Default chronological folds over the OHLCV history (2022-05 .. 2026-05).
# (name, start_frac, end_frac). 'recent' is the out-of-time test window —
# closest to the live-fail regime; an embargo gap separates it from 'older'.
DEFAULT_FOLDS: List[Tuple[str, float, float]] = [
    ("older",  0.00, 0.55),   # ~2022-05 .. ~2024-09
    ("mid",    0.55, 0.78),   # ~2024-09 .. ~2025-08
    ("recent", 0.82, 1.00),   # ~2025-09 .. 2026-05  <- OUT-OF-TIME test
]


def _build_fold_pool(data_dir: str, pool_size: int, max_days: int, workers: int,
                     seed: int, start_frac: float, end_frac: float,
                     save_path: str, reuse: bool, model_live_exit: bool = False) -> int:
    from build_mr_signal_pool import build_pool

    if reuse and os.path.exists(save_path):
        import pickle
        with open(save_path, "rb") as f:
            n = len(pickle.load(f))
        print(f"   ♻️  reuse existing fold pool ({n} episodes): {save_path}")
        return n

    return build_pool(
        data_dir=data_dir, pool_size=pool_size, max_days=max_days,
        save_path=save_path, workers=workers, seed=seed,
        start_frac=start_frac, end_frac=end_frac, model_live_exit=model_live_exit,
    )


def main():
    p = argparse.ArgumentParser(description="Out-of-time walk-forward eval (Mode 1: eval deployed model per time-fold)")
    p.add_argument("--model", default=os.path.join(ROOT, "models/mr/best/ppo_mr_filter.zip"))
    p.add_argument("--vec_normalize", default=os.path.join(ROOT, "models/mr/best/vec_normalize_mr.pkl"))
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data", "ohlcv"))
    p.add_argument("--pool_size", type=int, default=600, help="episodes to build per fold")
    p.add_argument("--max_days", type=int, default=45)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_episodes", type=int, default=800, help="eval rollouts per fold")
    p.add_argument("--ml_threshold", type=float, default=0.30)
    p.add_argument("--risk_per_trade", type=float, default=0.0070)  # live default (v8.0.56)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--tmp_dir", default="/tmp/wf_pools")
    p.add_argument("--reuse", action="store_true", help="reuse fold pools already built in tmp_dir")
    p.add_argument("--model_live_exit", action="store_true",
                   help="build fold pools with LIVE exit (BE+partial) instead of 'ride' — measures the sim↔live exit gap")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)

    from holdout_eval import _eval_on_pool  # reuse the exact env-rollout eval

    os.makedirs(args.tmp_dir, exist_ok=True)

    print("=" * 78)
    print(" 🔭 Walk-Forward / Out-of-TIME Eval (Mode 1 — deployed model per time-fold)")
    print("=" * 78)
    print(f"   Model:       {os.path.relpath(args.model, ROOT)}")
    print(f"   Folds:       {[f[0] for f in DEFAULT_FOLDS]}")
    print(f"   Pool/fold:   {args.pool_size} episodes built | {args.n_episodes} eval rollouts")
    print(f"   risk/trade:  {args.risk_per_trade:.4f} | ml_threshold {args.ml_threshold}")
    print("=" * 78)

    results: Dict[str, Dict] = {}
    for name, lo, hi in DEFAULT_FOLDS:
        print(f"\n── Fold '{name}'  frac [{lo:.2f}, {hi:.2f}] ──────────────────────────")
        pool_path = os.path.join(args.tmp_dir, f"wf_{name}.pkl")
        n_eps = _build_fold_pool(
            args.data_dir, args.pool_size, args.max_days, args.workers,
            args.seed, lo, hi, pool_path, args.reuse, args.model_live_exit,
        )
        if n_eps == 0:
            print(f"   ⚠️ fold '{name}' produced 0 episodes — skipped")
            continue
        m = _eval_on_pool(
            args.model, args.vec_normalize, pool_path,
            args.n_episodes, args.ml_threshold, args.risk_per_trade,
        )
        # EV/trade proxy = per-episode profit / takes-per-episode
        ev_trade = m["profit_avg"] / m["trades_avg"] if m["trades_avg"] > 0 else 0.0
        m["ev_trade"] = ev_trade
        results[name] = m
        print(f"   Pass {m['pass_rate']*100:5.1f}% | Win {m['win_rate']*100:5.1f}% | "
              f"Take {m['take_rate']*100:5.1f}% | Profit/ep ${m['profit_avg']:+,.0f} | "
              f"EV/trade ${ev_trade:+.2f} | DD {m['total_dd_max']*100:.1f}% | Breach {m['breach_rate']*100:.1f}%")

    # ---- Comparison + verdict ----
    print("\n" + "=" * 78)
    print(" 📊 Time-Fold Comparison")
    print("=" * 78)
    print(f"   {'fold':<8}{'Pass%':>8}{'Win%':>8}{'Take%':>8}{'Profit/ep':>12}{'EV/trade':>11}{'DD%':>7}{'Breach%':>9}")
    for name, _, _ in DEFAULT_FOLDS:
        if name not in results:
            continue
        m = results[name]
        print(f"   {name:<8}{m['pass_rate']*100:>8.1f}{m['win_rate']*100:>8.1f}"
              f"{m['take_rate']*100:>8.1f}{m['profit_avg']:>12,.0f}{m['ev_trade']:>11.2f}"
              f"{m['total_dd_max']*100:>7.1f}{m['breach_rate']*100:>9.1f}")

    print("\n" + "=" * 78)
    print(" 🎯 Out-of-Time Verdict (recent vs older)")
    print("=" * 78)
    if "recent" in results and "older" in results:
        dp = (results["older"]["pass_rate"] - results["recent"]["pass_rate"]) * 100
        de = results["older"]["ev_trade"] - results["recent"]["ev_trade"]
        print(f"   Δ Pass (older − recent): {dp:+6.1f} pp")
        print(f"   Δ EV/trade             : {de:+6.2f}")
        recent_ev = results["recent"]["ev_trade"]
        if recent_ev <= 0:
            verdict = "❌ NO EDGE on recent regime — EV/trade ≤ 0 out-of-time (matches live failure)"
        elif dp > 15:
            verdict = "🟠 REGIME-DEPENDENT — recent Pass collapses vs older (edge is regime-specific)"
        elif dp > 8:
            verdict = "🟡 MILD regime drift — watch"
        else:
            verdict = "✅ STABLE across time — edge holds out-of-time"
        print(f"   Verdict: {verdict}")
    else:
        print("   ⚠️ insufficient folds to judge")
    print("=" * 78)


if __name__ == "__main__":
    main()
