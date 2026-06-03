"""
===============================================================================
TF Signal Pool Builder (v8.1 Phase 3) — Pre-generate Episode Signals for TF
===============================================================================
Mirrors `scripts/build_mr_signal_pool.py` but uses `TrendFollowingBacktester`
(entry on H1, trend filter H4/D1, wide RR with runners). Start bars are sampled
on the H1 timeline (not M15).

Usage:
    python scripts/build_tf_signal_pool.py --pool_size 5000 --workers 8

Output: data/tf_signal_pool_<size>.pkl
===============================================================================
"""

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("SMC_QUIET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


_worker_backtester = None


def _init_worker(data_dir: str):
    global _worker_backtester
    import io, contextlib
    from ml.trend_following_backtester import TrendFollowingBacktester
    with contextlib.redirect_stdout(io.StringIO()):
        _worker_backtester = TrendFollowingBacktester(data_dir)


def _build_one_episode(task):
    global _worker_backtester
    symbol, start_bar, max_days, seed = task
    rng = np.random.default_rng(seed)
    try:
        signals = _worker_backtester.generate_episode_signals(
            symbol=symbol,
            h1_start_bar=start_bar,
            num_days=max_days,
            rng=rng,
        )
        return signals if signals else None
    except Exception as e:
        print(f"   ⚠️ worker error (symbol={symbol}, start={start_bar}): {e}")
        return None


def _h1_min_bars_for_episode(bt, max_days: int) -> int:
    """H1 bars needed for one episode: lookback + days + tail margin."""
    return bt.TF_LOOKBACK_H1 + max_days * bt.H1_PER_DAY + bt.H1_PER_DAY + bt.H1_PER_DAY


def _pick_pool_tasks(data_dir: str, pool_size: int, max_days: int, seed: int = 42):
    import io, contextlib
    from ml.trend_following_backtester import TrendFollowingBacktester
    with contextlib.redirect_stdout(io.StringIO()):
        bt = TrendFollowingBacktester(data_dir)

    needed = _h1_min_bars_for_episode(bt, max_days)
    symbols = [s for s in bt._available_symbols
               if len(bt._h1_cache.get(s, [])) >= needed]
    if not symbols:
        raise RuntimeError("No symbols with enough H1 data for TF episodes")

    rng = np.random.default_rng(seed)
    n_per_symbol = max(1, pool_size // len(symbols))
    tasks = []
    for symbol in symbols:
        h1_len = len(bt._h1_cache[symbol])
        min_start = bt.TF_LOOKBACK_H1
        max_start = h1_len - needed
        if max_start <= min_start:
            continue
        starts = np.linspace(min_start, max_start, n_per_symbol, dtype=int)
        for s in starts:
            jitter = int(rng.integers(-12, 12))
            start_bar = max(min_start, min(max_start - 1, int(s) + jitter))
            tasks.append((symbol, start_bar, max_days, int(rng.integers(0, 2**31))))

    tasks = tasks[:pool_size]
    del bt
    return tasks, symbols


def build_pool(data_dir, pool_size, max_days, save_path, workers=8, seed=42) -> int:
    print("=" * 72)
    print(" TF Signal Pool Builder (v8.1 Phase 3)")
    print("=" * 72)
    print(f"   Pool size: {pool_size} | Max days: {max_days} | Workers: {workers}")
    print(f"   Save path: {save_path}")
    print("=" * 72 + "\n")

    t0 = time.time()
    print("→ Picking (symbol, h1_start_bar) combinations...")
    tasks, symbols = _pick_pool_tasks(data_dir, pool_size, max_days, seed)
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Tasks:   {len(tasks)} episodes\n")

    print(f"→ Spawning {workers} workers...")
    pool = Pool(processes=workers, initializer=_init_worker, initargs=(data_dir,))
    print("→ Building TF episodes in parallel...\n")
    results = []
    last_log = 0
    try:
        for idx, signals in enumerate(pool.imap_unordered(_build_one_episode, tasks)):
            if signals:
                results.append(signals)
            if (idx + 1) - last_log >= 50 or (idx + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-6)
                eta = (len(tasks) - idx - 1) / max(rate, 1e-6)
                print(f"   [{idx+1:>5}/{len(tasks)}] valid={len(results):>5} "
                      f"elapsed={elapsed/60:>5.1f}min rate={rate:.2f}/s eta={eta/60:>5.1f}min")
                last_log = idx + 1
    finally:
        pool.close()
        pool.join()

    print(f"\n→ Saving pool ({len(results)} valid episodes)...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print("\n" + "=" * 72)
    print(f" Done — {len(results)} episodes in {(time.time()-t0)/60:.1f} min")
    print(f"   File: {save_path} ({size_mb:.1f} MB)")
    print("=" * 72 + "\n")
    return len(results)


def main():
    p = argparse.ArgumentParser(description="Build TF signal pool for RL training")
    p.add_argument("--pool_size", type=int, default=5000)
    p.add_argument("--max_days", type=int, default=45)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--save_path", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")
    if args.save_path is None:
        args.save_path = os.path.join(ROOT, "data", f"tf_signal_pool_{args.pool_size}.pkl")
    if not os.path.isdir(args.data_dir):
        print(f"❌ data_dir not found: {args.data_dir}")
        sys.exit(1)

    build_pool(args.data_dir, args.pool_size, args.max_days,
               args.save_path, args.workers, args.seed)


if __name__ == "__main__":
    main()
