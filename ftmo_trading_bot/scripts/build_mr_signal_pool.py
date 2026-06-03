"""
===============================================================================
MR Signal Pool Builder (v8.0) — Pre-generate Episode Signals for MR Training
===============================================================================
Mirrors `scripts/build_signal_pool.py` but uses `MeanReversionBacktester` so
the offline pool reflects mean-reversion signals (BB extremes + RSI confirm
+ ADX H1 trend block).

Usage:
    python scripts/build_mr_signal_pool.py --pool_size 5000
    python scripts/build_mr_signal_pool.py --pool_size 10000 --workers 8

Output: data/mr_signal_pool_<size>.pkl
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


def _init_worker(data_dir: str, model_live_exit: bool = False):
    """Worker initializer — load MR backtester once per process.

    model_live_exit (2026-06-03): when True, episode outcomes are resolved via the
    LIVE exit machinery (BE@0.3R + 33% partial @0.8R + Stage-2 lock) instead of the
    full-position "ride" — used by walk_forward_eval to measure the sim↔live gap.
    """
    global _worker_backtester
    import io, contextlib
    from ml.mean_reversion_backtester import MeanReversionBacktester

    with contextlib.redirect_stdout(io.StringIO()):
        _worker_backtester = MeanReversionBacktester(data_dir)
        _worker_backtester.model_live_exit = bool(model_live_exit)


def _build_one_episode(task):
    global _worker_backtester
    symbol, start_bar, max_days, seed = task

    rng = np.random.default_rng(seed)
    try:
        signals = _worker_backtester.generate_episode_signals(
            symbol=symbol,
            m15_start_bar=start_bar,
            num_days=max_days,
            rng=rng,
        )
        return signals if signals else None
    except Exception as e:
        print(f"   ⚠️ worker error (symbol={symbol}, start={start_bar}): {e}")
        return None


def _pick_pool_tasks(data_dir: str, pool_size: int, max_days: int, seed: int = 42,
                     start_frac: float = 0.0, end_frac: float = 1.0):
    """Pick (symbol, start_bar) tasks for episode generation.

    start_frac / end_frac (in [0, 1]) restrict the chronological window of M15
    start-bars sampled per symbol — used by walk_forward_eval.py for out-of-TIME
    folds (M15 bars are time-ordered, so a bar-index fraction == a time slice).
    Defaults (0.0, 1.0) reproduce the original full-history behaviour byte-for-byte.
    """
    import io, contextlib
    from ml.mean_reversion_backtester import MeanReversionBacktester

    with contextlib.redirect_stdout(io.StringIO()):
        bt = MeanReversionBacktester(data_dir)

    symbols = bt.get_sequential_symbols(max_days)
    if not symbols:
        raise RuntimeError("No symbols with enough data")

    start_frac = max(0.0, min(1.0, float(start_frac)))
    end_frac = max(start_frac, min(1.0, float(end_frac)))

    rng = np.random.default_rng(seed)
    n_per_symbol = max(1, pool_size // len(symbols))
    tasks = []
    for symbol in symbols:
        m15_len = len(bt._m15_cache[symbol])
        needed = bt.get_min_bars_for_episode(max_days)
        usable_lo = bt.MIN_M15_BARS
        usable_hi = m15_len - needed
        if usable_hi <= usable_lo:
            continue
        # Restrict start-bar range to the chronological [start_frac, end_frac] window
        span = usable_hi - usable_lo
        min_start = usable_lo + int(span * start_frac)
        max_start = usable_lo + int(span * end_frac)
        if max_start <= min_start:
            continue
        starts = np.linspace(min_start, max_start, n_per_symbol, dtype=int)
        for s in starts:
            jitter = int(rng.integers(-48, 48))
            start_bar = max(min_start, min(max_start - 1, int(s) + jitter))
            task_seed = int(rng.integers(0, 2**31))
            tasks.append((symbol, start_bar, max_days, task_seed))

    tasks = tasks[:pool_size]
    del bt
    return tasks, symbols


def build_pool(data_dir: str, pool_size: int, max_days: int, save_path: str,
               workers: int = 8, seed: int = 42,
               start_frac: float = 0.0, end_frac: float = 1.0,
               model_live_exit: bool = False) -> int:
    print("=" * 72)
    print(" MR Signal Pool Builder (v8.0)")
    print("=" * 72)
    print(f"   Pool size:  {pool_size}")
    print(f"   Max days:   {max_days}")
    print(f"   Workers:    {workers}")
    print(f"   Save path:  {save_path}")
    print("=" * 72 + "\n")

    t_start = time.time()

    if (start_frac, end_frac) != (0.0, 1.0):
        print(f"   Time window: frac [{start_frac:.2f}, {end_frac:.2f}] (chronological M15 slice)")
    print("→ Picking (symbol, start_bar) combinations...")
    tasks, symbols = _pick_pool_tasks(data_dir, pool_size, max_days, seed, start_frac, end_frac)
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Tasks:   {len(tasks)} episodes\n")

    if model_live_exit:
        print("   ⚙️ model_live_exit=ON → outcomes resolved via LIVE exit (BE/partial), not 'ride'")
    print(f"→ Spawning {workers} workers (loads MR backtester ~5s each)...")
    pool = Pool(processes=workers, initializer=_init_worker, initargs=(data_dir, model_live_exit))

    print(f"→ Building MR episodes in parallel...\n")
    results = []
    last_log = 0
    try:
        for idx, signals in enumerate(pool.imap_unordered(_build_one_episode, tasks)):
            if signals:
                results.append(signals)
            if (idx + 1) - last_log >= 50 or (idx + 1) == len(tasks):
                elapsed = time.time() - t_start
                rate = (idx + 1) / max(elapsed, 1e-6)
                remaining = (len(tasks) - idx - 1) / max(rate, 1e-6)
                print(
                    f"   [{idx+1:>5}/{len(tasks)}] "
                    f"valid={len(results):>5} "
                    f"elapsed={elapsed/60:>5.1f}min "
                    f"rate={rate:.2f}/s "
                    f"eta={remaining/60:>5.1f}min"
                )
                last_log = idx + 1
    finally:
        pool.close()
        pool.join()

    print(f"\n→ Saving pool ({len(results)} valid episodes)...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f, protocol=4)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    elapsed_total = time.time() - t_start

    print("\n" + "=" * 72)
    print(f" Done — {len(results)} episodes in {elapsed_total/60:.1f} min")
    print(f"   File: {save_path} ({size_mb:.1f} MB)")
    print("=" * 72 + "\n")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Build MR signal pool for RL training")
    parser.add_argument("--pool_size", type=int, default=5000)
    parser.add_argument("--max_days", type=int, default=45)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_frac", type=float, default=0.0,
                        help="Chronological window start (0-1) of M15 bars to sample from (walk-forward folds)")
    parser.add_argument("--end_frac", type=float, default=1.0,
                        help="Chronological window end (0-1) of M15 bars to sample from")
    parser.add_argument("--model_live_exit", action="store_true",
                        help="Resolve outcomes via LIVE exit (BE+partial) instead of full 'ride' — sim↔live gap test")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")
    if args.save_path is None:
        args.save_path = os.path.join(ROOT, "data", f"mr_signal_pool_{args.pool_size}.pkl")

    if not os.path.isdir(args.data_dir):
        print(f"❌ data_dir not found: {args.data_dir}")
        sys.exit(1)

    build_pool(
        data_dir=args.data_dir,
        pool_size=args.pool_size,
        max_days=args.max_days,
        save_path=args.save_path,
        workers=args.workers,
        seed=args.seed,
        start_frac=args.start_frac,
        end_frac=args.end_frac,
        model_live_exit=args.model_live_exit,
    )


if __name__ == "__main__":
    main()
