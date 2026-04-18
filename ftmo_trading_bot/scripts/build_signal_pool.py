"""
===============================================================================
Signal Pool Builder — Pre-generate Episode Signals for Training
===============================================================================
สร้าง pool ของ episode signals ล่วงหน้าเพื่อ cache ไว้บน disk
ทำให้ training reset เร็วขึ้น 50-100× (จาก ~6 วิ → <1 ms)

Usage:
    python scripts/build_signal_pool.py --pool_size 3000
    python scripts/build_signal_pool.py --pool_size 5000 --workers 8 --max_days 45

Output: data/signal_pool_<size>.pkl

Architecture:
  • ใช้ multiprocessing.Pool (8 workers) → ลด build time ~8×
  • แต่ละ worker โหลด backtester ครั้งเดียวตอน init
  • Stratified sampling: episodes กระจายข้าม symbols + start bars
===============================================================================
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

# Silence strategy debug prints (ก่อน import bot_config / strategy modules)
os.environ.setdefault("SMC_QUIET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Worker-local backtester cache (avoid re-loading per episode)
_worker_backtester = None


def _init_worker(data_dir: str):
    """Pool worker initializer — load backtester once per worker."""
    global _worker_backtester
    import io
    import contextlib
    from ml.strategy_backtester import StrategyBacktester

    with contextlib.redirect_stdout(io.StringIO()):
        _worker_backtester = StrategyBacktester(data_dir)


def _build_one_episode(task):
    """Build a single episode's signals (runs in worker process)."""
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


def _pick_pool_tasks(data_dir: str, pool_size: int, max_days: int, seed: int = 42):
    """
    Stratified sampling: กระจาย episodes ข้าม symbols + start bars ให้หลากหลาย
    """
    import io
    import contextlib
    from ml.strategy_backtester import StrategyBacktester

    with contextlib.redirect_stdout(io.StringIO()):
        bt = StrategyBacktester(data_dir)

    symbols = bt.get_sequential_symbols(max_days)
    if not symbols:
        raise RuntimeError("No symbols with enough data")

    rng = np.random.default_rng(seed)
    n_per_symbol = max(1, pool_size // len(symbols))
    tasks = []

    for symbol in symbols:
        m15_len = len(bt._m15_cache[symbol])
        needed = bt.get_min_bars_for_episode(max_days)
        min_start = bt.MIN_M15_BARS
        max_start = m15_len - needed

        if max_start <= min_start:
            continue

        # Evenly spaced starts + jitter เพื่อ diversity
        starts = np.linspace(min_start, max_start, n_per_symbol, dtype=int)
        for i, s in enumerate(starts):
            jitter = int(rng.integers(-48, 48))
            start_bar = max(min_start, min(max_start - 1, int(s) + jitter))
            task_seed = int(rng.integers(0, 2**31))
            tasks.append((symbol, start_bar, max_days, task_seed))

    # Trim to exact pool_size
    tasks = tasks[:pool_size]
    del bt
    return tasks, symbols


def build_pool(
    data_dir: str,
    pool_size: int,
    max_days: int,
    save_path: str,
    workers: int = 8,
    seed: int = 42,
) -> int:
    """
    สร้าง signal pool + save ลง disk
    Returns: จำนวน episodes ที่ build ได้ (อาจน้อยกว่า pool_size ถ้ามี signals ว่าง)
    """
    print(f"═══════════════════════════════════════════════════════════════")
    print(f" Signal Pool Builder")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"   Pool size:  {pool_size}")
    print(f"   Max days:   {max_days}")
    print(f"   Workers:    {workers}")
    print(f"   Save path:  {save_path}")
    print(f"═══════════════════════════════════════════════════════════════\n")

    t_start = time.time()

    # Pick tasks
    print("→ Picking (symbol, start_bar) combinations...")
    tasks, symbols = _pick_pool_tasks(data_dir, pool_size, max_days, seed)
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Tasks:   {len(tasks)} episodes to generate\n")

    # Spawn workers
    print(f"→ Spawning {workers} workers (each loads backtester ~5s)...")
    pool = Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(data_dir,),
    )

    # Build
    print(f"→ Building episodes in parallel...\n")
    results = []
    last_log = 0

    try:
        for idx, signals in enumerate(pool.imap_unordered(_build_one_episode, tasks)):
            if signals:
                results.append(signals)

            # Progress log every 50 episodes
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

    # Save
    print(f"\n→ Saving pool ({len(results)} valid episodes)...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    elapsed_total = time.time() - t_start

    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f" Done — {len(results)} episodes in {elapsed_total/60:.1f} min")
    print(f"   File: {save_path} ({size_mb:.1f} MB)")
    print(f"═══════════════════════════════════════════════════════════════\n")

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Build signal pool for RL training")
    parser.add_argument("--pool_size", type=int, default=3000,
                        help="จำนวน episodes (ค่าเริ่มต้น: 3000)")
    parser.add_argument("--max_days", type=int, default=45,
                        help="วันต่อ episode (ค่าเริ่มต้น: 45 = FTMO challenge length)")
    parser.add_argument("--workers", type=int, default=8,
                        help="จำนวน workers (ค่าเริ่มต้น: 8)")
    parser.add_argument("--data_dir", default=None,
                        help="Path ไปยัง OHLCV CSV")
    parser.add_argument("--save_path", default=None,
                        help="Path สำหรับ save pool (default: data/signal_pool_<size>.pkl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Default paths
    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")
    if args.save_path is None:
        args.save_path = os.path.join(ROOT, "data", f"signal_pool_{args.pool_size}.pkl")

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
    )


if __name__ == "__main__":
    main()
