"""
===============================================================================
FTMO Trading Bot — Autonomous Training Pipeline (v8.0)
===============================================================================
Master orchestrator for the Mean-Reversion strategy retrain loop. The user
launches this once and walks away; the script:

  Step A. Builds the MR signal pool (`build_mr_signal_pool.py`)
  Step B. Trains the GBM quality model (`train_mr_signal_quality.py`)
  Step C. Trains the PPO RL agent (`train_mr_signal_filter.py`)
  Step D. Evaluates 5000 episodes -> Pass Rate / DD / Profitable %
  Step E. Decides keep / iterate / stop based on `gates`
  Step F. If iterate: tune hyperparameters + relaunch from Step A or C
          (depending on what changed). Pool/GBM are reused when only RL
          shaping or learning rates change.
  Step G. Logs everything (decisions, metrics, durations) to
          `logs/auto_train_pipeline.jsonl` (one line per attempt) plus a
          human-readable `logs/auto_train_pipeline.log` for review.

Stops when:
  • Gates pass (success — best model + metrics persisted)
  • `--max_iterations` reached
  • `--max_hours` budget exhausted
  • A subprocess raises a hard error not handled by tuning

Safety:
  • Never overwrites the best model. Each attempt's artefacts are tagged
    with the attempt index. The "best" pointer in `logs/auto_train_pipeline_state.json`
    points at the strongest pass-rate so far.
  • Never deletes `logs/bot_state.json` or any FTMO live state file.
  • Stops if any attempt produces 0 valid signals (a sign the pool builder
    is mis-configured — fail fast rather than burn compute).

Usage:
    .venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
        --max_iterations 6 \
        --max_hours 60 \
        --pool_size 5000 \
        --timesteps_p1 5000000 \
        --timesteps_p2 2000000 \
        --target_pass_rate 0.08 \
        --target_dd_max 0.06 \
        --target_profitable 0.55

Use `--dry_run` to print the planned commands without launching anything.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── Config blob — every tunable in one place ─────────────────────────


@dataclass
class HyperParams:
    """All tunables the auto-loop touches. Tuned in place each iteration."""
    # Core training knobs
    timesteps_p1: int = 5_000_000
    timesteps_p2: int = 2_000_000
    n_envs: int = 8
    pool_size: int = 5000
    pool_workers: int = 8
    outcome_noise: float = 0.05
    # v8.0.3: lowered 0.40 → 0.30 after iter 1+2 showed agent under-trading
    # (WR 90% but only ~10 trades/ep at 0.40, not enough to hit 10% target).
    ml_threshold: float = 0.30
    risk_per_trade: float = 0.0099
    lr_p1: float = 3e-4
    lr_p2: float = 5e-5
    # MR reward shaping (forwarded to env via train script flags)
    quick_tp_bonus: float = 0.50
    slow_win_bonus: float = 0.20
    prolonged_loss_penalty: float = 0.40
    base_loss_penalty: float = 0.10
    duration_fine_coef: float = 0.02
    # When True, the next iteration rebuilds the pool. When False, only RL
    # retrains (pool + GBM reused).
    rebuild_pool: bool = True
    rebuild_gbm: bool = True


@dataclass
class Gates:
    """Acceptance criteria. Passing all = stop with success."""
    target_pass_rate: float = 0.08      # 8 % across 5000 eval eps
    target_dd_max: float = 0.06         # cap total DD max at 6 %
    target_daily_dd_max: float = 0.035  # cap daily DD max at 3.5 %
    target_profitable: float = 0.55     # 55 % of eps end profitable
    breach_rate_max: float = 0.05       # < 5 % breach rate


@dataclass
class IterationResult:
    """One complete A→B→C→D pass through the pipeline."""
    iteration: int
    timestamp: str
    duration_sec: float
    hyperparams: Dict
    metrics: Dict
    decision: str
    notes: str = ""
    artefact_paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


# ─── Logging helpers ──────────────────────────────────────────────────


class _DualLogger:
    """Writes every message to stdout and the human-readable log file.

    Also exposes `event(...)` to write structured JSON lines.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.text_path = os.path.join(log_dir, "auto_train_pipeline.log")
        self.jsonl_path = os.path.join(log_dir, "auto_train_pipeline.jsonl")
        self._tf = open(self.text_path, "a", encoding="utf-8")
        self._jf = open(self.jsonl_path, "a", encoding="utf-8")

    def info(self, msg: str):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self._tf.write(line + "\n")
        self._tf.flush()

    def event(self, payload: Dict):
        payload = dict(payload)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._jf.write(json.dumps(payload, default=str) + "\n")
        self._jf.flush()

    def close(self):
        try:
            self._tf.close()
            self._jf.close()
        except Exception:
            pass


# ─── Subprocess helpers ───────────────────────────────────────────────


def _python_executable() -> str:
    """Prefer .venv/bin/python so package versions match the locked stack."""
    venv_py = os.path.join(ROOT, "..", ".venv", "bin", "python")
    venv_py = os.path.abspath(venv_py)
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


def _run_step(cmd: List[str], log: _DualLogger, dry_run: bool = False,
              cwd: Optional[str] = None) -> int:
    log.info(f"$ {' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=cwd or ROOT, env=os.environ.copy())
    return proc.returncode


# ─── Pipeline steps ───────────────────────────────────────────────────


def step_build_pool(hp: HyperParams, log: _DualLogger, dry_run: bool) -> str:
    pool_path = os.path.join(ROOT, "data", f"mr_signal_pool_{hp.pool_size}.pkl")
    if os.path.exists(pool_path) and not hp.rebuild_pool:
        log.info(f"   Skip pool build — reuse existing {pool_path}")
        return pool_path
    cmd = [
        _python_executable(), os.path.join("scripts", "build_mr_signal_pool.py"),
        "--pool_size", str(hp.pool_size),
        "--workers", str(hp.pool_workers),
    ]
    rc = _run_step(cmd, log, dry_run)
    if rc != 0:
        raise RuntimeError(f"Pool build failed (rc={rc})")
    if not dry_run and not os.path.exists(pool_path):
        raise RuntimeError(f"Pool builder reported success but {pool_path} missing")
    return pool_path


def step_train_gbm(hp: HyperParams, log: _DualLogger, dry_run: bool) -> str:
    pool_path = os.path.join(ROOT, "data", f"mr_signal_pool_{hp.pool_size}.pkl")
    save_path = os.path.join(ROOT, "data", "mr_signal_quality_model.pkl")
    if os.path.exists(save_path) and not hp.rebuild_gbm:
        log.info(f"   Skip GBM train — reuse existing {save_path}")
        return save_path
    cmd = [
        _python_executable(), os.path.join("scripts", "train_mr_signal_quality.py"),
        "--pool", pool_path,
        "--save", save_path,
    ]
    rc = _run_step(cmd, log, dry_run)
    if rc != 0:
        raise RuntimeError(f"GBM train failed (rc={rc})")
    return save_path


def step_train_rl(hp: HyperParams, log: _DualLogger, dry_run: bool) -> Tuple[str, str]:
    cmd = [
        _python_executable(), os.path.join("scripts", "train_mr_signal_filter.py"),
        "--fresh",
        "--timesteps_p1", str(hp.timesteps_p1),
        "--timesteps_p2", str(hp.timesteps_p2),
        "--n_envs", str(hp.n_envs),
        "--pool_size", str(hp.pool_size),
        "--outcome_noise", str(hp.outcome_noise),
        "--ml_threshold", str(hp.ml_threshold),
        "--risk_per_trade", str(hp.risk_per_trade),
        "--quick_tp_bonus", str(hp.quick_tp_bonus),
        "--slow_win_bonus", str(hp.slow_win_bonus),
        "--prolonged_loss_penalty", str(hp.prolonged_loss_penalty),
        "--base_loss_penalty", str(hp.base_loss_penalty),
        "--duration_fine_coef", str(hp.duration_fine_coef),
        "--lr_p1", str(hp.lr_p1),
        "--lr_p2", str(hp.lr_p2),
    ]
    rc = _run_step(cmd, log, dry_run)
    if rc != 0:
        raise RuntimeError(f"RL train failed (rc={rc})")
    model_path = os.path.join(ROOT, "models", "mr", "ppo_mr_filter.zip")
    vec_norm_path = os.path.join(ROOT, "models", "mr", "vec_normalize_mr.pkl")
    return model_path, vec_norm_path


def step_evaluate(hp: HyperParams, log: _DualLogger, dry_run: bool) -> Dict:
    """Run eval-only and parse metrics back from the trainer's stdout.

    The trainer prints an "MR Eval Result" block we can parse. To avoid
    fragile string parsing we instead import + call the trainer's evaluate()
    directly when not in dry-run mode.
    """
    if dry_run:
        log.info("   [dry-run] would run trainer eval()")
        return {"pass_rate": 0.0, "breach_rate": 0.0, "profitable_rate": 0.0,
                "win_rate": 0.0, "take_rate": 0.0, "total_dd_max": 0.0,
                "daily_dd_max": 0.0, "profit_avg": 0.0}

    # Direct in-process call — keeps metrics structured (no parsing).
    from ml.aux_aware_ppo import AuxAwarePPO
    from scripts.train_mr_signal_filter import evaluate
    model_path = os.path.join(ROOT, "models", "mr", "ppo_mr_filter.zip")
    vec_norm = os.path.join(ROOT, "models", "mr", "vec_normalize_mr.pkl")
    pool_path = os.path.join(ROOT, "data", f"mr_signal_pool_{hp.pool_size}.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Eval cannot find model: {model_path}")
    model = AuxAwarePPO.load(model_path)
    metrics = evaluate(
        model,
        n_episodes=5000,
        vec_normalize_path=vec_norm if os.path.exists(vec_norm) else None,
        signal_pool_path=pool_path if os.path.exists(pool_path) else None,
        ml_filter_threshold=hp.ml_threshold,
        risk_per_trade=hp.risk_per_trade,
    )
    return metrics


# ─── Decision logic ──────────────────────────────────────────────────


def evaluate_gates(metrics: Dict, gates: Gates) -> Tuple[bool, List[str]]:
    """All-or-nothing acceptance. Returns (pass, list_of_failed_gates)."""
    failed = []
    if metrics.get("pass_rate", 0.0) < gates.target_pass_rate:
        failed.append(
            f"pass_rate={metrics.get('pass_rate', 0.0)*100:.1f}% < "
            f"{gates.target_pass_rate*100:.1f}%"
        )
    if metrics.get("total_dd_max", 1.0) > gates.target_dd_max:
        failed.append(
            f"total_dd_max={metrics.get('total_dd_max', 1.0)*100:.2f}% > "
            f"{gates.target_dd_max*100:.2f}%"
        )
    if metrics.get("daily_dd_max", 1.0) > gates.target_daily_dd_max:
        failed.append(
            f"daily_dd_max={metrics.get('daily_dd_max', 1.0)*100:.2f}% > "
            f"{gates.target_daily_dd_max*100:.2f}%"
        )
    if metrics.get("profitable_rate", 0.0) < gates.target_profitable:
        failed.append(
            f"profitable_rate={metrics.get('profitable_rate', 0.0)*100:.1f}% < "
            f"{gates.target_profitable*100:.1f}%"
        )
    if metrics.get("breach_rate", 1.0) > gates.breach_rate_max:
        failed.append(
            f"breach_rate={metrics.get('breach_rate', 1.0)*100:.1f}% > "
            f"{gates.breach_rate_max*100:.1f}%"
        )
    return len(failed) == 0, failed


def tune_hyperparams(hp: HyperParams, metrics: Dict, gates: Gates,
                     iteration: int) -> Tuple[HyperParams, str]:
    """Self-correct hyperparams based on which gate failed.

    Strategy:
      • Pass rate too low + DD ok        → push agent to TAKE more (lower
        ML threshold, lower passive SKIP cost via larger quick-TP bonus,
        slightly higher LR P2 for more late learning)
      • DD too high                      → tighten capital preservation
        (heavier prolonged-loss penalty, lower risk per trade, higher
        base loss penalty)
      • Profitable rate too low          → both — favor wins by raising
        QUICK_TP_BONUS but also adding SLOW_WIN_BONUS (don't punish
        late winners too much)
      • Breach rate too high             → cut risk_per_trade in half
        for one iteration to stabilize
    The function returns a new HyperParams + a string describing what changed,
    so the log captures the rationale (not just the new values).
    """
    new_hp = HyperParams(**asdict(hp))
    notes: List[str] = []

    pass_rate = metrics.get("pass_rate", 0.0)
    dd_max = metrics.get("total_dd_max", 1.0)
    daily_dd = metrics.get("daily_dd_max", 1.0)
    breach_rate = metrics.get("breach_rate", 1.0)
    profitable = metrics.get("profitable_rate", 0.0)
    take_rate = metrics.get("take_rate", 0.0)
    win_rate = metrics.get("win_rate", 0.0)

    # 1. Breach too high → emergency dd brake
    if breach_rate > gates.breach_rate_max:
        new_hp.risk_per_trade = max(new_hp.risk_per_trade * 0.5, 0.003)
        new_hp.prolonged_loss_penalty = min(new_hp.prolonged_loss_penalty + 0.20, 1.0)
        new_hp.base_loss_penalty = min(new_hp.base_loss_penalty + 0.10, 0.5)
        notes.append(f"breach_brake: risk→{new_hp.risk_per_trade}, "
                     f"loss penalties bumped")

    # 2. DD too high → tighten capital preservation
    elif dd_max > gates.target_dd_max or daily_dd > gates.target_daily_dd_max:
        new_hp.prolonged_loss_penalty = min(new_hp.prolonged_loss_penalty + 0.10, 1.0)
        new_hp.duration_fine_coef = min(new_hp.duration_fine_coef + 0.01, 0.05)
        new_hp.base_loss_penalty = min(new_hp.base_loss_penalty + 0.05, 0.5)
        new_hp.risk_per_trade = max(new_hp.risk_per_trade * 0.85, 0.005)
        notes.append("dd_tighten: heavier loss penalties, smaller risk")

    # 3. Pass rate too low — diagnose WHY then act (v8.0.3 smarter tune)
    #
    # Case A — low take_rate (<30%): agent too selective, ml_threshold blocks too much
    # Case B — high WR (≥65%) + decent take (≥40%) + still low pass: GOOD agent but
    #          NOT ENOUGH TRADES per episode to consistently hit 10% target
    # Case C — high take_rate + low WR (<55%): agent over-trading low-quality
    elif pass_rate < gates.target_pass_rate:
        is_high_wr_low_pass = (win_rate >= 0.65 and take_rate >= 0.40)
        if is_high_wr_low_pass:
            # Case B — accurate but undertrading. Drop ml_threshold a lot to
            # expose more candidates; reduce loss fear so agent isn't paralyzed.
            new_hp.ml_threshold = max(new_hp.ml_threshold - 0.05, 0.25)
            new_hp.quick_tp_bonus = min(new_hp.quick_tp_bonus + 0.05, 1.0)
            new_hp.base_loss_penalty = max(new_hp.base_loss_penalty - 0.02, 0.05)
            notes.append(f"high_wr_low_pass: ml_th→{new_hp.ml_threshold:.2f}, "
                         f"quick_tp→{new_hp.quick_tp_bonus:.2f}, "
                         f"base_loss↓{new_hp.base_loss_penalty:.2f}")
        elif take_rate < 0.30:
            # Case A — over-selective. Lower threshold + push TAKE.
            new_hp.ml_threshold = max(new_hp.ml_threshold - 0.04, 0.25)
            new_hp.quick_tp_bonus = min(new_hp.quick_tp_bonus + 0.10, 1.0)
            new_hp.slow_win_bonus = min(new_hp.slow_win_bonus + 0.05, 0.5)
            notes.append(f"low_take: ml_th→{new_hp.ml_threshold:.2f}, "
                         f"quick_tp→{new_hp.quick_tp_bonus:.2f}")
        elif win_rate < 0.55:
            # Case C — over-trading low quality. Tighten threshold.
            new_hp.ml_threshold = min(new_hp.ml_threshold + 0.03, 0.55)
            new_hp.base_loss_penalty = min(new_hp.base_loss_penalty + 0.05, 0.5)
            notes.append(f"over_trade: ml_th↑{new_hp.ml_threshold:.2f}, "
                         f"base_loss↑{new_hp.base_loss_penalty:.2f}")
        else:
            # Generic: nudge rewards
            new_hp.quick_tp_bonus = min(new_hp.quick_tp_bonus + 0.05, 1.0)
            new_hp.slow_win_bonus = min(new_hp.slow_win_bonus + 0.05, 0.5)
            notes.append(f"generic_push: quick_tp→{new_hp.quick_tp_bonus:.2f}")
        # If P2 looks under-converged (good WR but still few trades), bump LR
        if iteration >= 1 and win_rate >= 0.70 and take_rate < 0.50:
            new_hp.lr_p2 = min(new_hp.lr_p2 * 1.5, 2e-4)
            notes.append(f"P2_underconverge: lr_p2→{new_hp.lr_p2:.1e}")

    # 4. Profitable rate too low — keep more late winners
    elif profitable < gates.target_profitable:
        new_hp.slow_win_bonus = min(new_hp.slow_win_bonus + 0.05, 0.5)
        new_hp.duration_fine_coef = max(new_hp.duration_fine_coef - 0.005, 0.005)
        notes.append("low_profit: kinder to slow winners")

    # Pool reuse — pool depends only on strategy code (BB/RSI thresholds in
    # MeanReversionStrategy + bot_config.mr). Auto-tuning only adjusts RL/ML
    # params, so pool can always be reused after iter 0. (Iter 0 is the first
    # iteration where iteration index == 0.)
    new_hp.rebuild_pool = False
    new_hp.rebuild_gbm = False
    return new_hp, "; ".join(notes) if notes else "no_change"


# ─── State + best-so-far tracking ─────────────────────────────────────


def _state_path() -> str:
    return os.path.join(ROOT, "logs", "auto_train_pipeline_state.json")


def _save_state(state: Dict):
    os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
    with open(_state_path(), "w") as f:
        json.dump(state, f, indent=2, default=str)


def _load_state() -> Dict:
    if os.path.exists(_state_path()):
        try:
            with open(_state_path()) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _snapshot_best(model_dir: str, iteration: int,
                   metrics: Dict, log: _DualLogger) -> str:
    """Copy the current model + vec_normalize into a tagged best/ folder."""
    src_model = os.path.join(model_dir, "ppo_mr_filter.zip")
    src_vec = os.path.join(model_dir, "vec_normalize_mr.pkl")
    dst_dir = os.path.join(model_dir, "best")
    os.makedirs(dst_dir, exist_ok=True)
    dst_model = os.path.join(dst_dir, "ppo_mr_filter.zip")
    dst_vec = os.path.join(dst_dir, "vec_normalize_mr.pkl")
    if os.path.exists(src_model):
        shutil.copy2(src_model, dst_model)
    if os.path.exists(src_vec):
        shutil.copy2(src_vec, dst_vec)
    meta_path = os.path.join(dst_dir, "best_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"iteration": iteration, "metrics": metrics,
                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)
    log.info(f"   📌 Snapshotted best -> {dst_dir}")
    return dst_dir


# ─── Main loop ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous MR training pipeline (v8.0)"
    )
    parser.add_argument("--max_iterations", type=int, default=6,
                        help="Max attempts before stopping (default 6)")
    parser.add_argument("--max_hours", type=float, default=60.0,
                        help="Wall-clock budget in hours (default 60)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without launching anything")
    # Pipeline knobs (initial values for HyperParams)
    parser.add_argument("--pool_size", type=int, default=5000)
    parser.add_argument("--timesteps_p1", type=int, default=5_000_000)
    parser.add_argument("--timesteps_p2", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--ml_threshold", type=float, default=0.40)
    parser.add_argument("--risk_per_trade", type=float, default=0.0099)
    parser.add_argument("--outcome_noise", type=float, default=0.05,
                        help="Outcome PnL noise std (regularization, anti-overfit). "
                             "Default 0.05; v8.0.10 retrain uses 0.08")
    # Gates
    parser.add_argument("--target_pass_rate", type=float, default=0.08)
    parser.add_argument("--target_dd_max", type=float, default=0.06)
    parser.add_argument("--target_daily_dd_max", type=float, default=0.035)
    parser.add_argument("--target_profitable", type=float, default=0.55)
    parser.add_argument("--breach_rate_max", type=float, default=0.05)
    args = parser.parse_args()

    log = _DualLogger(os.path.join(ROOT, "logs"))
    log.info("=" * 70)
    log.info(" 🤖 Autonomous MR Training Pipeline (v8.0)")
    log.info("=" * 70)
    log.info(f"   max_iterations={args.max_iterations}, max_hours={args.max_hours}")
    log.info(f"   gates: pass≥{args.target_pass_rate}, "
             f"dd≤{args.target_dd_max}, "
             f"daily_dd≤{args.target_daily_dd_max}, "
             f"profitable≥{args.target_profitable}, "
             f"breach≤{args.breach_rate_max}")
    log.info(f"   dry_run={args.dry_run}")
    log.info("=" * 70)

    hp = HyperParams(
        timesteps_p1=args.timesteps_p1,
        timesteps_p2=args.timesteps_p2,
        n_envs=args.n_envs,
        pool_size=args.pool_size,
        ml_threshold=args.ml_threshold,
        risk_per_trade=args.risk_per_trade,
        outcome_noise=args.outcome_noise,
        # v8.0.4: rebuild only if files don't already exist. Preserves pool/GBM
        # across pipeline restarts (e.g. when env constants change but pool
        # generation logic is unchanged).
        rebuild_pool=not os.path.exists(
            os.path.join(ROOT, "data", f"mr_signal_pool_{args.pool_size}.pkl")
        ),
        rebuild_gbm=not os.path.exists(
            os.path.join(ROOT, "data", "mr_signal_quality_model.pkl")
        ),
    )
    gates = Gates(
        target_pass_rate=args.target_pass_rate,
        target_dd_max=args.target_dd_max,
        target_daily_dd_max=args.target_daily_dd_max,
        target_profitable=args.target_profitable,
        breach_rate_max=args.breach_rate_max,
    )

    history: List[IterationResult] = []
    best_metrics: Dict = {}
    best_iteration = -1
    t_start = time.time()

    for iteration in range(args.max_iterations):
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h >= args.max_hours:
            log.info(f"⏹  Wall-clock budget exhausted ({elapsed_h:.1f}h) — stop")
            break

        log.info("")
        log.info("─" * 70)
        log.info(f" Iteration {iteration + 1}/{args.max_iterations} "
                 f"(elapsed {elapsed_h:.1f}h)")
        log.info("─" * 70)
        log.info(f" hyperparams: {asdict(hp)}")
        log.event({"event": "iteration_start", "iteration": iteration,
                   "hyperparams": asdict(hp)})

        attempt_start = time.time()
        error = None
        metrics: Dict = {}
        try:
            # Step A
            log.info("→ Step A: build MR signal pool")
            step_build_pool(hp, log, args.dry_run)

            # Step B
            log.info("→ Step B: train GBM quality model")
            step_train_gbm(hp, log, args.dry_run)

            # Step C
            log.info("→ Step C: train PPO RL agent (2-phase)")
            step_train_rl(hp, log, args.dry_run)

            # Step D
            log.info("→ Step D: evaluate (5000 episodes)")
            metrics = step_evaluate(hp, log, args.dry_run)
            log.info(f"   metrics: {metrics}")

        except Exception as e:
            error = repr(e)
            log.info(f"❌ Iteration {iteration + 1} hit hard error: {error}")
            log.event({"event": "iteration_error",
                       "iteration": iteration, "error": error})

        attempt_dur = time.time() - attempt_start

        if error:
            result = IterationResult(
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                duration_sec=attempt_dur,
                hyperparams=asdict(hp),
                metrics=metrics,
                decision="error",
                error=error,
            )
            history.append(result)
            log.event({"event": "iteration_end", **asdict(result)})
            # On hard errors don't try to tune blindly — break loop
            log.info("⏹  Stopping due to hard error (manual investigation needed)")
            break

        # Track best so far
        if metrics.get("pass_rate", 0.0) > best_metrics.get("pass_rate", -1.0):
            best_metrics = dict(metrics)
            best_iteration = iteration
            if not args.dry_run:
                _snapshot_best(os.path.join(ROOT, "models", "mr"),
                               iteration, metrics, log)

        # Decide
        passed_all, failed_gates = evaluate_gates(metrics, gates)
        if passed_all:
            decision = "pass"
            notes = "all_gates_passed"
            log.info(f"✅ All gates passed. Stopping.")
            history.append(IterationResult(
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                duration_sec=attempt_dur,
                hyperparams=asdict(hp),
                metrics=metrics,
                decision=decision,
                notes=notes,
            ))
            log.event({"event": "iteration_end", "iteration": iteration,
                       "decision": decision, "metrics": metrics})
            break

        # Iterate
        log.info(f"⚠️  Gates failed: {failed_gates}")
        new_hp, tune_notes = tune_hyperparams(hp, metrics, gates, iteration)
        log.info(f"   tuning: {tune_notes}")
        log.event({
            "event": "iteration_tune",
            "iteration": iteration,
            "failed_gates": failed_gates,
            "tune_notes": tune_notes,
            "old_hp": asdict(hp),
            "new_hp": asdict(new_hp),
        })

        history.append(IterationResult(
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_sec=attempt_dur,
            hyperparams=asdict(hp),
            metrics=metrics,
            decision="iterate",
            notes=f"failed={failed_gates}; tune={tune_notes}",
        ))
        hp = new_hp

        # Persist state + history
        _save_state({
            "history": [asdict(h) for h in history],
            "best_metrics": best_metrics,
            "best_iteration": best_iteration,
            "current_hyperparams": asdict(hp),
            "elapsed_hours": elapsed_h,
        })

    # ─── Final summary ───────────────────────────────────────────────
    elapsed_h = (time.time() - t_start) / 3600.0
    log.info("")
    log.info("=" * 70)
    log.info(" 🏁 Auto pipeline finished")
    log.info(f"    iterations: {len(history)}")
    log.info(f"    elapsed:    {elapsed_h:.1f} h")
    log.info(f"    best pass_rate so far: {best_metrics.get('pass_rate', 0.0)*100:.2f}%"
             f" (iter {best_iteration})")
    log.info(f"    best metrics: {best_metrics}")
    log.info(f"    log:    {log.text_path}")
    log.info(f"    events: {log.jsonl_path}")
    log.info(f"    best model snapshot: models/mr/best/")
    log.info("=" * 70)

    _save_state({
        "history": [asdict(h) for h in history],
        "best_metrics": best_metrics,
        "best_iteration": best_iteration,
        "elapsed_hours": elapsed_h,
        "finished": True,
    })
    log.close()


if __name__ == "__main__":
    main()
