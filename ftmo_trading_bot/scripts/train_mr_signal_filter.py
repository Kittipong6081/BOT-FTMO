"""
===============================================================================
MR Signal Filter RL Trainer (v8.0) — 2-Phase PPO with Aux Task
===============================================================================
Mirrors `scripts/train_signal_filter.py` but instantiates
`MeanReversionFilterEnv` instead of `FTMOSignalFilterEnv`. Default pool path
points at `data/mr_signal_pool_<N>.pkl` and default model path lives in
`models/mr/` to keep the SMC artefacts pristine.

Same 2-phase curriculum:
  • Phase 1 (Alpha) — `enable_risk_penalty=False`, learn entry quality
  • Phase 2 (Risk)  — `enable_risk_penalty=True`,  enforce DD discipline

Usage:
    python scripts/train_mr_signal_filter.py --fresh \
        --timesteps_p1 5000000 --timesteps_p2 2000000 \
        --n_envs 8 --pool_size 5000 --ml_threshold 0.40 --risk_per_trade 0.0099

Output:
    models/mr/ppo_mr_filter.zip
    models/mr/vec_normalize_mr.pkl
===============================================================================
"""

import argparse
import os
import sys
import time

os.environ.setdefault("SMC_QUIET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.aux_aware_ppo import AuxAwarePPO
from ml.aux_aware_policy import AuxAwareACPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ml.mean_reversion_env import MeanReversionFilterEnv


# ─── Callbacks (identical to SMC trainer — reused intentionally) ──────

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial=0.05, final=0.005, verbose=0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = self.final + (self.initial - self.final) * (1.0 - progress)
        return True


class EarlyStopOnValueLoss(BaseCallback):
    def __init__(self, threshold=10.0, patience=5, warmup_steps=0, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.warmup_steps = warmup_steps
        self._consecutive = 0

    def _on_step(self):
        if self.num_timesteps < self.warmup_steps:
            self._consecutive = 0
            return True
        infos = self.logger.name_to_value
        vl = infos.get("train/value_loss", 0.0)
        if vl > self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.patience:
                print(f"\n[Early Stop] value_loss={vl:.2f} > {self.threshold} "
                      f"x{self.patience}")
                return False
        else:
            self._consecutive = 0
        return True


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, print_every=10, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self._ep_count = 0
        self._balances, self._profits, self._dds = [], [], []
        self._take_rates, self._win_rates = [], []
        self._passes = 0
        self._breaches = 0
        self._survives = 0
        self._last_print = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            summary = info.get("episode_summary")
            if summary is None:
                continue
            self._ep_count += 1
            self._balances.append(summary["balance"])
            self._profits.append(summary["profit"])
            self._dds.append(summary["total_dd_pct"])
            self._take_rates.append(summary["take_rate"])
            self._win_rates.append(summary["win_rate"])
            if summary["breached"]:
                self._breaches += 1
            elif summary["passed"]:
                self._passes += 1
            else:
                self._survives += 1
            if self._ep_count - self._last_print >= self.print_every:
                self._print_summary()
                self._last_print = self._ep_count
        return True

    def _print_summary(self):
        n = len(self._balances)
        recent = min(self.print_every, n)
        r_bal = self._balances[-recent:]
        r_pnl = self._profits[-recent:]
        r_dd = self._dds[-recent:]
        r_tr = self._take_rates[-recent:]
        r_wr = self._win_rates[-recent:]
        total_ep = self._passes + self._breaches + self._survives
        step = self.num_timesteps
        print(f"\n{'='*65}")
        print(f" [MR] Episode {self._ep_count} | Step {step:,}")
        print(f"{'='*65}")
        print(f"   Balance   : ${np.mean(r_bal):>10,.2f}")
        print(f"   Profit    : ${np.mean(r_pnl):>+10,.2f}")
        print(f"   DD        : {np.mean(r_dd):>8.2%}    (max {np.max(r_dd):>8.2%})")
        print(f"   Take Rate : {np.mean(r_tr):>8.1%}    Win Rate : {np.mean(r_wr):>8.1%}")
        print(f"   Pass: {self._passes}  Breach: {self._breaches}  Survive: {self._survives}")
        if total_ep > 0:
            print(f"   Pass Rate : {self._passes/total_ep:.1%}")
        print(f"{'='*65}")


# ─── Helpers ───────────────────────────────────────────────────────────

def make_env(data_dir=None, enable_risk_penalty=True, signal_pool_path=None,
             outcome_noise_std=0.05, ml_filter_threshold=0.0,
             risk_per_trade=None,
             quick_tp_bonus=None, slow_win_bonus=None,
             prolonged_loss_penalty=None, base_loss_penalty=None,
             duration_fine_coef=None):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return MeanReversionFilterEnv(
            data_dir=data_dir,
            verbose=False,
            enable_risk_penalty=enable_risk_penalty,
            signal_pool_path=signal_pool_path,
            outcome_noise_std=outcome_noise_std,
            ml_filter_threshold=ml_filter_threshold,
            risk_per_trade=risk_per_trade,
            quick_tp_bonus=quick_tp_bonus,
            slow_win_bonus=slow_win_bonus,
            prolonged_loss_penalty=prolonged_loss_penalty,
            base_loss_penalty=base_loss_penalty,
            duration_fine_coef=duration_fine_coef,
        )


def ensure_signal_pool(data_dir, pool_size, max_days, workers):
    pool_path = os.path.join(ROOT, "data", f"mr_signal_pool_{pool_size}.pkl")
    if os.path.exists(pool_path):
        size_mb = os.path.getsize(pool_path) / (1024 * 1024)
        print(f"   ✓ MR pool ready: {pool_path} ({size_mb:.1f} MB)")
        return pool_path

    print("\n" + "=" * 65)
    print(f" 📦 MR pool missing — building first time")
    print(f"    Will take ~{pool_size//200} min depending on machine")
    print("=" * 65 + "\n")
    from scripts.build_mr_signal_pool import build_pool
    build_pool(
        data_dir=data_dir,
        pool_size=pool_size,
        max_days=max_days,
        save_path=pool_path,
        workers=workers,
    )
    return pool_path


def evaluate(model, n_episodes=5000, vec_normalize_path=None,
             signal_pool_path=None, ml_filter_threshold=0.0,
             risk_per_trade=None):
    raw_env = DummyVecEnv([lambda: make_env(
        signal_pool_path=signal_pool_path,
        ml_filter_threshold=ml_filter_threshold,
        risk_per_trade=risk_per_trade,
    )])
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_env

    passes = breaches = survive = 0
    profits, win_rates, take_rates = [], [], []
    dds, daily_dds_max = [], []
    profitable_count = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            done = dones[0]
            last_info = infos[0] if infos else {}
        summary = last_info.get("episode_summary", {})
        take_rates.append(summary.get("take_rate", 0.0))
        profits.append(summary.get("profit", 0.0))
        win_rates.append(summary.get("win_rate", 0.0))
        dds.append(summary.get("total_dd_pct", 0.0))
        daily_dds_max.append(summary.get("max_daily_dd_pct", 0.0))
        if summary.get("profit", 0.0) > 0:
            profitable_count += 1
        if summary.get("breached", False):
            breaches += 1
        elif summary.get("passed", False):
            passes += 1
        else:
            survive += 1

    pass_rate = passes / n_episodes
    profitable_pct = profitable_count / n_episodes
    print("\n" + "=" * 65)
    print(f" MR Eval Result ({n_episodes} eps)")
    print("=" * 65)
    print(f"   Pass Rate         : {pass_rate*100:>6.1f}% ({passes}/{n_episodes})")
    print(f"   Breach Rate       : {breaches/n_episodes*100:>6.1f}%")
    print(f"   Profitable Rate   : {profitable_pct*100:>6.1f}%")
    print(f"   Win Rate (avg)    : {np.mean(win_rates)*100:>6.1f}%")
    print(f"   Take Rate (avg)   : {np.mean(take_rates)*100:>6.1f}%")
    print(f"   Profit (avg)      : ${np.mean(profits):>+10,.2f}")
    print(f"   Total DD max      : {np.max(dds)*100:>6.2f}%   (limit 8%)")
    print(f"   Daily DD max      : {np.max(daily_dds_max)*100:>6.2f}%   (limit 4%)")
    print("=" * 65)
    return {
        "pass_rate": pass_rate,
        "breach_rate": breaches / n_episodes,
        "profitable_rate": profitable_pct,
        "win_rate": float(np.mean(win_rates)),
        "take_rate": float(np.mean(take_rates)),
        "total_dd_max": float(np.max(dds)),
        "daily_dd_max": float(np.max(daily_dds_max)),
        "profit_avg": float(np.mean(profits)),
    }


# ─── Main 2-Phase Trainer ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MR Signal Filter (v8.0)")
    parser.add_argument("--timesteps_p1", type=int, default=5_000_000)
    parser.add_argument("--timesteps_p2", type=int, default=2_000_000)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--pool_size", type=int, default=5000)
    parser.add_argument("--pool_workers", type=int, default=8)
    parser.add_argument("--outcome_noise", type=float, default=0.05)
    parser.add_argument("--ml_threshold", type=float, default=0.30)
    parser.add_argument("--risk_per_trade", type=float, default=0.0085)  # v8.0.43c sync
    # MR-tunable shaping (auto_train_pipeline can override)
    parser.add_argument("--quick_tp_bonus", type=float, default=None)
    parser.add_argument("--slow_win_bonus", type=float, default=None)
    parser.add_argument("--prolonged_loss_penalty", type=float, default=None)
    parser.add_argument("--base_loss_penalty", type=float, default=None)
    parser.add_argument("--duration_fine_coef", type=float, default=None)
    parser.add_argument("--lr_p1", type=float, default=3e-4)
    parser.add_argument("--lr_p2", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=None,
                        help="PPO/env seed for reproducible runs (e.g. A/B comparisons). "
                             "None = SB3 default (random). Set equal across arms to isolate "
                             "the data effect from training noise.")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")

    model_dir = os.path.join(ROOT, "models", "mr")
    os.makedirs(model_dir, exist_ok=True)

    model_path_p1 = os.path.join(model_dir, "ppo_mr_filter_p1.zip")
    model_path_final = os.path.join(model_dir, "ppo_mr_filter.zip")
    vec_norm_p1 = os.path.join(model_dir, "vec_normalize_mr_p1.pkl")
    vec_norm_final = os.path.join(model_dir, "vec_normalize_mr.pkl")

    ckpt_dir = os.path.join(model_dir, "checkpoints_mr")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_log_dir = os.path.join(ROOT, "logs", "tb_mr_filter")
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.fresh:
        for p in [model_path_p1, model_path_final, vec_norm_p1, vec_norm_final]:
            if os.path.exists(p):
                backup = p + f".bak_{int(time.time())}"
                os.rename(p, backup)
                print(f"   Backup: {os.path.basename(p)} -> {os.path.basename(backup)}")

    if args.eval_only:
        if not os.path.exists(model_path_final):
            print(f"❌ Model not found: {model_path_final}")
            sys.exit(1)
        model = AuxAwarePPO.load(model_path_final)
        eval_pool = os.path.join(ROOT, "data", f"mr_signal_pool_{args.pool_size}.pkl")
        if not os.path.exists(eval_pool):
            eval_pool = None
        result = evaluate(model, n_episodes=5000, vec_normalize_path=vec_norm_final,
                          signal_pool_path=eval_pool,
                          ml_filter_threshold=args.ml_threshold,
                          risk_per_trade=args.risk_per_trade)
        return result

    # Verify data
    _check_env = make_env(args.data_dir)
    _n_symbols = len(_check_env._seq_symbols)
    del _check_env
    print(f"   Symbols available: {_n_symbols}")

    pool_path = None
    if args.pool_size > 0:
        pool_path = ensure_signal_pool(
            data_dir=args.data_dir,
            pool_size=args.pool_size,
            max_days=45,
            workers=args.pool_workers,
        )

    # ─── Phase 1 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" PHASE 1: MR Alpha ({args.timesteps_p1:,} steps)")
    print("=" * 65 + "\n")

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_vec_p1 = vec_cls([
        lambda erp=False, pp=pool_path, ns=args.outcome_noise, mt=args.ml_threshold,
        rpt=args.risk_per_trade,
        qt=args.quick_tp_bonus, sw=args.slow_win_bonus,
        pl=args.prolonged_loss_penalty, bl=args.base_loss_penalty,
        df=args.duration_fine_coef:
            make_env(args.data_dir, enable_risk_penalty=erp,
                     signal_pool_path=pp, outcome_noise_std=ns,
                     ml_filter_threshold=mt, risk_per_trade=rpt,
                     quick_tp_bonus=qt, slow_win_bonus=sw,
                     prolonged_loss_penalty=pl, base_loss_penalty=bl,
                     duration_fine_coef=df)
        for _ in range(args.n_envs)
    ])
    vec_env_p1 = VecNormalize(
        raw_vec_p1, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=20.0, gamma=0.99,
    )

    model_p1 = AuxAwarePPO(
        AuxAwareACPolicy,
        vec_env_p1,
        aux_loss_weight=0.5,
        learning_rate=args.lr_p1,
        seed=args.seed,
        n_steps=8192, batch_size=512, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.05, max_grad_norm=0.5,
        verbose=1, tensorboard_log=tb_log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            optimizer_kwargs=dict(weight_decay=1e-5),
        ),
    )
    ckpt_cb_p1 = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=ckpt_dir, name_prefix="ppo_mr_p1",
    )

    t0 = time.time()
    model_p1.learn(
        total_timesteps=args.timesteps_p1,
        callback=[
            EntropyScheduleCallback(initial=0.05, final=0.015),
            EarlyStopOnValueLoss(threshold=10.0, patience=5),
            EpisodeStatsCallback(print_every=20),
            ckpt_cb_p1,
        ],
        progress_bar=True,
        log_interval=1,
        tb_log_name="phase1_mr_alpha",
    )
    elapsed_p1 = time.time() - t0
    model_p1.save(model_path_p1)
    vec_env_p1.save(vec_norm_p1)
    print(f"\n Phase 1 done ({elapsed_p1/60:.1f} min)")

    # ─── Phase 2 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" PHASE 2: MR Risk ({args.timesteps_p2:,} steps)")
    print("=" * 65 + "\n")

    raw_vec_p2 = vec_cls([
        lambda erp=True, pp=pool_path, ns=args.outcome_noise, mt=args.ml_threshold,
        rpt=args.risk_per_trade,
        qt=args.quick_tp_bonus, sw=args.slow_win_bonus,
        pl=args.prolonged_loss_penalty, bl=args.base_loss_penalty,
        df=args.duration_fine_coef:
            make_env(args.data_dir, enable_risk_penalty=erp,
                     signal_pool_path=pp, outcome_noise_std=ns,
                     ml_filter_threshold=mt, risk_per_trade=rpt,
                     quick_tp_bonus=qt, slow_win_bonus=sw,
                     prolonged_loss_penalty=pl, base_loss_penalty=bl,
                     duration_fine_coef=df)
        for _ in range(args.n_envs)
    ])
    vec_env_p2 = VecNormalize.load(vec_norm_p1, raw_vec_p2)
    vec_env_p2.training = True
    vec_env_p2.norm_reward = True
    vec_env_p2.clip_reward = 20.0

    model_p2 = AuxAwarePPO.load(model_path_p1, env=vec_env_p2)
    if args.seed is not None:
        model_p2.set_random_seed(args.seed)  # reproducible Phase 2 (A/B isolation)
    # v8.0.43c: LR cosine decay from lr_p2 → lr_p2 × 0.2 (settle Phase 2)
    # ลด std oscillation, ช่วย converge ช่วงท้าย (เห็น STD ขึ้น 1.16→1.76 ใน v8.0.43b)
    import math as _math
    _lr_start = float(args.lr_p2)
    _lr_end = float(args.lr_p2) * 0.2  # 5e-5 → 1e-5
    def _cosine_lr(progress_remaining: float) -> float:
        # progress_remaining: 1.0 (start) → 0.0 (end)
        progress = 1.0 - progress_remaining
        cos_factor = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return _lr_end + (_lr_start - _lr_end) * cos_factor
    model_p2.learning_rate = _lr_start
    model_p2.lr_schedule = _cosine_lr
    for _pg in model_p2.policy.optimizer.param_groups:
        _pg["lr"] = _lr_start
    model_p2.ent_coef = 0.02
    model_p2.gamma = 0.99
    model_p2.tensorboard_log = tb_log_dir

    ckpt_cb_p2 = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=ckpt_dir, name_prefix="ppo_mr_p2",
    )
    t0 = time.time()
    model_p2.learn(
        total_timesteps=args.timesteps_p2,
        callback=[
            EntropyScheduleCallback(initial=0.02, final=0.010),
            EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000),
            EpisodeStatsCallback(print_every=20),
            ckpt_cb_p2,
        ],
        progress_bar=True,
        log_interval=1,
        tb_log_name="phase2_mr_risk",
        reset_num_timesteps=True,
    )
    elapsed_p2 = time.time() - t0
    model_p2.save(model_path_final)
    vec_env_p2.save(vec_norm_final)
    print(f"\n Phase 2 done ({elapsed_p2/60:.1f} min)")

    # ─── Final eval ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" Final MR Eval (5000 eps) — total train time: "
          f"{(elapsed_p1+elapsed_p2)/60:.1f} min")
    print("=" * 65)
    return evaluate(model_p2, n_episodes=5000, vec_normalize_path=vec_norm_final,
                    signal_pool_path=pool_path,
                    ml_filter_threshold=args.ml_threshold,
                    risk_per_trade=args.risk_per_trade)


if __name__ == "__main__":
    main()
