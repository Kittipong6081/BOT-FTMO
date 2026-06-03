"""
===============================================================================
TF Signal Filter RL Trainer (v8.1 Phase 3) — 2-Phase PPO with Aux Task
===============================================================================
Clone of `train_mr_signal_filter.py` that instantiates `TrendFollowingFilterEnv`.
Default pool = data/tf_signal_pool_<N>.pkl, models live in models/tf/.

Same 2-phase curriculum (Phase 1 Alpha → Phase 2 Risk). TF reward shaping params
(runner_bonus / slow_win_bonus / late_entry_penalty / base_loss_penalty) are
forwarded so auto_train_pipeline_tf can tune them.

Usage:
    python scripts/train_tf_signal_filter.py --fresh \
        --timesteps_p1 5000000 --timesteps_p2 2000000 \
        --n_envs 8 --pool_size 5000 --ml_threshold 0.30 --risk_per_trade 0.0060

Output: models/tf/ppo_tf_filter.zip + models/tf/vec_normalize_tf.pkl
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ml.trend_following_env import TrendFollowingFilterEnv


# ─── Callbacks (reused from MR trainer) ──────────────────────────────────

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial=0.05, final=0.005, verbose=0):
        super().__init__(verbose)
        self.initial, self.final = initial, final

    def _on_step(self):
        progress = self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = self.final + (self.initial - self.final) * (1.0 - progress)
        return True


class EarlyStopOnValueLoss(BaseCallback):
    def __init__(self, threshold=10.0, patience=5, warmup_steps=0, verbose=0):
        super().__init__(verbose)
        self.threshold, self.patience, self.warmup_steps = threshold, patience, warmup_steps
        self._consecutive = 0

    def _on_step(self):
        if self.num_timesteps < self.warmup_steps:
            self._consecutive = 0
            return True
        vl = self.logger.name_to_value.get("train/value_loss", 0.0)
        if vl > self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.patience:
                print(f"\n[Early Stop] value_loss={vl:.2f} > {self.threshold} x{self.patience}")
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
        self._passes = self._breaches = self._survives = 0
        self._last_print = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
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
        recent = min(self.print_every, len(self._balances))
        total_ep = self._passes + self._breaches + self._survives
        print(f"\n{'='*65}")
        print(f" [TF] Episode {self._ep_count} | Step {self.num_timesteps:,}")
        print(f"   Balance ${np.mean(self._balances[-recent:]):>10,.2f}  "
              f"Profit ${np.mean(self._profits[-recent:]):>+10,.2f}  "
              f"DD {np.mean(self._dds[-recent:]):.2%}")
        print(f"   Take {np.mean(self._take_rates[-recent:]):.1%}  "
              f"Win {np.mean(self._win_rates[-recent:]):.1%}  "
              f"Pass {self._passes} Breach {self._breaches} Survive {self._survives}")
        if total_ep > 0:
            print(f"   Pass Rate: {self._passes/total_ep:.1%}")
        print(f"{'='*65}")


# ─── Helpers ───────────────────────────────────────────────────────────

def make_env(data_dir=None, enable_risk_penalty=True, signal_pool_path=None,
             outcome_noise_std=0.05, ml_filter_threshold=0.0, risk_per_trade=None,
             runner_bonus=None, slow_win_bonus=None,
             late_entry_penalty=None, base_loss_penalty=None):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return TrendFollowingFilterEnv(
            data_dir=data_dir, verbose=False,
            enable_risk_penalty=enable_risk_penalty,
            signal_pool_path=signal_pool_path,
            outcome_noise_std=outcome_noise_std,
            ml_filter_threshold=ml_filter_threshold,
            risk_per_trade=risk_per_trade,
            runner_bonus=runner_bonus, slow_win_bonus=slow_win_bonus,
            late_entry_penalty=late_entry_penalty, base_loss_penalty=base_loss_penalty,
        )


def ensure_signal_pool(data_dir, pool_size, max_days, workers):
    pool_path = os.path.join(ROOT, "data", f"tf_signal_pool_{pool_size}.pkl")
    if os.path.exists(pool_path):
        print(f"   ✓ TF pool ready: {pool_path} "
              f"({os.path.getsize(pool_path)/(1024*1024):.1f} MB)")
        return pool_path
    print(f"\n📦 TF pool missing — building (~{pool_size//200} min)\n")
    from scripts.build_tf_signal_pool import build_pool
    build_pool(data_dir=data_dir, pool_size=pool_size, max_days=max_days,
               save_path=pool_path, workers=workers)
    return pool_path


def evaluate(model, n_episodes=5000, vec_normalize_path=None,
             signal_pool_path=None, ml_filter_threshold=0.0, risk_per_trade=None):
    raw_env = DummyVecEnv([lambda: make_env(
        signal_pool_path=signal_pool_path, ml_filter_threshold=ml_filter_threshold,
        risk_per_trade=risk_per_trade)])
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_env

    passes = breaches = survive = profitable_count = 0
    profits, win_rates, take_rates, dds, daily_dds_max = [], [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            done = dones[0]
            last_info = infos[0] if infos else {}
        s = last_info.get("episode_summary", {})
        take_rates.append(s.get("take_rate", 0.0))
        profits.append(s.get("profit", 0.0))
        win_rates.append(s.get("win_rate", 0.0))
        dds.append(s.get("total_dd_pct", 0.0))
        daily_dds_max.append(s.get("max_daily_dd_pct", 0.0))
        if s.get("profit", 0.0) > 0:
            profitable_count += 1
        if s.get("breached", False):
            breaches += 1
        elif s.get("passed", False):
            passes += 1
        else:
            survive += 1

    pass_rate = passes / n_episodes
    print("\n" + "=" * 65)
    print(f" TF Eval Result ({n_episodes} eps)")
    print(f"   Pass {pass_rate*100:.1f}%  Breach {breaches/n_episodes*100:.1f}%  "
          f"Profitable {profitable_count/n_episodes*100:.1f}%")
    print(f"   Win {np.mean(win_rates)*100:.1f}%  Take {np.mean(take_rates)*100:.1f}%  "
          f"Profit ${np.mean(profits):+,.2f}")
    print(f"   Total DD max {np.max(dds)*100:.2f}%  Daily DD max {np.max(daily_dds_max)*100:.2f}%")
    print("=" * 65)
    return {
        "pass_rate": pass_rate, "breach_rate": breaches / n_episodes,
        "profitable_rate": profitable_count / n_episodes,
        "win_rate": float(np.mean(win_rates)), "take_rate": float(np.mean(take_rates)),
        "total_dd_max": float(np.max(dds)), "daily_dd_max": float(np.max(daily_dds_max)),
        "profit_avg": float(np.mean(profits)),
    }


def main():
    p = argparse.ArgumentParser(description="Train TF Signal Filter (v8.1)")
    p.add_argument("--timesteps_p1", type=int, default=5_000_000)
    p.add_argument("--timesteps_p2", type=int, default=2_000_000)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--pool_size", type=int, default=5000)
    p.add_argument("--pool_workers", type=int, default=8)
    p.add_argument("--outcome_noise", type=float, default=0.05)
    p.add_argument("--ml_threshold", type=float, default=0.30)
    p.add_argument("--risk_per_trade", type=float, default=0.0060)  # TF risk (vs MR 0.0070)
    p.add_argument("--runner_bonus", type=float, default=None)
    p.add_argument("--slow_win_bonus", type=float, default=None)
    p.add_argument("--late_entry_penalty", type=float, default=None)
    p.add_argument("--base_loss_penalty", type=float, default=None)
    p.add_argument("--lr_p1", type=float, default=3e-4)
    p.add_argument("--lr_p2", type=float, default=5e-5)
    args = p.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")

    model_dir = os.path.join(ROOT, "models", "tf")
    os.makedirs(model_dir, exist_ok=True)
    model_path_p1 = os.path.join(model_dir, "ppo_tf_filter_p1.zip")
    model_path_final = os.path.join(model_dir, "ppo_tf_filter.zip")
    vec_norm_p1 = os.path.join(model_dir, "vec_normalize_tf_p1.pkl")
    vec_norm_final = os.path.join(model_dir, "vec_normalize_tf.pkl")
    ckpt_dir = os.path.join(model_dir, "checkpoints_tf")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_log_dir = os.path.join(ROOT, "logs", "tb_tf_filter")
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.fresh:
        for pth in [model_path_p1, model_path_final, vec_norm_p1, vec_norm_final]:
            if os.path.exists(pth):
                bak = pth + f".bak_{int(time.time())}"
                os.rename(pth, bak)
                print(f"   Backup: {os.path.basename(pth)} -> {os.path.basename(bak)}")

    if args.eval_only:
        if not os.path.exists(model_path_final):
            print(f"❌ Model not found: {model_path_final}")
            sys.exit(1)
        model = AuxAwarePPO.load(model_path_final)
        eval_pool = os.path.join(ROOT, "data", f"tf_signal_pool_{args.pool_size}.pkl")
        if not os.path.exists(eval_pool):
            eval_pool = None
        return evaluate(model, 5000, vec_norm_final, eval_pool,
                        args.ml_threshold, args.risk_per_trade)

    _check = make_env(args.data_dir)
    print(f"   Symbols available: {len(_check._seq_symbols)}")
    del _check

    pool_path = None
    if args.pool_size > 0:
        pool_path = ensure_signal_pool(args.data_dir, args.pool_size, 45, args.pool_workers)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv

    def _env_factory(erp):
        return lambda: make_env(
            args.data_dir, enable_risk_penalty=erp, signal_pool_path=pool_path,
            outcome_noise_std=args.outcome_noise, ml_filter_threshold=args.ml_threshold,
            risk_per_trade=args.risk_per_trade,
            runner_bonus=args.runner_bonus, slow_win_bonus=args.slow_win_bonus,
            late_entry_penalty=args.late_entry_penalty, base_loss_penalty=args.base_loss_penalty)

    # ─── Phase 1 (Alpha) ───
    print("\n" + "=" * 65 + f"\n PHASE 1: TF Alpha ({args.timesteps_p1:,} steps)\n" + "=" * 65)
    raw_vec_p1 = vec_cls([_env_factory(False) for _ in range(args.n_envs)])
    vec_env_p1 = VecNormalize(raw_vec_p1, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=20.0, gamma=0.99)
    model_p1 = AuxAwarePPO(
        AuxAwareACPolicy, vec_env_p1, aux_loss_weight=0.5, learning_rate=args.lr_p1,
        n_steps=8192, batch_size=512, n_epochs=5, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.05, max_grad_norm=0.5, verbose=1,
        tensorboard_log=tb_log_dir, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                           optimizer_kwargs=dict(weight_decay=1e-5)))
    t0 = time.time()
    model_p1.learn(total_timesteps=args.timesteps_p1, callback=[
        EntropyScheduleCallback(0.05, 0.015),
        EarlyStopOnValueLoss(threshold=10.0, patience=5),
        EpisodeStatsCallback(print_every=20),
        CheckpointCallback(max(50_000 // args.n_envs, 1), ckpt_dir, name_prefix="ppo_tf_p1"),
    ], progress_bar=True, log_interval=1, tb_log_name="phase1_tf_alpha")
    elapsed_p1 = time.time() - t0
    model_p1.save(model_path_p1)
    vec_env_p1.save(vec_norm_p1)
    print(f"\n Phase 1 done ({elapsed_p1/60:.1f} min)")

    # ─── Phase 2 (Risk) ───
    print("\n" + "=" * 65 + f"\n PHASE 2: TF Risk ({args.timesteps_p2:,} steps)\n" + "=" * 65)
    raw_vec_p2 = vec_cls([_env_factory(True) for _ in range(args.n_envs)])
    vec_env_p2 = VecNormalize.load(vec_norm_p1, raw_vec_p2)
    vec_env_p2.training = True
    vec_env_p2.norm_reward = True
    vec_env_p2.clip_reward = 20.0
    model_p2 = AuxAwarePPO.load(model_path_p1, env=vec_env_p2)
    import math as _math
    _lr_start, _lr_end = float(args.lr_p2), float(args.lr_p2) * 0.2
    def _cosine_lr(progress_remaining):
        progress = 1.0 - progress_remaining
        return _lr_end + (_lr_start - _lr_end) * 0.5 * (1.0 + _math.cos(_math.pi * progress))
    model_p2.learning_rate = _lr_start
    model_p2.lr_schedule = _cosine_lr
    for _pg in model_p2.policy.optimizer.param_groups:
        _pg["lr"] = _lr_start
    model_p2.ent_coef = 0.02
    model_p2.gamma = 0.99
    model_p2.tensorboard_log = tb_log_dir
    t0 = time.time()
    model_p2.learn(total_timesteps=args.timesteps_p2, callback=[
        EntropyScheduleCallback(0.02, 0.010),
        EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000),
        EpisodeStatsCallback(print_every=20),
        CheckpointCallback(max(50_000 // args.n_envs, 1), ckpt_dir, name_prefix="ppo_tf_p2"),
    ], progress_bar=True, log_interval=1, tb_log_name="phase2_tf_risk", reset_num_timesteps=True)
    elapsed_p2 = time.time() - t0
    model_p2.save(model_path_final)
    vec_env_p2.save(vec_norm_final)
    print(f"\n Phase 2 done ({elapsed_p2/60:.1f} min)")

    print("\n" + "=" * 65 + f"\n Final TF Eval (5000 eps) — train {(elapsed_p1+elapsed_p2)/60:.1f} min\n" + "=" * 65)
    return evaluate(model_p2, 5000, vec_norm_final, pool_path,
                    args.ml_threshold, args.risk_per_trade)


if __name__ == "__main__":
    main()
