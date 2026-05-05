"""
===============================================================================
Train Signal Filter Agent for FTMO Challenge — 2-Phase Curriculum Learning
===============================================================================
Phase 1 (Alpha):  enable_risk_penalty=False — เรียนรู้เลือก signal ที่ดี (raw PnL)
Phase 2 (Risk):   enable_risk_penalty=True  — เรียนรู้จัดการ DD/risk (FTMO rules)

Usage:
    python scripts/train_signal_filter.py                              # default
    python scripts/train_signal_filter.py --timesteps_p1 500000 --timesteps_p2 300000
    python scripts/train_signal_filter.py --eval_only
    python scripts/train_signal_filter.py --fresh

หลัง train เสร็จ: model อยู่ที่ models/ppo_signal_filter.zip
===============================================================================
"""
import argparse
import os
import sys
import time

# Silence strategy debug prints (BEFORE importing bot_config / strategy modules)
# ต้องตั้งก่อน import เพราะ bot_config อ่าน env var ตอน singleton init
os.environ.setdefault("SMC_QUIET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.aux_aware_ppo import AuxAwarePPO
from ml.aux_aware_policy import AuxAwareACPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ml.signal_filter_env import FTMOSignalFilterEnv


# ═══════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════

class EntropyScheduleCallback(BaseCallback):
    """ลด ent_coef แบบ linear ตลอด training (explore -> exploit)"""
    def __init__(self, initial: float = 0.05, final: float = 0.005, verbose: int = 0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = self.final + (self.initial - self.final) * (1.0 - progress)
        return True


class EarlyStopOnValueLoss(BaseCallback):
    def __init__(self, threshold: float = 10.0, patience: int = 5,
                 warmup_steps: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.warmup_steps = warmup_steps   # v7.0.4: grace period สำหรับ phase boundary
        self._consecutive = 0

    def _on_step(self) -> bool:
        # v7.0.4 (2026-05-02): warmup grace — ปล่อย value_loss spike ตอน reward
        # distribution shift (Phase 1 → Phase 2 เปิด DD penalty + activity floor).
        # Value head ของ Phase 1 ทำนายตามช่วง [-2, 5] เจอ Phase 2 reward [-15, 5]
        # → MSE spike ชั่วคราว 30-80k steps จนกว่า re-fit. warmup_steps=50_000 ใน Phase 2
        # = 1% ของ target → ปลอดภัย.
        if self.num_timesteps < self.warmup_steps:
            self._consecutive = 0   # reset counter ระหว่าง warmup
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
    """Print episode stats to console"""

    def __init__(self, print_every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self._ep_count = 0
        self._balances = []
        self._profits = []
        self._dds = []
        self._take_rates = []
        self._win_rates = []
        self._passes = 0
        self._breaches = 0
        self._survives = 0
        self._last_print = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            summary = info.get("episode_summary")
            if summary is None:
                continue

            self._ep_count += 1
            self._balances.append(summary['balance'])
            self._profits.append(summary['profit'])
            self._dds.append(summary['total_dd_pct'])
            self._take_rates.append(summary['take_rate'])
            self._win_rates.append(summary['win_rate'])

            if summary['breached']:
                self._breaches += 1
            elif summary['passed']:
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
        print(f" Episode {self._ep_count} | Step {step:,}")
        print(f"{'='*65}")
        print(f"   Balance   : ${np.mean(r_bal):>10,.2f}  (min ${np.min(r_bal):>10,.2f} / max ${np.max(r_bal):>10,.2f})")
        print(f"   Profit    : ${np.mean(r_pnl):>+10,.2f}  (min ${np.min(r_pnl):>+10,.2f} / max ${np.max(r_pnl):>+10,.2f})")
        print(f"   DD        : {np.mean(r_dd):>8.2%}    (max {np.max(r_dd):>8.2%})")
        print(f"   Take Rate : {np.mean(r_tr):>8.1%}    Win Rate : {np.mean(r_wr):>8.1%}")
        print(f"   Pass: {self._passes}  Breach: {self._breaches}  Survive: {self._survives}  (total {total_ep} eps)")
        if total_ep > 0:
            print(f"   Pass Rate : {self._passes/total_ep:.1%}  |  Breach Rate : {self._breaches/total_ep:.1%}")
        print(f"{'='*65}")

    def _on_training_end(self):
        if self._ep_count > self._last_print:
            self._print_summary()


class FTMOTradingCallback(BaseCallback):
    """
    Log FTMO trading metrics to TensorBoard grouped by category.
    Reads episode_summary from info dict at episode end.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._total_episodes = 0
        self._passes = 0
        self._breaches = 0
        self._survives = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            summary = info.get("episode_summary")
            if summary is None:
                continue

            self._total_episodes += 1

            if summary['breached']:
                self._breaches += 1
            elif summary['passed']:
                self._passes += 1
            else:
                self._survives += 1

            # Cumulative
            self.logger.record("ftmo/total_episodes", self._total_episodes)
            self.logger.record("ftmo/passes", self._passes)
            self.logger.record("ftmo/breaches", self._breaches)
            self.logger.record("ftmo/survives", self._survives)
            if self._total_episodes > 0:
                self.logger.record("ftmo/pass_rate",
                                   self._passes / self._total_episodes)
                self.logger.record("ftmo/breach_rate",
                                   self._breaches / self._total_episodes)

            # 1_Balance/
            self.logger.record("1_Balance/Current", summary['balance'])
            self.logger.record("1_Balance/Min",
                               summary.get('min_balance', summary['balance']))
            self.logger.record("1_Balance/Max", summary['peak_balance'])

            # 2_Profit/
            self.logger.record("2_Profit/Current", summary['profit'])
            self.logger.record("2_Profit/Min",
                               summary.get('min_profit', summary['profit']))
            self.logger.record("2_Profit/Max",
                               summary.get('max_profit',
                                           summary['peak_balance'] - 100_000))

            # 3_Drawdown/
            self.logger.record("3_Drawdown/Daily_Current",
                               summary.get('daily_dd_pct', 0.0))
            self.logger.record("3_Drawdown/Daily_Max",
                               summary.get('max_daily_dd_pct', 0.0))
            self.logger.record("3_Drawdown/Total", summary['total_dd_pct'])

            # 4_Performance/
            self.logger.record("4_Performance/Take_Rate", summary['take_rate'])
            self.logger.record("4_Performance/Win_Rate", summary['win_rate'])

            # 5_Stats/
            self.logger.record("5_Stats/Pass_Count", self._passes)
            self.logger.record("5_Stats/Breach_Count", self._breaches)
            if self._total_episodes > 0:
                self.logger.record("5_Stats/Pass_Rate",
                                   self._passes / self._total_episodes)
                self.logger.record("5_Stats/Breach_Rate",
                                   self._breaches / self._total_episodes)

        return True


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_env(data_dir=None, enable_risk_penalty=True, signal_pool_path=None,
             outcome_noise_std=0.02, ml_filter_threshold=0.0,
             risk_per_trade=None):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return FTMOSignalFilterEnv(
            data_dir=data_dir,
            verbose=False,
            enable_risk_penalty=enable_risk_penalty,
            signal_pool_path=signal_pool_path,
            outcome_noise_std=outcome_noise_std,
            ml_filter_threshold=ml_filter_threshold,
            risk_per_trade=risk_per_trade,
        )


def ensure_signal_pool(data_dir, pool_size, max_days, workers):
    """
    Check & build signal pool (one-time preprocessing).
    ลด reset time ใน env จาก ~6 วิ → <1 ms ระหว่าง training
    """
    pool_path = os.path.join(ROOT, "data", f"signal_pool_{pool_size}.pkl")

    if os.path.exists(pool_path):
        size_mb = os.path.getsize(pool_path) / (1024 * 1024)
        print(f"   ✓ Signal pool พร้อมใช้: {pool_path} ({size_mb:.1f} MB)")
        return pool_path

    print(f"\n{'='*65}")
    print(f" 📦 Signal Pool ยังไม่มี — สร้างครั้งแรก")
    print(f"    ใช้เวลาประมาณ 20-40 นาที (แล้วแต่เครื่อง)")
    print(f"    ทำครั้งเดียว — ครั้งหน้า train จะเร็วขึ้น 10×")
    print(f"{'='*65}\n")

    # Import และเรียก builder
    from scripts.build_signal_pool import build_pool
    build_pool(
        data_dir=data_dir,
        pool_size=pool_size,
        max_days=max_days,
        save_path=pool_path,
        workers=workers,
    )
    return pool_path


def evaluate(model, n_episodes: int = 5000, vec_normalize_path: str = None,
             signal_pool_path: str = None, ml_filter_threshold: float = 0.0,
             risk_per_trade: float = None):
    # Pass all training params so eval matches training distribution
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

    passes, breaches, survive = 0, 0, 0
    rewards = []
    take_rates = []
    ep_lengths = []
    balances = []
    profits = []
    win_rates = []
    dds = []
    daily_dds_max = []

    passed_days = []
    passed_trades = []
    survive_profits = []
    survive_days = []
    survive_trades = []
    all_days = []
    all_trades = []

    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            ep_reward += r[0]
            steps += 1
            done = dones[0]
            last_info = infos[0] if infos else {}
        rewards.append(ep_reward)
        ep_lengths.append(steps)

        summary = last_info.get('episode_summary', {})
        take_rates.append(summary.get('take_rate', 0.0))
        balances.append(summary.get('balance', 100_000))
        profits.append(summary.get('profit', 0.0))
        win_rates.append(summary.get('win_rate', 0.0))
        dds.append(summary.get('total_dd_pct', 0.0))
        daily_dds_max.append(summary.get('max_daily_dd_pct', 0.0))

        days_traded = summary.get('days_traded', 0)
        total_trades = summary.get('total_trades', 0)
        ep_profit = summary.get('profit', 0.0)
        all_days.append(days_traded)
        all_trades.append(total_trades)

        if summary.get('breached', False):
            breaches += 1
        elif summary.get('passed', False):
            passes += 1
            passed_days.append(days_traded)
            passed_trades.append(total_trades)
        else:
            survive += 1
            survive_profits.append(ep_profit)
            survive_days.append(days_traded)
            survive_trades.append(total_trades)

    print(f"\n{'='*65}")
    print(f" Eval Result ({n_episodes} episodes)")
    print(f"{'='*65}")
    print(f"   Pass (hit 10%):     {passes:3d}  ({passes/n_episodes*100:.1f}%)")
    print(f"   Breach (DD limit):  {breaches:3d}  ({breaches/n_episodes*100:.1f}%)")
    print(f"   Survive, no target: {survive:3d}  ({survive/n_episodes*100:.1f}%)")
    print(f"{'='*65}")
    print(f"   Balance (avg):      ${np.mean(balances):>12,.2f}")
    print(f"   Balance (min/max):  ${np.min(balances):>12,.2f} / ${np.max(balances):>12,.2f}")
    print(f"   Profit (avg):       ${np.mean(profits):>+12,.2f}  ({np.mean(profits)/1000:.2f}%)")
    print(f"   Profit (min/max):   ${np.min(profits):>+12,.2f} / ${np.max(profits):>+12,.2f}")
    print(f"{'='*65}")
    dds_arr = np.array(dds)
    dd_daily_arr = np.array(daily_dds_max)
    print(f"   Total DD:  avg {np.mean(dds_arr):>6.2%}  | p95 {np.percentile(dds_arr, 95):>6.2%}  | "
          f"p99 {np.percentile(dds_arr, 99):>6.2%}  | max {np.max(dds_arr):>6.2%}")
    print(f"   Daily DD:  avg {np.mean(dd_daily_arr):>6.2%}  | p95 {np.percentile(dd_daily_arr, 95):>6.2%}  | "
          f"p99 {np.percentile(dd_daily_arr, 99):>6.2%}  | max {np.max(dd_daily_arr):>6.2%}")
    total_ratio = np.max(dds_arr) / 0.08 * 100
    daily_ratio = np.max(dd_daily_arr) / 0.04 * 100
    print(f"   → Total DD max / 8% limit = {total_ratio:.0f}%  |  Daily DD max / 4% limit = {daily_ratio:.0f}%")
    print(f"   Take Rate (avg):    {np.mean(take_rates):>8.1%}")
    print(f"   Win Rate (avg):     {np.mean(win_rates):>8.1%}")
    print(f"   Ep Length (avg):     {np.mean(ep_lengths):>8.1f}")
    print(f"   Reward (avg):        {np.mean(rewards):>8.2f} (std {np.std(rewards):.2f})")
    print(f"{'='*65}")

    # === Per-category breakdown: days to target / profit at end ===
    if all_days:
        avg_days_all = np.mean(all_days)
        avg_trades_all = np.mean(all_trades)
        orders_per_day = avg_trades_all / max(avg_days_all, 1e-9)
        print(f"\n📊 Episode Breakdown:")
        print(f"   All eps     — avg days: {avg_days_all:>5.1f} | avg orders: {avg_trades_all:>5.1f} | orders/day: {orders_per_day:.2f}")

    if passed_days:
        print(f"\n🎯 Passed eps (n={len(passed_days)}) — ใช้กี่วันถึง 10%:")
        print(f"   Days to target: avg {np.mean(passed_days):>5.1f} | median {np.median(passed_days):>5.1f} | "
              f"min {np.min(passed_days):>3d} | max {np.max(passed_days):>3d}")
        print(f"   Trades to target: avg {np.mean(passed_trades):>5.1f} | "
              f"min {np.min(passed_trades):>3d} | max {np.max(passed_trades):>3d}")
    else:
        print(f"\n🎯 Passed eps: 0 (ไม่มี ep ไหน hit 10%)")

    if survive_profits:
        sp = np.array(survive_profits)
        sp_pct = sp / 1000.0  # 100k balance → pct
        print(f"\n⏳ Survive eps (n={len(survive_profits)}) — กำไรเมื่อครบ 45 วัน:")
        print(f"   Profit: avg ${np.mean(sp):>+9,.2f} ({np.mean(sp_pct):+.2f}%) | "
              f"median ${np.median(sp):>+9,.2f} ({np.median(sp_pct):+.2f}%)")
        print(f"   Profit: min ${np.min(sp):>+9,.2f} ({np.min(sp_pct):+.2f}%) | "
              f"max ${np.max(sp):>+9,.2f} ({np.max(sp_pct):+.2f}%)")
        print(f"   Days used: avg {np.mean(survive_days):>5.1f} | trades: avg {np.mean(survive_trades):>5.1f}")
        profitable = int((sp > 0).sum())
        print(f"   Profitable: {profitable}/{len(sp)} ({profitable/len(sp)*100:.1f}%)")

    print(f"{'='*65}")
    return passes / n_episodes


# ═══════════════════════════════════════════════════════════════
# Main — 2-Phase Curriculum Training
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train FTMO Signal Filter (2-Phase Curriculum)")
    parser.add_argument("--timesteps_p1", type=int, default=300_000,
                        help="Phase 1 (Alpha) timesteps")
    parser.add_argument("--timesteps_p2", type=int, default=200_000,
                        help="Phase 2 (Risk) timesteps")
    parser.add_argument("--fresh", action="store_true",
                        help="Train from scratch (backup old model)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluate existing model without training")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--pool_size", type=int, default=3000,
                        help="Signal pool size (0 = disable pool, use slow on-the-fly)")
    parser.add_argument("--pool_workers", type=int, default=8,
                        help="Workers ตอน build signal pool (ครั้งแรกเท่านั้น)")
    parser.add_argument("--outcome_noise", type=float, default=0.05,
                        help="Gaussian noise std on outcome (0=off, 0.02=2pct, 0.05=5pct default v6.13)")
    parser.add_argument("--ml_threshold", type=float, default=0.36,
                        help="ML pre-filter threshold (0.36 = production calibrated default v6.13)")
    parser.add_argument("--risk_per_trade", type=float, default=0.007,
                        help="Risk per trade fraction (default 0.007=0.7pct sync กับ live v6.13)")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(ROOT, "data", "ohlcv")

    # ─── Paths ────────────────────────────────────────────────
    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path_p1 = os.path.join(model_dir, "ppo_signal_filter_p1.zip")
    model_path_final = os.path.join(model_dir, "ppo_signal_filter.zip")
    vec_norm_p1 = os.path.join(model_dir, "vec_normalize_sf_p1.pkl")
    vec_norm_final = os.path.join(model_dir, "vec_normalize_sf.pkl")

    ckpt_dir = os.path.join(model_dir, "checkpoints_sf")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_log_dir = os.path.join(ROOT, "logs", "tb_signal_filter")
    os.makedirs(tb_log_dir, exist_ok=True)

    # ─── Backup old model if --fresh ──────────────────────────
    if args.fresh:
        for p in [model_path_p1, model_path_final, vec_norm_p1, vec_norm_final]:
            if os.path.exists(p):
                backup = p + f".bak_{int(time.time())}"
                os.rename(p, backup)
                print(f"   Backup: {os.path.basename(p)} -> {os.path.basename(backup)}")

    # ─── Eval-only mode ───────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(model_path_final):
            print(f"Model not found: {model_path_final}")
            sys.exit(1)
        # v6.9 E2: AuxAwarePPO subclass — load with AuxAwarePPO to keep aux head
        model = AuxAwarePPO.load(model_path_final)
        # Use same pool as training (if exists) for distribution match
        eval_pool = os.path.join(ROOT, "data", f"signal_pool_{args.pool_size}.pkl")
        if not os.path.exists(eval_pool):
            eval_pool = None
        evaluate(model, n_episodes=5000, vec_normalize_path=vec_norm_final,
                 signal_pool_path=eval_pool,
                 ml_filter_threshold=args.ml_threshold,
                 risk_per_trade=args.risk_per_trade)
        return

    # ─── Check data availability ──────────────────────────────
    _check_env = make_env(args.data_dir)
    _n_symbols = len(_check_env._seq_symbols)
    del _check_env
    print(f"   Symbols available: {_n_symbols}")

    # ─── Ensure signal pool exists (build if missing) ─────────
    pool_path = None
    if args.pool_size > 0:
        pool_path = ensure_signal_pool(
            data_dir=args.data_dir,
            pool_size=args.pool_size,
            max_days=45,
            workers=args.pool_workers,
        )

    # ═════════════════════════════════════════════════════════════
    # PHASE 1: Alpha Generation (no risk penalty)
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f" PHASE 1: Alpha Training ({args.timesteps_p1:,} steps)")
    print(f"   enable_risk_penalty = False")
    print(f"   LR=3e-4, ent=0.05->0.005, gamma=0.99, clip_reward=20")
    print(f"   network = [256, 128]  (ใหญ่ขึ้นเพื่อ approximate long-horizon value)")
    print(f"   outcome_noise = {args.outcome_noise:.3f}  (ลดจาก 0.05 → reward variance ต่ำลง)")
    print(f"   ml_threshold  = {args.ml_threshold:.3f}  {'⭐ Hybrid mode' if args.ml_threshold > 0 else '(ปิด filter)'}")
    _risk_show = args.risk_per_trade if args.risk_per_trade is not None else 0.003
    print(f"   risk_per_trade= {_risk_show:.4f}  ({_risk_show*100:.2f}%)")
    print(f"   signal_pool = {args.pool_size if pool_path else 'OFF (slow)'}")
    print(f"{'='*65}\n")

    # SubprocVecEnv: spawn 1 process ต่อ env → ใช้หลาย core จริง
    # (DummyVecEnv เดิมรันเรียงใน 1 process = 1 core)
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_vec_p1 = vec_cls([
        lambda erp=False, pp=pool_path, ns=args.outcome_noise, mt=args.ml_threshold, rpt=args.risk_per_trade:
            make_env(args.data_dir, enable_risk_penalty=erp,
                     signal_pool_path=pp, outcome_noise_std=ns,
                     ml_filter_threshold=mt, risk_per_trade=rpt)
        for _ in range(args.n_envs)
    ])
    vec_env_p1 = VecNormalize(
        raw_vec_p1, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=20.0, gamma=0.99,
    )

    # v7.1.7b (2026-05-05) — MPS rollback: small network [256,128] = MPS overhead > benefit
    # Tested v7.1.7a: MPS 1535 steps/sec vs CPU baseline 10000 steps/sec = 6× SLOWER
    # → ใช้ CPU สำหรับ small network (PPO inference per-step ดี on CPU)
    # MPS เหมาะกับ network ใหญ่ (millions params) ไม่ใช่ small policy network
    # Other v7.1.7 optimizations เก็บไว้: n_steps 8192, weight_decay 1e-5, slower entropy
    _device = 'cpu'
    print(f"   Device: {_device} (CPU — small network, MPS overhead > benefit)")

    # v6.9 E2: AuxAwarePPO + AuxAwareACPolicy — auxiliary regression head on signal outcome
    model_p1 = AuxAwarePPO(
        AuxAwareACPolicy,
        vec_env_p1,
        aux_loss_weight=0.5,   # weight for MSE(predict_aux, outcome_pnl_ratio)
        learning_rate=3e-4,
        n_steps=8192,          # v7.1.7: 4096 → 8192 — larger batch สำหรับ pool diversity
        batch_size=512,        # v7.1.7: 256 → 512 — match larger n_steps (8192/16 mini-batches)
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.05,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tb_log_dir,
        device=_device,        # v7.1.7: enable MPS Apple Silicon
        # Network [256, 128] — ใหญ่พอ approximate value function ระยะยาว (γ=0.99)
        # v7.1.7: เพิ่ม weight_decay 1e-5 → regularize policy network สำหรับ pool 10k diversity
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            optimizer_kwargs=dict(weight_decay=1e-5),
        ),
    )

    ckpt_cb_p1 = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_sf_p1",
    )

    t0 = time.time()
    model_p1.learn(
        total_timesteps=args.timesteps_p1,
        callback=[
            EntropyScheduleCallback(initial=0.05, final=0.015),  # v7.1.7: 0.005 → 0.015 (slower decay = กัน entropy collapse)
            EarlyStopOnValueLoss(threshold=10.0, patience=5),
            EpisodeStatsCallback(print_every=20),
            FTMOTradingCallback(),
            ckpt_cb_p1,
        ],
        progress_bar=True,
        log_interval=1,
        tb_log_name="phase1_alpha",
    )
    elapsed_p1 = time.time() - t0

    model_p1.save(model_path_p1)
    vec_env_p1.save(vec_norm_p1)
    print(f"\n Phase 1 done ({elapsed_p1/60:.1f} min)")
    print(f"   Model -> {model_path_p1}")
    print(f"   VecNormalize -> {vec_norm_p1}")

    # ═════════════════════════════════════════════════════════════
    # PHASE 2: Risk Management (with risk penalty)
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f" PHASE 2: Risk Training ({args.timesteps_p2:,} steps)")
    print(f"   enable_risk_penalty = True")
    print(f"   LR=2e-4, ent=0.03->0.008, gamma=0.99, clip_reward=20")
    print(f"   load Phase 1 model (keep exploration to avoid SKIP-all collapse)")
    print(f"{'='*65}\n")

    # SubprocVecEnv: ใช้หลาย core จริงเหมือน Phase 1
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_vec_p2 = vec_cls([
        lambda erp=True, pp=pool_path, ns=args.outcome_noise, mt=args.ml_threshold, rpt=args.risk_per_trade:
            make_env(args.data_dir, enable_risk_penalty=erp,
                     signal_pool_path=pp, outcome_noise_std=ns,
                     ml_filter_threshold=mt, risk_per_trade=rpt)
        for _ in range(args.n_envs)
    ])
    vec_env_p2 = VecNormalize.load(vec_norm_p1, raw_vec_p2)
    vec_env_p2.training = True
    vec_env_p2.norm_reward = True
    vec_env_p2.clip_reward = 20.0   # เผื่อ DD penalty สูงช่วง breach

    # v6.9 E2: load with AuxAwarePPO (preserves aux_head + buffer class)
    model_p2 = AuxAwarePPO.load(model_path_p1, env=vec_env_p2)
    # v6.8 P2 stability fix — value_loss explosion ทำให้ early-stop เร็วเกิน
    # หลังเปลี่ยน reward distribution (calibration / ฯลฯ) — ลด LR + ent_coef
    # v7.0.5 (2026-05-02): Fix LR schedule properly — SB3 PPO `_setup_model()` wrap
    # `lr_schedule` ตอน init เท่านั้น. การ set `model.learning_rate = X` หลัง
    # `AuxAwarePPO.load()` ไม่ rebuild schedule → optimizer ใช้ค่า Phase 1 default (3e-4)
    # → ทำให้ Phase 2 ใช้ LR 6× สูงกว่า intended (latent bug ตั้งแต่ v6.x)
    # Fix: rebuild lr_schedule + update optimizer param_groups ตรงๆ
    model_p2.learning_rate = 5e-5   # ↓ จาก 2e-4 (4x conservative) — match value-loss scale
    model_p2.lr_schedule = FloatSchedule(5e-5)   # v7.0.5: rebuild schedule (proper fix)
    for _pg in model_p2.policy.optimizer.param_groups:
        _pg['lr'] = 5e-5             # v7.0.5: update optimizer immediately (กัน lag 1 rollout)
    model_p2.ent_coef = 0.02        # ↓ จาก 0.03 — ลด exploration variance ใน P2
    model_p2.gamma = 0.99           # Phase 2 inherit, แต่ explicit set ให้ชัด
    model_p2.tensorboard_log = tb_log_dir

    ckpt_cb_p2 = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_sf_p2",
    )

    t0 = time.time()
    model_p2.learn(
        total_timesteps=args.timesteps_p2,
        callback=[
            EntropyScheduleCallback(initial=0.02, final=0.010),  # v7.1.7: 0.005 → 0.010 (slower decay)
            # v6.8: threshold 10 → 20 — value_loss spike ชั่วคราวยอม, ปกป้องเฉพาะ true divergence
            # v7.0.4: warmup_steps=50_000 — ปล่อยให้ value head re-fit reward distribution Phase 2
            # ก่อน enable check (กัน trigger เร็วเกินตอน phase boundary, observed v7.0.3 spike 31)
            # v7.0.6 ทดลอง threshold 30 → Pass Rate ตก 10.7% → 7.4% (Phase 2 รันเต็ม 5M
            # = over-tune toward safety, ลด pass-rate aggressiveness)
            # v7.0.7 (2026-05-02): revert threshold 30 → 20 — early stop ที่ value_loss=20.45
            # ของ v7.0.5 = inflection point detector จริง ไม่ใช่ false positive. threshold 20
            # หลัง LR fix proper = right calibration (engineered sweet spot @ Phase 2 ~70%)
            EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000),
            EpisodeStatsCallback(print_every=20),
            FTMOTradingCallback(),
            ckpt_cb_p2,
        ],
        progress_bar=True,
        log_interval=1,
        tb_log_name="phase2_risk",
        reset_num_timesteps=True,
    )
    elapsed_p2 = time.time() - t0

    model_p2.save(model_path_final)
    vec_env_p2.save(vec_norm_final)
    print(f"\n Phase 2 done ({elapsed_p2/60:.1f} min)")
    print(f"   Model -> {model_path_final}")
    print(f"   VecNormalize -> {vec_norm_final}")

    # ═════════════════════════════════════════════════════════════
    # Final Evaluation
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f" Final Evaluation (5000 episodes)")
    print(f"   Total training time: {(elapsed_p1 + elapsed_p2)/60:.1f} min")
    print(f"{'='*65}")
    evaluate(model_p2, n_episodes=5000, vec_normalize_path=vec_norm_final,
             signal_pool_path=pool_path,
             ml_filter_threshold=args.ml_threshold,
             risk_per_trade=args.risk_per_trade)


if __name__ == "__main__":
    main()
