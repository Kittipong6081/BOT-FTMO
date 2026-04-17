"""
===============================================================================
Train PPO Agent for FTMO Challenge (V3 — 16 obs dims, 5 action dims)
===============================================================================
ฝึก PPO agent ด้วย FTMOOptimizationEnv V3 (T12-T16 reward + obs + action)
- Default: 500k timesteps + checkpoint ทุก 50k steps
- Eval callback ทุก 25k steps (100 episodes deterministic)

Usage:
    python scripts/train_ppo.py                 # 500k timesteps default
    python scripts/train_ppo.py --timesteps 1000000 --fresh
    python scripts/train_ppo.py --eval_only     # eval current model only

หลัง train เสร็จ: model อยู่ที่ ftmo_trading_bot/models/ppo_ftmo_agent.zip
===============================================================================
"""
import argparse
import os
import sys
import time

import numpy as np

# เพิ่ม project root เข้า sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ml.rl_environment import FTMOOptimizationEnv


class EarlyStopOnValueLoss(BaseCallback):
    """T10: หยุด train เมื่อ value_loss สูงเกินไปติดต่อกัน — ป้องกัน divergence"""

    def __init__(self, threshold: float = 10.0, patience: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self._consecutive = 0

    def _on_step(self) -> bool:
        infos = self.logger.name_to_value
        vl = infos.get("train/value_loss", 0.0)
        if vl > self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.patience:
                print(f"\n⛔ Early Stop: value_loss={vl:.2f} > {self.threshold} "
                      f"ติดต่อกัน {self.patience} ครั้ง — หยุด train")
                return False
        else:
            self._consecutive = 0
        return True


def make_env(data_dir=None, verbose=False):
    """สร้าง env — verbose=False ซ่อน log ทั้งหมด (init + step) ตอน train"""
    import io, contextlib
    if verbose:
        return FTMOOptimizationEnv(data_dir=data_dir, verbose=True)
    # ซ่อน init prints (MT5 warning, Strategy init, Backtester ready) ใน subprocess
    with contextlib.redirect_stdout(io.StringIO()):
        return FTMOOptimizationEnv(data_dir=data_dir, verbose=False)


def evaluate(model, n_episodes: int = 100, vec_normalize_path: str = None):
    """รัน episode แล้วรายงาน pass-rate (ไม่ breach + ถึง 10%) — ใช้ VecNormalize ถ้ามี"""
    raw_env = DummyVecEnv([lambda: make_env()])
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = raw_env

    passes, breaches, no_target = 0, 0, 0
    rewards = []
    ep_lengths = []

    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            ep_reward += r[0]
            steps += 1
            done = dones[0]
        rewards.append(ep_reward)
        ep_lengths.append(steps)

        inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
        stats = inner_env._get_stats()
        if stats.get('total_dd_pct', 0) >= 0.08 or stats.get('daily_dd_pct', 0) >= 0.04:
            breaches += 1
        elif stats.get('target_progress_pct', 0) >= 100.0:
            passes += 1
        else:
            no_target += 1

    print(f"\n📊 Eval ({n_episodes} episodes):")
    print(f"   ✅ Pass (hit 10%):     {passes:3d}  ({passes/n_episodes*100:.1f}%)")
    print(f"   ❌ Breach (DD limit):  {breaches:3d}  ({breaches/n_episodes*100:.1f}%)")
    print(f"   ➖ Survive, no target: {no_target:3d}  ({no_target/n_episodes*100:.1f}%)")
    print(f"   💰 Mean reward:        {np.mean(rewards):.2f} (std {np.std(rewards):.2f})")
    print(f"   📏 Mean ep_length:     {np.mean(ep_lengths):.1f}")
    return passes / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--fresh", action="store_true",
                        help="ลบ model เก่าและ train ใหม่ทั้งหมด")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="จำนวน parallel envs (มาก = เร็วแต่กิน RAM)")
    parser.add_argument("--data_dir", default=None,
                        help="โฟลเดอร์ OHLCV CSV (default: data/ohlcv)")
    args = parser.parse_args()

    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_ftmo_agent.zip")
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_log_dir = os.path.join(ROOT, "logs", "tb_ppo")
    os.makedirs(tb_log_dir, exist_ok=True)

    if args.fresh and os.path.exists(model_path):
        backup = model_path + f".bak_{int(time.time())}"
        os.rename(model_path, backup)
        print(f"🗄️  Backup model เก่า → {backup}")

    if args.eval_only:
        if not os.path.exists(model_path):
            print(f"❌ ไม่พบ model: {model_path}")
            sys.exit(1)
        model = PPO.load(model_path)
        vnp = os.path.join(model_dir, "vec_normalize.pkl")
        evaluate(model, n_episodes=100, vec_normalize_path=vnp)
        return

    # Vectorized envs (เร่ง training + ลด correlation)
    # verbose=False → ซ่อน log วิเคราะห์กราฟทั้งหมดใน subprocess
    print(f"🔧 กำลังสร้าง {args.n_envs} training envs + 1 eval env ...")
    raw_vec_env = SubprocVecEnv([lambda: make_env(args.data_dir) for _ in range(args.n_envs)])
    raw_eval_env = make_vec_env(lambda: make_env(args.data_dir), n_envs=1)

    # ตรวจสอบข้อมูลจาก 1 env เพื่อแสดงสรุป
    _check_env = make_env(args.data_dir, verbose=False)
    _bt = "SMC Strategy จริง" if (_check_env._backtester and _check_env._backtester.is_available) else "Statistical Simulation"
    _data = "Real OHLCV" if _check_env.has_real_data else "Synthetic"
    _syms = len(_check_env._ohlcv_cache) if _check_env.has_real_data else 0
    del _check_env
    print(f"   📊 Data: {_data} ({_syms} symbols) | Engine: {_bt}")

    # VecNormalize — normalize obs + reward อัตโนมัติ
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    # T6: เปิด norm_reward=True — stabilize value function (เดิม False ทำให้ value_loss ระเบิด)
    if os.path.exists(vec_normalize_path) and not args.fresh:
        print(f"   📂 โหลดประวัติ VecNormalize จาก {vec_normalize_path}")
        vec_env = VecNormalize.load(vec_normalize_path, raw_vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        eval_env = VecNormalize.load(vec_normalize_path, raw_eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print(f"   🆕 สร้าง VecNormalize ใหม่")
        vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        eval_env = VecNormalize(raw_eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
    print(f"   ✅ VecNormalize: obs=True, reward=True, clip=10.0")

    # Load หรือสร้าง model
    if os.path.exists(model_path):
        try:
            print(f"📦 Continue training จาก {model_path}")
            model = PPO.load(model_path, env=vec_env)
            model.learning_rate = 3e-4
        except Exception as e:
            print(f"⚠️  โหลดไม่ได้ ({e}) — สร้างใหม่")
            model = None
    else:
        model = None

    if model is None:
        print("🆕 สร้าง PPO model ใหม่ (V4 — Stable Training จากผล TensorBoard)")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.97,
            gae_lambda=0.95,
            clip_range=0.15,
            clip_range_vf=0.2,
            ent_coef=0.005,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tb_log_dir,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 64], vf=[128, 64]),  # T4: ลดจาก [512,256,128] — ลด overfit + value_loss
            ),
        )

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_ftmo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(ROOT, "logs", "eval"),
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=50,   # T11: เพิ่มจาก 20 — eval ที่เชื่อถือได้มากขึ้น
        deterministic=True,
        render=False,
    )

    # T10: Early stopping เมื่อ value_loss ระเบิด
    early_stop_cb = EarlyStopOnValueLoss(threshold=10.0, patience=5, verbose=1)

    print(f"\n{'='*60}")
    print(f"🚀 เริ่ม training: {args.timesteps:,} timesteps × {args.n_envs} envs")
    print(f"   LR={model.learning_rate}, batch={model.batch_size}, "
          f"ent={model.ent_coef}, gamma={model.gamma}")
    print(f"   net_arch=pi:[128,64] vf:[128,64] | clip_range_vf=0.2")
    print(f"   norm_reward=True | early_stop: value_loss>10 × 5")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb, early_stop_cb],
        progress_bar=True,
        reset_num_timesteps=args.fresh or (not os.path.exists(model_path))
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"⏱️  Training เสร็จ — ใช้เวลา {elapsed/60:.1f} นาที")
    print(f"{'='*60}")

    model.save(model_path)
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    vec_env.save(vec_normalize_path)
    print(f"💾 Model → {model_path}")
    print(f"💾 VecNormalize → {vec_normalize_path}")

    # Sync normalization stats ไป eval env ก่อน final eval
    eval_env.obs_rms = vec_env.obs_rms
    eval_env.ret_rms = vec_env.ret_rms

    # Final eval — ผลสอบ FTMO
    print(f"\n{'='*60}")
    print(f"🏁 Final Evaluation (100 episodes)")
    print(f"{'='*60}")
    evaluate(model, n_episodes=100, vec_normalize_path=vec_normalize_path)


if __name__ == "__main__":
    main()
