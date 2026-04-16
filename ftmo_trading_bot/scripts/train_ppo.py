"""
===============================================================================
Train PPO Agent for FTMO Challenge (V2 — 13 obs dims)
===============================================================================
ฝึก PPO agent ด้วย FTMOOptimizationEnv ใหม่ (V2 reward + obs)
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from ml.rl_environment import FTMOOptimizationEnv


def make_env(data_dir=None):
    return FTMOOptimizationEnv(data_dir=data_dir)


def evaluate(model, n_episodes: int = 100):
    """รัน episode แล้วรายงาน pass-rate (ไม่ breach + ถึง 10%)"""
    env = make_env()
    passes, breaches, no_target = 0, 0, 0
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)

        stats = env._get_stats()
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
        evaluate(model, n_episodes=100)
        return

    # Vectorized envs (เร่ง training + ลด correlation)
    vec_env = make_vec_env(lambda: make_env(args.data_dir), n_envs=args.n_envs)
    eval_env = make_vec_env(lambda: make_env(args.data_dir), n_envs=1)

    # Load หรือสร้าง model
    if os.path.exists(model_path):
        try:
            print(f"📦 Continue training จาก {model_path}")
            model = PPO.load(model_path, env=vec_env)
        except Exception as e:
            print(f"⚠️  โหลดไม่ได้ ({e}) — สร้างใหม่")
            model = None
    else:
        model = None

    if model is None:
        print("🆕 สร้าง PPO model ใหม่")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,  # ลดจาก 3e-4 — ป้องกัน policy diverge (std ระเบิด)
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,         # FTMO เป็น long-horizon
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,   # clip value loss — กัน value function ระเบิด
            ent_coef=0.001,      # ต่ำมาก — กัน premature convergence แต่ไม่ diverge
            verbose=1,
            tensorboard_log=tb_log_dir,
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
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    print(f"🚀 เริ่ม training: {args.timesteps:,} timesteps × {args.n_envs} envs")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n⏱️  ใช้เวลา {elapsed/60:.1f} นาที")

    model.save(model_path)
    print(f"💾 บันทึก model → {model_path}")

    # Final eval
    evaluate(model, n_episodes=100)


if __name__ == "__main__":
    main()
