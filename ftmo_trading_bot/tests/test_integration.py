"""End-to-end smoke test — รัน full episode + sanity check"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from ml.rl_environment import FTMOOptimizationEnv


def test_random_policy_full_episode_no_crash():
    """random agent run ครบ 30 วัน → ไม่ crash + reward เป็นตัวเลขถูก"""
    env = FTMOOptimizationEnv(max_steps=30)
    env.reset(seed=123)
    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        steps += 1
        assert np.isfinite(r), f"reward not finite at step {steps}: {r}"
        assert obs.shape == (13,)
        if term or trunc:
            break

    assert 1 <= steps <= 30
    assert isinstance(total_reward, float)


def test_ppo_can_instantiate_with_v2_env():
    """PPO ต้องสร้างได้กับ obs space 13 dims (V2)"""
    from stable_baselines3 import PPO
    env = FTMOOptimizationEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    # smoke: predict ครั้งเดียว
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (4,)


def test_ppo_short_train_run():
    """ฝึก 1024 steps สั้นๆ → ไม่ crash + model save/load ได้"""
    import tempfile, os
    from stable_baselines3 import PPO
    env = FTMOOptimizationEnv()
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256)
    model.learn(total_timesteps=1024)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_ppo.zip")
        model.save(path)
        loaded = PPO.load(path)
        obs, _ = env.reset(seed=0)
        action, _ = loaded.predict(obs)
        assert action.shape == (4,)
