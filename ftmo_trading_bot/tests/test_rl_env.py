"""ทดสอบ FTMOOptimizationEnv — observation/action space + episode loop"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from ml.rl_environment import FTMOOptimizationEnv


def test_obs_dim_is_13():
    env = FTMOOptimizationEnv()
    assert env.observation_space.shape == (13,)


def test_action_dim_is_4():
    env = FTMOOptimizationEnv()
    assert env.action_space.shape == (4,)


def test_reset_returns_valid_obs():
    env = FTMOOptimizationEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()


def test_step_returns_valid_obs_and_reward():
    env = FTMOOptimizationEnv()
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (13,)
    assert not np.isnan(obs).any()
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_full_episode_terminates():
    """run จนจบ episode → ต้องไม่ crash + ต้อง terminate ภายใน max_steps"""
    env = FTMOOptimizationEnv(max_steps=30)
    env.reset(seed=42)
    steps = 0
    while steps < 50:
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert terminated or truncated
    assert steps <= 30


def test_obs_within_clip_bounds():
    """obs ทุก dim ต้องอยู่ใน [-5, 5] ตาม observation_space"""
    env = FTMOOptimizationEnv()
    env.reset(seed=0)
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert (obs >= -5.0).all() and (obs <= 5.0).all(), f"obs out of bounds: {obs}"
        if term or trunc:
            break


def test_stats_expose_v2_fields():
    env = FTMOOptimizationEnv()
    env.reset()
    env.step(env.action_space.sample())
    stats = env._get_stats()
    for key in ['trading_days', 'consecutive_losses', 'is_final_step',
                'intraday_excursion_pct', 'day_end_loss_pct']:
        assert key in stats, f"missing stat: {key}"
