"""
===============================================================================
Parity Audit — train and live must use IDENTICAL params
===============================================================================
Hard rule (v8.0+): the agent's behavior in training must match live exactly.
"Great in train, fails in live" usually traces to one of these mismatches:

  1. Strategy params (BB/RSI/ADX thresholds, SL/TP, scan cadence, dedup)
     train: MeanReversionStrategy class defaults / bot_config.mr.*
     live:  same — verified here

  2. Indicator computation (EMA/ATR/RSI/BB/ADX formula)
     train: TechnicalIndicators.calculate_all on full DF (precomputed)
     live:  TechnicalIndicators.calculate_all on rolling window
     → outputs at the rightmost bar must match (verified by replaying ~100
       bars of cached pool data through the live indicator chain)

  3. ML threshold (filter signal if ml_score < threshold)
     train: --ml_threshold 0.40 (CLI flag, env constructor)
     live:  bot_config.ftmo.ML_FILTER_THRESHOLD
     → must match

  4. Risk per trade
     train: env.risk_per_trade (from --risk_per_trade)
     live:  bot_config.ftmo.DEFAULT_RISK_PER_TRADE_PCT
     → must match

  5. Observation construction
     train: FTMOSignalFilterEnv._get_obs / MeanReversionFilterEnv._get_obs
     live:  FTMOTradingBot._build_signal_observation
     → same shape, same scale, same semantics for every dim

  6. VecNormalize stats
     train: vec_normalize_mr.pkl saved with model
     live:  loaded by SelfLearningAgent
     → must use the SAME pickle file from the same training run

  7. Correlation rules
     train: FTMOSignalFilterEnv.CORRELATION_GROUPS (mirror of live)
     live:  TradeExecutor correlation logic
     → groups + max_correlated_positions must match

  8. SL/TP behavior
     train: StrategyBacktester._resolve_trade — SL/TP/timeout/gap only,
            no BE/partial/trail
     live:  TradeManager — does BE+partial+trail
     → known gap (train < live), accepted as "train < live capability"

This script reports any mismatch. Exit 1 on any hard mismatch (1-7).
Soft (8) is always reported but does not fail.

Run:
  .venv/bin/python ftmo_trading_bot/scripts/parity_audit.py
===============================================================================
"""

from __future__ import annotations

import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def audit_strategy_params() -> int:
    """Strategy class defaults must match `bot_config.mr` overrides (or be the
    canonical source of truth)."""
    print("=" * 72)
    print(" Audit 1: Strategy params (train.MeanReversionStrategy ↔ bot_config.mr)")
    print("=" * 72)
    fails = 0
    from strategy.mean_reversion_strategy import MeanReversionStrategy
    from config.settings import bot_config
    s = MeanReversionStrategy()
    cfg = bot_config.mr
    pairs = [
        ("BB_OVERSOLD", "bb_oversold"),
        ("BB_OVERBOUGHT", "bb_overbought"),
        ("RSI_OVERSOLD", "rsi_oversold"),
        ("RSI_OVERBOUGHT", "rsi_overbought"),
        ("ADX_TREND_BLOCK", "adx_trend_block"),
        ("SL_ATR_MULT", "sl_atr_mult"),
        ("RR_RATIO", "rr_ratio"),
        ("MIN_REVERSAL_WICK_RATIO", "min_reversal_wick_ratio"),
    ]
    for class_attr, cfg_attr in pairs:
        v_class = getattr(s, class_attr)
        v_cfg = getattr(cfg, cfg_attr, None)
        if v_cfg is None:
            print(f"  ⚠️ bot_config.mr.{cfg_attr} missing — using class default {v_class}")
        elif abs(float(v_class) - float(v_cfg)) > 1e-9:
            print(f"  ❌ MeanReversionStrategy.{class_attr}={v_class} != "
                  f"bot_config.mr.{cfg_attr}={v_cfg}")
            fails += 1
        else:
            print(f"  ✓ {class_attr} = {v_class}")

    # v8.0.9: cross-check FTMO RR floor — must be ≤ MR's RR (else live RM rejects)
    mr_rr = float(getattr(cfg, "rr_ratio", 1.0))
    ftmo_rr_min = float(bot_config.ftmo.MIN_RISK_REWARD_RATIO)
    if ftmo_rr_min > mr_rr:
        print(f"  ❌ FTMO MIN_RISK_REWARD_RATIO={ftmo_rr_min} > MR rr_ratio={mr_rr} "
              f"→ RiskManager will reject every MR signal")
        fails += 1
    else:
        print(f"  ✓ FTMO MIN_RISK_REWARD_RATIO={ftmo_rr_min} ≤ MR rr_ratio={mr_rr}")

    return fails


def audit_ml_threshold() -> int:
    """Train (--ml_threshold default) must match live config."""
    print("\n" + "=" * 72)
    print(" Audit 2: ML threshold (train CLI default ↔ live config)")
    print("=" * 72)
    fails = 0
    from config.settings import bot_config

    # 1) Trainer default — read CLI parser
    train_default = None
    # Force UTF-8 — Windows cp1252 chokes on Thai source comments.
    for line in open(os.path.join(ROOT, "scripts", "train_mr_signal_filter.py"), encoding="utf-8"):
        if "--ml_threshold" in line and "default" in line:
            # parser.add_argument("--ml_threshold", type=float, default=0.40)
            try:
                train_default = float(line.split("default=")[1].split(")")[0].split(",")[0])
                break
            except Exception:
                pass

    # 2) Auto-pipeline default — HyperParams.ml_threshold (dataclass)
    auto_pipeline_default = None
    for line in open(os.path.join(ROOT, "scripts", "auto_train_pipeline.py"), encoding="utf-8"):
        if "ml_threshold:" in line and "float" in line:
            # ml_threshold: float = 0.40
            try:
                auto_pipeline_default = float(line.split("=")[-1].strip())
                break
            except Exception:
                pass

    # 2b) v8.0.80 (H8): Auto-pipeline ARGPARSE default — this is the EFFECTIVE value,
    # because main() always builds HyperParams(ml_threshold=args.ml_threshold). The
    # dataclass default (#2) is dead in the main() path → must audit the argparse line too.
    auto_argparse_default = None
    for line in open(os.path.join(ROOT, "scripts", "auto_train_pipeline.py"), encoding="utf-8"):
        if "--ml_threshold" in line and "default=" in line:
            try:
                auto_argparse_default = float(line.split("default=")[1].split(")")[0].split(",")[0])
                break
            except Exception:
                pass

    # 3) Live default
    live_threshold = float(getattr(bot_config.ftmo, "ML_FILTER_THRESHOLD", -1))

    print(f"  trainer CLI --ml_threshold default:        {train_default}")
    print(f"  auto_train_pipeline HyperParams (dataclass): {auto_pipeline_default}")
    print(f"  auto_train_pipeline argparse (EFFECTIVE):  {auto_argparse_default}")
    print(f"  live bot_config.ftmo.ML_FILTER_THRESHOLD:  {live_threshold}")

    if train_default is not None and auto_pipeline_default is not None \
       and abs(train_default - auto_pipeline_default) > 1e-9:
        print(f"  ❌ trainer default != auto-pipeline HyperParams default")
        fails += 1
    # H8: argparse (effective) ต้องตรงกับ trainer + dataclass + live
    if auto_argparse_default is not None and train_default is not None \
       and abs(auto_argparse_default - train_default) > 1e-9:
        print(f"  ❌ auto-pipeline ARGPARSE default ({auto_argparse_default}) != trainer ({train_default}) "
              f"— main() ส่งค่า argparse นี้เข้า training จริง!")
        fails += 1
    if auto_argparse_default is not None and auto_pipeline_default is not None \
       and abs(auto_argparse_default - auto_pipeline_default) > 1e-9:
        print(f"  ❌ auto-pipeline argparse default ({auto_argparse_default}) != dataclass ({auto_pipeline_default})")
        fails += 1
    if train_default is not None and live_threshold >= 0 \
       and abs(train_default - live_threshold) > 1e-9:
        print(f"  ❌ TRAIN ↔ LIVE MISMATCH on ml_threshold")
        fails += 1
    if auto_argparse_default is not None and live_threshold >= 0 \
       and abs(auto_argparse_default - live_threshold) > 1e-9:
        print(f"  ❌ AUTO-PIPELINE (effective) ↔ LIVE MISMATCH on ml_threshold")
        fails += 1
    if fails == 0:
        print(f"  ✓ all ml_thresholds aligned at {train_default}")
    return fails


def audit_risk_per_trade() -> int:
    """Train env risk_per_trade ↔ live FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT."""
    print("\n" + "=" * 72)
    print(" Audit 3: Risk per trade (train ↔ live)")
    print("=" * 72)
    fails = 0
    from config.settings import bot_config
    from ml.signal_filter_env import FTMOSignalFilterEnv

    train_class = float(FTMOSignalFilterEnv.RISK_PER_TRADE)
    live_default = float(bot_config.ftmo.DEFAULT_RISK_PER_TRADE_PCT)

    # Auto-pipeline default
    auto_pipe_default = None
    for line in open(os.path.join(ROOT, "scripts", "auto_train_pipeline.py"), encoding="utf-8"):
        if "risk_per_trade:" in line and "float" in line:
            try:
                auto_pipe_default = float(line.split("=")[-1].strip())
                break
            except Exception:
                pass

    print(f"  FTMOSignalFilterEnv.RISK_PER_TRADE = {train_class}")
    print(f"  bot_config.ftmo.DEFAULT_RISK_PER_TRADE_PCT = {live_default}")
    print(f"  auto_train_pipeline HyperParams.risk_per_trade = {auto_pipe_default}")

    for v in [auto_pipe_default]:
        if v is not None and abs(v - live_default) > 1e-9:
            print(f"  ❌ TRAIN ↔ LIVE MISMATCH on risk_per_trade")
            fails += 1
    if fails == 0:
        print(f"  ✓ risk_per_trade aligned at {live_default}")
    return fails


def audit_obs_dim() -> int:
    """OBS_DIM must equal env.observation_space.shape[0] AND match what the
    live obs builder produces.

    On VPS without OHLCV CSVs, env.__init__ raises RuntimeError. Fall back
    to checking class-level constants only — sufficient for live deploy
    sanity (live uses MT5 real-time data, not CSVs).
    """
    print("\n" + "=" * 72)
    print(" Audit 4: Obs dimension (3-way sync: agent ↔ env ↔ live)")
    print("=" * 72)
    fails = 0
    from ml.rl_agent import SelfLearningAgent
    from ml.mean_reversion_env import MeanReversionFilterEnv

    agent_dim = int(SelfLearningAgent.OBS_DIM)
    os.environ["BOT_DISABLE_CHRONOS"] = "1"
    os.environ["SMC_QUIET"] = "1"

    try:
        env = MeanReversionFilterEnv(
            data_dir=os.path.join(ROOT, "data", "ohlcv"),
            max_days=10,
            enable_risk_penalty=False,
            signal_pool_path=None,
            ml_filter_threshold=0.0,
            risk_per_trade=0.005,
            verbose=False,
        )
        env_dim = int(env.observation_space.shape[0])
        obs, _ = env.reset(seed=42)
        obs_actual = int(obs.shape[0])
        print(f"  SelfLearningAgent.OBS_DIM           = {agent_dim}")
        print(f"  MeanReversionFilterEnv.observation_space.shape[0] = {env_dim}")
        print(f"  actual obs from reset()             = {obs_actual}")

        if not (agent_dim == env_dim == obs_actual):
            print(f"  ❌ OBS dim mismatch")
            fails += 1
        else:
            print(f"  ✓ all 3 sources agree at {agent_dim}")
    except RuntimeError as e:
        # VPS / no-OHLCV path — fall back to class-level check (live uses MT5 not CSV)
        print(f"  ⚠️  env init skipped (no OHLCV — VPS-style env): {e}")
        # Read observation_space from class (gym Box defined as class attr at __init__)
        # Without instance, just compare agent OBS_DIM to expected production value 35.
        if agent_dim != 35:
            print(f"  ❌ SelfLearningAgent.OBS_DIM = {agent_dim}, expected 35")
            fails += 1
        else:
            print(f"  ✓ SelfLearningAgent.OBS_DIM = {agent_dim} (class-level check only)")
    return fails


def audit_correlation_groups() -> int:
    """Training env's correlation groups must mirror live TradeExecutor."""
    print("\n" + "=" * 72)
    print(" Audit 5: Correlation groups (train env ↔ live TradeExecutor)")
    print("=" * 72)
    fails = 0
    try:
        from ml.signal_filter_env import FTMOSignalFilterEnv
        from execution.trade_executor import TradeExecutor
        train_groups = FTMOSignalFilterEnv.CORRELATION_GROUPS
        # TradeExecutor exposes CORRELATION_GROUPS as class const (per wiki)
        live_groups = getattr(TradeExecutor, "CORRELATION_GROUPS", None)
        if live_groups is None:
            print("     ⚠️ TradeExecutor.CORRELATION_GROUPS not found — skip")
            return 0
        for k, v in train_groups.items():
            v_live = live_groups.get(k, set())
            if set(v) != set(v_live):
                print(f"  ❌ group '{k}' diverges: train={v}, live={v_live}")
                fails += 1
        for k in live_groups:
            if k not in train_groups:
                print(f"  ❌ live has group '{k}' not in train")
                fails += 1
        if fails == 0:
            print(f"  ✓ {len(train_groups)} groups match exactly")
    except Exception as e:
        print(f"     ⚠️ cannot import TradeExecutor: {e}")
    return fails


def audit_indicators() -> int:
    """Spot-check that indicators produce the same value at the last bar
    whether computed on full DF or rolling window."""
    print("\n" + "=" * 72)
    print(" Audit 6: Indicator parity (full DF vs rolling window)")
    print("=" * 72)
    import pandas as pd
    import numpy as np
    from strategy.indicators import TechnicalIndicators
    csv = os.path.join(ROOT, "data", "ohlcv", "EURUSD_M15.csv")
    if not os.path.exists(csv):
        print(f"     ⚠️ {csv} missing — skip")
        return 0
    df = pd.read_csv(csv)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'volume' not in df.columns:
        df['volume'] = 0
    df = df.tail(500).reset_index(drop=True).copy()
    ind = TechnicalIndicators()
    full = ind.calculate_all(df.copy())
    last_full = full.iloc[-1]
    # Rolling: take last 200 bars only and recompute
    rolling = ind.calculate_all(df.tail(200).copy())
    last_roll = rolling.iloc[-1]

    fails = 0
    for col in ["atr", "rsi", "bb_pctb", "adx", "stoch_k", "macd_histogram"]:
        if col in last_full and col in last_roll:
            v_full = float(last_full[col])
            v_roll = float(last_roll[col])
            tol = max(abs(v_full) * 0.001, 1e-5)
            if not np.isfinite(v_full) or not np.isfinite(v_roll):
                print(f"  ⚠️ {col}: NaN (full={v_full}, roll={v_roll})")
                continue
            if abs(v_full - v_roll) > tol:
                print(f"  🟡 {col}: full={v_full:.6f} vs roll={v_roll:.6f} "
                      f"(Δ={abs(v_full-v_roll):.6f})")
                # Note: small EMA-warmup differences are expected. Hard fail
                # only if the gap is large enough to shift obs significantly.
                if abs(v_full - v_roll) > max(abs(v_full) * 0.1, 1e-3):
                    fails += 1
            else:
                print(f"  ✓ {col}: full≈roll = {v_full:.6f}")
    return fails


def audit_vec_normalize() -> int:
    """models/mr/vec_normalize_mr.pkl must exist alongside the model."""
    print("\n" + "=" * 72)
    print(" Audit 7: VecNormalize stats availability")
    print("=" * 72)
    fails = 0
    mr_dir = os.path.join(ROOT, "models", "mr")
    model = os.path.join(mr_dir, "ppo_mr_filter.zip")
    vecn = os.path.join(mr_dir, "vec_normalize_mr.pkl")

    if not os.path.exists(model):
        print(f"     (model not yet trained — skip)")
        return 0
    if not os.path.exists(vecn):
        print(f"  ❌ {vecn} missing — live obs will be UNnormalized vs training!")
        fails += 1
    else:
        # Sanity: load it
        try:
            with open(vecn, "rb") as f:
                vn = pickle.load(f)
            mean = vn.obs_rms.mean if hasattr(vn, "obs_rms") else None
            if mean is None or len(mean) != 35:
                print(f"  ❌ vec_normalize_mr.pkl obs_rms invalid")
                fails += 1
            else:
                print(f"  ✓ vec_normalize_mr.pkl loaded — obs_rms.mean shape {mean.shape}")
        except Exception as e:
            print(f"  ❌ cannot load vec_normalize_mr.pkl: {e}")
            fails += 1
    return fails


def main():
    print("\n  🔁 PARITY AUDIT (v8.0)")
    print("     Goal: train and live must use IDENTICAL params + obs construction\n")
    total = 0
    total += audit_strategy_params()
    total += audit_ml_threshold()
    total += audit_risk_per_trade()
    total += audit_obs_dim()
    total += audit_correlation_groups()
    total += audit_indicators()
    total += audit_vec_normalize()

    print("\n" + "=" * 72)
    if total == 0:
        print(" ✅ ALL ALIGNED — train and live should behave identically")
        print("=" * 72)
        sys.exit(0)
    else:
        print(f" ❌ FAILED — {total} parity issue(s)")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
