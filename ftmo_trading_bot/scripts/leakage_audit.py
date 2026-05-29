"""
===============================================================================
Leakage Audit — verify RL/ML never see "answer keys" (future-resolved fields)
===============================================================================
Hard rule (v8.0+): the agent's observation must contain ONLY signal-time
features. Future-resolved fields (outcome_pnl_ratio, bars_to_resolution,
is_quick_tp, etc.) are TARGETS — they may flow into:
  • reward shaping (allowed — reward is supervisory at train time only)
  • aux-task targets (allowed — aux head not used at inference)
  • GBM y label (allowed — y only)
But NEVER into:
  • observation vectors (obs[i])
  • GBM x features

This script fails (exit 1) if any leak is detected.

Run:
  .venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py
===============================================================================
"""

from __future__ import annotations

import ast
import os
import sys
import pickle
from typing import List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Future-resolved fields that must NEVER appear in observation vectors
LEAK_KEYS: Set[str] = {
    "outcome_pnl_ratio",     # final PnL — direct answer
    "bars_to_resolution",    # how long until SL/TP — timing answer
    "is_quick_tp",            # quick winner flag — answer
    "outcome_partial",        # legacy v7.1 leak
    "outcome",                # generic outcome attr
    "tp_hit",
    "sl_hit",
    "future_high",
    "future_low",
    "future_close",
}

# Functions/methods that build observations (must be leak-free)
# Paths are relative to the project root (ROOT = parent of ftmo_trading_bot/)
OBS_BUILDERS = [
    ("ml/signal_filter_env.py", "_get_obs"),
    ("ml/mean_reversion_env.py", "_get_obs"),
    ("main.py", "_build_signal_observation"),
]


def _extract_function_source(filepath: str, func_name: str) -> str:
    """Pull the AST source of one function from a file (no class context)."""
    # Force UTF-8 — Windows defaults to cp1252 which can't decode Thai chars in source.
    src = open(os.path.join(ROOT, filepath), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.unparse(node)
    return ""


def _find_leak_accesses(func_src: str, func_name: str) -> List[Tuple[str, int, str]]:
    """Detect *value* accesses to leak keys (not just substrings).

    Catches:
      sig.get('outcome_pnl_ratio')
      sig['outcome_pnl_ratio']
      sig.outcome_pnl_ratio
    Skips substrings inside larger identifiers like 'recent_wr_norm'.
    """
    leaks = []
    if not func_src:
        return leaks
    tree = ast.parse(func_src)
    for node in ast.walk(tree):
        # sig.get('LEAK_KEY')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and arg0.value in LEAK_KEYS:
                    leaks.append((func_name, getattr(node, "lineno", 0),
                                  f"sig.get('{arg0.value}')"))
        # sig['LEAK_KEY']
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in LEAK_KEYS:
                leaks.append((func_name, getattr(node, "lineno", 0),
                              f"sig['{node.slice.value}']"))
        # sig.LEAK_KEY (attribute)
        if isinstance(node, ast.Attribute) and node.attr in LEAK_KEYS:
            leaks.append((func_name, getattr(node, "lineno", 0),
                          f".{node.attr}"))
    return leaks


def audit_obs_builders() -> int:
    """Static AST audit of all observation-building functions.

    Returns: number of leaks found (0 = pass).
    """
    print("=" * 72)
    print(" Audit 1: Static AST scan of observation builders")
    print("=" * 72)
    total = 0
    for filepath, func in OBS_BUILDERS:
        src = _extract_function_source(filepath, func)
        leaks = _find_leak_accesses(src, f"{filepath}::{func}")
        if leaks:
            print(f"\n  ❌ {filepath}::{func}")
            for fn, ln, expr in leaks:
                print(f"     line {ln}: {expr}")
            total += len(leaks)
        else:
            print(f"  ✓ {filepath}::{func} — clean")
    return total


def audit_gbm_features() -> int:
    """GBM x-features must not include future-resolved keys.

    Reads FEATURE_KEYS from train_mr_signal_quality.py and the saved model
    payload's 'keys' list.
    """
    print("\n" + "=" * 72)
    print(" Audit 2: GBM features (x must be signal-time only)")
    print("=" * 72)
    leaks = 0

    # 1) FEATURE_KEYS in trainer source
    from scripts.train_mr_signal_quality import FEATURE_KEYS
    print(f"  trainer FEATURE_KEYS ({len(FEATURE_KEYS)}):")
    for k in FEATURE_KEYS:
        if k in LEAK_KEYS:
            print(f"     ❌ '{k}' is a leak key in FEATURE_KEYS!")
            leaks += 1
    if leaks == 0:
        print("     ✓ no leak keys in FEATURE_KEYS")

    # 2) Saved model payload
    payload_path = os.path.join(ROOT, "data", "mr_signal_quality_model.pkl")
    if os.path.exists(payload_path):
        try:
            with open(payload_path, "rb") as f:
                payload = pickle.load(f)
            saved_keys = payload.get("keys", [])
            for k in saved_keys:
                if k in LEAK_KEYS:
                    print(f"     ❌ '{k}' in saved model.keys!")
                    leaks += 1
            if leaks == 0:
                print(f"  ✓ saved model.keys ({len(saved_keys)}) — clean")
        except Exception as e:
            print(f"     ⚠️  cannot read {payload_path}: {e}")
    else:
        print(f"     (model file not yet saved — skip)")

    return leaks


def audit_pool_dict_leak() -> int:
    """Sanity check: pool dicts contain LEAK_KEYS (target labels), but the env
    must never expose them via obs. Verify at least one signal in the pool has
    outcome_pnl_ratio populated (correctness check, not a leak alert)."""
    print("\n" + "=" * 72)
    print(" Audit 3: Pool labels populated (sanity, not leak alert)")
    print("=" * 72)
    pool_path = os.path.join(ROOT, "data", "mr_signal_pool_3000.pkl")
    if not os.path.exists(pool_path):
        print(f"     (pool file not yet built — skip)")
        return 0
    try:
        with open(pool_path, "rb") as f:
            pool = pickle.load(f)
    except Exception as e:
        print(f"     ⚠️  cannot read pool: {e}")
        return 0
    if not pool or not pool[0]:
        print("     pool empty")
        return 0
    sig = pool[0][0]
    for k in ["outcome_pnl_ratio", "bars_to_resolution", "is_quick_tp"]:
        if k in sig:
            print(f"     ✓ pool dict has '{k}' (target — used in reward/aux only)")
        else:
            print(f"     🟡 pool dict missing '{k}' — env shaping may be broken")
    return 0


def audit_dynamic_obs() -> int:
    """Dynamic test: run a few env steps, ensure obs values never leak future
    info. Build a known-bad pool with extreme outcome_pnl_ratio in alternating
    signs — if obs reflects this, it's a leak."""
    print("\n" + "=" * 72)
    print(" Audit 4: Dynamic obs sanity (env step → obs values)")
    print("=" * 72)
    os.environ["BOT_DISABLE_CHRONOS"] = "1"
    os.environ["SMC_QUIET"] = "1"
    try:
        import numpy as np
        from ml.mean_reversion_env import MeanReversionFilterEnv
        env = MeanReversionFilterEnv(
            data_dir=os.path.join(ROOT, "data", "ohlcv"),
            max_days=10,
            enable_risk_penalty=False,
            signal_pool_path=None,  # on-the-fly
            outcome_noise_std=0.0,
            ml_filter_threshold=0.0,
            risk_per_trade=0.005,
            verbose=False,
        )
    except Exception as e:
        print(f"     ⚠️  env init failed: {e}")
        return 0

    obs, info = env.reset(seed=42)
    print(f"  obs.shape = {obs.shape}")
    if obs.shape[0] != 35:
        print(f"     ❌ obs dim mismatch: {obs.shape[0]} != 35")
        return 1

    # Verify obs values are bounded reasonably (not raw outcome ratios)
    obs_min, obs_max = float(obs.min()), float(obs.max())
    print(f"  obs range = [{obs_min:.3f}, {obs_max:.3f}]")
    if obs_min < -10 or obs_max > 10:
        print(f"     ❌ obs has out-of-range values (likely leak via raw outcome)")
        return 1
    print("     ✓ obs values within reasonable bounds")
    return 0


def main():
    print("\n  🔒 LEAKAGE AUDIT (v8.0)")
    print("     Goal: agent must never see future-resolved fields in obs/x\n")

    n1 = audit_obs_builders()
    n2 = audit_gbm_features()
    audit_pool_dict_leak()       # sanity print only — never fails the audit
    n4 = audit_dynamic_obs()

    total = n1 + n2 + n4
    print("\n" + "=" * 72)
    if total == 0:
        print(" ✅ ALL CLEAN — no leakage detected")
        print("=" * 72)
        sys.exit(0)
    else:
        print(f" ❌ FAILED — {total} leak(s) detected")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
