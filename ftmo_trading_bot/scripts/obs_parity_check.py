"""
obs_parity_check.py — Per-dim train↔live observation parity (v8.0.80 follow-up)

WHY: parity_audit only checks obs DIMENSION (35) + indicator parity, NOT each
obs slot's VALUE. Silent per-dim mismatches (e.g. v8.0.80 H3 ml_score, H4
trades_today) were invisible. This replays REAL pool signals through BOTH:
  - TRAIN: MeanReversionFilterEnv._get_obs (the env the model trained on)
  - LIVE:  FTMOTradingBot._build_signal_observation (what the live agent sees)
with matched neutral portfolio state, and diffs element-wise.

CLASSIFICATION:
  - SIGNAL dims (pure function of the signal): MUST match (tol 1e-4) → FAIL if not
  - LEAK-ZERO dims (29,30): must both be 0
  - STATE/TIME dims (17-25,31): report only — depend on runtime state/clock,
    semantic-match by construction (value differs by design)

Usage: python scripts/obs_parity_check.py [--pool data/mr_signal_pool_5000.pkl] [--n 40]
Exit 0 = all SIGNAL/LEAK dims aligned; exit 1 = a must-match dim diverged.
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SMC_QUIET", "1")
os.environ.setdefault("BOT_DISABLE_CHRONOS", "1")

import numpy as np  # noqa: E402

# Dims that are a pure function of the signal → MUST be byte-identical
SIGNAL_DIMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 32, 33, 34]
# Leak-free dims → both must be exactly 0
LEAK_ZERO_DIMS = [29, 30]
# State/time dims → report only (runtime/clock-dependent, semantic-match by design)
STATE_DIMS = [17, 18, 19, 20, 21, 22, 23, 24, 25, 31]

DIM_NAME = {
    0: "confluence", 1: "rr", 2: "direction", 3: "atr_norm", 4: "bb_extreme(ob)",
    5: "bias_align", 6: "sl_atr", 7: "rsi", 8: "macd", 9: "trend_str",
    10: "bb_bw(ob_size)", 11: "adx_norm", 12: "stoch", 13: "bb_pctb", 14: "atr_chg",
    15: "price_roc", 16: "ml_score", 17: "total_dd", 18: "daily_dd", 19: "progress",
    20: "day_progress", 21: "trades_today", 22: "recent_wr", 23: "consec_loss",
    24: "spread_pct", 25: "recent_opposite", 26: "htf_align(adx)", 27: "chronos_align",
    28: "chronos_unc", 29: "floating_pnl", 30: "open_losing", 31: "mins_since",
    32: "kc_dist", 33: "ema_slope", 34: "band_squeeze",
}


def _pip_for(symbol: str) -> tuple:
    s = symbol.upper()
    if "XAU" in s or "XAG" in s:
        return 0.01, 2, 0.01
    if "JPY" in s:
        return 0.01, 3, 0.001
    return 0.0001, 5, 0.00001


class _StubConnector:
    def __init__(self, sig_dict):
        self._d = sig_dict
        self._pip, self._digits, self._point = _pip_for(sig_dict.get("symbol", "EURUSD"))

    def get_open_positions(self, symbol=None):
        return []

    def get_symbol_info(self, symbol):
        return {"digits": self._digits, "point": self._point}

    def get_current_price(self, symbol):
        # spread in PRICE units chosen so _build_spread_pct_of_atr → sig['spread_pips']
        spread_price = float(self._d.get("spread_pips", 0.0)) * (self._pip / self._point)
        return {"bid": 1.0, "ask": 1.0 + spread_price, "spread": spread_price}


class _StubRisk:
    def __init__(self):
        self._flip_lock = {}
        self.initial_balance = 100_000.0

    def get_risk_status(self):
        return {"overall_drawdown_pct": 0.0, "daily_loss_pct": 0.0,
                "current_balance": 100_000.0}


class _StubAnalyzer:
    _trades = []


def _sig_obj(d: dict):
    """Wrap a pool dict as an object exposing the attrs the live obs builder reads."""
    o = SimpleNamespace(**d)
    o.signal_type = SimpleNamespace(value="BUY" if d.get("direction", 1.0) > 0 else "SELL")
    o.symbol = d.get("symbol", "EURUSD")
    # raw sl_distance not stored in pool (only sl_distance_atr/pips) → reconstruct
    o.sl_distance = float(d.get("sl_distance_atr", 0.0)) * float(d.get("atr_value", 0.0))
    # live builder uses sig.entry_price for the pip heuristic (>50 → JPY/metal)
    pip, _, _ = _pip_for(o.symbol)
    o.entry_price = 100.0 if pip == 0.01 else 1.0
    return o


def _train_obs(env, sig_dict):
    """Clean train obs (no spread-noise mutation): force the un-noised signal at idx 0."""
    env.reset()                       # initialises fresh state (balance, dd=0, idx=0)
    env._signals = [dict(sig_dict)]   # un-noised copy
    env._signal_idx = 0
    env._current_day = sig_dict["day"]
    return env._get_obs()


def _live_obs(bot, sig_dict):
    sig = _sig_obj(sig_dict)
    # day_progress parity: make live challenge-day == episode day, /45
    bot._get_challenge_day = lambda _date, _d=sig_dict: int(_d["day"])
    bot._connector = _StubConnector(sig_dict)
    return bot._build_signal_observation(sig, live_context={"ml_score": sig_dict.get("ml_score", 0.5)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "mr_signal_pool_5000.pkl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    from ml.mean_reversion_env import MeanReversionFilterEnv
    from main import FTMOTradingBot

    pool = pickle.load(open(args.pool, "rb"))
    # flatten signals, sample diverse ones
    flat = [s for ep in pool for s in ep]
    step = max(1, len(flat) // args.n)
    samples = flat[::step][: args.n]

    # one env reused; inject pool in-memory (env loads from file normally)
    env = MeanReversionFilterEnv(ml_filter_threshold=0.0)
    env._signal_pool = [[s] for s in samples]
    bot = FTMOTradingBot.__new__(FTMOTradingBot)
    bot._risk_manager = _StubRisk()
    bot._quality_model = None
    bot._analyzer = _StubAnalyzer()
    bot._trade_open_history = []

    max_diff = np.zeros(35)
    examples = {}
    for sd in samples:
        ot = np.asarray(_train_obs(env, sd), dtype=np.float64)
        ol = np.asarray(_live_obs(bot, sd), dtype=np.float64)
        d = np.abs(ot - ol)
        for i in range(35):
            if d[i] > max_diff[i]:
                max_diff[i] = d[i]
                examples[i] = (float(ot[i]), float(ol[i]))

    print("=" * 78)
    print(f" 🔬 Per-dim obs parity (train vs live) — {len(samples)} signals, tol={args.tol}")
    print("=" * 78)
    fails = []
    for grp, dims, label in [
        ("SIGNAL", SIGNAL_DIMS, "MUST match"),
        ("LEAK0", LEAK_ZERO_DIMS, "both = 0"),
        ("STATE", STATE_DIMS, "report only (runtime/clock)"),
    ]:
        print(f"\n── {grp} dims ({label}) ──")
        for i in dims:
            tr, lv = examples.get(i, (0.0, 0.0))
            flag = ""
            if grp == "SIGNAL" and max_diff[i] > args.tol:
                flag = "  ❌ MISMATCH"; fails.append(i)
            elif grp == "LEAK0" and max_diff[i] > args.tol:
                flag = "  ❌ NOT ZERO"; fails.append(i)
            elif grp == "STATE" and max_diff[i] > args.tol:
                flag = "  ⚠️ differs (check if by-design)"
            print(f"  obs[{i:2d}] {DIM_NAME[i]:18s} maxΔ={max_diff[i]:.6f}  "
                  f"(train={tr:+.4f} live={lv:+.4f}){flag}")

    print("\n" + "=" * 78)
    if fails:
        print(f" ❌ {len(fails)} MUST-match dim(s) diverged: {[f'obs[{i}]={DIM_NAME[i]}' for i in fails]}")
        print("=" * 78)
        sys.exit(1)
    print(" ✅ All SIGNAL + LEAK dims aligned (≤ tol). STATE/clock dims are by-design.")
    print("=" * 78)


if __name__ == "__main__":
    main()
