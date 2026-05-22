"""
Audit Blocked Symbols — Before/After Re-Evaluation Tool
========================================================
Purpose:
  Re-run the post-deploy audit for AUDUSD/USDCAD/USDCHF (or any blocked set)
  to decide whether to unblock based on statistical significance.

Usage:
  .venv/bin/python scripts_local/audit_blocked_symbols.py [CUTOFF_DATE]

  CUTOFF_DATE: optional ISO date (default: 2026-05-22). Trades with Open Time
               >= CUTOFF are treated as NEW logic period.

Output:
  - Overall NEW vs OLD comparison (all symbols)
  - Per-blocked-symbol stats (WR, Net, EV, Realized R, PF)
  - Statistical tests (one-sample t-test, Welch t-test, Cohen's d, 95% CI)
  - Unblock recommendation per symbol (with gate criteria)

Unblock criteria (all must pass for a symbol):
  - n_new >= 15 trades
  - Net P/L > 0
  - 95% CI lower bound for Realized R > 0
  - Profit Factor >= 1.20

Author: Lead Quant Auditor
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scistats

ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "ftmo_trading_bot" / "logs" / "ftmo_trades.xlsx"

# Read blocked_symbols from config (single source of truth)
sys.path.insert(0, str(ROOT / "ftmo_trading_bot"))
try:
    from config.settings import bot_config
    BLOCKED = list(bot_config.symbols.blocked_symbols)
except Exception:
    BLOCKED = ["AUDUSD", "USDCAD", "USDCHF"]

# Gate criteria for unblock
GATE_MIN_N = 15
GATE_MIN_NET = 0.0
GATE_MIN_CI_LOWER = 0.0    # 95% CI lower bound for Realized R
GATE_MIN_PF = 1.20


def stats_block(df: pd.DataFrame, label: str) -> dict:
    """Compute summary stats for a trade subset."""
    n = len(df)
    if n == 0:
        print(f"  {label}: 0 trades")
        return {"n": 0}
    wins = (df["P/L ($)"] > 0).sum()
    wr = wins / n * 100
    net = df["P/L ($)"].sum()
    avg_w = df[df["P/L ($)"] > 0]["P/L ($)"].mean() if wins else 0
    avg_l = df[df["P/L ($)"] < 0]["P/L ($)"].mean() if (n - wins) else 0
    r = df["Realized_R"].mean()
    r_std = df["Realized_R"].std() if n > 1 else 0
    risk = df["Risk$"].mean()
    pf = (
        df[df["P/L ($)"] > 0]["P/L ($)"].sum()
        / abs(df[df["P/L ($)"] < 0]["P/L ($)"].sum())
    ) if (df["P/L ($)"] < 0).any() else float("inf")
    # 95% CI for mean realized R
    if n >= 2:
        ci_half = 1.96 * r_std / np.sqrt(n)
        ci_low, ci_high = r - ci_half, r + ci_half
    else:
        ci_low, ci_high = r, r
    print(f"  {label}: n={n}, WR={wr:.1f}%, Net=${net:.2f}")
    print(f"     Avg Win=${avg_w:.2f}, Avg Loss=${avg_l:.2f}, Avg Risk=${risk:.2f}")
    print(f"     Realized R={r:+.3f} (95% CI [{ci_low:+.3f}, {ci_high:+.3f}]), PF={pf:.2f}, EV=${net/n:.2f}/trade")
    return {
        "n": n, "wr": wr, "net": net, "avg_w": avg_w, "avg_l": avg_l,
        "r_mean": r, "r_std": r_std, "ci_low": ci_low, "ci_high": ci_high,
        "pf": pf, "ev": net / n,
    }


def main(cutoff_str: str = "2026-05-22"):
    print("=" * 76)
    print(f" AUDIT BLOCKED SYMBOLS — cutoff: {cutoff_str}")
    print(f" Blocked set: {BLOCKED}")
    print("=" * 76)

    df = pd.read_excel(XLSX_PATH, sheet_name="Trades")
    df = df[df["Close Time"].notna() & df["P/L ($)"].notna()].copy()
    df["P/L ($)"] = pd.to_numeric(df["P/L ($)"], errors="coerce")
    df = df[df["P/L ($)"].notna()].copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df["Realized_R"] = df["P/L ($)"] / df["Risk$"]

    cutoff = pd.Timestamp(cutoff_str)
    old_all = df[df["Open Time"] < cutoff].copy()
    new_all = df[df["Open Time"] >= cutoff].copy()
    print(f"\nOverall split:")
    print(f"  OLD (< {cutoff_str}): {len(old_all)} trades, Net ${old_all['P/L ($)'].sum():+.2f}, WR {(old_all['P/L ($)']>0).mean()*100:.1f}%")
    print(f"  NEW (>= {cutoff_str}): {len(new_all)} trades, Net ${new_all['P/L ($)'].sum():+.2f}, WR {(new_all['P/L ($)']>0).mean()*100:.1f}%")

    old_b = old_all[old_all["Symbol"].isin(BLOCKED)].copy()
    new_b = new_all[new_all["Symbol"].isin(BLOCKED)].copy()

    print(f"\n{'='*76}\n  BLOCKED SYMBOLS — COMBINED\n{'='*76}")
    stats_block(old_b, "OLD")
    stats_block(new_b, "NEW")

    # Statistical significance (combined)
    if len(new_b) >= 2 and len(old_b) >= 2:
        t1, p1 = scistats.ttest_1samp(new_b["Realized_R"], 0)
        t2, p2 = scistats.ttest_ind(old_b["Realized_R"], new_b["Realized_R"], equal_var=False)
        effect = abs((new_b["Realized_R"].mean() - old_b["Realized_R"].mean())
                     / np.sqrt((new_b["Realized_R"].var() + old_b["Realized_R"].var()) / 2))
        n_req = max(8, int(round(15.7 / effect ** 2))) if effect > 0 else 99999
        print(f"\n  Statistical tests (combined):")
        print(f"    One-sample t-test (NEW vs 0):  t={t1:.2f}, p={p1:.4f}  {'✅ sig' if p1 < 0.05 else '🟡 NOT sig'}")
        print(f"    Welch's t-test (OLD vs NEW):    t={t2:.2f}, p={p2:.4f}  {'✅ sig' if p2 < 0.05 else '🟡 NOT sig'}")
        print(f"    Cohen's d effect size:          {effect:.2f}")
        print(f"    Sample size needed per group:   ~{n_req}")

    # Per-symbol gate check
    print(f"\n{'='*76}\n  PER-SYMBOL UNBLOCK GATE\n{'='*76}")
    print(f"  Gate criteria: n>={GATE_MIN_N}, Net>${GATE_MIN_NET}, 95% CI lower R>{GATE_MIN_CI_LOWER}, PF>={GATE_MIN_PF}")
    print()
    recommendations = []
    for sym in BLOCKED:
        sub_new = new_b[new_b["Symbol"] == sym]
        sub_old = old_b[old_b["Symbol"] == sym]
        print(f"\n* {sym}:")
        s_old = stats_block(sub_old, "  OLD") if len(sub_old) else {"n": 0}
        s_new = stats_block(sub_new, "  NEW") if len(sub_new) else {"n": 0}

        # Gate check
        if s_new.get("n", 0) < GATE_MIN_N:
            verdict = "🔴 KEEP BLOCKED"
            reason = f"n={s_new.get('n', 0)} < {GATE_MIN_N} (need more data)"
        else:
            checks = []
            checks.append(("Net P/L > 0", s_new["net"] > GATE_MIN_NET))
            checks.append(("95% CI lower > 0", s_new["ci_low"] > GATE_MIN_CI_LOWER))
            checks.append(("PF >= 1.20", s_new["pf"] >= GATE_MIN_PF))
            passed = sum(1 for _, ok in checks if ok)
            if passed == 3:
                verdict = "🟢 UNBLOCK"
                reason = "All gates passed"
            else:
                verdict = "🔴 KEEP BLOCKED"
                reason = " / ".join(f"{name}: {'✓' if ok else '✗'}" for name, ok in checks)
        print(f"     → {verdict}  ({reason})")
        recommendations.append((sym, verdict))

    # Final config patch suggestion
    print(f"\n{'='*76}\n  CONFIG ACTION\n{'='*76}")
    keep = [s for s, v in recommendations if "KEEP" in v]
    unblock = [s for s, v in recommendations if "UNBLOCK" in v]
    if not unblock:
        print(f"  No change needed — keep blocked_symbols = {BLOCKED}")
    else:
        new_set = [s for s in BLOCKED if s not in unblock]
        print(f"  Suggested edit in config/settings.py:")
        print(f"    blocked_symbols: List[str] = field(default_factory=lambda: {new_set})")
        print(f"  → Unblock: {unblock}")
        print(f"  → Keep blocked: {keep}")
    print("=" * 76)


if __name__ == "__main__":
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2026-05-22"
    main(cutoff)
