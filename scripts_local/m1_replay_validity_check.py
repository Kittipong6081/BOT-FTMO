"""
M1 OHLCV Replay Validity Check (Plan D)
========================================
Purpose:
  Eliminate counterfactual bias in trade replay by walking REAL M1 high/low bars
  for each closed trade — replaces the unreliable polled MFE column from the bot's
  5-second scan loop.

Inputs:
  - ftmo_trading_bot/logs/ftmo_trades.xlsx (Trades sheet)
  - ftmo_trading_bot/data/ohlcv/{Symbol}_M1.csv

Outputs (stdout):
  1. Data coverage report
  2. TRUE MFE distribution + bias vs polled MFE
  3. Per-symbol continuation rate (% of TP-hits that reached 1.5R / 2.0R)
  4. Two-scenario comparison (Phase 1 only vs Phase 1+2 Full), -$10 spread/trade
  5. Final deployment gate decision (≥ 40% continuation = deploy RR 1.5)

Author: Lead Quant Researcher
Date: 2026-05-23
"""
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "ftmo_trading_bot" / "logs" / "ftmo_trades.xlsx"
OHLCV_DIR = ROOT / "ftmo_trading_bot" / "data" / "ohlcv"

BLOCKED_SYMBOLS = {"AUDUSD", "USDCAD", "USDCHF"}
NEW_RISK_USD = 70.0    # 0.7% × $10k FTMO
SPREAD_COST_USD = 10.0 # Fixed -$10/trade (per user spec)
GATE_CONTINUATION_PCT = 40.0  # ≥40% continuation rate → deploy RR 1.5

# Pip size lookup
def pip_size(symbol: str) -> float:
    if "JPY" in symbol or "XAU" in symbol:
        return 0.01
    return 0.0001


# ---------------------------------------------------------------------------
# M1 data loader (cached per symbol)
# ---------------------------------------------------------------------------
_M1_CACHE: dict[str, pd.DataFrame] = {}

def load_m1(symbol: str) -> pd.DataFrame | None:
    if symbol in _M1_CACHE:
        return _M1_CACHE[symbol]
    path = OHLCV_DIR / f"{symbol}_M1.csv"
    if not path.exists():
        _M1_CACHE[symbol] = None
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    _M1_CACHE[symbol] = df
    return df


# ---------------------------------------------------------------------------
# Compute TRUE MFE for one trade
# ---------------------------------------------------------------------------
def compute_true_mfe_r(row) -> dict:
    """Return dict with TRUE_MFE_R + reached_X flags.
    Returns {} if M1 data missing or trade outside coverage.
    """
    sym = row["Symbol"]
    m1 = load_m1(sym)
    if m1 is None:
        return {"status": "no_m1_csv"}

    open_t = pd.to_datetime(row["Open Time"])
    close_t = pd.to_datetime(row["Close Time"])

    # Filter bars in window
    bars = m1[(m1["time"] >= open_t.floor("min")) & (m1["time"] <= close_t.ceil("min"))]
    if bars.empty:
        return {"status": "no_bars_in_window"}

    entry = float(row["Entry"])
    sl = float(row["SL"])
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return {"status": "zero_sl_distance"}

    trade_type = str(row["Type"]).upper()
    if trade_type == "BUY":
        peak = bars["high"].max()
        true_mfe_dist = peak - entry
    elif trade_type == "SELL":
        trough = bars["low"].min()
        true_mfe_dist = entry - trough
    else:
        return {"status": f"unknown_type_{trade_type}"}

    true_mfe_r = max(true_mfe_dist / sl_dist, 0.0)
    return {
        "status": "ok",
        "n_bars": len(bars),
        "TRUE_MFE_R": true_mfe_r,
        "reached_0.3R": true_mfe_r >= 0.3,
        "reached_0.8R": true_mfe_r >= 0.8,
        "reached_1.0R": true_mfe_r >= 1.0,
        "reached_1.5R": true_mfe_r >= 1.5,
        "reached_2.0R": true_mfe_r >= 2.0,
    }


# ---------------------------------------------------------------------------
# Scenario logic — outcome in R units (deterministic from TRUE_MFE_R)
# ---------------------------------------------------------------------------
def outcome_phase1_only(true_mfe_r: float) -> float:
    """Phase 1 Only: BE@0.3R, Partial 33% @ 0.8R, TP @ 1.0R"""
    if true_mfe_r < 0.3:
        return -1.0                              # SL hit, no BE
    elif true_mfe_r < 0.8:
        return 0.0                               # BE save
    elif true_mfe_r < 1.0:
        return 0.33 * 0.8 + 0.67 * 0.0           # Partial @0.8R + 67% revert to BE
    else:
        return 0.33 * 0.8 + 0.67 * 1.0           # Partial + 67% hit TP @1.0R = +0.934R

def outcome_phase1_plus_2(true_mfe_r: float) -> float:
    """Phase 1+2 Full: BE@0.3R, Partial 33% @ 1.0R, TP @ 1.5R"""
    if true_mfe_r < 0.3:
        return -1.0                              # SL hit
    elif true_mfe_r < 1.0:
        return 0.0                               # BE save (partial trigger moved to 1.0R)
    elif true_mfe_r < 1.5:
        return 0.33 * 1.0 + 0.67 * 0.0           # Partial @1.0R + 67% revert to BE = +0.33R
    else:
        return 0.33 * 1.0 + 0.67 * 1.5           # Partial + 67% hit TP @1.5R = +1.335R


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" M1 OHLCV REPLAY VALIDITY CHECK — Plan D")
    print("=" * 80)

    # Load trades
    df = pd.read_excel(XLSX_PATH, sheet_name="Trades")
    df = df[df["Close Time"].notna() & df["P/L ($)"].notna()].copy()
    df["P/L ($)"] = pd.to_numeric(df["P/L ($)"], errors="coerce")
    df = df[df["P/L ($)"].notna()].copy()
    df = df.reset_index(drop=True)
    n_total = len(df)
    print(f"\n📂 Loaded {n_total} closed trades from {XLSX_PATH.name}")

    # Compute TRUE MFE per trade
    results = df.apply(compute_true_mfe_r, axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)

    # Coverage report
    print("\n" + "=" * 80)
    print(" [1] DATA COVERAGE")
    print("=" * 80)
    print(df["status"].value_counts().to_string())
    df_ok = df[df["status"] == "ok"].copy()
    print(f"\n✅ Usable trades: {len(df_ok)} / {n_total} ({len(df_ok)/n_total*100:.1f}%)")

    # Coverage per symbol
    print("\nPer-symbol coverage:")
    cov = df.groupby("Symbol").agg(
        total=("Symbol", "count"),
        ok=("status", lambda x: (x == "ok").sum())
    )
    cov["pct"] = (cov["ok"] / cov["total"] * 100).round(1)
    print(cov.to_string())

    # ---------------------------------------------------------------------------
    # [2] TRUE MFE distribution + bias vs polled
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [2] TRUE MFE vs POLLED MFE BIAS")
    print("=" * 80)
    # Polled MFE_R
    def polled_mfe_r(row):
        pip = pip_size(row["Symbol"])
        sl_pips = abs(row["Entry"] - row["SL"]) / pip
        mfe = row["MFE"] if not pd.isna(row["MFE"]) else 0.0
        return mfe / sl_pips if sl_pips > 0 else 0.0
    df_ok["MFE_R_polled"] = df_ok.apply(polled_mfe_r, axis=1)
    df_ok["bias_R"] = df_ok["TRUE_MFE_R"] - df_ok["MFE_R_polled"]
    print(f"  TRUE_MFE_R    : mean={df_ok['TRUE_MFE_R'].mean():.3f}  median={df_ok['TRUE_MFE_R'].median():.3f}  max={df_ok['TRUE_MFE_R'].max():.3f}")
    print(f"  POLLED_MFE_R  : mean={df_ok['MFE_R_polled'].mean():.3f}  median={df_ok['MFE_R_polled'].median():.3f}  max={df_ok['MFE_R_polled'].max():.3f}")
    print(f"  BIAS (T-P)    : mean={df_ok['bias_R'].mean():+.3f}  median={df_ok['bias_R'].median():+.3f}  → polled underestimates by this much")
    underpolled = (df_ok["bias_R"] > 0.05).sum()
    print(f"  Trades where TRUE > polled (poll missed): {underpolled} / {len(df_ok)} = {underpolled/len(df_ok)*100:.1f}%")

    # ---------------------------------------------------------------------------
    # [3] Continuation rate per symbol
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [3] CONTINUATION RATE PER SYMBOL (TRUE M1 data)")
    print("=" * 80)
    print(" Of trades that reached 1.0R, what % continued to 1.5R / 2.0R?")
    print()
    # Cast booleans → int so .sum() returns counts (pandas otherwise may collapse to bool)
    for col in ["reached_0.3R", "reached_0.8R", "reached_1.0R", "reached_1.5R", "reached_2.0R"]:
        df_ok[col] = df_ok[col].astype(int)
    cont = df_ok.groupby("Symbol").agg(
        n=("Symbol", "count"),
        reached_1R=("reached_1.0R", "sum"),
        reached_15R=("reached_1.5R", "sum"),
        reached_2R=("reached_2.0R", "sum"),
    )
    cont["cont_to_1.5R_%"] = pd.Series(
        np.where(cont["reached_1R"] > 0, cont["reached_15R"] / cont["reached_1R"] * 100, np.nan),
        index=cont.index,
    ).round(1)
    cont["cont_to_2.0R_%"] = pd.Series(
        np.where(cont["reached_1R"] > 0, cont["reached_2R"] / cont["reached_1R"] * 100, np.nan),
        index=cont.index,
    ).round(1)
    cont["blocked"] = cont.index.isin(BLOCKED_SYMBOLS)
    print(cont.to_string())

    # Average continuation rate (non-blocked only, weighted by reached_1R count)
    non_blocked = cont[~cont["blocked"]]
    total_reached_1R = non_blocked["reached_1R"].sum()
    total_reached_15R = non_blocked["reached_15R"].sum()
    total_reached_2R = non_blocked["reached_2R"].sum()
    avg_cont_15 = total_reached_15R / total_reached_1R * 100 if total_reached_1R else 0
    avg_cont_2 = total_reached_2R / total_reached_1R * 100 if total_reached_1R else 0

    print(f"\n📊 NON-BLOCKED symbols aggregate:")
    print(f"   Trades reached 1.0R: {total_reached_1R}")
    print(f"   → Continued to 1.5R: {total_reached_15R} = {avg_cont_15:.1f}%  ⬅ KEY METRIC")
    print(f"   → Continued to 2.0R: {total_reached_2R} = {avg_cont_2:.1f}%")

    # ---------------------------------------------------------------------------
    # [4] Scenario comparison (after -$10 spread/trade)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [4] SCENARIO COMPARISON (after -$10 spread per trade)")
    print("=" * 80)

    def run_scenario(name, outcome_fn, exclude_blocked=True):
        d = df_ok.copy()
        if exclude_blocked:
            d = d[~d["Symbol"].isin(BLOCKED_SYMBOLS)]
        d["outcome_R"] = d["TRUE_MFE_R"].apply(outcome_fn)
        d["pnl_$"] = d["outcome_R"] * NEW_RISK_USD - SPREAD_COST_USD
        n = len(d)
        wins = (d["pnl_$"] > 0).sum()
        net = d["pnl_$"].sum()
        ev = net / n if n else 0
        wr = wins / n * 100 if n else 0
        win_sum = d[d["pnl_$"] > 0]["pnl_$"].sum()
        loss_sum = abs(d[d["pnl_$"] < 0]["pnl_$"].sum())
        pf = win_sum / loss_sum if loss_sum else float("inf")
        return {"name": name, "n": n, "wr": wr, "net": net, "ev": ev, "pf": pf, "d": d}

    # OLD baseline (real history, no recompute)
    old_n = len(df)
    old_w = (df["P/L ($)"] > 0).sum()
    old_net = df["P/L ($)"].sum()
    old_wsum = df[df["P/L ($)"] > 0]["P/L ($)"].sum()
    old_lsum = abs(df[df["P/L ($)"] < 0]["P/L ($)"].sum())
    old_pf = old_wsum / old_lsum if old_lsum else float("inf")

    p1 = run_scenario("Phase 1 Only", outcome_phase1_only)
    p12 = run_scenario("Phase 1+2 Full (RR 1.5)", outcome_phase1_plus_2)

    print(f"\n{'Scenario':<32} {'N':>4} {'WR':>7} {'Net P/L':>10} {'EV/trd':>8} {'PF':>6}")
    print(f"{'-'*32} {'-'*4} {'-'*7} {'-'*10} {'-'*8} {'-'*6}")
    print(f"{'OLD baseline (real history)':<32} {old_n:>4} {old_w/old_n*100:>6.1f}% ${old_net:>8.0f} ${old_net/old_n:>6.2f} {old_pf:>6.2f}")
    print(f"{p1['name']:<32} {p1['n']:>4} {p1['wr']:>6.1f}% ${p1['net']:>8.0f} ${p1['ev']:>6.2f} {p1['pf']:>6.2f}")
    print(f"{p12['name']:<32} {p12['n']:>4} {p12['wr']:>6.1f}% ${p12['net']:>8.0f} ${p12['ev']:>6.2f} {p12['pf']:>6.2f}")

    # Per-symbol net P/L
    print("\n📍 Per-symbol Net P/L (Phase 1 vs Phase 1+2):")
    syms = sorted(df["Symbol"].unique())
    print(f"  {'Symbol':<8} {'OLD':>10} {'Phase 1':>10} {'Phase 1+2':>11}")
    for s in syms:
        old_s = df[df["Symbol"] == s]["P/L ($)"].sum()
        if s in BLOCKED_SYMBOLS:
            print(f"  {s:<8} {'$'+str(round(old_s)):>10} {'BLOCKED':>10} {'BLOCKED':>11}")
        else:
            p1_s = p1["d"][p1["d"]["Symbol"] == s]["pnl_$"].sum() if s in p1["d"]["Symbol"].values else 0
            p12_s = p12["d"][p12["d"]["Symbol"] == s]["pnl_$"].sum() if s in p12["d"]["Symbol"].values else 0
            print(f"  {s:<8} {'$'+str(round(old_s)):>10} {'$'+str(round(p1_s)):>10} {'$'+str(round(p12_s)):>11}")

    # ---------------------------------------------------------------------------
    # [5] FINAL DEPLOYMENT GATE
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [5] FINAL DEPLOYMENT GATE DECISION")
    print("=" * 80)
    print(f"\n  Continuation Rate (1.0R → 1.5R) avg = {avg_cont_15:.1f}%")
    print(f"  Gate threshold                       = {GATE_CONTINUATION_PCT:.0f}%")

    if avg_cont_15 >= GATE_CONTINUATION_PCT:
        decision = "✅ DEPLOY Phase 1+2 Full (v8.0.60, RR 1.5)"
        reason = f"Continuation {avg_cont_15:.1f}% ≥ {GATE_CONTINUATION_PCT}% → RR 1.5 expected to compound wins"
    else:
        decision = "🔴 REVERT RR 1.5 → 1.0, deploy Phase 1 ONLY (v8.0.52 model + Phase 1 fixes)"
        reason = f"Continuation {avg_cont_15:.1f}% < {GATE_CONTINUATION_PCT}% → RR 1.5 wins won't materialize (MR mean-reverts)"

    print(f"\n  DECISION: {decision}")
    print(f"  REASON  : {reason}")
    print(f"\n  Best scenario by Net P/L (after spread):")
    print(f"    Phase 1 Only : ${p1['net']:+.0f} (EV ${p1['ev']:+.2f}/trade)")
    print(f"    Phase 1+2    : ${p12['net']:+.0f} (EV ${p12['ev']:+.2f}/trade)")
    winner = "Phase 1+2" if p12['net'] > p1['net'] else "Phase 1 Only"
    print(f"    → Winner: {winner}")

    print("\n" + "=" * 80)
    print(" REPORT END")
    print("=" * 80)


if __name__ == "__main__":
    main()
