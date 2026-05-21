"""Validate RL agent against live xlsx log — BLIND TEST (no answer key, no training).

Loads each trade's obs at entry from xlsx, asks the agent to decide TAKE/SKIP,
then compares against the actual live outcome. Pure read-only — no model
updates, no pool changes.

Usage:
  .venv/bin/python ftmo_trading_bot/scripts/validate_live_xlsx.py [YYYY-MM-DD]
  Default: all trades. With date: only trades on that day.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import openpyxl
from ml.rl_agent import SelfLearningAgent

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    xlsx_path = ROOT / "logs" / "ftmo_trades.xlsx"
    # MR model dir
    model_dir = str(ROOT / "models" / "mr")

    print("=" * 70)
    print(" 🎓 BLIND VALIDATION — Agent vs Live xlsx (no answer key)")
    print("=" * 70)

    # Load agent (read-only)
    print(f"\n→ Loading model from: {model_dir}")
    agent = SelfLearningAgent(model_dir=model_dir, verbose=0)
    # Override paths to MR model files
    agent.model_path = str(Path(model_dir) / "ppo_mr_filter.zip")
    agent.vec_normalize_path = str(Path(model_dir) / "vec_normalize_mr.pkl")
    agent.initialize_model(strict=True)
    print(f"   ✓ Model loaded (deterministic=True, read-only)")

    # Load xlsx
    print(f"\n→ Loading xlsx: {xlsx_path.name}")
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb["Trades"]
    rows = list(ws.iter_rows(values_only=True))
    data = rows[1:]
    print(f"   ✓ {len(data)} total trades")

    # Optional date filter from CLI arg
    date_filter = sys.argv[1] if len(sys.argv) > 1 else None
    if date_filter:
        print(f"   📅 Filtering trades on: {date_filter}")

    # Filter rows with obs + outcome (+ optional date)
    valid = []
    for r in data:
        obs_str = r[55]
        pl = r[15]
        open_t = r[12]
        if obs_str and pl is not None:
            if date_filter and (not open_t or date_filter not in str(open_t)):
                continue
            try:
                obs = json.loads(obs_str)
                if len(obs) == 32:
                    valid.append((r, np.array(obs, dtype=np.float32)))
            except Exception:
                continue
    print(f"   ✓ {len(valid)} closed trades with valid obs")

    if not valid:
        print("⚠️ No valid trades to evaluate")
        return

    # === BLIND PREDICT — agent sees only obs ===
    print(f"\n→ Predicting (blind, no outcome reveal)...")
    correct_take = 0
    take_wins = 0
    take_losses = 0
    skip_wins_missed = 0   # bot SKIP but live won
    skip_losses_saved = 0  # bot SKIP and live lost
    total_take = 0
    total_skip = 0

    per_trade = []
    for (row, obs) in valid:
        # Use agent's internal predict (already deterministic + normalized)
        # should_take_signal returns True (TAKE) / False (SKIP)
        take = agent.should_take_signal(obs)
        action = 1 if take else 0

        pl = row[15]
        win = pl > 0

        if action == 1:  # TAKE
            total_take += 1
            if win:
                take_wins += 1
            else:
                take_losses += 1
        else:  # SKIP
            total_skip += 1
            if win:
                skip_wins_missed += 1
            else:
                skip_losses_saved += 1

        per_trade.append({
            "sym": row[1],
            "type": row[2],
            "agent": "TAKE" if action == 1 else "SKIP",
            "live_pl": pl,
            "live_outcome": "WIN" if win else "LOSS",
            "match": (action == 1) == win,  # TAKE+win or SKIP+loss = match
        })

    # === ANALYSIS ===
    n = len(valid)
    print()
    print("=" * 70)
    print(" 📊 RESULTS")
    print("=" * 70)
    print(f"\nTotal trades evaluated: {n}")
    print(f"\nAgent decisions:")
    print(f"  TAKE: {total_take} ({total_take/n*100:.0f}%)")
    print(f"  SKIP: {total_skip} ({total_skip/n*100:.0f}%)")

    print(f"\nWhen agent said TAKE ({total_take} trades):")
    if total_take:
        print(f"  → Live won:  {take_wins} ({take_wins/total_take*100:.0f}% — 🟢 GOOD)")
        print(f"  → Live lost: {take_losses} ({take_losses/total_take*100:.0f}% — 🔴 BAD)")
        print(f"  Take precision: {take_wins/total_take*100:.0f}%")

    print(f"\nWhen agent said SKIP ({total_skip} trades):")
    if total_skip:
        print(f"  → Live won (missed!):  {skip_wins_missed} ({skip_wins_missed/total_skip*100:.0f}% — 🟡 missed opportunity)")
        print(f"  → Live lost (saved!):  {skip_losses_saved} ({skip_losses_saved/total_skip*100:.0f}% — 🟢 correctly avoided)")
        print(f"  Skip precision: {skip_losses_saved/total_skip*100:.0f}%")

    # Overall alignment
    matches = sum(1 for t in per_trade if t["match"])
    print(f"\n=== Alignment Score: {matches}/{n} = {matches/n*100:.0f}% ===")
    print(f"   (TAKE+win) + (SKIP+loss) = matches")

    # Per-symbol breakdown
    print(f"\n=== Per-symbol breakdown ===")
    syms = {}
    for t in per_trade:
        syms.setdefault(t["sym"], []).append(t)
    for sym, trades in sorted(syms.items()):
        n_s = len(trades)
        m_s = sum(1 for t in trades if t["match"])
        takes = sum(1 for t in trades if t["agent"] == "TAKE")
        print(f"  {sym:<8} {n_s} trades | TAKE {takes}/{n_s} | Align {m_s}/{n_s} = {m_s/n_s*100:.0f}%")

    # Disagreements (interesting cases)
    disagree = [t for t in per_trade if not t["match"]]
    if disagree:
        print(f"\n=== Disagreements ({len(disagree)} cases — bot vs live ===")
        for t in disagree[-15:]:
            mark = "❌ agent TAKE → loss" if t["agent"] == "TAKE" else "🟡 agent SKIP → missed win"
            print(f"  {t['sym']:<8} {t['type']:<5} agent={t['agent']:<5} live=${t['live_pl']:+.0f}  {mark}")

    print("\n" + "=" * 70)
    print(" ✅ Validation complete — agent received ZERO learning signal")
    print("=" * 70)


if __name__ == "__main__":
    main()
