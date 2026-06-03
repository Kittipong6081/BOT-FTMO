"""
===============================================================================
FTMO Trading Bot — Per-Strategy Risk Book (v8.1 Phase 2, dual-strategy MR+TF)
===============================================================================
Tracks each strategy's daily loss budget SEPARATELY so one strategy can halt
itself (hit its sub-budget) without halting the other. Sits BELOW the global
FTMO guards:

    [1] FTMO 4% hard breach guard (RiskManager.check_risk)   — closes the portfolio
    [2] Global soft cap 3.0% (dual mode) / 2.5% (single)     — locks the portfolio
    [3] Per-strategy sub-budget: MR 2.0% / TF 1.5%  ← THIS   — halts one strategy

Attribution is by MT5 magic (source of truth): realized P/L is recorded at close
time (strategy_id passed by the executor), floating P/L is computed live from
open positions' magic → never drifts from the broker on restart.

Only consulted when `bot_config.tf.enabled` — single-strategy MR is unaffected.
===============================================================================
"""

from __future__ import annotations

from typing import Dict, Optional

from config.settings import bot_config


class StrategyRiskBook:
    """Per-strategy daily realized/floating P/L + self-halt on sub-budget."""

    def __init__(self, connector):
        self._connector = connector
        self._cfg = bot_config.ftmo
        # realized (closed) P/L today, per strategy. floating is computed live.
        self._daily_realized_pnl: Dict[str, float] = {}
        self._strategy_halted: Dict[str, bool] = {}
        self._day: Optional[str] = None

    # ─── day lifecycle ────────────────────────────────────────────────────

    def reset_day(self, day_iso: Optional[str] = None):
        """Clear realized P/L + halt flags at the broker day rollover."""
        self._daily_realized_pnl = {}
        self._strategy_halted = {}
        self._day = day_iso

    # ─── realized (recorded at close) ─────────────────────────────────────

    def record_realized(self, strategy_id: str, pnl: float):
        sid = strategy_id or "MR"
        self._daily_realized_pnl[sid] = self._daily_realized_pnl.get(sid, 0.0) + float(pnl)

    # ─── floating (computed live from broker + magic) ─────────────────────

    def _floating_by_strategy(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            magic_to_sid = {int(m): sid for sid, m in self._cfg.STRATEGY_MAGIC.items()}
            for pos in self._connector.get_open_positions():
                sid = magic_to_sid.get(int(pos.get("magic", 0)))
                if sid is None:
                    continue
                out[sid] = out.get(sid, 0.0) + float(pos.get("profit", 0.0)) + float(pos.get("swap", 0.0))
        except Exception:
            pass
        return out

    # ─── budget math ──────────────────────────────────────────────────────

    def daily_pnl(self, strategy_id: str) -> float:
        """realized + floating for one strategy today (negative = net loss)."""
        return (self._daily_realized_pnl.get(strategy_id, 0.0)
                + self._floating_by_strategy().get(strategy_id, 0.0))

    def sub_budget_usd(self, strategy_id: str, initial_balance: float) -> float:
        pct = self._cfg.STRATEGY_SUB_BUDGET_PCT.get(strategy_id)
        if pct is None:
            return float("inf")          # no sub-budget configured → unlimited
        return float(initial_balance) * float(pct)

    def remaining_budget(self, strategy_id: str, initial_balance: float) -> float:
        budget = self.sub_budget_usd(strategy_id, initial_balance)
        if budget == float("inf"):
            return float("inf")
        loss = max(0.0, -self.daily_pnl(strategy_id))   # only losses consume budget
        return max(0.0, budget - loss)

    def is_halted(self, strategy_id: str, initial_balance: float) -> bool:
        """True once this strategy's loss reaches its sub-budget (self-halt).

        Latches the halt flag so a brief floating recovery doesn't un-halt
        mid-day (mirrors the global _daily_loss_locked behaviour).
        """
        if self._strategy_halted.get(strategy_id):
            return True
        budget = self.sub_budget_usd(strategy_id, initial_balance)
        if budget == float("inf"):
            return False
        if -self.daily_pnl(strategy_id) >= budget:
            self._strategy_halted[strategy_id] = True
            return True
        return False

    # ─── persistence (RiskManager bot_state.json schema 9) ────────────────

    def to_dict(self) -> Dict:
        return {
            "daily_realized_pnl": dict(self._daily_realized_pnl),
            "strategy_halted": dict(self._strategy_halted),
            "day": self._day,
        }

    def from_dict(self, data: Optional[Dict]):
        if not data:
            return
        self._daily_realized_pnl = dict(data.get("daily_realized_pnl", {}) or {})
        self._strategy_halted = dict(data.get("strategy_halted", {}) or {})
        self._day = data.get("day")
