"""
===============================================================================
FTMO Trading Bot — Mean Reversion Filter Environment (v8.0)
===============================================================================
RL environment for the MR-strategy pivot. Inherits the SMC env's plumbing
(observation builder, FTMO state machine, correlation simulator, episode
machinery) and overrides:

  • `step()`  — reward shaping per the v8.0 spec
                  - Quick TP bonus: large reward when TP hits in <= 5 M15 bars
                  - Slow win bonus: smaller bonus for late TP wins
                  - Prolonged floating loss penalty: severe penalty for trades
                    that bleed for many bars before SL
                  - Capital-preservation focus: every losing trade pays a
                    "duration fine" proportional to how long it floated red
  • `_get_obs()` — keeps 32-dim shape (so the trainer + VecNormalize stay
                   compatible) but reinterprets a few SMC slots:
                       obs[4]  = bb_extreme score      (was ob_norm)
                       obs[10] = bb_band_width / ATR   (was ob_size_atr)
                       obs[26] = trend_strength_inverse for MR
                                (high when ADX low — ranging markets are
                                 friendly to MR)
  • Pool path defaults to `data/mr_signal_pool_*.pkl` so the SMC pool stays
    untouched.

The env relies on extra fields produced by `MeanReversionBacktester`:
  - `bars_to_resolution`  (int)   — speed of TP/SL hit
  - `is_quick_tp`         (bool)  — winner that resolved in <= 5 bars
  - `bb_extreme`          (float) — strength of BB band penetration
  - `bb_band_width_atr`   (float) — BB band width / ATR (vol regime)
  - `reversal_wick_ratio` (float) — rejection wick / candle body
===============================================================================
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ml.signal_filter_env import FTMOSignalFilterEnv


class MeanReversionFilterEnv(FTMOSignalFilterEnv):
    """Mean-reversion variant — quick-TP bonus + prolonged-loss penalty.

    Drop-in replacement for `FTMOSignalFilterEnv` consumed by the same
    PPO trainer. Same obs shape (32,), same action shape (1,) so the
    network architecture stays identical.
    """

    # ─── MR reward-shaping parameters (override via constructor) ──────
    QUICK_TP_BARS: int = 5             # winner resolved in <= 5 bars = "quick"
    QUICK_TP_BONUS: float = 0.50       # reward bonus for quick TP wins
    SLOW_WIN_BONUS: float = 0.20       # smaller bonus for slow wins
    PROLONGED_LOSS_BARS: int = 12      # losing trade floating > 12 bars = bad
    PROLONGED_LOSS_PENALTY: float = 0.40
    BASE_LOSS_PENALTY: float = 0.10    # baseline (any losing trade)
    DURATION_FINE_COEF: float = 0.02   # per-bar fine for losing trades
    DURATION_FINE_CAP: float = 0.30    # cap so it doesn't dominate
    ADX_VIOLATION_PENALTY: float = 0.30  # taking trade when ADX > 25 (block)
    ADX_VIOLATION_THRESHOLD: float = 25.0

    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_days: int = 45,
        verbose: bool = False,
        enable_risk_penalty: bool = True,
        signal_pool_path: Optional[str] = None,
        outcome_noise_std: float = 0.05,
        ml_filter_threshold: float = 0.0,
        risk_per_trade: Optional[float] = None,
        # MR-specific overrides
        quick_tp_bonus: Optional[float] = None,
        slow_win_bonus: Optional[float] = None,
        prolonged_loss_penalty: Optional[float] = None,
        base_loss_penalty: Optional[float] = None,
        duration_fine_coef: Optional[float] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            max_days=max_days,
            verbose=verbose,
            enable_risk_penalty=enable_risk_penalty,
            signal_pool_path=signal_pool_path,
            outcome_noise_std=outcome_noise_std,
            ml_filter_threshold=ml_filter_threshold,
            risk_per_trade=risk_per_trade,
        )
        # Override reward params if caller provided
        if quick_tp_bonus is not None:
            self.QUICK_TP_BONUS = float(quick_tp_bonus)
        if slow_win_bonus is not None:
            self.SLOW_WIN_BONUS = float(slow_win_bonus)
        if prolonged_loss_penalty is not None:
            self.PROLONGED_LOSS_PENALTY = float(prolonged_loss_penalty)
        if base_loss_penalty is not None:
            self.BASE_LOSS_PENALTY = float(base_loss_penalty)
        if duration_fine_coef is not None:
            self.DURATION_FINE_COEF = float(duration_fine_coef)

    # ─── Override observation builder (semantic only — same shape) ─────

    def _get_obs(self) -> np.ndarray:
        """35-dim obs — parent's 35 slots + MR reinterpretation of three.

        Slots reinterpreted:
          obs[4]  = bb_extreme           (replaces ob_norm)
          obs[10] = bb_band_width_atr/3  (replaces ob_size_atr; clipped 0..1)
          obs[26] = adx_inverse_norm     (1.0 when ADX low = ranging = MR-friendly)
        New v8.0.74 slots (populated by parent from sig dict):
          obs[32] = kc_distance_norm     (ATR band distance)
          obs[33] = ema_slope_norm       (EMA slope steepness)
          obs[34] = band_squeeze_ratio   (band squeeze detection)
        """
        obs = super()._get_obs()
        if self._signal_idx >= len(self._signals):
            return obs
        sig = self._signals[self._signal_idx]

        obs[4] = float(np.clip(sig.get("bb_extreme", 0.0), 0.0, 1.0))

        bbw_atr = float(sig.get("bb_band_width_atr", 0.0))
        obs[10] = float(np.clip(bbw_atr / 3.0, 0.0, 1.0))

        adx = float(sig.get("adx", 0.0))
        obs[26] = float(np.clip(1.0 - adx / 50.0, 0.0, 1.0))

        return obs

    # ─── Override step() — same plumbing, MR-specific reward ───────────

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        take = float(action[0]) > 0.0

        sig = self._signals[self._signal_idx]
        pnl = 0.0
        was_clamped = False

        # Day rollover
        if sig["day"] != self._current_day:
            self._update_daily_dd()
            self._current_day = sig["day"]
            self._start_new_day()

        # Drop stale open positions (correlation simulator)
        if self._open_positions:
            cutoff = self._signal_idx - self.HOLD_SIGNALS_APPROX
            self._open_positions = [
                op for op in self._open_positions if op["opened_at_idx"] >= cutoff
            ]

        # Floating PnL state — kept zeroed (same leak-free policy as v7.2.1)
        self._floating_pnl_norm = 0.0
        self._open_losing_count_norm = 0.0

        # Correlation block — forced SKIP
        correlation_forced_skip = False
        if take and self._is_correlation_blocked(sig):
            take = False
            correlation_forced_skip = True
            self._correlation_forced_skips += 1

        outcome = float(sig.get("outcome_pnl_ratio", 0.0))
        ml_score = float(sig.get("ml_score", 0.5))
        bars_to_res = int(sig.get("bars_to_resolution", self.QUICK_TP_BARS + 1))
        is_quick_tp = bool(sig.get("is_quick_tp", False))
        adx_value = float(sig.get("adx", 0.0))

        # ─── 1. TAKE branch ───────────────────────────────────────────────
        if take and self._trades_today < self.MAX_TRADES_PER_DAY:
            risk_amount = self.balance * self.risk_per_trade

            if self._signal_pool and self.outcome_noise_std > 0:
                noise = float(self.np_random.normal(0.0, self.outcome_noise_std))
                outcome_perturbed = outcome * (1.0 + noise)
            else:
                outcome_perturbed = outcome

            raw_pnl = risk_amount * outcome_perturbed
            if self.enable_risk_penalty:
                pnl = self._clamp_pnl_to_guard(raw_pnl)
                was_clamped = (pnl != raw_pnl)
            else:
                pnl = raw_pnl

            self.balance += pnl
            self._trades_today += 1
            self._trade_results.append(pnl)
            self._total_takes += 1

            self._last_closed_direction = float(sig.get("direction", 0.0))
            self._last_closed_signal_step = int(self._signal_idx)
            self._open_positions.append({
                "symbol": sig.get("symbol", ""),
                "direction": 1 if sig.get("direction", 0) > 0 else -1,
                "opened_at_idx": self._signal_idx,
            })

            if pnl > 0:
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1

            self.peak_balance = max(self.peak_balance, self.balance)
            total_loss = max(0.0, self.INITIAL_BALANCE - self.balance)
            self.total_dd_pct = total_loss / self.INITIAL_BALANCE
            day_loss = max(0.0, self.daily_start_balance - self.balance)
            self.daily_dd_pct = day_loss / max(self.daily_start_balance, 1.0)
            self.min_balance = min(self.min_balance, self.balance)
            self.max_daily_dd_pct = max(self.max_daily_dd_pct, self.daily_dd_pct)

            profit = self.balance - self.INITIAL_BALANCE
            target_amount = self.INITIAL_BALANCE * self.TARGET_PCT
            self.target_progress_pct = (profit / target_amount) * 100.0

            # ─── MR-specific reward (the core pivot) ────────────────────
            pnl_norm = pnl / max(risk_amount, 1.0)

            # Spread cost (same as parent — keep cost awareness)
            spread_pips_trade = float(sig.get("spread_pips", 0.0))
            sl_pips_trade = float(sig.get("sl_distance_pips", 0.0))
            if sl_pips_trade > 0 and spread_pips_trade > 0:
                spread_cost_R = min(spread_pips_trade / sl_pips_trade, 1.0)
                pnl_norm -= spread_cost_R * 0.5

            reward = float(np.clip(pnl_norm, -1.0, 3.0))

            # Win shaping
            if pnl_norm > 0:
                # Capital preservation reward: dense, scales with speed
                if is_quick_tp or bars_to_res <= self.QUICK_TP_BARS:
                    reward += self.QUICK_TP_BONUS
                else:
                    reward += self.SLOW_WIN_BONUS
                # ML quality bonus carried over (policy learns to align with
                # quality model when both agree on a winner)
                if ml_score >= 0.40:
                    reward += 0.15
            else:
                # Loss shaping — capital-preservation pressure
                # Base penalty for any losing trade
                reward -= self.BASE_LOSS_PENALTY
                # Duration fine — proportional to how long the trade bled red.
                # bars_to_res is bounded by the resolution window; cap fine.
                duration_fine = min(
                    self.DURATION_FINE_COEF * max(0, bars_to_res - 1),
                    self.DURATION_FINE_CAP,
                )
                reward -= duration_fine
                # Prolonged-loss penalty — severe, fires when bars >= threshold
                if bars_to_res >= self.PROLONGED_LOSS_BARS:
                    reward -= self.PROLONGED_LOSS_PENALTY

            # Trend-filter discipline: penalize trades taken when ADX > 25
            # (strategy already vetoes most of these, but pool may contain
            # signals from edge cases or noisy ADX)
            if adx_value > self.ADX_VIOLATION_THRESHOLD:
                reward -= self.ADX_VIOLATION_PENALTY

            # Phase-wise activity / DD shaping (kept from parent — works for MR)
            if not self.enable_risk_penalty:
                reward += 0.02
            else:
                reward += 0.04
                reward += self._dd_penalty()
                if was_clamped:
                    reward -= 0.25

        # ─── 2. SKIP branch — Oracle feedback (chart-reading label) ────
        else:
            self._total_skips += 1
            reward = 0.0

            is_p1 = not self.enable_risk_penalty
            # MR-tuned oracle weights — leaner than SMC because RR 1:1 means
            # missed winners hurt less (smaller absolute outcome).
            if outcome >= 0.5:
                reward -= 0.20 if is_p1 else 0.55
                if ml_score >= 0.40:
                    reward -= 0.15 if is_p1 else 0.30
            elif outcome >= 0.1:
                reward -= 0.05 if is_p1 else 0.10
            elif outcome <= -0.5:
                # Smart skip — reward MORE for MR because capital preservation
                # is the headline objective
                reward += 0.30 if is_p1 else 0.45
                if ml_score < 0.36:
                    reward += 0.15
            elif outcome <= -0.1:
                reward += 0.10

            # Passive SKIP cost — slightly higher than SMC env so policy
            # doesn't drift to SKIP-all under MR's lower per-trade reward.
            reward -= 0.012

        # ─── 3. Progress shaping + milestones ──────────────────────────
        progress_delta = self.target_progress_pct - self._last_progress
        if progress_delta > 0:
            reward += 0.05 * progress_delta
        elif progress_delta < -5.0:
            reward += 0.005 * progress_delta
        self._last_progress = self.target_progress_pct

        if self.enable_risk_penalty:
            if (
                not self._mid_check_day10_fired
                and self._current_day >= 10
                and self.target_progress_pct < 20.0
                and self._total_takes < 3
            ):
                reward -= 0.2
                self._mid_check_day10_fired = True
            if (
                not self._mid_check_day20_fired
                and self._current_day >= 20
                and self.target_progress_pct < 40.0
                and self._total_takes < 6
            ):
                reward -= 0.3
                self._mid_check_day20_fired = True
            if (
                not self._mid_check_day35_fired
                and self._current_day >= 35
                and self.target_progress_pct < 60.0
                and self._total_takes < 12
            ):
                reward -= 0.7
                self._mid_check_day35_fired = True

        if self.target_progress_pct >= 30.0 and not self._milestone_30_given:
            reward += 0.5
            self._milestone_30_given = True
        if self.target_progress_pct >= 60.0 and not self._milestone_60_given:
            reward += 1.0
            self._milestone_60_given = True
        if self.target_progress_pct >= 90.0 and not self._milestone_90_given:
            reward += 1.5
            self._milestone_90_given = True

        terminated = False
        if self.target_progress_pct >= 100.0:
            if not self._target_bonus_given:
                reward += 4.0
                self._target_bonus_given = True
                self._peak_passed = True
            if self.enable_risk_penalty:
                terminated = True

        # ─── 4. Episode end ──────────────────────────────────────────
        self._signal_idx += 1

        truncated = False
        if self._signal_idx >= len(self._signals):
            truncated = True
            take_rate = self._total_takes / max(self._total_takes + self._total_skips, 1)
            if self.enable_risk_penalty:
                if take_rate < 0.05:
                    reward -= 1.0
                elif take_rate < 0.12:
                    reward -= 0.3
                elif self.target_progress_pct < 30.0:
                    reward -= 0.3
                if (
                    self._current_day >= 40
                    and (self._total_takes < 15)
                    and not self._peak_passed
                ):
                    reward -= 1.0
            else:
                if take_rate < 0.03:
                    reward -= 0.8

        info = {
            "action": "TAKE" if take else "SKIP",
            "signal_confluence": sig["confluence_score"],
            "signal_rr": sig["rr_ratio"],
            "pnl": pnl,
            "balance": self.balance,
            "total_dd_pct": self.total_dd_pct,
            "daily_dd_pct": self.daily_dd_pct,
            "progress_pct": self.target_progress_pct,
            "day": sig["day"],
            "total_takes": self._total_takes,
            "total_skips": self._total_skips,
            "clamped": was_clamped,
            "aux_target": float(sig.get("outcome_pnl_ratio", 0.0)),
            "bars_to_resolution": bars_to_res,
            "is_quick_tp": is_quick_tp,
        }

        if terminated or truncated:
            stats = self.get_stats()
            profit = self.balance - self.INITIAL_BALANCE
            info["episode_summary"] = {
                "balance": self.balance,
                "profit": profit,
                "profit_pct": profit / self.INITIAL_BALANCE * 100,
                "total_dd_pct": self.total_dd_pct,
                "peak_balance": self.peak_balance,
                "total_takes": self._total_takes,
                "total_skips": self._total_skips,
                "take_rate": stats["take_rate"],
                "win_rate": stats["win_rate"],
                "total_trades": stats["total_trades"],
                "target_progress_pct": self.target_progress_pct,
                "breached": (
                    self.total_dd_pct >= self.TOTAL_DD_LIMIT
                    or self.daily_dd_pct >= self.DAILY_DD_LIMIT
                ),
                "passed": self._peak_passed or self.target_progress_pct >= 100.0,
                "days_traded": self._current_day + 1,
                "daily_dd_pct": self.daily_dd_pct,
                "min_balance": self.min_balance,
                "min_profit": self.min_balance - self.INITIAL_BALANCE,
                "max_profit": self.peak_balance - self.INITIAL_BALANCE,
                "max_daily_dd_pct": self.max_daily_dd_pct,
                "correlation_forced_skips": self._correlation_forced_skips,
            }

        info["correlation_forced_skip"] = correlation_forced_skip
        return self._get_obs(), float(reward), terminated, truncated, info
