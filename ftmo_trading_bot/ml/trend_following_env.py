"""
===============================================================================
FTMO Trading Bot — Trend Following Filter Environment (v8.1 Phase 3)
===============================================================================
RL env for the TF strategy. Inherits all FTMOSignalFilterEnv plumbing (obs
builder, FTMO state machine, correlation simulator, episode machinery) and
overrides reward shaping with INVERTED incentives vs MR:

  • MR rewards quick small TPs + penalizes prolonged holds.
  • TF rewards RUNNERS (outcome >= 2R, scaled with size) + small bonus for
    modest wins, and penalizes losing trades entered on an AGED trend
    (late entry). The "let winners run" behaviour is baked into the
    TrendFollowingBacktester trail (Stage-2 cap disabled) → the agent's job is
    to TAKE the trend-continuation setups that turn into runners and SKIP the
    choppy/late ones.

Same 35-dim obs shape (so VecNormalize + network arch stay identical) — three
slots reinterpreted for TF semantics (tf_v1 layout):
    obs[4]  = trend_strength / 100          (was bb_extreme)
    obs[10] = trend_age_bars / 30           (was bb_band_width/ATR)
    obs[26] = adx / 50  (trending = good)   (MR used 1 - adx/50)

Reads TF-specific pool fields produced by TrendFollowingBacktester:
  - `trend_age_bars` (int), `pullback_depth_atr` (float), `adx_at_entry` (float),
    `is_runner` (bool) — plus the shared `outcome_pnl_ratio`, `bars_to_resolution`.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ml.signal_filter_env import FTMOSignalFilterEnv


class TrendFollowingFilterEnv(FTMOSignalFilterEnv):
    """Trend-following variant — runner bonus + late-entry penalty."""

    # ─── TF reward-shaping parameters (override via constructor) ──────
    # v8.1 runner-tune: favour big runners over small +1R wins so the agent
    # prefers setups that turn into trends (RUNNER_BONUS↑, SLOW_WIN_BONUS↓).
    RUNNER_R: float = 2.0              # outcome >= 2R counts as a "runner"
    RUNNER_BONUS: float = 0.70         # base bonus for a runner win (was 0.50)
    RUNNER_SCALE: float = 0.25         # extra per R above RUNNER_R (capped)
    RUNNER_SCALE_CAP: float = 0.75
    SLOW_WIN_BONUS: float = 0.10       # modest win (0 < outcome < RUNNER_R) (was 0.15)
    BASE_LOSS_PENALTY: float = 0.10    # any losing trade
    LATE_ENTRY_TREND_AGE: int = 60     # H1 bars — trend "too mature" to enter
    LATE_ENTRY_PENALTY: float = 0.20   # losing trade entered on an aged trend

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
        # TF-specific overrides
        runner_bonus: Optional[float] = None,
        slow_win_bonus: Optional[float] = None,
        late_entry_penalty: Optional[float] = None,
        base_loss_penalty: Optional[float] = None,
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
        if runner_bonus is not None:
            self.RUNNER_BONUS = float(runner_bonus)
        if slow_win_bonus is not None:
            self.SLOW_WIN_BONUS = float(slow_win_bonus)
        if late_entry_penalty is not None:
            self.LATE_ENTRY_PENALTY = float(late_entry_penalty)
        if base_loss_penalty is not None:
            self.BASE_LOSS_PENALTY = float(base_loss_penalty)

    # ─── Observation builder (semantic only — same 35-dim shape, tf_v1) ─────

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        if self._signal_idx >= len(self._signals):
            return obs
        sig = self._signals[self._signal_idx]
        obs[4] = float(np.clip(sig.get("trend_strength", 0.0) / 100.0, 0.0, 1.0))
        obs[10] = float(np.clip(sig.get("trend_age_bars", 0) / 30.0, 0.0, 1.0))
        adx = float(sig.get("adx", 0.0))
        obs[26] = float(np.clip(adx / 50.0, 0.0, 1.0))   # trending = good (vs MR inverse)
        # tf_v2 (early-entry): repurpose the dead Chronos slots [27]/[28] for the leading
        # signals so the RL can rank early/building vs late/exhausted entries (the base
        # _get_obs left these = 0; MR never overrides them → MR obs unchanged).
        obs[27] = float(np.clip(sig.get("di_spread", 0.0) / 50.0, -1.0, 1.0))
        obs[28] = float(np.clip(sig.get("adx_slope", 0.0) / 10.0, -1.0, 1.0))
        return obs

    # ─── step() — same plumbing, TF reward (runners > quick TP) ────────────

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        take = float(action[0]) > 0.0

        sig = self._signals[self._signal_idx]
        pnl = 0.0
        was_clamped = False

        if sig["day"] != self._current_day:
            self._update_daily_dd()
            self._current_day = sig["day"]
            self._start_new_day()

        if self._open_positions:
            cutoff = self._signal_idx - self.HOLD_SIGNALS_APPROX
            self._open_positions = [
                op for op in self._open_positions if op["opened_at_idx"] >= cutoff
            ]

        self._floating_pnl_norm = 0.0
        self._open_losing_count_norm = 0.0

        correlation_forced_skip = False
        if take and self._is_correlation_blocked(sig):
            take = False
            correlation_forced_skip = True
            self._correlation_forced_skips += 1

        # TF has no MR-style entry-confirm gate; pool sets entry_confirm_passed=True.
        # Keep the hook for parity with parent (no-op for TF).
        entry_confirm_forced_skip = False
        if take and not sig.get("entry_confirm_passed", True):
            take = False
            entry_confirm_forced_skip = True
            self._entry_confirm_forced_skips += 1

        outcome = float(sig.get("outcome_pnl_ratio", 0.0))
        ml_score = float(sig.get("ml_score", 0.5))
        bars_to_res = int(sig.get("bars_to_resolution", 0))
        trend_age = int(sig.get("trend_age_bars", 0))

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

            # ─── TF-specific reward (runners > quick TP) ────────────────
            pnl_norm = pnl / max(risk_amount, 1.0)

            spread_pips_trade = float(sig.get("spread_pips", 0.0))
            sl_pips_trade = float(sig.get("sl_distance_pips", 0.0))
            if sl_pips_trade > 0 and spread_pips_trade > 0:
                spread_cost_R = min(spread_pips_trade / sl_pips_trade, 1.0)
                pnl_norm -= spread_cost_R * 0.5

            reward = float(np.clip(pnl_norm, -1.0, 3.0))

            if pnl_norm > 0:
                # Reward RUNNERS: base bonus + scale with how far it ran
                if outcome >= self.RUNNER_R:
                    reward += self.RUNNER_BONUS + min(
                        (outcome - self.RUNNER_R) * self.RUNNER_SCALE,
                        self.RUNNER_SCALE_CAP,
                    )
                else:
                    reward += self.SLOW_WIN_BONUS
                if ml_score >= 0.40:
                    reward += 0.15
            else:
                # Loss shaping — penalize entering an already-mature trend that reversed
                reward -= self.BASE_LOSS_PENALTY
                if trend_age >= self.LATE_ENTRY_TREND_AGE:
                    reward -= self.LATE_ENTRY_PENALTY

            if not self.enable_risk_penalty:
                reward += 0.02
            else:
                reward += 0.04
                reward += self._dd_penalty()
                if was_clamped:
                    reward -= 0.25

        # ─── 2. SKIP branch — oracle feedback ──────────────────────────────
        else:
            self._total_skips += 1
            reward = 0.0
            is_p1 = not self.enable_risk_penalty
            # TF oracle: missing a RUNNER is the costliest mistake (TF's raison d'être).
            if outcome >= self.RUNNER_R:
                reward -= 0.30 if is_p1 else 0.70
                if ml_score >= 0.40:
                    reward -= 0.15 if is_p1 else 0.30
            elif outcome >= 0.5:
                reward -= 0.10 if is_p1 else 0.20
            elif outcome <= -0.5:
                reward += 0.30 if is_p1 else 0.45
                if ml_score < 0.36:
                    reward += 0.15
            elif outcome <= -0.1:
                reward += 0.10
            reward -= 0.012

        # ─── 3. Progress shaping + milestones (identical to parent/MR) ─────
        progress_delta = self.target_progress_pct - self._last_progress
        if progress_delta > 0:
            reward += 0.05 * progress_delta
        elif progress_delta < -5.0:
            reward += 0.005 * progress_delta
        self._last_progress = self.target_progress_pct

        if self.enable_risk_penalty:
            if (not self._mid_check_day10_fired and self._current_day >= 10
                    and self.target_progress_pct < 20.0 and self._total_takes < 3):
                reward -= 0.2
                self._mid_check_day10_fired = True
            if (not self._mid_check_day20_fired and self._current_day >= 20
                    and self.target_progress_pct < 40.0 and self._total_takes < 6):
                reward -= 0.3
                self._mid_check_day20_fired = True
            if (not self._mid_check_day35_fired and self._current_day >= 35
                    and self.target_progress_pct < 60.0 and self._total_takes < 12):
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

        # ─── 4. Episode end ──────────────────────────────────────────────
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
                if (self._current_day >= 40 and self._total_takes < 15
                        and not self._peak_passed):
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
            "trend_age_bars": trend_age,
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
                "entry_confirm_forced_skips": self._entry_confirm_forced_skips,
            }

        info["correlation_forced_skip"] = correlation_forced_skip
        info["entry_confirm_forced_skip"] = entry_confirm_forced_skip
        return self._get_obs(), float(reward), terminated, truncated, info
