# 03 — RL Training (Obs 32 dims v7.1 staged, Reward, PPO + Auxiliary Task)
> Last Updated: 2026-05-04 (v7.1.2 staged — reward re-balance, RL retrain pending) | Scope: RL env, obs space v7.1 (+ portfolio realtime + session timing), reward shaping, PPO hyperparams, curriculum, aux task (E2).
>
> **v7.1.2 reward re-balance (after v7.1.1 eval Pass 1.4%)**:
> - missed-winner P2: −0.65 → **−0.85** (push TAKE back, ใกล้ v7.0.x แต่ไม่ถึง −0.90 wipeout level)
> - missed-winner P2 + ml ≥ 0.40: −0.40 → **−0.50**
> - Chronos disagreement (ml<0.55): −0.40 → **−0.30**
> - Chronos disagreement (ml≥0.55): −0.15 → **−0.10**
> - Concurrent loss penalty: −0.25 → **−0.20**
>
> **v7.1 changes (kept in v7.1.2)**:
> - **Obs 29 → 32**: `[29] floating_pnl_norm`, `[30] open_losing_count_norm`, `[31] mins_since_session_norm` (training simulator + live ผ่าน `RiskManager.get_unrealized_drawdown_pct`)
> - **GBM features 17 → 24**: เพิ่ม `hour_of_day_sin/cos`, `day_of_week`, `minutes_since_session_start`, `is_post_weekend_first_hour`, `volatility_regime_score`, `atr_zscore_30bars` (`compute_temporal_features` helper)
> - **Chronos formula**: `clip((q90-q10)/(atr×√8), 0, 3)` → `clip(log1p(...)/2, 0, 3)` (กัน saturation ที่ 3.0)

## TL;DR (30-second scan)

- Obs = **29 dims** (v7, 2026-05-01) — adds `chronos_alignment`, `chronos_uncertainty_norm` on top of v6 27 (Amazon Chronos 2 zero-shot M15 forecast).
- Action continuous [−1, 1] — `>0 = TAKE`, `≤0 = SKIP`.
- **2-phase curriculum**: Phase 1 (Alpha, no DD penalty, oracle SKIP) → Phase 2 (Risk, DD penalty active).
- **Phase E2 — Auxiliary Task** (verified Pass Rate **9.7 %** at v6.13, 2026-04-29; Phase E2 architecture intact since 2026-04-25): policy has aux head that predicts `outcome_pnl_ratio`. MSE loss × 0.5 added to PPO loss. Forces representation to encode signal quality.
- ML threshold = **0.36** (calibrated probability). Signals below this are rejected before RL.
- Live inference must normalize obs with the `vec_normalize_sf.pkl` stats captured during training.
- VecNormalize: `norm_obs=True`, `norm_reward=True`, `clip_obs=10.0`, `clip_reward=20.0`.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Obs dims | **29** (v7) | `SelfLearningAgent.OBS_DIM`, `FTMOSignalFilterEnv.observation_space` |
| Chronos forecaster | `ChronosForecaster` (`amazon/chronos-bolt-small`) | `ml/chronos_forecaster.py` |
| Action space | Box(−1, 1, shape=(1,)) | `FTMOSignalFilterEnv.action_space` |
| VecNormalize stats | `models/vec_normalize_sf.pkl` | loaded by `SelfLearningAgent._load_normalize_stats` |
| RL model | `models/ppo_signal_filter.zip` (aux-aware policy) | loaded by `SelfLearningAgent.initialize_model` |
| RL training PPO | `AuxAwarePPO` (aux loss weight = 0.5) | `ml/aux_aware_ppo.py` |
| RL policy | `AuxAwareACPolicy` (actor + value + `aux_head`) | `ml/aux_aware_policy.py` |
| Aux target | `info['aux_target']` = `outcome_pnl_ratio` | `FTMOSignalFilterEnv.step()` |
| Pool | `data/signal_pool_3000.pkl` | loaded by `FTMOSignalFilterEnv` |
| ML threshold | **0.40 (v7.1.3)** ↑ จาก 0.36 (calibrated) | CLI `--ml_threshold 0.40` |

---

## Observation Space Layout (29 dims)

**Must be kept in sync in three places**:

1. `FTMOSignalFilterEnv._get_obs` in `ftmo_trading_bot/ml/signal_filter_env.py`
2. `FTMOTradingBot._build_signal_observation` in `ftmo_trading_bot/main.py`
3. `SelfLearningAgent.OBS_DIM` in `ftmo_trading_bot/ml/rl_agent.py`

### Signal Core [0–11]

| Idx | Feature | Source (signal dict key) | Normalization |
|-----|---------|---------------------------|---------------|
| 0 | `confluence_norm` | `confluence_score` | `(x-50)/50` |
| 1 | `rr_norm` | `rr_ratio` | `(x-1)/4` |
| 2 | `direction` | BUY = +1 / SELL = −1 | raw |
| 3 | `atr_norm` | `atr_pips` | `(x-15)/10` |
| 4 | `ob_norm` | `ob_score` | `x/100` |
| 5 | `bias_align` | direction × `market_bias` | ±1 |
| 6 | `sl_atr` | `sl_distance / atr_value` | raw |
| 7 | `rsi_norm` | `rsi_value` | `(x-50)/50` |
| 8 | `macd_norm` | `macd_histogram / atr_value` | raw |
| 9 | `trend_str` | `trend_strength` | `x/100` |
| 10 | `ob_size_atr` | OB body / atr | raw |
| 11 | `adx_norm` | `adx` | `x/100` |

### Market Regime [12–15]

| Idx | Feature | Normalization |
|-----|---------|---------------|
| 12 | `stoch_norm` (`stoch_k`) | `(x-50)/50` |
| 13 | `bb_pctb` | raw |
| 14 | `atr_chg` (`atr_change_ratio`) | raw |
| 15 | `price_roc` | raw |

### ML Quality [16] ⭐

| Idx | Feature | Normalization |
|-----|---------|---------------|
| 16 | `ml_score_norm` (GBM `P(win)`) | `(p-0.5)×2` maps [0,1] → [−1, +1] |

### Portfolio State [17–23]

| Idx | Feature | Normalization |
|-----|---------|---------------|
| 17 | `total_dd_norm` | `-total_dd / 0.10` |
| 18 | `daily_dd_norm` | `-daily_dd / 0.05` |
| 19 | `progress_norm` | `profit / target × 100 / 100` |
| 20 | `day_progress` | `challenge_day / max_days` |
| 21 | `trades_today` | `open_positions / MAX_TRADES_PER_DAY` |
| 22 | `recent_wr_norm` | last 10 WR → `wr×2 − 1` |
| 23 | `consec_losses` | `min(consec, 5) / 5` |

### v6 Cost / Flip / HTF [24–26] (2026-04-22)

| Idx | Feature | Source | Normalization |
|-----|---------|--------|---------------|
| 24 | `spread_pct_of_atr` | `FTMOTradingBot._build_spread_pct_of_atr` | `spread_pips / atr_pips` (clip 0–3) |
| 25 | `has_opposite_recently_closed` | `FTMOTradingBot._has_opposite_recently_closed` | flip-lock flag (0 / 1) |
| 26 | `htf_trend_alignment` | uses `bias_align` as proxy | ±1 (clip) |

**Why v6**: addresses whipsaw + spread awareness — the agent learns to avoid low-RR setups when spread is wide (GBPJPY) and becomes aware of recent flips, reducing revenge trading.

### v7 Chronos Forecast [27–28] (2026-05-01, formula refactor v7.0.2)

| Idx | Feature | Source | Normalization (v7.0.2) |
|-----|---------|--------|---------------|
| 27 | `chronos_alignment` | `ChronosForecaster.forecast_features` (Amazon Chronos 2 zero-shot, M15 × 8 ahead) | `direction × sign(close − median_h+8)`, clip ±1 — flipped จาก v7.0 หลังเจอ corr ติดลบ |
| 28 | `chronos_uncertainty_norm` | `ChronosForecaster.forecast_features` (q90 − q10) | `(q90 − q10) / (atr_value × √8)`, clip [0, 3] — Brownian-scaled, ไม่ saturate |

**Why v7**: gives the agent forward-looking probabilistic context. Chronos 2 is a zero-shot foundation model — no fine-tune needed. Cache key = `(symbol, last_bar_ts)` so per-scan latency stays ~100 ms cold / <5 ms warm. Disable via `bot_config.ml.CHRONOS_ENABLED = False` or env `BOT_DISABLE_CHRONOS=1` → obs[27,28] = 0.0 (graceful degradation to v6.14 behavior).

**Why v7.0.2 formula refactor** — pool diagnostic หลัง v7.0 retrain (Pass Rate 9.7% → 4.0%) เผย:

- `chronos_uncertainty_norm` saturated ที่ 3.0 ใน 96.2% ของ signals → useless dim. สูตรเดิม `(q90-q10)/atr` ไม่คิด √horizon scaling (forecast variance ของ time-series model ขยายตาม √time per Brownian motion).
- `chronos_alignment` corr กับ outcome = **−0.0178** (anti-signal). SMC ค้าขาย swing/reversal สวนเทรนด์ระยะสั้น แต่ Chronos forecast เทรนด์ → "Chronos agree with SMC" = SMC ตามเทรนด์ = ช้าเกิน, "Chronos disagree" = SMC จับ reversal = profitable.

→ flip sign ของ alignment + Brownian-scale uncertainty (factor √8 ≈ 2.83).

**Interpretation (v7.0.2)**:

- `alignment = +1` → SMC + Chronos contrarian (e.g. SMC BUY + Chronos median ต่ำกว่า close) = good reversal setup
- `alignment = −1` → SMC + Chronos agree on trend direction = SMC late, weak setup
- `uncertainty ≈ 1.0` = forecast band ตามที่ ATR คาด, `< 0.5` = market quiet (Chronos มั่นใจ), `> 2.0` = high vol (uncertainty band กว้างกว่าปกติ)

**Pool path** — `StrategyBacktester.generate_episode_signals` calls `ChronosForecaster.forecast_features(symbol, ltf_slice, direction, atr_val)` using closed-bar slice only (anti-lookahead). Result stored in signal dict as `chronos_alignment` + `chronos_uncertainty_norm` for `FTMOSignalFilterEnv._get_obs` to read.

**Live path** — `FTMOTradingBot._build_signal_observation` reads `self._strategy._ltf_data` (M15 cache, refreshed each scan ~60 s) and calls the same `forecast_features` to compute obs[27,28]. `_build_live_context` mirrors the same values into Excel `Signals`/`Trades` sheets.

**Accuracy benchmark (v7.0.7)** — see [`scripts/test_chronos_accuracy.py`](../ftmo_trading_bot/scripts/test_chronos_accuracy.py). Rolling-window backtest บน 10 symbols × 100 forecasts:

| Metric | Result | Interpretation |
|---|---|---|
| Direction Accuracy | **50.5%** | ~random (M15 noise สูง) |
| Quantile Coverage | **79.8%** | calibrated เป๊ะ (target 80%) |
| MAPE | **0.12%** | ทำนาย "level" แม่นมาก |

→ Chronos contribute **มาก** ผ่าน `chronos_uncertainty_norm` (calibrated regime detector) และ **น้อย** ผ่าน `chronos_alignment` (direction marginal). Sub-population ที่มี edge: AUDUSD (56.4%), EURJPY (53.5%).

**Multi-TF consensus benchmark** — see [`scripts/test_chronos_mtf.py`](../ftmo_trading_bot/scripts/test_chronos_mtf.py). Forecast บน H4+H1+M15 พร้อมกัน → consensus direction:

| Consensus | N | Accuracy | Verdict |
|---|---|---|---|
| ALL UP (3/3 ขึ้น) | 148 | 48.0% | similar baseline |
| ALL DOWN (3/3 ลง) | 42 | 35.7% | **anti-signal** ❌ |
| Majority 2/3 | 316 | 46.8% | similar |
| ALL_AGREE 3/3 (รวม) | 190 | **45.3%** | **แย่กว่า baseline** |
| Baseline M15 alone | 506 | 48.2% | reference |

→ **Multi-TF ไม่ improve overall** (45.3% < baseline 48.2%). Sub-population ที่ดีขึ้น: USDCHF (+11 pp), NZDUSD (+4.4 pp), EURUSD (+3.5 pp). แต่ symbol อื่นแย่ลงหนัก (USDCAD -18.9, GBPUSD -18.2). **Conclusion**: ไม่ integrate เข้า v7.0.7 — inference cost 3× ไม่คุ้มกับ uncertain gain.

---

## Reward Structure

Located in `FTMOSignalFilterEnv.step`. See the block comments labelled `Phase 1` / `Phase 2`:

### Phase 1 (Alpha) — learn chart reading + profit taking

```
TAKE (v6.13 — equalize @ ml ≥ 0.36):
    clip(pnl_norm, -1.0, 3.0) - spread_cost*0.5
    + 0.30 if win AND ml_score >= 0.36   ⭐ uniform (เดิม 0.35 vs 0.15 split)
    - 0.35 if big-loss AND ml_score < 0.35   (rare — gate ที่ 0.36)
    + 0.02 action nudge (P1 only) / 0.04 + dd_penalty (P2)

SKIP (Oracle reward — POLICY ไม่เห็น outcome ใน obs, reward ใช้ตอน train เท่านั้น):
    outcome >=  0.5 : -0.30 (P1) / **-0.90** (P2, v6.13: -0.70 → -0.90)
        AND ml >= 0.40: extra -0.25 (P1) / **-0.55** (P2, v6.13: -0.40 → -0.55)
    outcome >=  0.1 : -0.10 (P1) / -0.20 (P2)
    outcome <= -0.5 : +0.30 (P1) / **+0.35** (P2, v6.13: +0.20 → +0.35)
    outcome <= -0.1 : +0.10 (P1) / **+0.10** (P2, v6.13: +0.06 → +0.10)
    + passive cost -0.010/step
```

### Phase 2 (Risk) — add DD management

```
TAKE: Phase 1 reward
    + dd_penalty(total) = -0.5 · (exp(3·ratio) - 1)  when total DD > 4%
    + dd_penalty(daily) = -0.3 · (exp(3·ratio) - 1)  when daily DD > 2.5%
    + -0.5 if risk guard clamped PnL

Activity floor (Phase 2 only, v6.3 B1v2):
    take_rate < 5%                              : -1.0
    take_rate < 12%                             : -0.3
    progress < 30%                              : -0.3
    days >= 40 & takes < 15 & not passed        : -1.0   ← undertrading (terminal)

Mid-episode undertrading checks (Phase 2, sticky one-shot):
    day >= 10 & progress < 20% & takes < 3      : -0.2   ⭐ v6.13 NEW (early signal)
    day >= 20 & progress < 40% & takes < 6      : -0.3   (v6.3 B1v2)
    day >= 35 & progress < 60% & takes < 12     : -0.7   (v6.3 B1v2)
```

**Progress shaping (v6.3 B1v2 — multiplier 2.5× stronger than pre-B1v2):**

```
progress_delta > 0     : reward += 0.05 × progress_delta   (was 0.02 before B1v2)
progress_delta < -5.0  : reward += 0.005 × progress_delta  (drawdown nudge, unchanged)
```

**Progress bonuses (v6.3 B1 — sticky, fires once per milestone):**

```
30% progress : +0.5
60% progress : +1.0
90% progress : +1.5
100% target  : +4.0     ← was +2.0 before B1
Sum if pass  : +7.0 max
```

**Passive SKIP cost**: `-0.010` per step (v6.3 B1, down from `-0.015`) — gives room to SKIP low-quality signals without stacking punishment.

### Outcome Perturbation

- `outcome_noise_std=0.05` (v6.13 default — เพิ่มจาก 0.02) — 5 % Gaussian noise on pool outcomes during env step.
- Purpose: regularization (prevent overfit to fixed pool outcomes; simulate live slippage + spread variability).

---

## PPO Hyperparameters (`scripts/train_signal_filter.py`)

| Phase | Phase 1 (Alpha) | Phase 2 (Risk) |
|-------|-----------------|----------------|
| `enable_risk_penalty` | False | True |
| Steps (default) | 10M | 5M |
| `learning_rate` | 3e-4 | **5e-5** (v7.0.5 — proper fix via `FloatSchedule`) |
| `ent_coef` schedule | 0.05 → 0.005 | 0.01 → 0.002 |
| `EarlyStopOnValueLoss` | threshold=10, patience=5, **warmup=0** | threshold=**20** (v7.0.7 revert จาก 30 หลัง v7.0.6 ตก), patience=5, **warmup=50,000** (v7.0.4) |
| Output | `ppo_signal_filter_p1.zip` | `ppo_signal_filter.zip` |

**Shared**:

- `gamma=0.99` (long-horizon credit)
- `n_steps=4096`, `batch_size=256`, `n_epochs=5`
- `clip_range=0.2`, `gae_lambda=0.95`
- `policy_kwargs.net_arch = dict(pi=[256,128], vf=[256,128])`

**VecNormalize**:

- `norm_obs=True`, `norm_reward=True`
- `clip_obs=10.0`, `clip_reward=20.0` (widened from 10 to accommodate the DD breach penalty)

**SubprocVecEnv**: used when `n_envs > 1` → real multi-core parallelism.

---

## Training Pipeline (run order)

```bash
# Step 1: Pool (~8 min, 8 workers)
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py --pool_size 3000 --workers 8

# Step 2: ML GBM + Isotonic Calibrator (~5 min, re-scores pool in place)
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py

# Step 3: RL PPO + Auxiliary Task (~30–40 hr CPU @ n_envs=8, 2 phases)
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.05 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

The trainer auto-uses `AuxAwarePPO` + `AuxAwareACPolicy` (aux loss weight = 0.5). P2 stability tuning: LR 5e-5, ent_coef 0.02, EarlyStopOnValueLoss threshold 20.

**Evaluation** (default 5000 episodes):

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.36 --risk_per_trade 0.007
```

⚠️ Always invoke via `.venv/bin/python` (or activate venv) — system `python` may resolve to a different interpreter with non-pinned package versions, shifting Pass Rate by 1–3 pp due to RNG init / BLAS differences.

**Targets** (5000-eps evaluation):

- Pass rate ≥ 5 % (excellent > 8 %)
- Breach rate < 2 %
- Win rate > 45 %
- Take rate 50–60 % (ML filter active)

**Verified v6.13 (Phase E2 + Combined Patch)** — risk 0.7 %, ml_threshold 0.36, outcome_noise 0.05, 5000 eps, 2026-04-29: **Pass Rate 9.7 %** ⭐⭐⭐ (vs v6.11.3 baseline 3.4 % = +185 %). Leak-free verified via 5-point audit (obs/GBM/aux head/SKIP-oracle/eval sampling).

**Phase progression history**:

| Phase | Pass Rate | Note |
|-------|----------:|------|
| Old "Option B" | 12.5 % | Leaky baseline (eval seeded with same pool used for GBM training) |
| Honest baseline (B1v2) | 3.5–3.7 % | Leak removed |
| Phase C (SMC 4 principles) | 1.5 % | Reverted — over-filtered pool |
| Phase D (BE+partial+trail in train) | 0.2 % | Reverted — capped winner tail |
| Phase E1 (calibration) | 3.0 % | Calibrator stable, but no Pass Rate boost |
| Phase E2 (auxiliary task, 2026-04-25) | 10.0 % | Pre-v6.11 SMC overhaul |
| v6.11 SMC overhaul (hard gates) | 0.0 % | Pool collapsed -99 % — partial rollback |
| v6.11.2 (rollback) | 2.7 % | Soft bonuses instead of hard gates |
| v6.11.3 (mild relax IDM/ADX) | 3.4 % | Baseline before v6.13 |
| **v6.13 (combined patch, 2026-04-29)** | **9.7 %** | Current verified ⭐ |

---

## Live Inference Flow

```
TradeSignal (from SMCStrategy)
    ↓
SignalQualityModel.score(sig) → ml_score
    ↓
FTMOTradingBot._build_signal_observation(sig) → obs (27 dims)
    ↓
SelfLearningAgent.should_take_signal(obs)
    ├─→ _prepare_obs: validates size == OBS_DIM
    ├─→ _normalize_obs: (obs - _obs_mean) / sqrt(_obs_var + 1e-8)
    ├─→ clip [-clip_obs, +clip_obs]
    └─→ model.predict(deterministic=True)
        └─→ return action[0] > 0.0  (TAKE / SKIP)
```

⚠️ **Obs mismatch**: `SelfLearningAgent._prepare_obs` raises `ValueError` immediately.

---

## Cross-links

- Pipeline architecture → [01-architecture.md](01-architecture.md)
- Module responsibilities → [02-modules.md](02-modules.md)
- Live risk state machine + FTMO rules → [04-operations.md](04-operations.md)
- Red-flag rules (sync rules) → [05-invariants.md](05-invariants.md)

## Invariants & Gotchas

- ⛔ Changing obs dim without retraining breaks the model immediately (mismatch error).
- ⛔ Changing feature order silently breaks the model — no exception; agent just returns wrong values.
- ⚠️ Training risk and live risk must match (`DEFAULT_RISK_PER_TRADE_PCT` in `settings.py` ↔ `--risk_per_trade` at training time).
- ⚠️ `vec_normalize_sf.pkl` must match the RL model's training run (obs normalization stats).
