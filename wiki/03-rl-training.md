# 03 — RL Training (Obs 27 dims, Reward, PPO + Auxiliary Task)
> Last Updated: 2026-04-26 | Scope: RL env, obs space v6, reward shaping, PPO hyperparams, curriculum, aux task (E2)

## TL;DR (30-second scan)

- Obs = **27 dims** (v6, 2026-04-22) — adds `spread_pct_of_atr`, `has_opposite_recently_closed`, `htf_trend_alignment` on top of the previous 24.
- Action continuous [−1, 1] — `>0 = TAKE`, `≤0 = SKIP`.
- **2-phase curriculum**: Phase 1 (Alpha, no DD penalty, oracle SKIP) → Phase 2 (Risk, DD penalty active).
- **Phase E2 — Auxiliary Task** (verified 2026-04-25, Pass Rate **10.0 %**): policy has aux head that predicts `outcome_pnl_ratio`. MSE loss × 0.5 added to PPO loss. Forces representation to encode signal quality.
- ML threshold = **0.36** (calibrated probability). Signals below this are rejected before RL.
- Live inference must normalize obs with the `vec_normalize_sf.pkl` stats captured during training.
- VecNormalize: `norm_obs=True`, `norm_reward=True`, `clip_obs=10.0`, `clip_reward=20.0`.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Obs dims | 27 | `SelfLearningAgent.OBS_DIM`, `FTMOSignalFilterEnv.observation_space` |
| Action space | Box(−1, 1, shape=(1,)) | `FTMOSignalFilterEnv.action_space` |
| VecNormalize stats | `models/vec_normalize_sf.pkl` | loaded by `SelfLearningAgent._load_normalize_stats` |
| RL model | `models/ppo_signal_filter.zip` (aux-aware policy) | loaded by `SelfLearningAgent.initialize_model` |
| RL training PPO | `AuxAwarePPO` (aux loss weight = 0.5) | `ml/aux_aware_ppo.py` |
| RL policy | `AuxAwareACPolicy` (actor + value + `aux_head`) | `ml/aux_aware_policy.py` |
| Aux target | `info['aux_target']` = `outcome_pnl_ratio` | `FTMOSignalFilterEnv.step()` |
| Pool | `data/signal_pool_3000.pkl` | loaded by `FTMOSignalFilterEnv` |
| ML threshold | 0.36 (calibrated) | CLI `--ml_threshold 0.36` |

---

## Observation Space Layout (27 dims)

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

---

## Reward Structure

Located in `FTMOSignalFilterEnv.step`. See the block comments labelled `Phase 1` / `Phase 2`:

### Phase 1 (Alpha) — learn chart reading + profit taking

```
TAKE:
    clip(pnl_norm, -1.5, 3.0)
    + chart_reading_bonus (confluence × outcome alignment)
    + 0.02 action nudge

SKIP (Oracle feedback — agent sees outcome in training):
    outcome >=  0.5 : -0.50  (missed big win; -0.80 if conf >= 75)
    outcome >=  0.1 : -0.16
    outcome <= -0.5 : +0.30  (smart skip of big loss)
    outcome <= -0.1 : +0.10
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

Mid-episode undertrading checks (Phase 2, v6.3 B1v2, sticky one-shot):
    day >= 20 & progress < 40% & takes < 6      : -0.3
    day >= 35 & progress < 60% & takes < 12     : -0.7
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

- `outcome_noise_std=0.02` — 2 % Gaussian noise on pool outcomes during env step.
- Purpose: regularization (prevent overfit to fixed pool outcomes; simulate live slippage).

---

## PPO Hyperparameters (`scripts/train_signal_filter.py`)

| Phase | Phase 1 (Alpha) | Phase 2 (Risk) |
|-------|-----------------|----------------|
| `enable_risk_penalty` | False | True |
| Steps (default) | 10M | 5M |
| `learning_rate` | 3e-4 | 1e-4 |
| `ent_coef` schedule | 0.05 → 0.005 | 0.01 → 0.002 |
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

# Step 3: RL PPO + Auxiliary Task (~2–3 hr CPU, 2 phases)
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 300000 --timesteps_p2 200000 \
    --n_envs 4 --pool_size 3000 --outcome_noise 0.02 \
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

**Verified Phase E2 (Auxiliary Task)** — risk 0.7 %, ml_threshold 0.36, 5000 eps, 2026-04-25: **Pass Rate 10.0 %** ⭐ (3× honest baseline 3.5 %). Verified leak-free via runtime hook + obs feature audit.

**Phase progression history**:

| Phase | Pass Rate | Note |
|-------|----------:|------|
| Old "Option B" | 12.5 % | Leaky baseline (eval seeded with same pool used for GBM training) |
| Honest baseline (B1v2) | 3.5–3.7 % | Leak removed |
| Phase C (SMC 4 principles) | 1.5 % | Reverted — over-filtered pool |
| Phase D (BE+partial+trail in train) | 0.2 % | Reverted — capped winner tail |
| Phase E1 (calibration) | 3.0 % | Calibrator stable, but no Pass Rate boost |
| **Phase E2 (auxiliary task)** | **10.0 %** | Current verified |

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
