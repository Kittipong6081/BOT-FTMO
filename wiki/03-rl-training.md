# 03 — RL Training (Obs 32 dims, MR Reward Shaping, PPO + Auxiliary Task)
> Last Updated: 2026-05-19 (v8.0.43 — Risk 0.7% + Option X trail in pool sim) | Scope: RL env, obs space, reward shaping (MR-specific), PPO hyperparams, curriculum, aux task.

## TL;DR (30-second scan)

- Obs = **32 dims** (v7.1+, production model trained at 32 dims).
- Action continuous [−1, 1] — `>0 = TAKE`, `≤0 = SKIP`.
- **2-phase curriculum**: Phase 1 (Alpha, no DD penalty, oracle SKIP) → Phase 2 (Risk, DD penalty active + MR-specific shaping).
- **Phase E2 — Auxiliary Task**: policy has aux head that predicts `outcome_pnl_ratio`. MSE loss × 0.5 added to PPO loss. Forces representation to encode signal quality.
- **v8.0.5 verified**: Pass Rate **59.30 %** (5000-eps eval), Profitable 89.10 %, Breach 0 %, Total DD max 5.80 %, Daily DD max 3.00 %.
- ML threshold = **0.30** (calibrated probability, v8.0.3). Signals below this are rejected before RL.
- Live inference must normalize obs with the `vec_normalize_mr.pkl` stats captured during training.
- VecNormalize: `norm_obs=True`, `norm_reward=True`, `clip_obs=10.0`, `clip_reward=20.0`.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Obs dims | **32** | `SelfLearningAgent.OBS_DIM`, `FTMOSignalFilterEnv.observation_space` |
| Chronos forecaster | `ChronosForecaster` (`amazon/chronos-bolt-small`, optional) | `ml/chronos_forecaster.py` |
| Action space | Box(−1, 1, shape=(1,)) | `FTMOSignalFilterEnv.action_space` |
| **VecNormalize stats** | `models/mr/best/vec_normalize_mr.pkl` | loaded by `SelfLearningAgent._load_normalize_stats` |
| **RL model** | `models/mr/best/ppo_mr_filter.zip` (aux-aware policy) | loaded by `SelfLearningAgent.initialize_model` |
| RL training PPO | `AuxAwarePPO` (aux loss weight = 0.5) | `ml/aux_aware_ppo.py` |
| RL policy | `AuxAwareACPolicy` (actor + value + `aux_head`) | `ml/aux_aware_policy.py` |
| Aux target | `info['aux_target']` = `outcome_pnl_ratio` | `FTMOSignalFilterEnv.step()` |
| **MR env** | `MeanReversionFilterEnv(FTMOSignalFilterEnv)` | `ml/mean_reversion_env.py` |
| **Pool** | `data/mr_signal_pool_3000.pkl` (~309 MB, gitignored) | `MeanReversionBacktester` |
| **ML threshold** | **0.30** (live = `FTMOConfig.ML_FILTER_THRESHOLD` — train must match) | CLI `--ml_threshold 0.30` |
| **DAILY_DD_GUARD (env)** | 0.030 (3.0%) | `FTMOSignalFilterEnv.DAILY_DD_GUARD` |
| **TOTAL_DD_GUARD (env)** | 0.058 (5.8%) | `FTMOSignalFilterEnv.TOTAL_DD_GUARD` |

---

## Observation Space Layout (32 dims)

**Must be kept in sync in three places**:

1. `FTMOSignalFilterEnv._get_obs` in `ftmo_trading_bot/ml/signal_filter_env.py`
2. `MeanReversionFilterEnv._get_obs` in `ftmo_trading_bot/ml/mean_reversion_env.py` (overrides obs[4], obs[10], obs[26])
3. `FTMOTradingBot._build_signal_observation` in `ftmo_trading_bot/main.py`
4. `SelfLearningAgent.OBS_DIM` in `ftmo_trading_bot/ml/rl_agent.py`

**MR-specific obs slot reinterpretation (v8.0)**:

| Slot | Original (SMC) | MR semantic |
|------|----------------|-------------|
| obs[4] | `ob_norm` (Order Block score) | `bb_extreme` (BB band penetration depth, 0..1) |
| obs[10] | `ob_size_atr` (OB body / ATR) | `bb_band_width_atr / 3` (BB band width / ATR, clip 0..1) |
| obs[26] | `htf_trend_alignment` | `adx_inverse_norm` (1.0 when ADX low = ranging = MR-friendly) |
| obs[27,28] | Chronos forecast features | (same — disabled in MR pipeline by default, set to 0) |

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

### v7 Chronos Forecast [27–28] (2026-05-06, semantics refactor v7.2)

| Idx | Feature | Source | Normalization (v7.2) |
|-----|---------|--------|---------------|
| 27 | `chronos_alignment` | `ChronosForecaster.forecast_features` (Amazon Chronos 2 zero-shot, M15 × 8 ahead) | `direction × sign(median_h+8 − close)`, clip ±1 — un-flipped vs v7.0.2 (live audit fix) |
| 28 | `chronos_uncertainty_norm` | `ChronosForecaster.forecast_features` (q90 − q10) | `log1p((q90 − q10) / (atr_value × √8)) / 2`, clip [0, 3] — Brownian-scaled + log1p (v7.1) |

**Why v7**: gives the agent forward-looking probabilistic context. Chronos 2 is a zero-shot foundation model — no fine-tune needed. Cache key = `(symbol, last_bar_ts)` so per-scan latency stays ~100 ms cold / <5 ms warm. Disable via `bot_config.ml.CHRONOS_ENABLED = False` or env `BOT_DISABLE_CHRONOS=1` → obs[27,28] = 0.0 (graceful degradation to v6.14 behavior).

**Why v7.2 un-flip** — live audit ของ `logs/ftmo_trades.xlsx` 2026-05-05 (524 signals, strong DOWN-trend day, 100% SELL) พบ `chronos_alignment = -1` ติดทุก signal:

- ตลาดลง → Chronos median forecast ก็ลง → `last_close > median_h` → `delta > 0` → v7.0.2 formula `forecast_dir = +1` → `alignment = SELL(-1) × +1 = -1`
- v7.0.2 design assumed SMC = contrarian/reversal strategy → alignment +1 = "Chronos disagree = good reversal setup"
- ความจริง: SMC ที่ใช้จริงรัน **trend-following** (Tier 1 hard veto Counter-D1, HTF bias filter) — ไม่ใช่ contrarian
- Reward penalty (`signal_filter_env.py:677-682`) `if chronos_align < 0 → reward -= 0.30 (ml<0.55) หรือ -0.10` → ลงทุก TAKE → agent learned skip-default
- Live consequence: SKIP signal ML 0.693 (สูงสุด!), TAKE สูงสุดแค่ 0.598 = discrimination ผิดทาง

→ **Fix v7.2**: un-flip → `delta = median_h - last_close`. Reward penalty ไม่แก้ — semantics ใหม่ -1 = warning จริง (counter-trend) → penalty ใช้งานได้ปกติ

**Interpretation (v7.2)**:

- `alignment = +1` → SMC + Chronos agree on direction (good — trend-following confirmed, e.g. SMC SELL + Chronos median ต่ำกว่า close)
- `alignment = −1` → SMC สวน Chronos forecast (warning — counter-trend setup, reward penalty active)
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

## Reward Structure (v8.0 — MR-specific)

Located in `MeanReversionFilterEnv.step` (overrides `FTMOSignalFilterEnv.step`). MR shaping replaces SMC's confluence-bonus logic with capital-preservation focused signals.

### TAKE branch (MR-specific shaping)

```text
Base PnL (always):
    pnl_norm = pnl / risk_amount
    pnl_norm -= spread_cost_R × 0.5   (clip 0..1)
    reward = clip(pnl_norm, -1.0, 3.0)

Win shaping (pnl_norm > 0):
    + QUICK_TP_BONUS = 0.50  if bars_to_resolution ≤ 5 (quick TP win)
    + SLOW_WIN_BONUS = 0.20  otherwise (slow win)
    + 0.15  if ml_score ≥ 0.40 (quality alignment)

Loss shaping (pnl_norm ≤ 0):
    - BASE_LOSS_PENALTY = 0.10  baseline
    - DURATION_FINE_COEF × (bars_to_resolution - 1)  per-bar fine
                                                    (cap at DURATION_FINE_CAP = 0.30)
    - PROLONGED_LOSS_PENALTY = 0.40  if bars_to_resolution ≥ 12

Trend filter discipline:
    - ADX_VIOLATION_PENALTY = 0.30  if signal.adx > 25 (defense in depth)

Phase 1: + 0.02 activity nudge
Phase 2: + 0.04 nudge + DD penalty + clamp penalty (-0.25 if guard fired)
```

### SKIP branch (Oracle reward, agent doesn't see outcome in obs)

```text
outcome ≥  0.5 (would-win big):
    -0.20 (P1) / -0.55 (P2)
    +ml ≥ 0.40 extra: -0.15 (P1) / -0.30 (P2)
outcome ≥  0.1 :  -0.05 (P1) / -0.10 (P2)
outcome ≤ -0.5 (smart skip — capital preservation reward, MR-tuned):
    +0.30 (P1) / +0.45 (P2)
    +ml < 0.36 extra: +0.15
outcome ≤ -0.1 :  +0.10 (P2 only)

Passive SKIP cost: -0.012 / step (MR slightly higher than SMC's -0.010 to push more TAKE)
```

### Phase 2 (Risk) — add DD management

```text
TAKE: Phase 1 reward
    + dd_penalty(total) = -0.5 · (exp(3·ratio) - 1)  when total DD > 4%
    + dd_penalty(daily) = -0.3 · (exp(3·ratio) - 1)  when daily DD > 2.5%
    + -0.25 if risk guard clamped PnL

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
- **v8.0.10 anti-overfit retrain**: bumped to **0.08** alongside larger pool (5000) + longer P2 (5M) — fights episode-memorization shown by holdout_eval gap.
- Purpose: regularization (prevent overfit to fixed pool outcomes; simulate live slippage + spread variability).

---

## PPO Hyperparameters (`scripts/train_mr_signal_filter.py`, v8.0)

| Phase | Phase 1 (Alpha) | Phase 2 (Risk) |
|-------|-----------------|----------------|
| `enable_risk_penalty` | False | True |
| Steps (default) | **5M** (v8.0 reduced) | **2M** (v8.0 reduced) |
| `learning_rate` | 3e-4 | **5e-5** (proper fix via `FloatSchedule`) |
| `ent_coef` schedule | 0.05 → 0.015 | 0.02 → 0.010 |
| `EarlyStopOnValueLoss` | threshold=10, patience=5, **warmup=0** | threshold=20, patience=5, **warmup=50,000** |
| Output | `models/mr/ppo_mr_filter_p1.zip` (gitignored) | `models/mr/ppo_mr_filter.zip` |

**Shared (v8.0)**:

- `gamma=0.99` (long-horizon credit)
- `n_steps=8192` (v7.1.7+, larger batch), `batch_size=512`, `n_epochs=5`
- `clip_range=0.2`, `gae_lambda=0.95`
- `policy_kwargs.net_arch = dict(pi=[256,128], vf=[256,128])`
- `policy_kwargs.optimizer_kwargs = dict(weight_decay=1e-5)`
- Aux loss weight = 0.5

**VecNormalize**:

- `norm_obs=True`, `norm_reward=True`
- `clip_obs=10.0`, `clip_reward=20.0` (widened from 10 to accommodate the DD breach penalty)

**SubprocVecEnv**: used when `n_envs > 1` → real multi-core parallelism.

---

## Training Pipeline (v8.0 — autonomous orchestrator)

**Recommended path** — autonomous loop (build → GBM → RL → eval → self-correct):

```bash
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 10 --max_hours 60 \
    --pool_size 3000 --timesteps_p1 5000000 --timesteps_p2 2000000 \
    --target_pass_rate 0.08 --target_dd_max 0.06 \
    --target_daily_dd_max 0.035 --target_profitable 0.55
```

**v8.0.10 anti-overfit retrain** — larger pool + more noise + longer P2:

```bash
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 3 --max_hours 12 \
    --pool_size 5000 --timesteps_p1 5000000 --timesteps_p2 5000000 \
    --outcome_noise 0.08 \
    --target_pass_rate 0.08 --target_dd_max 0.06 \
    --target_daily_dd_max 0.035 --target_profitable 0.55
```

**Anti-overfit verification** (run after pipeline finishes):

```bash
.venv/bin/python ftmo_trading_bot/scripts/holdout_eval.py \
    --train_pool data/mr_signal_pool_5000.pkl \
    --holdout_pool data/mr_signal_pool_holdout.pkl \
    --n_episodes 2000
```

Verdict: Δ Pass Rate ≤ 5 pp = HEALTHY, 5-10 pp = MILD, > 10 pp = OVERFIT (exits 1).

**Manual path** (3 steps):

```bash
# Step 1: Pool (~11 min, 8 workers)
.venv/bin/python ftmo_trading_bot/scripts/build_mr_signal_pool.py \
    --pool_size 3000 --workers 8

# Step 2: GBM + Isotonic Calibrator (~2 min, re-scores pool in place)
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_quality.py

# Step 3: RL PPO + Auxiliary Task (~10-12 min CPU @ n_envs=8, 2 phases)
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py --fresh \
    --timesteps_p1 5000000 --timesteps_p2 2000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.05 \
    --ml_threshold 0.30 --risk_per_trade 0.0099
```

**Evaluation** (default 5000 episodes):

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.30 --risk_per_trade 0.0099
```

⚠️ Always invoke via `.venv/bin/python` (or activate venv) — system `python` may resolve to a different interpreter with non-pinned package versions, shifting Pass Rate by 1–3 pp due to RNG init / BLAS differences.

**Eval gates (autonomous loop stops when all pass)**:

- Pass rate ≥ 8 %
- Total DD max ≤ 6 % (env guard 5.8 %)
- Daily DD max ≤ 3.5 % (env guard 3.0 %)
- Profitable rate ≥ 55 %
- Breach rate ≤ 5 %

**Verified v8.0.5** — pool_size 3000, ml_threshold 0.30, risk 0.0099, env guards 3.0/5.8, 5000 eps, 2026-05-07: **Pass Rate 59.30 %** ⭐⭐⭐, Profitable 89.10 %, Breach 0.00 %, Total DD max 5.80 %, Daily DD max 3.00 %, Profit avg +7.23 %.

**Phase progression history (v8.0 path)**:

| Version | Pass Rate | Note |
|-------|----------:|------|
| **v8.0.5 (locked-in 2026-05-07)** | **59.30 %** | All 5 gates passed iter 1, 12 min training ⭐ |
| v8.0.4 (DAILY_DD_GUARD 4%→3%) | 58.32 % | Daily DD freed from 4% pin |
| v8.0.3 (ml_threshold 0.40→0.30) | 61.70 % | Smart auto-tune kicked in |
| v8.0.2 (relax BB/RSI + scan 48/day) | 2.52 % | Bot under-trading at ml=0.40 |
| v8.0 (initial) | n/a | pool yield too low, never converged |

(For pre-v8.0 SMC progression see `git log -- wiki/05-invariants.md` — Phase C/D/E1/E2/v6.11/v6.13.)

---

## Live Inference Flow (v8.0+)

```text
MRSignal (from LiveMRScanner.scan_all_symbols → MeanReversionStrategy.analyze_with_data)
    ↓
SignalQualityModel.score(sig) → ml_score    (loads data/mr_signal_quality_model.pkl)
    ↓
ML gate: if ml_score < bot_config.ftmo.ML_FILTER_THRESHOLD (0.30) → log "ML_FILTERED" + skip
    ↓
FTMOTradingBot._build_signal_observation(sig) → obs (32 dims)
    obs[4]=bb_extreme, obs[10]=bb_band_width/3, obs[26]=adx_inverse_norm
    ↓
SelfLearningAgent.should_take_signal(obs)
    ├─→ _prepare_obs: validates size == OBS_DIM (32)
    ├─→ _normalize_obs: (obs - _obs_mean) / sqrt(_obs_var + 1e-8)
        (uses models/mr/best/vec_normalize_mr.pkl stats)
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
