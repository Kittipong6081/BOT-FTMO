# 01 — Architecture (3-Brain MR Pipeline + Chronos Forecaster)
> Last Updated: 2026-05-30 (v8.1-phase1 — optional dual-strategy regime router, default OFF) | Scope: system overview + data flow

## TL;DR (30-second scan)

- **v8.1 dual-strategy (opt-in, `bot_config.tf.enabled`, default OFF)**: when ON, a `StrategyRouter` + `MarketRegimeClassifier` sits BEFORE the scanners — each symbol's H1 regime arms exactly one strategy (RANGING→MR, TRENDING→TF, AMBIGUOUS→none). MR (`LiveMRScanner`, M15) and TF (`TrendFollowingScanner`, H1) own separate per-symbol caches; signals carry `strategy_id` and route obs/context to their producing scanner via `FTMOTradingBot._strategy_for(sig)`. **Default OFF → the flow below is exactly single-strategy MR.** TF is paper-mode until Phase 2. Details → [05-invariants.md v8.1](05-invariants.md).

- Three brains + 1 forecaster: **MR Strategy** (rule-based, BB + RSI + ADX trend filter) → **ML Quality** (GBM + Isotonic calibrator) → **Chronos Forecaster** (Amazon Chronos 2 zero-shot, feeds obs[27,28] when enabled) → **RL Agent** (PPO + Auxiliary Task — TAKE/SKIP).
- Live entry point: `FTMOTradingBot` in `ftmo_trading_bot/main.py`. Calls `LiveMRScanner.scan_all_symbols()` (drop-in for legacy `SMCStrategy.scan_all_symbols`).
- Training pipeline: `build_mr_signal_pool.py` → `train_mr_signal_quality.py` → `train_mr_signal_filter.py` (orchestrated by `auto_train_pipeline.py`).
- Observation = **32 dims** (v8 production model). Must stay in sync across `FTMOSignalFilterEnv._get_obs`, `MeanReversionFilterEnv._get_obs` (overrides 3 slots), `FTMOTradingBot._build_signal_observation`, and `SelfLearningAgent.OBS_DIM`.
- Every live decision flows through gates in this order: **Risk → Time (rollover/daily-close/weekend) → News → MR Strategy → ML gate → RL → [Execute pre-gates: Spread Spike → Entry Confirmation] → Order → Manage**.
- **v8.0.55 Execute pre-gates** (NEW): `TradeExecutor._check_spread_spike` (median 30 bars, > 2x → SKIP, broker-agnostic) → `TradeExecutor._check_entry_confirmation` (slip 0.30R + M1 direction + BB %B still extreme). `RiskManager` cluster cooldown gate (300s / 600s same theme) also added at Risk step.
- **v8.0.5 verified**: Pass Rate **59.30 %** (5000-eps eval), Profitable Rate 89.10 %, Breach 0 %, Total DD max 5.80 % (env guard 5.8%), Daily DD max 3.00 % (env guard 3.0%), Profit avg +7.23 %. PPO + auxiliary head predicting `outcome_pnl_ratio` (MSE weight=0.5).

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Live entry class | `FTMOTradingBot` | `main.py` |
| **Strategy brain (v8.0+)** | `LiveMRScanner` (wraps `MeanReversionStrategy`) | `strategy/mean_reversion_strategy.py` |
| ML brain | `SignalQualityModel` (GBM + isotonic calibrator) | `ml/signal_quality.py` |
| RL brain | `SelfLearningAgent` (PPO inference, path-aware: `models/mr/` first) | `ml/rl_agent.py` |
| **RL training env (v8.0+)** | `MeanReversionFilterEnv(FTMOSignalFilterEnv)` (shape=(32,)) | `ml/mean_reversion_env.py` |
| Chronos forecaster | `ChronosForecaster` (`amazon/chronos-bolt-small`, optional) | `ml/chronos_forecaster.py` |
| RL training PPO | `AuxAwarePPO` (PPO + aux MSE loss) | `ml/aux_aware_ppo.py` |
| RL policy | `AuxAwareACPolicy` (actor + value + aux head) | `ml/aux_aware_policy.py` |
| RL rollout buffer | `AuxRolloutBuffer` (adds `aux_targets`) | `ml/aux_rollout_buffer.py` |
| **Pool generator (v8.0+)** | `MeanReversionBacktester(StrategyBacktester)` | `ml/mean_reversion_backtester.py` |
| Risk gate | `RiskManager` | `core/risk_manager.py` |
| Order gate | `TradeExecutor` | `execution/trade_executor.py` |
| Position lifecycle | `TradeManager` | `execution/trade_manager.py` |
| **Live logger (v8.0.6)** | `TradeLogger` (Excel: 4 sheets, 58+20 cols, `_COL`/`_SCOL` lookup) | `analytics/trade_logger.py` |
| **Audits** | `leakage_audit.py` + `parity_audit.py` (mandatory before commit) | `scripts/` |
| **Autonomous orchestrator** | `auto_train_pipeline.py` (Build → GBM → RL → Eval → Self-correct loop) | `scripts/` |

---

## Data Flow (Live, v8.0+)

```text
OHLCV (M15/H1) from MT5Connector  (H4 not used in MR — kept for indicator parity)
        ↓
LiveMRScanner.scan_all_symbols() → MeanReversionStrategy.analyze_with_data()
   └─→ MRSignal (BB %B + RSI + ADX H1 ≤ 30 + reversal-wick + ATR floor)
        ↓
SignalQualityModel.score(sig)   (loads data/mr_signal_quality_model.pkl)
   └─→ ml_score ∈ [0, 1]  (GBM P(win), AUC ~0.59)
        ↓
ML gate: filter signals with ml_score < 0.30 → log as "ML_FILTERED" and skip
        ↓
ChronosForecaster.forecast_features(...)   (optional — env BOT_DISABLE_CHRONOS=1 skips)
   └─→ chronos_alignment, chronos_uncertainty_norm
        ↓
FTMOTradingBot._build_signal_observation(sig)
   └─→ obs (32 dims) — signal core + market regime + ML + portfolio + cost/flip + chronos + realtime
        Reinterpreted slots for MR: obs[4]=bb_extreme, obs[10]=bb_band_width/3,
        obs[26]=adx_inverse_norm
        ↓
SelfLearningAgent.should_take_signal(obs)   (loads models/mr/best/ppo_mr_filter.zip)
   └─→ TAKE / SKIP
        ↓ (if TAKE)
TradeExecutor.execute_signal(sig, live_context)
   └─→ MT5 market order (always carries SL=1.0×ATR, TP=1.0×SL — RR 1:1 quick TP)
        ↓
TradeManager.update_positions()
   └─→ BE @ RR 1.0, trailing, partial close, session close
```

---

## Loop Priority (every tick of `FTMOTradingBot.run`, v8.0.5)

Do not reorder — the Risk gate must come first:

1. **`RiskManager.check_state()`** — skip the tick if state is DAILY_HALT / MAX_DRAWDOWN_HALT / MANUAL_HALT.
2. **Time gates** (in order) —
   - `TimeManager.is_friday_close_time()` — Friday >= 20:45 EET (FTMO weekend rule)
   - `TimeManager.is_weekend()` — Sat/Sun (announce-once, sleep until Mon 00:00)
   - `TimeManager.is_daily_close_time()` — Mon-Thu 23:30-23:55 EET (FTMO zero-overnight)
   - `TimeManager.is_rollover_period()` — 23:55-01:05 EET (spread-spike protection)
3. **`NewsCalendarScheduler.check_and_run()`** — import CSV on Sunday 23:30 EET; blackout 30 min before / 15 min after high-impact news.
4. **`LiveMRScanner.scan_all_symbols()`** — produce `MRSignal` per symbol (every 12 loops ≈ 1 min, not every tick). MR scans 10 symbols on M15 + H1 ADX filter.
5. **ML pre-filter** — `bot_config.ftmo.ML_FILTER_THRESHOLD = 0.30`. Signals with ml_score below threshold logged as "ML_FILTERED" and skipped before agent.
6. **`SelfLearningAgent.should_take_signal(obs)`** — decide TAKE / SKIP from 32-dim obs.
7. **`TradeExecutor.execute_signal()`** — receives `live_context` (ml_score, ADX, account state, **`obs_27_json` (32 dims, key kept for back-compat)** for retrain) → final risk check + lot sizing + send order. Logs to `TradeLogger` Trades sheet (58 cols).
8. **`TradeManager.update_positions()`** — trailing / BE / partial close per position.

**Console quiet mode:** idle states print **once on entry** via `_*_announced` flags. Per-signal `AGENT_SKIP` / `ML_FILTERED` / `NO_AGENT` go to `Signals` Excel sheet only. Only `📡 [Agent] TAKE` + trade open/close events surface to console. See [04-operations.md § Quiet Mode](04-operations.md).

---

## Training Pipeline (Offline, v8.0+)

Three steps, orchestrated by `auto_train_pipeline.py`. The order matters: pool → GBM → RL (RL uses `ml_score` as an obs feature).

| Step | Script | Class / function | Output |
|------|--------|------------------|--------|
| 1 | `scripts/build_mr_signal_pool.py` | `MeanReversionBacktester.generate_episode_signals` × N (8 workers) | `data/mr_signal_pool_<N>.pkl` (~309 MB at N=3000, gitignored) |
| 2 | `scripts/train_mr_signal_quality.py` | GBM + `GroupKFold cross_val_predict` (OOF) → `IsotonicRegression` calibrator + drift baseline | `data/mr_signal_quality_model.pkl` + in-place pool re-score |
| 3 | `scripts/train_mr_signal_filter.py` | `AuxAwarePPO` + `AuxAwareACPolicy` + `MeanReversionFilterEnv` (2-phase curriculum, aux loss weight=0.5) | `models/mr/ppo_mr_filter.zip` + `models/mr/vec_normalize_mr.pkl` |
| ⓘ | `scripts/auto_train_pipeline.py` | Orchestrator: runs 1-3 → evals → tunes hyperparams via `tune_hyperparams()` → loops up to `--max_iterations` or `--max_hours` budget | `logs/auto_train_pipeline_state.json` + `models/mr/best/` (best snapshot) |

Audits before deploy (mandatory):

```bash
.venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py    # exit 0 = no leakage
.venv/bin/python ftmo_trading_bot/scripts/parity_audit.py     # exit 0 = train↔live aligned
```

Details on PPO config, reward, and curriculum → [03-rl-training.md](03-rl-training.md).

---

## Directory Layout

```text
ftmo_trading_bot/
├── main.py                  ← FTMOTradingBot (live entry)
├── config/                  ← settings.py + news_*.py
├── strategy/                ← MR brain (mean_reversion_strategy + indicators) — SMC removed v8.0.6
├── ml/                      ← ML+RL brain (signal_quality, rl_agent, env, backtester)
├── core/                    ← risk_manager, mt5_connector, time_manager, position_sizer, news_scheduler, notifier
├── execution/               ← trade_executor, trade_manager
├── analytics/               ← performance + trade_logger (Excel: Trades 64 cols / Signals 21 cols / Daily / Stats)
├── scripts/                 ← training + data fetch scripts
├── data/                    ← OHLCV CSVs + signal_pool + ml_model pkl (with isotonic calibrator)
├── models/                  ← ppo_signal_filter.zip + vec_normalize_sf.pkl (aux-aware policy weights)
└── logs/                    ← bot_state.json + ftmo_trades.xlsx + tensorboard + news_scheduler_state
```

---

## Cross-links

- Module details → [02-modules.md](02-modules.md)
- RL obs layout, reward, PPO config → [03-rl-training.md](03-rl-training.md)
- Live operations (loop, risk state machine, news, FTMO) → [04-operations.md](04-operations.md)
- Red-flag rules + version log → [05-invariants.md](05-invariants.md)

## Invariants & Gotchas

- ⛔ Do not reorder loop priority — Risk gate stays first.
- ⚠️ Obs 27 dims must be synced in all 3 places (env / main / agent). See [05-invariants.md](05-invariants.md).
- ⚠️ Pool / ML model / RL model form a dependency chain: changing obs requires rebuilding all three.
