# 01 — Architecture (3-Brain Pipeline + v7 Chronos Forecaster)
> Last Updated: 2026-05-01 (v7.0) | Scope: system overview + data flow

## TL;DR (30-second scan)

- Three brains + 1 forecaster: **SMC Strategy** (rule-based) → **ML Quality** (GBM + Isotonic calibrator) → **Chronos Forecaster** (v7, Amazon Chronos 2 zero-shot, feeds obs[27,28]) → **RL Agent** (PPO + Auxiliary Task — TAKE/SKIP).
- Live entry point: `FTMOTradingBot` in `ftmo_trading_bot/main.py`.
- Training pipeline: `build_signal_pool.py` → `train_signal_quality.py` → `train_signal_filter.py`.
- Observation = **29 dims** (v7, 2026-05-01 — adds Chronos forecast features). Must stay in sync across `FTMOSignalFilterEnv._get_obs`, `FTMOTradingBot._build_signal_observation`, and `SelfLearningAgent.OBS_DIM`.
- Every live decision flows through gates in this order: **Risk → Session → Strategy → ML gate (v6.12) → RL → Execute → Manage**.
- v6.13 verified: Pass Rate **9.7 %** (5000-eps eval) via PPO + auxiliary head predicting `outcome_pnl_ratio` (MSE weight=0.5). +185 % vs v6.11.3 baseline 3.4 %.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Live entry class | `FTMOTradingBot` | `main.py` |
| Strategy brain | `SMCStrategy` | `strategy/smc_strategy.py` |
| ML brain | `SignalQualityModel` (GBM + isotonic calibrator) | `ml/signal_quality.py` |
| RL brain | `SelfLearningAgent` (PPO inference) | `ml/rl_agent.py` |
| RL training env | `FTMOSignalFilterEnv` (shape=(29,) v7) | `ml/signal_filter_env.py` |
| Chronos forecaster (v7) | `ChronosForecaster` (`amazon/chronos-bolt-small`) | `ml/chronos_forecaster.py` |
| RL training PPO | `AuxAwarePPO` (PPO + aux MSE loss) | `ml/aux_aware_ppo.py` |
| RL policy | `AuxAwareACPolicy` (actor + value + aux head) | `ml/aux_aware_policy.py` |
| RL rollout buffer | `AuxRolloutBuffer` (adds `aux_targets`) | `ml/aux_rollout_buffer.py` |
| Pool generator | `StrategyBacktester` | `ml/strategy_backtester.py` |
| Risk gate | `RiskManager` | `core/risk_manager.py` |
| Order gate | `TradeExecutor` | `execution/trade_executor.py` |
| Position lifecycle | `TradeManager` | `execution/trade_manager.py` |
| Live logger | `TradeLogger` (Excel: 4 sheets, 64+21 cols) | `analytics/trade_logger.py` |

---

## Data Flow (Live)

```text
OHLCV (M15/H1/H4) from MT5Connector
        ↓
SMCStrategy.scan_signal()
   └─→ TradeSignal + features (confluence, OB, FVG, BOS, RSI, ADX, ...)
        ↓
SignalQualityModel.score(sig)
   └─→ ml_score ∈ [0, 1]  (GBM P(win))
        ↓
ChronosForecaster.forecast_features(symbol, m15_df, direction, atr)   ← v7
   └─→ chronos_alignment, chronos_uncertainty_norm  (cached per (symbol, last_bar_ts))
        ↓
FTMOTradingBot._build_signal_observation(sig)
   └─→ obs (29 dims, v7)  — signal core + market regime + ML + portfolio + cost/flip/HTF + chronos
        ↓
SelfLearningAgent.should_take_signal(obs)
   └─→ TAKE / SKIP
        ↓ (if TAKE)
TradeExecutor.execute_trade(sig)
   └─→ MT5 market order (always carries SL/TP)
        ↓
TradeManager.update_positions()
   └─→ trailing SL, partial close, session close
```

---

## Loop Priority (every tick of `FTMOTradingBot.run`)

Do not reorder — the Risk gate must come first:

1. **`RiskManager.check_state()`** — skip the tick if state is DAILY_HALT / MAX_DRAWDOWN_HALT / MANUAL_HALT.
2. **`TimeManager.is_trading_session()`** — verify session window (London/NY) and Friday cutoff.
3. **`NewsCalendarScheduler.check_and_run()`** — import CSV on Sunday 23:30 EET; blackout around news events.
4. **`SMCStrategy.scan_all_symbols()`** — generate signals that pass `MIN_CONFLUENCE_SCORE` (every 12 loops ≈ 1 min, not every tick).
5. **`SignalQualityModel.score()`** — attach `ml_score` to each signal (GBM raw → isotonic calibrator → calibrated probability).
6. **`SelfLearningAgent.should_take_signal()`** — decide TAKE / SKIP.
7. **`TradeExecutor.execute_signal()`** — receives `live_context` (ml_score, ADX, biases, account state, **`obs_27_json`** for retrain) → final risk check + lot sizing + send order. Logs to `TradeLogger` Trades sheet.
8. **`TradeManager.update_positions()`** — trailing / BE / partial close for every open position; mirrors state to `ExecutedTrade.be_moved`/`partial_closed_flag`/`trailing_active` for logging.

**Console quiet mode (v6.9):** idle states (Daily Halt / Friday close / Weekend / Daily Close 23:30 / Rollover) print **once on entry** via `_*_announced` flags, not per-loop. Per-signal `AGENT_SKIP` and `NO_AGENT` fall-through go to `Signals` Excel sheet only. Only `📡 [Agent] TAKE` and trade open/close events surface to console. See [04-operations.md § Quiet Mode](04-operations.md).

---

## Training Pipeline (Offline)

Three steps, in order. The order matters: pool must exist before ML training; ML must exist before RL training (because RL uses `ml_score` as an obs feature).

| Step | Script | Class / function | Output |
|------|--------|------------------|--------|
| 1 | `scripts/build_signal_pool.py` | `StrategyBacktester.generate_episode_signals` × N (multiprocessing) | `data/signal_pool_3000.pkl` |
| 2 | `scripts/train_signal_quality.py` | GBM + `GroupKFold cross_val_predict` (OOF) → `IsotonicRegression` calibrator | `data/signal_quality_model.pkl` (model + calibrator) + in-place pool re-score |
| 3 | `scripts/train_signal_filter.py` | `AuxAwarePPO` + `AuxAwareACPolicy` + `FTMOSignalFilterEnv` (2-phase curriculum, aux loss weight=0.5) | `models/ppo_signal_filter.zip` + `models/vec_normalize_sf.pkl` |

Details on PPO config, reward, and curriculum → [03-rl-training.md](03-rl-training.md).

---

## Directory Layout

```text
ftmo_trading_bot/
├── main.py                  ← FTMOTradingBot (live entry)
├── config/                  ← settings.py + news_*.py
├── strategy/                ← SMC brain (smc_strategy + 5 detectors)
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
