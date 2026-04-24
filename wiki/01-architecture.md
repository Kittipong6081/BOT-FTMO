# 01 — Architecture (3-Brain Pipeline)
> Last Updated: 2026-04-24 | Scope: system overview + data flow

## TL;DR (30-second scan)

- Three brains: **SMC Strategy** (rule-based) → **ML Quality** (GBM filter) → **RL Agent** (PPO TAKE/SKIP).
- Live entry point: `FTMOTradingBot` in `ftmo_trading_bot/main.py`.
- Training pipeline: `build_signal_pool.py` → `train_signal_quality.py` → `train_signal_filter.py`.
- Observation = 27 dims (v6, 2026-04-22). Must stay in sync across `FTMOSignalFilterEnv._get_obs`, `FTMOTradingBot._build_signal_observation`, and `SelfLearningAgent.OBS_DIM`.
- Every live decision flows through gates in this order: **Risk → Session → Strategy → ML → RL → Execute → Manage**.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Live entry class | `FTMOTradingBot` | `main.py` |
| Strategy brain | `SMCStrategy` | `strategy/smc_strategy.py` |
| ML brain | `SignalQualityModel` (GBM) | `ml/signal_quality.py` |
| RL brain | `SelfLearningAgent` (PPO inference) | `ml/rl_agent.py` |
| RL training env | `FTMOSignalFilterEnv` | `ml/signal_filter_env.py` |
| Pool generator | `StrategyBacktester` | `ml/strategy_backtester.py` |
| Risk gate | `RiskManager` | `core/risk_manager.py` |
| Order gate | `TradeExecutor` | `execution/trade_executor.py` |
| Position lifecycle | `TradeManager` | `execution/trade_manager.py` |

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
FTMOTradingBot._build_signal_observation(sig)
   └─→ obs (27 dims)  — signal core + market regime + ML + portfolio + cost/flip/HTF
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
4. **`SMCStrategy.scan_all_symbols()`** — generate signals that pass `MIN_CONFLUENCE_SCORE`.
5. **`SignalQualityModel.score()`** — attach `ml_score` to each signal.
6. **`SelfLearningAgent.should_take_signal()`** — decide TAKE / SKIP.
7. **`TradeExecutor.execute_trade()`** — final risk check + lot sizing + send order.
8. **`TradeManager.update_positions()`** — trailing / BE / partial close for every open position.

---

## Training Pipeline (Offline)

Three steps, in order. The order matters: pool must exist before ML training; ML must exist before RL training (because RL uses `ml_score` as an obs feature).

| Step | Script | Class / function | Output |
|------|--------|------------------|--------|
| 1 | `scripts/build_signal_pool.py` | `StrategyBacktester.generate_episode_signals` × N (multiprocessing) | `data/signal_pool_3000.pkl` |
| 2 | `scripts/train_signal_quality.py` | `SignalQualityModel.train_from_pool` | `data/signal_quality_model.pkl` + in-place pool re-score |
| 3 | `scripts/train_signal_filter.py` | PPO + `FTMOSignalFilterEnv` (2-phase curriculum) | `models/ppo_signal_filter.zip` + `models/vec_normalize_sf.pkl` |

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
├── analytics/               ← performance, trade_logger
├── scripts/                 ← training + data fetch scripts
├── data/                    ← OHLCV CSVs + signal_pool + ml_model pkl
├── models/                  ← ppo_signal_filter.zip + vec_normalize_sf.pkl
└── logs/                    ← bot_state.json + tensorboard + news_scheduler_state
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
