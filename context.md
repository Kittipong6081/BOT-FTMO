# CONTEXT — FTMO Trading Bot (LLM Wiki Hub)
> Last Updated: 2026-04-25 | Scope: Hub / Index — read this first, then drill into wiki/*

## TL;DR (LLM read first — 30-second scan)

- **Goal**: pass the FTMO 2-step Standard Challenge (10 % profit, 4 % daily DD, 8 % total DD).
- **3 brains**: `SMCStrategy` (rules) → `SignalQualityModel` (GBM, AUC ~0.59) → `SelfLearningAgent` (PPO TAKE/SKIP).
- **Live entry**: `python main.py` → `FTMOTradingBot.run` loops every 5 s.
- **Obs = 27 dims** (v6, 2026-04-22). Must stay in sync across three places: `FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM`.
- **Runs on**: macOS/Linux (train + backtest), Windows + MT5 (live).
- **Wiki Sync Protocol**: editing `.py` files under `ftmo_trading_bot/` requires updating `wiki/` + `context.md` + `readme.md` (when user-facing) in the same turn. See `CLAUDE.md`.

## Headline Numbers

| Metric | Value | Source (symbol) |
|--------|-------|-----------------|
| Symbols | **10** (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, XAUUSD) | `SymbolConfig.symbols` |
| Timeframes | M15 (entry) / H1 (structure) / H4 (HTF bias) | `SymbolConfig.primary/structure/higher_timeframe` |
| Profit target | 10 % | `FTMOConfig.PROFIT_TARGET_PCT` |
| Daily DD stop | 4 % | `FTMOConfig.DAILY_LOSS_HARD_STOP_PCT` |
| Total DD stop | 8 % | `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` |
| Default risk per trade | **0.7 %** (verified optimal) | `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` |
| Risk floor / cap | 0.5 % / 0.8 % | `FTMOConfig.MIN/MAX_RISK_PER_TRADE_PCT` |
| Min confluence | 70 | `FTMOConfig.MIN_CONFLUENCE_SCORE` |
| Max open positions | 3 | `FTMOConfig.MAX_OPEN_POSITIONS` |
| ATR floor (signal gate, per-symbol) | 3-8 pips FX, 500 ticks XAUUSD | `SymbolConfig.symbol_overrides[X].atr_floor_pips` |
| MIN_SL guard (per-symbol, v6.2) | 10-20 pips FX, 300 ticks XAUUSD | `SymbolConfig.symbol_overrides[X].min_sl_pips` |
| SL base multiplier (global) | 1.5 × ATR | `bot_config.indicators.atr_sl_multiplier` |
| Obs dims | **27** | `SelfLearningAgent.OBS_DIM` |
| RL model | `models/ppo_signal_filter.zip` + `models/vec_normalize_sf.pkl` | `SelfLearningAgent` |
| ML model | `data/signal_quality_model.pkl` | `SignalQualityModel` |
| Pool | `data/signal_pool_3000.pkl` (~158k signals) | `StrategyBacktester` |
| FTMO program | 2-step Standard (no Consistency Rule → threshold = 1.0) | `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` |

## Verified Performance (Option B, risk 0.7 %, 5000 eps, 2026-04-20)

| Metric | Value |
|--------|-------|
| Pass Rate | **12.5 %** |
| Profit avg | +2.59 % |
| Win Rate | 46.2 % |
| DD avg | 0.47 % |
| DD max | 8.50 % |
| Breach rate | **0 %** |
| Days to pass (avg) | 29.5 |

---

## 🗺️ Wiki Navigation

| File | Read when you need to... |
|------|---------------------------|
| [wiki/01-architecture.md](wiki/01-architecture.md) | Understand the 3-brain pipeline, data flow, loop priority, and training pipeline overview |
| [wiki/02-modules.md](wiki/02-modules.md) | Find which class / method lives in which file (module map with symbol names) |
| [wiki/03-rl-training.md](wiki/03-rl-training.md) | Inspect the 27-dim obs layout, reward structure, PPO hyperparams, and curriculum |
| [wiki/04-operations.md](wiki/04-operations.md) | Understand main loop priority, FTMO state machine, news, sessions, cooldowns |
| [wiki/05-invariants.md](wiki/05-invariants.md) | ⛔ Rules that must not be broken, migration notes, and version log |

**Tip for LLM**: always start at `context.md`. Drill into the relevant wiki page only when needed. You do not need to read every file.

---

## Directory Layout (condensed)

```text
ftmo_trading_bot/
├── main.py                  ← FTMOTradingBot (live entry)
├── config/settings.py       ← MT5Config, FTMOConfig, SymbolConfig, bot_config
├── strategy/                ← SMCStrategy + 5 detectors (OB, FVG, Sweep, Structure, Indicators)
├── ml/                      ← SignalQualityModel, SelfLearningAgent, FTMOSignalFilterEnv, StrategyBacktester
├── core/                    ← RiskManager, MT5Connector, TimeManager, PositionSizer, NewsCalendarScheduler, DiscordNotifier
├── execution/               ← TradeExecutor, TradeManager
├── analytics/               ← PerformanceAnalyzer (+ TradeLogger disabled)
├── scripts/                 ← build_signal_pool, train_signal_quality, train_signal_filter, fetch_mt5_data
├── data/                    ← OHLCV CSVs + signal_pool + ml_model pkl
├── models/                  ← ppo_signal_filter.zip + vec_normalize_sf.pkl
└── logs/                    ← bot_state.json + tensorboard + news_scheduler_state
```

Full module details → [wiki/02-modules.md](wiki/02-modules.md).

---

## Quick Commands

**Training (3 steps, in order):**

```bash
python scripts/build_signal_pool.py --pool_size 3000 --workers 8
python scripts/train_signal_quality.py
python scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.33 --risk_per_trade 0.007
```

**Evaluation:**

```bash
python scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.33 --risk_per_trade 0.007 \
    --eval_episodes 500
```

**Live:**

```bash
python main.py
```

**TensorBoard:**

```bash
tensorboard --logdir logs/tb_signal_filter
```

---

## 🔴 Critical Invariants (must-read before editing)

- ⛔ **Obs 27 dims sync** — changing count or order of features requires retraining the whole pipeline.
- ⛔ **Position ID matching** — use `position_id`, never `ticket`.
- ⛔ **Do not delete `logs/bot_state.json`** mid-challenge.
- ⛔ **Timezone**: broker = EET, config = UTC — convert before comparing.
- ⛔ **Do not use `mt5.symbol_info_tick().time` directly** (FTMO quirk, +3 h drift). Use `datetime.now(Bucharest)` via `TimeManager`.
- ⛔ **Risk per trade** in live must match the training risk (`DEFAULT_RISK_PER_TRADE_PCT` ↔ `--risk_per_trade`).
- ⛔ **FTMO program type** = 2-step Standard → `CONSISTENCY_RULE_THRESHOLD = 1.0`.

Full list → [wiki/05-invariants.md](wiki/05-invariants.md).

---

## 📝 Wiki Maintenance Protocol

**When?** Every time `.py` files under `ftmo_trading_bot/` are edited.

**Update which files?**:

- Obs dim / feature / order → `wiki/03-rl-training.md` + `wiki/05-invariants.md` (version log)
- Config values (risk, symbols, DD thresholds) → `context.md` (Headline Numbers) + `wiki/04-operations.md` + `readme.md`
- Module signature / class name change → `wiki/02-modules.md`
- Loop / state machine change → `wiki/04-operations.md`
- User-facing change → `readme.md` (Thai)

**Last Updated** — bump the date on every file you touch (top of file).

**Source references** — always use class / method / variable names. Never line numbers (they rot quickly).

**Language** — docs in English; `readme.md` in Thai. See `CLAUDE.md`.

Details → [CLAUDE.md](CLAUDE.md).
