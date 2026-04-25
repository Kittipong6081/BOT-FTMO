# 02 — Modules Map (30+ files)
> Last Updated: 2026-04-25 | Scope: every module + key class / method / variable

## TL;DR (30-second scan)

- 7 layers: **strategy / ml / core / execution / config / analytics / scripts** + `main.py`.
- Finding code: look up the symbol in the tables below, then grep or jump to the file.
- Mission-critical modules: `RiskManager`, `MT5Connector`, `TradeExecutor`, `SMCStrategy`, `SelfLearningAgent`, `FTMOSignalFilterEnv`.
- Every config value and symbol is referenced by name — never by line number (line numbers rot quickly).

---

## 🎯 strategy/ — SMC Brain (rule-based)

| File | Key symbols | Role |
|------|-------------|------|
| `smc_strategy.py` | `SMCStrategy.scan_all_symbols`, `SMCStrategy.scan_signal`, `SignalType.BUY/SELL`, `TradeSignal` (dataclass) | Main engine: chains HTF → MTF → LTF, confluence scoring, signal building |
| `indicators.py` | `TechnicalIndicators.atr`, `.ema`, `.rsi`, `.macd`, `.adx`, `.stoch`, `.bb` | Indicator calculation on numpy arrays |
| `order_blocks.py` | `OrderBlockDetector.find_order_blocks`, `OrderBlockDetector.score_ob` | Fractal-based OB detection + impulse scoring [0–100] |
| `fair_value_gaps.py` | `FairValueGap.analyze`, `FVG` (dataclass) | 3-candle imbalance + fill tracking |
| `liquidity_sweeps.py` | `LiquiditySweepDetector.find_sweeps` | Swing high / low sweep detection |
| `market_structure.py` | `MarketStructure.detect_bos_choch`, `MarketStructure.htf_bias` | BOS / CHoCH (body-based, 5-bar lookback); HTF bias 5-bar window ≥3 bars |

**Confluence**: inside `SMCStrategy.scan_signal`, scores combine OB + FVG + Sweep + Structure + Indicators, capped at 100 (session multiplier × raw).

### SL Formula & ATR Floor (3 separate knobs — do not confuse)

SL is built inside `SMCStrategy.scan_signal` (BUY branch + SELL mirror) in three stacked steps:

1. **Base** — `sl_distance = atr_value × bot_config.indicators.atr_sl_multiplier` (default `1.5`).
2. **OB override** — if a recent Order Block is close and `ob_sl_distance < sl_distance × 1.5`, swap in `ob_sl_distance` (tighter SL when OB structure is near).
3. **MIN_SL guard (v6.2)** — if `sl_distance < min_sl_pips × pip_size`, bump up to the floor. Prevents spread from eating > ~15 % of SL.

Three knobs with distinct roles — **never conflate them**:

| Knob | Where | Role |
|------|-------|------|
| `SymbolConfig.symbol_overrides[X].atr_floor_pips` | gate at top of `SMCStrategy.scan_signal` | **Signal gate** — if `atr_pips < floor`, the signal is discarded (dead-market filter). Does **not** affect SL width of accepted trades. |
| `bot_config.indicators.atr_sl_multiplier` | base SL formula | ATR-to-SL ratio (1.5 × ATR). Global — touches every symbol. |
| `SymbolConfig.symbol_overrides[X].min_sl_pips` | clamp after OB override | **SL floor** — per-symbol minimum to keep spread-to-SL ratio sane (EURUSD 10, GBPJPY 20, XAUUSD 300 ticks). |

**Want SL narrower?** Lower `atr_sl_multiplier` or `min_sl_pips` — lowering `atr_floor_pips` only widens the accepted-signal population (indirect, clamped by `min_sl_pips`).

### v6.7 Rollback Phase D (2026-04-25)

Both Phase D variants (full BE+partial+trail, and BE-only) reduced Pass Rate below the B1v2 baseline (3.7 %). Reverted to v6.3 B1v2 backtester (`_resolve_trade` uses only SL / TP / timeout / force-close / gap — no BE, no partial, no trailing inside training). **Live `TradeManager` still performs BE + partial + trail** — train-live gap is accepted as "train < live" direction.

### v6.5 Phase D — Train-Live Alignment (REVERTED 2026-04-25)

**Kept as historical reference only.** Evidence: BE-only dropped pool mean outcome from `−0.0645` → `−0.1051` (8.4 pp of winners became 0R vs 9.6 pp of losers saved — net EV negative). Phase D full caps tail (winners capped at 1.5R instead of 2R+) which is the wrong direction for FTMO's tail-probability objective.

- **`StrategyBacktester._resolve_trade`** was rewritten as bar-by-bar state machine (removed in v6.7).
- Class constants `_BE_TRIGGER_RR`, `_PARTIAL_CLOSE_PCT`, `_PARTIAL_TRIGGER_RR`, `_TRAIL_ACTIVATION_RR`, `_TRAIL_ATR_MULTIPLIER` — removed in v6.7.

### v6.4 SMC Phase C — 4 Professional Principles (REVERTED 2026-04-25)

**This block was rolled back** after empirical results (Pass Rate 3.7 % → 1.5 %). Kept here as historical reference only; none of these changes exist in the current codebase.

- **`SMCStrategy._get_h4_poi_zones` / `_is_near_h4_poi`** (removed) — H4 POI cache + gate.
- **`SMCStrategy._ob_detector_h4` / `_fvg_detector_h4` / `_h4_poi_cache`** (removed).
- **`MarketStructure.is_valid_pullback`** (removed) — 3-gate validator.
- **IDM sweep gate** (removed) — sweep + OB bonus / old OB penalty.
- **FVG + BOS conjunction** (removed).
- **ADX floor** reverted 25 → 20.

### v6.3 SMC fixes (2026-04-24 audit)

- **SELL `atr_floor_pips` per-symbol override now applied** — previously `_evaluate_sell_signal` hardcoded 100/8, ignoring `SymbolConfig.symbol_overrides`. BUY + SELL now share the same gate.
- **BUY `timestamp` uses `TimeManager.get_server_time(symbol)`** — previously `datetime.now()` (local machine time). SELL was already correct.
- **HTF bias anti-lookahead** — `analyze_with_data` + `analyze` now use `htf_df["trend"].iloc[-6:-1]` (5 closed bars, excludes the currently-forming bar). Old `iloc[-5:]` leaked forming-bar data into backtests.
- **EMA200 NaN guard** — H1 EMA200 veto in BUY + SELL now wraps `pd.notna(ema200_h1) and ema200_h1 > 0 and ...`. NaN values (fresh MTF data < 200 bars) previously short-circuited the veto silently.
- **D1 bias cache invalidates on UTC day rollover** — `_get_d1_bias` no longer returns stale bias for up to 1 hour after midnight UTC (was previously caching by TTL alone).
- **`min_sl_pips` clamp now logs** — when SL width gets bumped up to the floor, `SMCStrategy` prints a warning (respects `bot_config.debug_mode`).

---

## 🔬🎓 ml/ — ML + RL Brain

| File | Key symbols | Role |
|------|-------------|------|
| `signal_quality.py` | `SignalQualityModel.score`, `SignalQualityModel.train_from_pool` | sklearn `GradientBoostingClassifier` wrapper — P(win) prediction (AUC ~0.59) |
| `strategy_backtester.py` | `StrategyBacktester.generate_episode_signals`, `StrategyBacktester._resolve_trade`, `StrategyBacktester._quality_model` | Pool generation + outcome resolution with **trade management** (BE move / partial close / trailing) mirroring `TradeManager`. v6.5 Phase D. |
| `signal_filter_env.py` | `FTMOSignalFilterEnv` (gym.Env), `._get_obs`, `.step`, `.reset` | Gymnasium env for RL training — pool sampler + reward shaper |
| `rl_agent.py` | `SelfLearningAgent.OBS_DIM` (= 27), `.should_take_signal`, `.initialize_model`, `._normalize_obs` | PPO inference wrapper for live — loads `vec_normalize_sf.pkl` so obs normalization matches training |

**Quality model loading**: `StrategyBacktester.__init__` auto-loads `data/signal_quality_model.pkl` if present, so pool signals always include `ml_score` (defaults to 0.5 = neutral when unavailable).

### v6.3 ML + RL fixes (2026-04-24 audit)

- **Anti-leakage GBM training** — `train_signal_quality.py` now uses `GroupKFold` (group = episode index) with `cross_val_predict` to produce OOF probabilities. The pool's `ml_score` field is re-scored with **OOF predictions**, not in-sample. Quoted AUC is now an unbiased OOS estimate.
- **Gap handling in `_resolve_trade`** — when `bar_open` is already past SL or TP (weekend gap, news), outcome is filled at `bar_open` with slippage/spread cost, not at SL/TP price.
- **Pool confluence threshold aligned with live** — `generate_episode_signals` now uses `bot_config.ftmo.MIN_CONFLUENCE_SCORE` (live threshold) instead of hardcoded 60. Pool + live see the same obs[0] distribution.
- **Spread noise in training env** — `FTMOSignalFilterEnv.reset` jitters `spread_pips` by `uniform(0.7, 1.5)` once per episode. obs[24] `spread_pct_of_atr` + reward `spread_cost` both see realistic variance.
- **`spread_cost_R` clamp** — reward deduction now bounded at `min(spread/sl, 1.0)` × 0.5, capping spread penalty at -0.5R (was unbounded, could exceed -1R on news-spike spreads).
- **`FTMOSignalFilterEnv.obs_dim()` returns 27** — was stale `24`. Matches `observation_space.shape[0]`.
- **`_has_opposite_recently_closed` logs flip_lock missing** — `main.py` warns once if `RiskManager._flip_lock` is None, preventing silent obs[25]=0 forever.

---

## 🛡️ core/ — Risk, Connection, Notification

### `risk_manager.py` — FTMO compliance engine (CRITICAL)

| Symbol | Role |
|--------|------|
| `BotState` (Enum) | ACTIVE / DAILY_HALT / MAX_DRAWDOWN_HALT / MANUAL_HALT / DISCONNECTED |
| `RiskManager.check_state` | Pre-trade gate — returns False when halted |
| `RiskManager.get_risk_status` | Returns dict: `daily_loss_pct`, `overall_drawdown_pct`, `current_balance` |
| `RiskManager._initial_balance` | FTMO anchor — total DD is measured against this |
| `RiskManager._daily_start_balance` | Daily anchor — daily DD is measured against this |
| `RiskManager._save_state` / `._load_state` | Persistence via `logs/bot_state.json` (schema v4) |

**State file schema** (`logs/bot_state.json`):

- `initial_balance`, `state`, `highest_balance`, `current_day`, `daily_closed_pnl`, `consecutive_losses`, `halt_until`, `daily_pnl_history`, `mt5_login` (v4), `challenge_start_date` (v4), `schema_version` (v4).
- **Validation**: `mt5_login` mismatch → reset; balance diff > 20 % → warn only (never auto-reset).

### `mt5_connector.py` — MT5 API wrapper

| Symbol | Role |
|--------|------|
| `MT5Connector.connect` / `.disconnect` | MT5 login (mock fallback when library missing) |
| `MT5Connector.get_account_info` | Balance, equity, margin |
| `MT5Connector.get_ohlcv` | Fetch candles (M15 / H1 / H4) |
| `MT5Connector.place_order` | Market order with SL/TP |
| `MT5Connector.close_all_positions` | Emergency halt |
| `MT5Connector.get_open_positions` | List currently open positions |
| `MT5Connector.get_symbol_info` | Cached — reduces API calls |

⚠️ **Position ID vs ticket**: use `position_id` for deal matching (never `order` or `ticket`).

### `time_manager.py` — Broker time + sessions

| Symbol | Role |
|--------|------|
| `TimeManager.get_server_time` | Returns `datetime` in Europe/Bucharest (EET/EEST) |
| `TimeManager.is_trading_session` | Checks session window (London / NY) |
| `TimeManager.is_friday_close_time` | True when Friday ≥ 20:45 EET |

⚠️ Use `datetime.now(Europe/Bucharest)` — **never** `mt5.symbol_info_tick().time` (FTMO returns broker-local epoch, and `fromtimestamp(tz=Bucharest)` double-adds the offset → +3 h drift). Requires NTP-synced VPS.

### `position_sizer.py` — Lot calculation

| Symbol | Role |
|--------|------|
| `PositionSizer.calculate_lot_size` | Returns `{lot_size, risk_usd, risk_pct}` |
| `PositionSizer._pip_value` | Three cases: quote = USD, base = USD (USDJPY / USDCHF), cross (EURJPY / GBPJPY uses USDJPY rate) |

Always floor-round the lot size (never ceil) — safety margin against broker steps.

### `news_scheduler.py` — Auto-import calendar weekly

| Symbol | Role |
|--------|------|
| `NewsCalendarScheduler.check_and_run` | Called every tick; imports CSV when it is Sunday 23:30 EET and the import has not run yet |
| `NewsCalendarScheduler._import_latest_csv` | Parses newest CSV in `config/news_inbox/` → writes `config/news_calendar.json` → moves CSV into `processed/` |
| State file | `logs/news_scheduler_state.json` (prevents double-runs) |

### `notifier.py` — Discord webhook

| Symbol | Role |
|--------|------|
| `DiscordNotifier.send_message`, `.send_alert` | Sends webhook |
| Rate limit | 20 req/min sliding window + min 1 s interval + 429 Retry-After + `threading.Lock` |

---

## 💱 execution/ — Order placement + position lifecycle

### `trade_executor.py`

| Symbol | Role |
|--------|------|
| `TradeExecutor.execute_trade` | Final gate: re-check risk → calc lot → send order → log |
| `ExecutedTrade` (dataclass v2) | Schema carries ML features (session, day_of_week, HTF bias, MAE/MFE, spread, slippage) for retraining |
| `TradeExecutor._check_correlation` | Groups: USD_WEAK / USD_STRONG / JPY_CROSS / EUR_PAIRS / GBP_PAIRS — `MAX_CORRELATED_POSITIONS` per group per direction |

**Hard rule**: every order must have SL — never send orderless.

### `trade_manager.py`

| Symbol | Role |
|--------|------|
| `TradeManager.update_positions` | Called every tick — syncs MT5 positions + trailing + partial close |
| `TrailingState` (dataclass) | Per-position state: `initial_sl`, `current_sl`, `best_price`, `trailing_active`, `partial_closed` |
| Constants | `BE_TRIGGER_RR=1.0` (move SL → entry), `PARTIAL_CLOSE_PCT=0.5` (50 % at 1:1), `TRAIL_ACTIVATION_RR=1.5` |

---

## ⚙️ config/ — Settings + News

### `settings.py` — Master config

Key dataclasses:

| Class | Key fields |
|-------|------------|
| `MT5Config` | `terminal_path`, `login`, `password`, `server`, `timeout` (loaded from `.env`) |
| `FTMOConfig` | `DAILY_LOSS_HARD_STOP_PCT=0.04`, `MAX_DRAWDOWN_HARD_STOP_PCT=0.08`, `PROFIT_TARGET_PCT=0.10`, `MIN_RISK_PER_TRADE_PCT=0.005`, `MAX_RISK_PER_TRADE_PCT=0.008`, `DEFAULT_RISK_PER_TRADE_PCT=0.007` ⭐, `MIN_CONFLUENCE_SCORE=70.0`, `CONSISTENCY_RULE_THRESHOLD=1.0` (2-step Standard → check off), `COOLDOWN_AFTER_LOSS_MIN=60`, `POST_TP_LOCK_TTL_MIN=60` |
| `SymbolConfig.symbols` | 10 symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, **XAUUSD** |
| `bot_config` (singleton) | Aggregates every dataclass — import target for other modules |

### `news_events.py` — Hardcoded fallback news

| Symbol | Role |
|--------|------|
| `RecurringNewsEvent` (dataclass) | `name`, `currencies` (Set), `weekday`, `day_of_month`, `time_utc`, `.occurs_on()` |
| `HIGH_IMPACT_EVENTS` (list) | NFP, ECB, BoE, FOMC, CPI, etc. — used when `news_calendar.json` is missing or expired |

⚠️ All times UTC (compared after EET → UTC conversion).

### `news_csv_parser.py` — ForexFactory CSV importer

| Symbol | Role |
|--------|------|
| `parse_forexfactory_csv(csv_path, tz_offset=0)` | Returns `List[Dict]` of high-impact events |
| Filter | Only impact ∈ {"high", "holiday"} |

---

## 📊 analytics/

### `performance.py`

| Symbol | Role |
|--------|------|
| `TradeResult` (dataclass) | ticket, symbol, entry/close, lot, risk, profit, rr_achieved, times, confluence |
| `PerformanceAnalyzer.calculate_basic` | Win rate, profit factor |
| `PerformanceAnalyzer.calculate_advanced` | Sharpe (annualized 252), Sortino |
| `PerformanceAnalyzer.calculate_risk_metrics` | Max DD, Calmar |
| `PerformanceAnalyzer._trades` | List of `TradeResult` (used for `recent_wr_norm` in `_build_signal_observation`) |

### `trade_logger.py`

⚠️ **Disabled** in `FTMOTradingBot.__init__` (`self._logger = None`) — trade history is read via MT5 `history_deals_get()` instead.
- To re-enable: schema v2 (30+ columns), requires `openpyxl`, monthly files at `logs/ftmo_trades_YYYY_MM.xlsx`.

---

## 🛠️ scripts/ — Training + data fetch

| Script | Role |
|--------|------|
| `build_signal_pool.py` | Multiprocessing 8 workers — calls `StrategyBacktester.generate_episode_signals` × N → `data/signal_pool_3000.pkl` |
| `train_signal_quality.py` | Trains GBM (`SignalQualityModel.train_from_pool`) → saves model + re-scores pool in place |
| `train_signal_filter.py` | PPO 2-phase curriculum (Phase 1 Alpha → Phase 2 Risk) → `models/ppo_signal_filter.zip` |
| `fetch_mt5_data.py` | (Windows only) fetches OHLCV from MT5 → CSV under `data/ohlcv/` |
| `import_forexfactory_csv.py` | Manual CSV → `news_calendar.json` import (alternative to `NewsCalendarScheduler`) |

---

## 🚀 main.py — Entry Point

| Symbol | Role |
|--------|------|
| `FTMOTradingBot.__init__` | Builds every subsystem (connector, risk, sizer, strategy, executor, manager, RL agent, ML model, news scheduler, notifier) |
| `FTMOTradingBot.connect` | MT5 login + load state |
| `FTMOTradingBot.run` | Main loop (runs every `main_loop_interval` seconds, default 5 s) |
| `FTMOTradingBot._build_signal_observation` | Builds the 27-dim obs from a `TradeSignal` + portfolio state — must match `FTMOSignalFilterEnv._get_obs` |
| `FTMOTradingBot._build_spread_pct_of_atr` | Computes obs[24] (cost awareness) |
| `FTMOTradingBot._has_opposite_recently_closed` | Computes obs[25] (flip-lock context) |
| `FTMOTradingBot.shutdown` | Graceful shutdown — saves state, closes connector |

---

## Cross-links

- 3-brain pipeline + loop priority → [01-architecture.md](01-architecture.md)
- Obs 27 dims layout + reward + PPO → [03-rl-training.md](03-rl-training.md)
- FTMO state machine + news + sessions → [04-operations.md](04-operations.md)
- Red-flag rules + version log → [05-invariants.md](05-invariants.md)
