# CONTEXT — FTMO Trading Bot (LLM Wiki Hub)
> Last Updated: 2026-04-30 (v6.14) | Scope: Hub / Index — read this first, then drill into wiki/*

## TL;DR (LLM read first — 30-second scan)

- **Goal**: pass the FTMO 2-step Standard Challenge (10 % profit, 4 % daily DD, 8 % total DD).
- **3 brains**: `SMCStrategy` (rules) → `SignalQualityModel` (GBM + Isotonic calibrator) → `SelfLearningAgent` (PPO + Auxiliary Task — TAKE/SKIP).
- **Live entry**: `python main.py` → `FTMOTradingBot.run` loops every 5 s. Console runs in **quiet mode** (announce-once for idle states; per-signal SKIP/NO_AGENT logged to Excel `Signals` sheet, not console).
- **Obs = 27 dims** (v6, 2026-04-22). Must stay in sync across three places: `FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM`.
- **Runs on**: macOS/Linux (train + backtest), Windows + MT5 (live).
- **Live logging**: `TradeLogger` (re-enabled v6.9, schema bumped v6.10) writes Excel — Trades 64 cols (incl. `Obs27 JSON` for retrain), Signals 21 cols (per-scan log), Daily, Stats.
- **v6.11 SMC overhaul (2026-04-29)**: Counter-D1 hard veto + Sweep within 8 bars + Fresh M15 BOS within 6 bars + ADX H4 ≥ 22 + Quiet-vol × off-overlap blocker + IDM detector + OB grading (Extreme/Decisional/Internal). BE trigger ใช้ `best_price` (rolling MFE). Per-component pts populate ใน TradeSignal → Trades sheet เห็น HTF/MTF/OB/FVG/Sweep pts จริง. ดู [`wiki/05-invariants.md` v6.11 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v6.11.1 (2026-04-29 evening)**: Post-impl audit fix — `StrategyBacktester._init_strategy` เพิ่ม `_idm_detector` init ให้ตรงกับ `SMCStrategy.__init__` (กัน AttributeError ตอน rebuild pool). Eval Pass Rate 11.2 % ที่ได้จาก cached pool (Apr 25, pre-v6.11) — **invalid สำหรับ v6.11**. ต้อง rebuild pool + GBM ก่อน eval ใหม่.
- **v6.11.2 (2026-04-29 evening)**: Partial rollback — Tier 2.2 (Sweep prereq) + Tier 2.3 (Fresh M15 BOS prereq) hard gates → soft bonuses. Pool หาย 99 % ภายใต้ v6.11 hard gates → 0.0 % Pass Rate. หย่อน 2 จุด → pool 78k signals, Pass Rate 2.7 %, WR 65.6 %.
- **v6.11.3 (2026-04-29 evening)**: Mild tune — IDM penalty 5→2 + ADX H4 floor 22→20. Pool 90k (+15 %), Pass Rate 2.7→**3.4 %**, WR 65.6→**68.8 %**, DD max 4.46→**3.23 %** (safer). Pure improvement ทุกมิติ. **KEEP + deploy demo**. Backups `*.bak_v6.11.2` พร้อมสำหรับ rollback.
- **v6.12 (2026-04-29 night)**: Live-vs-train sync fix — `FTMOConfig.ML_FILTER_THRESHOLD = 0.36` + ML gate ใน `FTMOTradingBot.run` ก่อน agent. ปิด silent regression จากการที่ live ไม่บังคับ ML threshold (training บังคับ). Logging fix — `ML Threshold` column ใน `Signals` sheet ตอนนี้แสดง 0.36 จริง (เดิม 0.0 เพราะ getattr ผิด attribute). ไม่ต้อง retrain.
- **v6.14 (2026-04-30) ⭐ current — Live demo log audit, 4 bug fixes** (silent regression vs v6.13 spec):
  - **Fix 1 — SL flow alignment (XAU)**: `SMCStrategy.scan_signal` BUY+SELL ใช้ per-symbol `sl_atr_multiplier` (XAU=1.8×, FX=1.5×) แทน global hardcoded 1.5× + เพิ่ม OB SL `ob_sl_floor = atr × 0.5` ป้องกัน clamp ลด SL ใกล้เกินไป
  - **Fix 1C — XAUUSD `min_sl_pips`**: 300 → **1000 ticks** ($3 → $10) — guard ชั้นสุดท้าย ไม่ให้ Gold SL ต่ำกว่า ~1.0×ATR
  - **Fix 2 — TradeLogger off-by-one**: close-row update path เขียน column 28-31 ทับ `DD@Entry % / MAE / MFE / Time-in-Trade` → แก้เป็น 29-32 ตรงกับ `TRADE_HEADERS`
  - **Fix 3 — PerformanceAnalyzer replay**: re-enable `load_from_excel(logs/ftmo_trades.xlsx)` ใน `FTMOTradingBot.initialize` ขั้นตอน 3.6 (เดิม `[DISABLED]` block) → Stats / Max DD / Sharpe ต่อเนื่องข้าม restart
  - **Symptom จาก live demo (04-29→04-30, 9 trades)**: XAU trade #0 [437211678] SL = 0.28×ATR → SL hit ใน 12s (-$103); MFE column ใน Excel เก็บค่า duration; Stats sheet "Total Trades 3" แทน 9
  - **ไม่ต้อง retrain**: train env (`StrategyBacktester`) ใช้ 1.8× อยู่แล้ว — Fix 1 ดึง live ให้ตามให้ทัน. หลัง deploy ต้องตรวจ live SL/ATR ratio ของ XAU trade ใหม่ ≈ 1.8 (ไม่ใช่ 0.28 หรือ 1.5)
  - Backups `*.bak_v6.13` พร้อม rollback. ดู [`wiki/05-invariants.md` v6.14 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v6.13 (2026-04-29 night) — VERIFIED Pass Rate 9.7 %** (5000 eps eval):
  - **L1 Pause** 2→3, **L2 defaults safety** (env const 0.003→0.007, outcome_noise 0.02→0.05), **L3 reward rebalance** (TAKE equalize @ ml ≥ 0.36, P2 SKIP-oracle missed-winner −0.70→−0.90, smart-skip +0.20→+0.35, day-10 early undertrading check), **L4 XAU SL 1.5×→1.8×**
  - **Eval result**: Pass Rate **9.7 %** (vs baseline 3.4 % = +185 %), WR 64.8 %, Orders/ep 7.7, DD max 4.40 % (ห่าง 8 % limit), Breach 0 %, Profit avg +3.89 %
  - Pass Rate เกือบถึง FTMO 10 % target — ทะลุเป้าทุกมิติ. KEEP + deploy demo
  - **No-leak audit ผ่าน**: 27 obs dims + GBM 17 features = signal-time only. Aux head ใช้ outcome เป็น MSE target. SKIP-oracle เป็น reward shaping (policy ไม่เห็น outcome)
  - Backups `*.bak_v6.12` พร้อม rollback. ดู [`wiki/05-invariants.md` v6.13 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **Wiki Sync Protocol**: editing `.py` files under `ftmo_trading_bot/` requires updating `wiki/` + `context.md` + `readme.md` (when user-facing) in the same turn. Stop hook enforces (`decision: block`). See `CLAUDE.md`.

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
| MIN_SL guard (per-symbol, v6.2 / v6.14 XAU raise) | 10-20 pips FX, **1000 ticks XAUUSD** ($10) | `SymbolConfig.symbol_overrides[X].min_sl_pips` |
| SL base multiplier (per-symbol, v6.14 wired) | FX 1.5×, **XAUUSD 1.8×** | `SymbolConfig.symbol_overrides[X].sl_atr_multiplier` (fallback `bot_config.indicators.atr_sl_multiplier`) |
| OB SL clamp lower bound (v6.14) | 0.5 × ATR | hard-coded in `SMCStrategy.scan_signal` |
| Obs dims | **27** | `SelfLearningAgent.OBS_DIM` |
| RL model | `models/ppo_signal_filter.zip` + `models/vec_normalize_sf.pkl` | `SelfLearningAgent` |
| ML model | `data/signal_quality_model.pkl` | `SignalQualityModel` |
| Pool | `data/signal_pool_3000.pkl` (~158k signals) | `StrategyBacktester` |
| FTMO program | 2-step Standard (no Consistency Rule → threshold = 1.0) | `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` |

## Verified Performance (v6.13 — Combined patch + obs no-leak audit, risk 0.7 %, 5000 eps, 2026-04-29)

| Metric | v6.11.3 baseline | **v6.13** | Δ |
|--------|------------------|-----------|---|
| Pass Rate | 3.4 % | **9.7 %** ⭐⭐⭐ | **+185 %** |
| Win Rate | 68.8 % | 64.8 % | -4 pp |
| Orders/ep | 6.1 | **7.7** | +26 % |
| Total DD max | 3.23 % | 4.40 % | +37 % (ห่าง 8 % limit) |
| Daily DD max | 2.12 % | 2.15 % | similar |
| Breach Rate | 0 % | **0 %** | same |
| Profit avg (5000 eps) | — | **+3.89 %** | — |

**Phase progression** (each = 5000-eps eval): leaky baseline 12.5 % → honest baseline 3.5 % → Phase C 1.5 % → Phase D 0.2 % → Phase E1 (calibration) 3.0 % → Phase E2 (aux task) 10.0 % → v6.11 SMC overhaul 0.0 % → v6.11.2 (rollback) 2.7 % → v6.11.3 (mild relax) 3.4 % → **v6.13 (combined patch) 9.7 %**. Details in [wiki/05-invariants.md § Version Log](wiki/05-invariants.md).

**Note**: the old "Option B 12.5 %" baseline was leaky (eval seeded with same pool used for GBM training). Honest baseline = 3.5 %. E2 is verified leak-free via runtime hook + obs feature audit.

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
├── analytics/               ← PerformanceAnalyzer + TradeLogger (Excel: Trades 64 cols / Signals 21 cols / Daily / Stats)
├── scripts/                 ← build_signal_pool, train_signal_quality, train_signal_filter, fetch_mt5_data
├── data/                    ← OHLCV CSVs + signal_pool + ml_model pkl (with isotonic calibrator)
├── models/                  ← ppo_signal_filter.zip + vec_normalize_sf.pkl (aux-aware policy weights)
└── logs/                    ← bot_state.json + ftmo_trades.xlsx + tensorboard + news_scheduler_state
```

Full module details → [wiki/02-modules.md](wiki/02-modules.md).

---

## Quick Commands

**Training (3 steps, in order):**

```bash
python scripts/build_signal_pool.py --pool_size 3000 --workers 8
python scripts/train_signal_quality.py
python scripts/train_signal_filter.py --fresh \
    --timesteps_p1 300000 --timesteps_p2 200000 \
    --n_envs 4 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Phase E2 trainer uses `AuxAwarePPO` + `AuxAwareACPolicy` automatically (aux loss weight = 0.5).

**Evaluation:**

```bash
python scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.36 --risk_per_trade 0.007
```

Default 5000 episodes. Use `.venv/bin/python` (not bare `python`) — version mismatch can shift Pass Rate.

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
