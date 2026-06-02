# 04 — Live Operations (Loop, FTMO State, News, Sessions)
> Last Updated: 2026-06-02 (🧹 TF/dual-strategy removed — MR-only) | Scope: main loop, RiskManager state machine, FTMO rules, news, trading sessions, console quiet mode, live logging
>
> **v8.0.80 loop/state changes** (→ [`05-invariants.md` v8.0.80](05-invariants.md#-version-log-entry--v8080-2026-05-30--production-audit-remediation-execution-safety--ftmo-breach-guard--trainlive-parity-retrain-required-for-c3h6)): (C2) `RiskManager.check_risk` runs an **always-on FTMO 4% daily-breach emergency close** before the state-gated `_check_daily_loss` — so a `DAILY_HALT` set by consec-loss/stop-out (which don't close positions) can no longer let open positions bleed past 4%. (H2) `max_trades_per_day` now uses a per-symbol counter. `main.run()` hourly stats / GBM-drift / periodic-status now fire on **wall-clock gates** (not `_loop_count % 720`), so the v8.0.70 adaptive 1s loop no longer fires them ~5× too often.

## TL;DR (30-second scan)

- Entry: `python main.py` — builds `FTMOTradingBot` and loops every 5 s. Strategy = **MR (Mean Reversion)** via `LiveMRScanner` (v8.0+).
- FTMO program = **2-step Standard** → `CONSISTENCY_RULE_THRESHOLD = 1.0` (check disabled).
- Risk hard stops: **4 % daily DD** (buffer vs FTMO 5 %), **10 % total DD** (v8.1.1 — = full FTMO rule, no buffer; soft warn at 8 %), target **10 % profit**.
- Default risk per trade = **0.7 %** (v8.0.43 Option X — paired with trail).
- **ML threshold = 0.30** (v8.0.3, sync `bot_config.ftmo.ML_FILTER_THRESHOLD` ↔ trainer ↔ HyperParams).
- **v8.0.55 pre-execution gates** (in addition to RiskManager + correlation):
  - `TradeExecutor._check_spread_spike` — `current_spread / median(SPREAD_SPIKE_LOOKBACK_BARS=30) > SPREAD_SPIKE_RATIO_LIMIT (2.0)` → SKIP. Warmup: `SPREAD_SPIKE_MIN_SAMPLES=10` (falls back to fixed `max_spread_points`).
  - `TradeExecutor._check_entry_confirmation` — slip ≤ `ENTRY_CONFIRM_MAX_SLIP_R (0.30)` (M1 last-bar direction check removed 2026-06-02 — was live-only, training proxy is slip-only); KC distance `base 0.35 × clip(1+atr_z·0.20, 0.6, 1.3)` (v8.0.77 RF-2: was 0.60 base / 1.5 max — volatile crowded the norm clip ceiling); KC slope cap `0.35 × clip(1+atr_z·0.15, 0.7, 1.6)` (v8.0.77 RF-3: was 0.15 — too tight, saturated ±1.0); consec_outside ≥ `KC_CONSEC_OUTSIDE_MAX=3`.
  - `RiskManager` cluster cooldown — `CLUSTER_COOLDOWN_ANY_SEC (300)` global, `CLUSTER_COOLDOWN_SAME_THEME_SEC (600)` for same USD/JPY/METAL theme. Extends `v8.0.26 MIN_SECONDS_BETWEEN_OPENS_SEC (60s)`.
  - `TradeExecutor._check_correlation_risk` — duplicate-symbol block; USD-theme cap `MAX_USD_THEME_POSITIONS (2)`; non-USD currency-leg cap `MAX_SAME_CURRENCY_LEG_POSITIONS` (**2** — was 1 in v8.0.79; raised to 2 on 2026-06-02 per user request → allows up to 2 same-direction on a shared non-USD leg, e.g. 2×EURUSD BUY or EURUSD SELL + EURJPY SELL) via `_non_usd_legs`. Per-group guard `MAX_CORRELATED_POSITIONS=99` is off.
- All internal times are **EET** (Europe/Bucharest) via `TimeManager.get_server_time()`.
- **No session block** (v8.0.6 SessionConfig cleanup) — bot trades 24/5 except: rollover (23:55-01:05 EET), Friday >= 20:45 EET, weekend, news blackout. **(v8.1.2: Mon-Thu 23:30 daily overnight close DISABLED — `enforce_daily_close=False` → holds positions overnight on weekdays; Friday close unchanged.)**
- **Console quiet mode**: idle-state prints use announce-once flags; per-signal SKIP/NO_AGENT goes to Excel `Signals` sheet, not console.
- **Live logging (v8.0.6)**: `TradeLogger` writes `logs/ftmo_trades.xlsx` (4 sheets: **Trades 58 cols**, **Signals 20 cols**, Daily, Stats). Auto-archives legacy schema (66/23 cols) on first launch. Uses `_COL`/`_SCOL` name-based column lookup. Includes `Obs JSON` (32 dims) for offline retrain.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Main loop interval | 5 s default, **1 s when ANY position open** (v8.0.70 — was v8.0.46 ≥ 0.5R only; full-lifecycle 1s ensures BE @ 0.3R + Partial @ 0.8R + Stage 2/3 fire without 5s window miss) | `bot_config.main_loop_interval`, adaptive code in `FTMOTradingBot.run` |
| Signal scan cadence | **60 s wall-clock** (v8.0.70 — decoupled from loop count). Was `loop_count % 12 == 0` = 60 s @ 5 s loop but would become 12 s @ 1 s loop. Time-based gate keeps scan at 1/min regardless of loop speed | `FTMOTradingBot._last_signal_scan_ts` |
| **Stepwise Trail (v8.0.73)** | Stage 1@0.5R partial+BE; Stage 2@0.8R **SL→0.5R lock** (TP unchanged at 1.0R per v8.0.73 — removed extension to 1.5R, continuation 26.9% per M1 replay); Stage 3@1.0R SL→1R + trail chase (rarely fires now since TP fills at 1.0R first) | `TradeManager.TP_STEP_*` + `TRAIL_*` constants |
| Symbols | 10 (incl. XAUUSD) | `SymbolConfig.symbols` |
| Daily DD stop | 4 % | `FTMOConfig.DAILY_LOSS_HARD_STOP_PCT` |
| Total DD stop | **10 %** (v8.1.1 — full FTMO rule, no buffer; soft warn at 8 %) | `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` |
| Profit target | 10 % | `FTMOConfig.PROFIT_TARGET_PCT` |
| Default risk / trade | 0.7 % | `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` |
| Max open positions | 3 (v8.0.69 — was 2 in v8.0.56) | `FTMOConfig.MAX_OPEN_POSITIONS` |
| ML threshold | 0.30 | `FTMOConfig.ML_FILTER_THRESHOLD` |
| Cooldown after loss | 60 min | `FTMOConfig.COOLDOWN_AFTER_LOSS_MIN` |
| Pause / Halt counts (v6.13) | 3 / 4 consec losses | `FTMOConfig.CONSECUTIVE_LOSS_PAUSE_COUNT/HALT_COUNT` |
| Post-TP lock TTL | 60 min | `FTMOConfig.POST_TP_LOCK_TTL_MIN` |
| Consistency threshold | 1.0 (off, 2-step Standard) | `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` |

---

## Main Loop Priority (inside `FTMOTradingBot.run`)

**Do not reorder** — the Risk gate must come first:

| # | Action | Symbol | Skip condition |
|---|--------|--------|----------------|
| 1 | Risk check | `RiskManager.check_state` | state ≠ ACTIVE → skip tick |
| 2a | Friday close | `TimeManager.is_friday_close_time` | Friday >= 20:45 EET → close all + skip |
| 2b | Weekend | `TimeManager.is_weekend` | Sat/Sun → announce-once + sleep until Monday |
| 2c | Daily close | `TimeManager.is_daily_close_time` | **DISABLED v8.1.2** (`enforce_daily_close=False`) — holds overnight Mon-Thu; was 23:30-23:55 EET close-all. Re-enable: `enforce_daily_close=True` |
| 2d | Rollover | `TimeManager.is_rollover_period` | 23:55-01:05 EET → skip (spread spike protection) |
| 3 | News scheduler | `NewsCalendarScheduler.check_and_run` | Sunday 23:30 EET → auto-import CSV (non-blocking) |
| 4 | News filter | `news_events` / `news_calendar.json` | within ±30 / 15 min of high-impact event → skip + close vulnerable positions |
| 5 | **Strategy scan (v8.0)** | `LiveMRScanner.scan_all_symbols()` → `MeanReversionStrategy.analyze_with_data` | every 12 loops (~1 min); MR rules: BB %B extreme + RSI confirm + reversal-wick + ADX H1 ≤ 30 + ATR floor |
| 6 | ML quality | `SignalQualityModel.score` (loads `data/mr_signal_quality_model.pkl`) | populates `live_context["ml_score"]` (calibrated) |
| 6b | **ML gate (v8.0.3)** | `FTMOTradingBot.run` checks `ml_score < bot_config.ftmo.ML_FILTER_THRESHOLD` | logged as `ML_FILTERED` — **must equal `--ml_threshold` ตอน train (0.30)** |
| 7 | RL decision | `SelfLearningAgent.should_take_signal` (loads `models/mr/best/ppo_mr_filter.zip`) | SKIP → drop signal (logged as AGENT_SKIP) |
| 8 | Build live context | `FTMOTradingBot._build_live_context(sig)` | computes ml_score, ADX H1/H4, balance, overtrading metrics, **`obs_27_json` (32 dims, key kept for back-compat)** |
| 9 | Execute | `TradeExecutor.execute_signal(sig, live_context)` | final risk / correlation / cooldown check; logs to Trades sheet (58 cols) |
| 10 | Manage open | `TradeManager.manage_all_positions` → `check_news_close` → `check_session_close` | trailing/BE/partial → pre-news close (T-30 min ก่อนข่าวแรง) → Friday Force Close / Daily Overnight Close |

---

## Console Quiet Mode (v6.9 announce-once)

`FTMOTradingBot.__init__` initializes 5 announce-once flags:

| Flag | State | Auto-reset |
|------|-------|-----------|
| `_daily_halt_announced` | DAILY_HALT (FTMO daily DD reached) | when state ≠ DAILY_HALT |
| `_friday_announced` | Friday 20:45 EET force-close | when not Friday close time |
| `_weekend_announced` | Saturday/Sunday market closed | when not weekend |
| `_daily_close_announced` | Daily Close 23:30 EET (Mon-Thu, Zero-Overnight) | when not daily-close time |
| `_rollover_announced` | Rollover/spread expansion 23:55–01:05 EET | when not rollover |

Each idle-state guard prints **once on entry** (sets flag = True), then silences until the state exits (auto-resets in `else` branch). Discord risk alerts still fire once at entry (no regression).

**Removed prints (now silent — logged to Excel only):**

- `⏭️ [Agent] SKIP ...` per-signal — `Signals` sheet `AGENT_SKIP` row
- `📡 [Bot] สัญญาณ ... NO_AGENT` per-signal — `Signals` sheet `AGENT_TAKE` row (when no RL agent loaded)

**Kept prints:**

- `📡 [Agent] TAKE ...` — agent decided to open trade (event with consequence)
- `✅ [Bot] เปิดเทรดสำเร็จ: Ticket ...` — trade open confirm
- `🟢/🔴 [Logger] บันทึกเทรดปิด ... P/L=$...` — close confirm
- All errors / warnings / FTMO breach alerts — never silenced

---

## Live Logging (`TradeLogger` Excel)

`logs/ftmo_trades.xlsx` — auto-create on first scan/trade. 4 sheets:

| Sheet | Cols | What goes in |
|-------|-----:|--------------|
| `Trades` | 64 | Per-trade: ticket, entry/SL/TP, lot, RR, ML scores, agent decision, confluence breakdown, trade-mgmt state (`be_moved`, `partial_closed_flag`, `partial_close_skipped` v6.10, `trailing_active`), bid/ask @entry/exit, market context (ADX H1/H4, MTF/D1 bias), account state, overtrading metrics, **`Obs27 JSON`** (full obs vector at decision time) |
| `Signals` | 21 | Per-scan event: time, symbol, direction, result (`AGENT_TAKE`/`AGENT_SKIP`/`AGENT_TAKE_FAIL`/`REJECTED`/`NO_SIGNAL`), confluence, ml_score, agent decision, ADX, biases, reasons, `Executor Reject` (v6.10), **`Obs27 JSON`** |
| `Daily` | 11 | Date, Trades, Wins, Losses, WR%, Gross P/L, Net P/L, DD%, Daily DD%, Balance EOD |
| `Stats` | 2 | Metric / Value (Win Rate, Sharpe, Profit Factor, etc., refreshed by `update_stats_sheet`) |

**Win/Loss classification**: `profit > 0` on cumulative net (sum of all deals from `MT5Connector.get_deals_by_position(ticket)`). Partial close + BE-SL = WIN if cumulative > 0. See [05-invariants.md FAQ](05-invariants.md).

**Schema migration warning**: pre-v6.10 `ftmo_trades.xlsx` has 63 / 20 cols (and pre-v6.9 has 62 / 19 cols) → append misaligns. Rename or delete before first run.

---

## FTMO State Machine (`BotState` in `RiskManager`)

```
     ┌─────────────────────────────────────┐
     │                                     │
     ▼                                     │
   ACTIVE ─── daily loss ≥ 4% ──▶ DAILY_HALT
     │                            │
     │                            └── daily rollover ──┐
     │                                                 │
     ├─ total DD ≥ 10% ─▶ MAX_DRAWDOWN_HALT (permanent)│
     │                                                 │
     ├─ user stop ──▶ MANUAL_HALT                      │
     │                                                 │
     └─ MT5 disconnect ──▶ DISCONNECTED ─ reconnect ──┘
                                                       │
                                                       ▼
                                                    ACTIVE
```

**Key behaviours**:

- `RiskManager._daily_start_balance` resets at the broker day rollover (uses `TimeManager.get_server_time().date()`).
- `RiskManager._initial_balance` = FTMO anchor — does not change during the challenge.
- `RiskManager._peak_daily_equity` = high-water mark within the day (resets at rollover).
- **Position ID matching**: `RiskManager` + `TradeManager` use `position_id` (never `ticket`).

### State Persistence (`logs/bot_state.json`, schema v4)

| Field | Purpose |
|-------|---------|
| `initial_balance` | FTMO anchor |
| `state` | BotState enum string |
| `highest_balance` | High-water mark |
| `current_day` | Broker date |
| `daily_closed_pnl` | P/L for today |
| `consecutive_losses` | Anti-revenge counter |
| `halt_until` | Cooldown timestamp |
| `daily_pnl_history` | Kept for Consistency Rule (disabled on 2-step Standard, still tracked) |
| `mt5_login` (v4) | Validation — mismatch → reset |
| `challenge_start_date` (v4) | Anchor date |
| `schema_version` (v4) | Migration check |

**Validation on load** (`RiskManager._load_state`):

- `mt5_login` mismatch → reset state (new account).
- Balance diff > 20 % → **warn only** (no auto-reset — never reset during real DD).

---

## Cooldown / Anti-Revenge

In `RiskManager` + `TradeExecutor`:

| Trigger | Behaviour | Symbol |
|---------|-----------|--------|
| Loss on symbol X | Block same-direction entries on X for 60 min | `FTMOConfig.COOLDOWN_AFTER_LOSS_MIN` |
| **3 consecutive losses (v6.13)** | Pause entire bot for 60 min (DD trigger ~2.1 % ห่าง FTMO 4 % limit) | `FTMOConfig.CONSECUTIVE_LOSS_PAUSE_COUNT=3 / MIN=60` |
| **4 consecutive losses (v6.13)** | Halt for the remainder of the day | `FTMOConfig.CONSECUTIVE_LOSS_HALT_COUNT=4` |
| Loss < 0.05 % daily | Not counted as consecutive loss | `FTMOConfig.MIN_LOSS_TO_COUNT_PCT` (tick noise filter) |

**Post-TP Pullback Lock** (prevents chasing after TP hit):

- After a TP hit on `(symbol, direction)`, block the same direction until one of:
  - Price pulls back ≥ `POST_TP_ATR_BUFFER` (0.3 × ATR) from TP, or
  - Price touches EMA20 M15 within the last 3 bars, or
  - TTL expires (60 min).

---

## Trading Sessions & Friday Close

- Session config in `settings.py` is in **UTC** — convert before comparing to broker time (EET).
- **Friday 20:45 EET hard-close** — `TimeManager.is_friday_close_time` causes `TradeManager` to close every open position.
- London + NY overlap is the prime window (default).

### `TradeManager.check_session_close` — 3 trigger ตามลำดับ (2026-04-30 update)

| # | Trigger | Source | เวลา EET (server) | เวลา UTC | ปิดอะไร | กฎอ้างอิง |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Friday Force Close | `is_friday_close_time` | ศุกร์ 20:45 EET | — | ทุก position | กฎ FTMO weekend rule |
| 2 | Daily Overnight Close | `is_daily_close_time` | Mon-Thu 23:30 EET | — | ทุก position | **user policy** (กัน swap + gap) — ไม่ใช่กฎ FTMO 2-step Standard |
| 3 | Friday Warning | `friday_force_close − 15 min` | ศุกร์ 20:30 EET | — | ทุก position | soft wind-down ก่อน FTMO bell (EET-based, v7.0.1) |

⚠️ **Removed (2026-04-30)**: NY Session End (winners-only profit lock) ที่เคย trigger ตอน `newyork_end − 15 min` (= 16:45 UTC = 23:45 ICT) — block นี้ไม่มี upper bound → ปิด winners ตลอด ~7 ชั่วโมงต่อวัน ทับ logic BE/Partial/Trailing ใน `TradeManager`. ถอดออกทั้งหมด — TradeManager ดูแล position ผ่าน trailing/BE/partial ตามปกติจน hit SL/TP/timeout หรือเข้า trigger #1-3 ข้างบน

**FTMO 2-step Standard ไม่มีกฎห้ามถือ overnight Mon-Thu** — Daily Overnight Close (#2) เป็นการตัดสินใจของ project เพื่อกัน swap + gap ปิด/เปิดผ่าน `bot_config.sessions.enforce_daily_close` (default = True)

⚠️ **MT5 FTMO tick.time quirk**: do not use `mt5.symbol_info_tick().time` directly → FTMO-Demo returns broker-local epoch (EEST) → `fromtimestamp(tz=Bucharest)` double-adds the offset (+3 h) → use `datetime.now(Europe/Bucharest)` inside `TimeManager.get_server_time` → VPS must be NTP-synced.

---

## News Handling (2-tier)

### Priority 1: JSON calendar

- `config/news_calendar.json` (schema: `updated_at`, `valid_until`, `events[]`).
- Each event: `datetime_utc`, `currency`, `name`, `impact`.
- Auto-imported every Sunday 23:30 EET via `NewsCalendarScheduler.check_and_run`.
- **Entry block** — blocks NEW signals inside `[event − no_trade_before_news_minutes (60), event + no_trade_after_news_minutes (20)]`. (after-window was 45 pre-v8.0.78; before-window 60 kept for NFP/FOMC aftermath per v8.0.38.)
- **v7.1.10 / v8.0.78 — Pre-news position close**: `TradeManager.check_news_close()` ปิด position ที่เปิดอยู่ก่อนข่าว **`news_close_before_minutes` (v8.0.78 = 10, decoupled from the 60 entry-window)**, window_after = 0. ⚠️ **Invariant**: close-window MUST be `< no_trade_before_news_minutes` by a runway margin — เดิมทั้งคู่ = 60 → ไม้ที่เปิดช่วง 60-90 นาทีก่อนข่าวโดนปิดทันที (runway 0). 60 − 10 = runway ≥50 นาที. USD news → ปิด EURUSD/GBPUSD/USDJPY/AUDUSD/USDCAD/USDCHF/NZDUSD + **XAUUSD**. Close reason = `"Pre-news close"`

### Priority 2: Hardcoded fallback

- `config/news_events.py` → `HIGH_IMPACT_EVENTS` (list of `RecurringNewsEvent`).
- Used when the JSON is missing or expired (`valid_until` passed).
- Lower accuracy (~40–50 %) since exact dates/times are unknown.

### Weekly Update Workflow

1. Download CSV from ForexFactory (Time Zone = GMT/UTC, Impact = High, Next Week).
2. Drop the file into `config/news_inbox/` (any filename).
3. Wait until Sunday 23:30 EET — bot auto-imports and moves the CSV into `processed/`.

Manual alternative: `python scripts/import_forexfactory_csv.py <path/to/csv>`.

---

## Starting a New FTMO Challenge

**Steps**:

```bash
# 1. Stop the bot
# 2. Back up the old state
mv logs/bot_state.json logs/bot_state.json.bak_$(date +%s)
# 3. Update .env if using a new account: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
# 4. Start the bot — RiskManager._load_state creates a fresh state automatically
python main.py
```

**Do not**:

- ❌ Delete `bot_state.json` mid-challenge (DD anchor is lost).
- ❌ Hand-edit `initial_balance`.
- ❌ Change the MT5 account without resetting.

---

## Discord Notifications (`DiscordNotifier`)

- Enabled via `.env` (`DISCORD_ENABLE=True`, `DISCORD_WEBHOOK_URL`).
- Rate limit: 20 req/min sliding window + min 1 s interval.
- 429 handling: Retry-After header.
- Thread-safe via `threading.Lock`.

---

## Cross-links

- Pipeline architecture → [01-architecture.md](01-architecture.md)
- Module inventory → [02-modules.md](02-modules.md)
- RL obs / reward / PPO → [03-rl-training.md](03-rl-training.md)
- Red-flag rules → [05-invariants.md](05-invariants.md)

## Invariants & Gotchas

- ⛔ Do not reorder the main loop — Risk stays first.
- ⛔ Deleting `bot_state.json` mid-challenge destroys the DD anchor.
- ⚠️ `CONSISTENCY_RULE_THRESHOLD` must be 1.0 for 2-step Standard (change before switching programs).
- ⚠️ Session config is UTC but broker is EET — convert before comparing.
