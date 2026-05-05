# 04 — Live Operations (Loop, FTMO State, News, Sessions)
> Last Updated: 2026-05-04 (v7.1 staged) | Scope: main loop, RiskManager state machine, FTMO rules, news, trading sessions, console quiet mode, live logging
>
> **v7.1 changes** — `RiskManager.check_unrealized_circuit_breaker` (เรียกใน `can_open_trade` หลัง global pause) + cross-group `MAX_USD_THEME_POSITIONS` ใน `TradeExecutor._check_correlation_risk` + GBM drift monitor (`_check_gbm_drift`) ทุก 720 loops

## TL;DR (30-second scan)

- Entry: `python main.py` — builds `FTMOTradingBot` and loops every 5 s.
- FTMO program = **2-step Standard** → `CONSISTENCY_RULE_THRESHOLD = 1.0` (check disabled because the program has no Consistency Rule).
- Risk hard stops: **4 % daily DD**, **8 % total DD** (buffer vs FTMO 5 %/10 %), target **10 % profit**.
- Default risk per trade = **0.7 %** (verified at 5000 eps).
- All internal times are **EET** (Europe/Bucharest) via `TimeManager.get_server_time()`.
- News block: weekly auto-import every Sunday 23:30 EET from `config/news_inbox/`.
- **Console quiet mode (v6.9)**: idle-state prints use announce-once flags; per-signal SKIP/NO_AGENT goes to Excel `Signals` sheet, not console.
- **Live logging**: `TradeLogger` writes `logs/ftmo_trades.xlsx` (4 sheets: Trades 64 cols, Signals 21 cols, Daily, Stats). Schema bumped v6.10 (added `Partial Skipped` + `Executor Reject`). Includes `Obs27 JSON` for offline retrain.

## Quick Reference

| Item | Value | Source (symbol) |
|------|-------|-----------------|
| Main loop interval | 5 s (default) | `bot_config.main_loop_interval` |
| Symbols | 10 (incl. XAUUSD) | `SymbolConfig.symbols` |
| Daily DD stop | 4 % | `FTMOConfig.DAILY_LOSS_HARD_STOP_PCT` |
| Total DD stop | 8 % | `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` |
| Profit target | 10 % | `FTMOConfig.PROFIT_TARGET_PCT` |
| Default risk / trade | 0.7 % | `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` |
| Max open positions | 3 | `FTMOConfig.MAX_OPEN_POSITIONS` |
| Min confluence | 70 | `FTMOConfig.MIN_CONFLUENCE_SCORE` |
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
| 2 | Session check | `TimeManager.is_trading_session` | outside London/NY or after Friday cutoff → skip |
| 3 | News scheduler | `NewsCalendarScheduler.check_and_run` | Sunday 23:30 EET → auto-import CSV (non-blocking) |
| 4 | News filter | `news_events` / `news_calendar.json` | within ±30 / 15 min of a high-impact event → skip |
| 5 | Strategy scan | `SMCStrategy.scan_all_symbols` | every 12 loops (~1 min); confluence < `MIN_CONFLUENCE_SCORE` → drop |
| 6 | ML quality | `SignalQualityModel.score` | populates `live_context["ml_score"]` (calibrated) |
| 6b | **ML gate (v6.12)** | `FTMOTradingBot.run` checks `ml_score < bot_config.ftmo.ML_FILTER_THRESHOLD` | logged as `ML_FILTERED` in `Signals` sheet — **must equal `--ml_threshold` ตอน train (0.36)** |
| 7 | RL decision | `SelfLearningAgent.should_take_signal` | SKIP → drop signal (logged to `Signals` sheet as AGENT_SKIP) |
| 8 | Build live context | `FTMOTradingBot._build_live_context(sig)` | computes ml_score, ADX, biases, balance, overtrading metrics, **`obs_27_json`** |
| 9 | Execute | `TradeExecutor.execute_signal(sig, live_context)` | final risk / correlation / cooldown check; logs to Trades sheet |
| 10 | Manage open | `TradeManager.update_positions` | trailing / BE / partial / Friday Force Close (EET) / Daily Overnight Close (EET) / Friday warning (UTC) |

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
     ├─ total DD ≥ 8% ──▶ MAX_DRAWDOWN_HALT (permanent)│
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
- Blocks signals inside the window `[event − 30 min, event + 15 min]` (`no_trade_before_news_minutes`, `no_trade_after_news_minutes`).

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
