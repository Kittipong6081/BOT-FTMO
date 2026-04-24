# 04 — Live Operations (Loop, FTMO State, News, Sessions)
> Last Updated: 2026-04-24 | Scope: main loop, RiskManager state machine, FTMO rules, news, trading sessions

## TL;DR (30-second scan)

- Entry: `python main.py` — builds `FTMOTradingBot` and loops every 5 s.
- FTMO program = **2-step Standard** → `CONSISTENCY_RULE_THRESHOLD = 1.0` (check disabled because the program has no Consistency Rule).
- Risk hard stops: **4 % daily DD**, **8 % total DD** (buffer vs FTMO 5 %/10 %), target **10 % profit**.
- Default risk per trade = **0.7 %** (verified at 5000 eps).
- All internal times are **EET** (Europe/Bucharest) via `TimeManager.get_server_time()`.
- News block: weekly auto-import every Sunday 23:30 EET from `config/news_inbox/`.

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
| 5 | Strategy scan | `SMCStrategy.scan_all_symbols` | confluence < `MIN_CONFLUENCE_SCORE` → drop |
| 6 | ML quality | `SignalQualityModel.score` | never skips — just attaches `ml_score` |
| 7 | RL decision | `SelfLearningAgent.should_take_signal` | SKIP → drop signal |
| 8 | Execute | `TradeExecutor.execute_trade` | final risk / correlation / cooldown check |
| 9 | Manage open | `TradeManager.update_positions` | trailing / BE / partial / session close |

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
| 2 consecutive losses | Pause entire bot for 60 min | `FTMOConfig.CONSECUTIVE_LOSS_PAUSE_COUNT/MIN` |
| 3 consecutive losses | Halt for the remainder of the day | `FTMOConfig.CONSECUTIVE_LOSS_HALT_COUNT` |
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
