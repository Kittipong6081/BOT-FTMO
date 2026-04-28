# CLAUDE.md — Project Instructions

Claude Code loads this file automatically in every conversation for this project. Read and follow every time.

---

## 🌐 Language Policy (CRITICAL)

- **All documentation in English** — `context.md`, `wiki/*.md`, `CLAUDE.md`, and any new docs. Clear, concise.
- **Exception — `readme.md` MUST be in Thai.** User-facing end-user document.
- **Chat replies to the user in Thai.** Plan files under `~/.claude/plans/` in Thai.
- Class, method, and variable names stay in English regardless of surrounding prose language.

---

## 🚀 Conversation Start Protocol (MANDATORY)

**Before answering the first project-related question in any new conversation**, you must:

1. **Read `context.md` directly** with the `Read` tool — not via Explore/general-purpose subagent. `context.md` is the Hub/Index and is short (≤ 250 lines).
2. **Read the relevant `wiki/*.md` directly** with the `Read` tool, using the § Wiki Navigation table in `context.md` to pick the right file(s) for the topic. Do not rely on subagent summaries for wiki content.
3. **Subagents augment, they do not substitute.** You may use Explore for searching `.py` source, grep, file lookups, or non-wiki data files — but the main agent must read `context.md` and the relevant `wiki/*.md` itself before reasoning about the answer.

**Why** — subagent summaries are lossy and have caused real misses: e.g. reporting only `Trades` sheet 2 rows while missing `Signals` sheet 177 rows in `ftmo_trades.xlsx`, and missing the `Obs27 JSON for retrain` design intent stated in `context.md`. The user notices.

**Triggers** — any question that touches: code, data, architecture, modules, training, RL/ML, FTMO rules, MT5, config, obs dim/features, reward, invariants, operations, or workflow.

**Skip when** — pure chitchat unrelated to the project (greetings, current date, generic non-project questions).

---

## 📚 Project Overview

- **FTMO Trading Bot** — 3-brain system (SMC rules + ML GBM + RL PPO) for passing the FTMO 2-step Standard Challenge.
- **MANDATORY at conversation start**: Read `context.md` directly with the Read tool before answering the first project-related question (see § Conversation Start Protocol above). Drill into `wiki/*.md` for topic-specific detail using the same Read-direct rule.
- Live entry: `python main.py` → `FTMOTradingBot.run` loop every 5 s.

---

## 🔁 Wiki Sync Protocol (MANDATORY)

After editing any `.py` file under `ftmo_trading_bot/`, you **must** do all 4 steps in the same turn without being asked.

### Step 1 — Identify impact

Decide which wiki sections are affected:

| Change type | Files to update |
|-------------|-----------------|
| Obs dim / feature / order | `wiki/03-rl-training.md` + `wiki/05-invariants.md` (version log) + `context.md` (Headline Numbers) |
| Config values (risk %, symbols, DD thresholds, intervals) | `context.md` (Headline Numbers) + `wiki/04-operations.md` + `readme.md` (user-facing) |
| Add/remove/rename class/method/module | `wiki/02-modules.md` |
| Main loop / state machine | `wiki/01-architecture.md` + `wiki/04-operations.md` |
| Training pipeline / PPO hyperparams / reward | `wiki/03-rl-training.md` |
| User-facing flow (install, CLI, workflow) | `readme.md` (Thai) |

### Step 2 — Update wiki + context

- Edit affected values/text in the relevant files.
- Bump **Last Updated** at the top of each file touched (format `YYYY-MM-DD`).
- For obs / architecture / FTMO-level changes, add an entry to `wiki/05-invariants.md` Version Log.

### Step 3 — Update readme.md if needed

If the change touches user-facing flow (command, install, config the user sees), update `readme.md` (Thai) accordingly.

### Step 4 — Short report

Tell the user in 1–2 lines which docs were updated. Do not ask for approval.

---

## 📏 Wiki Writing Rules

### ✅ Reference code by symbol names

- Use: `` `SelfLearningAgent.OBS_DIM` in `ftmo_trading_bot/ml/rl_agent.py` ``
- Use: `` `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` ``
- Use: `` `FTMOTradingBot._build_signal_observation` ``

### ❌ Never use line numbers as source pointers

- ❌ Do not write: `rl_agent.py:34` or `main.py:412-525`
- Reason: line numbers rot the moment anyone edits the file → wiki looks outdated even when still correct.
- Only exception: in-wiki cross-links via markdown `#anchor` headers.

### 📝 Style

- Bullet-heavy. Prose paragraphs ≤ 3 lines.
- Every number / threshold must carry a source symbol next to it.
- Every file has: `Last Updated` + `TL;DR` + `Quick Reference` table + `Cross-links` + `Invariants & Gotchas`.
- `context.md` ≤ 250 lines. Each `wiki/*.md` ≤ 350 lines.

---

## 🔴 Critical Invariants (must-read)

Full list in `wiki/05-invariants.md`. Short version:

- ⛔ Obs 27 dims must be in sync in 3 places: `FTMOSignalFilterEnv._get_obs`, `FTMOTradingBot._build_signal_observation`, `SelfLearningAgent.OBS_DIM`.
- ⛔ FTMO anchor (`RiskManager._initial_balance`) must not change mid-challenge. Never delete `logs/bot_state.json` while running.
- ⛔ MT5 deal matching uses `position_id`, not `ticket`.
- ⛔ Timezone: broker = EET, config = UTC — convert before comparing. Do not use `mt5.symbol_info_tick().time` directly (FTMO quirk, +3 h drift).
- ⛔ FTMO program = 2-step Standard → `CONSISTENCY_RULE_THRESHOLD = 1.0`.

---

## 🧪 Before commit

- Confirm `wiki/`, `context.md`, and (if needed) `readme.md` are updated.
- The Stop hook warns when `.py` files changed but no doc files changed.
- Do not use `--no-verify`. If the hook warns, sync docs first.

---

## Entry Points

- Live: `python main.py` (runs `FTMOTradingBot.run`)
- Training pipeline (in order): `build_signal_pool.py` → `train_signal_quality.py` → `train_signal_filter.py`
- Evaluation: `train_signal_filter.py --eval_only`

Full reference: `context.md` + `wiki/`.
