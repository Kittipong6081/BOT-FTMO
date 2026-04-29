# 🤖 FTMO Trading Bot

> ระบบเทรด Forex อัตโนมัติเพื่อผ่าน **FTMO 2-Step Standard Challenge** (10% profit, 4% daily DD, 8% total DD) ใช้ **3 สมอง** ทำงานร่วมกัน: SMC Strategy + ML Quality Filter + RL Agent (PPO with Auxiliary Task)
>
> **Last Updated:** 2026-04-29 (v6.13) | **Verified:** Pass Rate **9.7%** (5000 eps eval, +185% จาก v6.11.3 baseline 3.4%)

---

## 📖 สารบัญ

- [โปรเจคนี้ทำอะไร](#-โปรเจคนี้ทำอะไร)
- [สถาปัตยกรรม 3 สมอง](#-สถาปัตยกรรม-3-สมอง)
- [โครงสร้างโปรเจค](#-โครงสร้างโปรเจค)
- [ติดตั้ง (สำคัญ — pin versions)](#-ติดตั้ง-สำคัญ--pin-versions)
- [Deploy บน VPS — ห้ามผิดเวอร์ชัน](#-deploy-บน-vps--ห้ามผิดเวอร์ชัน)
- [เตรียมข้อมูล (เฉพาะตอน train ใหม่)](#-เตรียมข้อมูล-เฉพาะตอน-train-ใหม่)
- [Training Pipeline](#-training-pipeline)
- [Evaluation](#-evaluation)
- [Live Trading](#-live-trading)
- [Live Logging (Excel)](#-live-logging-excel)
- [News Calendar Update](#-news-calendar-update)
- [เริ่ม FTMO Challenge ใหม่](#-เริ่ม-ftmo-challenge-ใหม่)
- [Configuration Reference](#-configuration-reference)
- [FAQ](#-faq)

---

## 🎯 โปรเจคนี้ทำอะไร

**FTMO** เป็น Prop Firm ที่ให้ทุน $100K ถ้าผ่านการสอบ:

- เป้าหมาย: **+10% profit** ภายใน ~45 วัน
- ห้าม Daily DD เกิน **5%** | ห้าม Total DD เกิน **10%**
- 90%+ ของผู้สอบไม่ผ่านเพราะความโลภ + เสียวินัย

**Bot ทำหน้าที่:**

- ใช้ **Smart Money Concepts (SMC)** หา setup เทรด (BOS/CHoCH, Order Blocks, FVG, Liquidity Sweeps)
- ใช้ **ML (GBM + Isotonic Calibration)** กรอง signal ที่มีโอกาสชนะสูง
- ใช้ **PPO RL Agent** + Auxiliary Task ตัดสินใจ TAKE/SKIP โดยมองทั้งคุณภาพ signal และสถานะบัญชี (DD, progress, trades today)
- ทำงาน **24/5** อัตโนมัติ + buffer ป้องกัน FTMO breach (4%/8% แทน 5%/10%)

---

## 🧠 สถาปัตยกรรม 3 สมอง

### 1. 🎯 SMC Strategy (Brain 1) — หา setup

ใน [`ftmo_trading_bot/strategy/`](ftmo_trading_bot/strategy/) มี:

- `smc_strategy.py` — main engine, scan 9 symbols ทุกนาที
- `market_structure.py` — BOS / CHoCH detection (D1 → H4 → H1 → M15)
- `order_blocks.py` — Order Block detection
- `fair_value_gaps.py` — FVG detection
- `liquidity_sweeps.py` — Liquidity Sweep detection
- `indicators.py` — RSI, MACD, ADX, ATR, Bollinger, Stochastic

**Output:** `Signal` (symbol, BUY/SELL, entry, SL, TP, confluence_score, RR)

### 2. 🔬 ML Quality Filter (Brain 2) — กลั่นกรอง

ใน [`ftmo_trading_bot/ml/signal_quality.py`](ftmo_trading_bot/ml/signal_quality.py):

- **GBM Classifier** (sklearn `GradientBoostingClassifier`) เรียนทำนาย P(win) ของแต่ละ signal
- **Isotonic Calibration** บนผล OOF (`GroupKFold cross_val_predict`) → ป้องกัน overconfident probabilities
- ใช้ feature 30+ ตัว: confluence, ATR, RR, ADX, RSI, MACD, BB %B, MTF/HTF bias, session, day-of-week, ฯลฯ
- **Threshold เริ่มต้น: 0.36** (`bot_config.ftmo.ML_FILTER_THRESHOLD`) — signal ที่ score < 0.36 ถูก reject ทันที (ก่อนถึง RL agent) ทั้งใน live และตอน train. **v6.12**: live ก็บังคับ threshold เดียวกันแล้ว (เดิม live ไม่ได้บังคับ — เป็น bug ที่ทำให้ distribution ต่างจากตอน train); reject log เป็น `Result = "ML_FILTERED"` ใน `Signals` sheet

**Output:** `ml_score ∈ [0, 1]` (calibrated)

### 3. 🎓 RL Agent — PPO + Auxiliary Task (Brain 3)

ใน [`ftmo_trading_bot/ml/`](ftmo_trading_bot/ml/) มี:

- `aux_aware_policy.py` — PPO policy พร้อม `aux_head: nn.Linear(latent_dim_pi, 1)` ทำนาย `outcome_pnl_ratio`
- `aux_aware_ppo.py` — PPO subclass เพิ่ม MSE aux_loss (weight=0.5)
- `aux_rollout_buffer.py` — RolloutBuffer พร้อม `aux_targets`
- `signal_filter_env.py` — Gymnasium env (state = 27-dim obs, action = TAKE/SKIP)
- `rl_agent.py` — wrapper สำหรับ inference (`should_take_signal`, `get_action_confidence`)

**Obs 27 dims** (ต้อง sync 3 จุด — ดู [`wiki/05-invariants.md`](wiki/05-invariants.md)):

```
[0]  ml_score              [9]  bb_pctb           [18] consec_norm
[1]  atr_pips              [10] atr_chg           [19] spread_pct_of_atr
[2]  sl_atr                [11] price_roc         [20] has_opposite_recently_closed
[3]  rsi_norm              [12] total_dd_n        [21] htf_trend_alignment
[4]  macd_norm             [13] daily_dd_n        [22] bias_align
[5]  trend_str             [14] progress_n        [23] direction
[6]  ob_size_atr           [15] day_progress      [24] rr_norm
[7]  adx_norm              [16] trades_today_n    [25] confluence_norm
[8]  stoch_norm            [17] recent_wr_norm    [26] (reserved)
```

**Reward:** ratio (P/L ÷ risk) + DD penalty + activity floor + auxiliary signal (Phase E2)

**Verified Pass Rate:** **10.0%** (5000-eps eval, 3× baseline 3.5%)

---

## 📂 โครงสร้างโปรเจค

```
BOT-FTMO/
├── readme.md                          # คู่มือผู้ใช้ (ภาษาไทย — ไฟล์นี้)
├── context.md                         # Project hub/index (English, สำหรับ AI assistants)
├── CLAUDE.md                          # AI assistant instructions
├── wiki/                              # Technical docs (English)
│   ├── 01-architecture.md
│   ├── 02-modules.md
│   ├── 03-rl-training.md
│   ├── 04-operations.md
│   └── 05-invariants.md
└── ftmo_trading_bot/                  # Source code
    ├── main.py                        # 🎯 Live bot loop
    ├── requirements.txt               # Pinned dependencies
    │
    ├── config/
    │   ├── settings.py                # FTMO/Symbol/Session config
    │   ├── news_calendar.json         # Imported news events
    │   └── news_inbox/                # Drop ForexFactory CSV here
    │
    ├── strategy/                      # 🎯 Brain 1: SMC
    │   ├── smc_strategy.py
    │   ├── market_structure.py
    │   ├── order_blocks.py
    │   ├── fair_value_gaps.py
    │   ├── liquidity_sweeps.py
    │   └── indicators.py
    │
    ├── ml/                            # 🔬🎓 Brain 2 + 3: ML + RL
    │   ├── signal_quality.py          # GBM wrapper (with calibrator)
    │   ├── rl_agent.py                # PPO inference wrapper
    │   ├── signal_filter_env.py       # Gymnasium env (27 dims)
    │   ├── aux_aware_policy.py        # PPO + aux head
    │   ├── aux_aware_ppo.py           # PPO subclass + aux loss
    │   ├── aux_rollout_buffer.py      # RolloutBuffer + aux_targets
    │   └── strategy_backtester.py     # Pool generator (run SMC on history)
    │
    ├── execution/                     # 💱 Trade execution + management
    │   ├── trade_executor.py          # Open/close orders + sync MT5 deals
    │   └── trade_manager.py           # BE move, partial close, trailing
    │
    ├── core/                          # 🛡️ Core infrastructure
    │   ├── mt5_connector.py           # MT5 API + Mock Mode
    │   ├── risk_manager.py            # FTMO anchor + DD + cooldown
    │   ├── position_sizer.py          # Lot calc (3-case JPY/USD/cross)
    │   ├── time_manager.py            # EET/UTC convert + sessions
    │   ├── news_scheduler.py          # Auto-import news CSV
    │   └── notifier.py                # Discord webhooks
    │
    ├── analytics/                     # 📊 Logging
    │   ├── trade_logger.py            # Excel logger (4 sheets)
    │   └── performance.py             # Win rate / Sharpe / Profit Factor
    │
    ├── scripts/                       # 🛠️ Training + utilities
    │   ├── fetch_mt5_data.py          # Step 0: ดึง OHLCV CSV
    │   ├── build_signal_pool.py       # Step 1: Generate pool
    │   ├── train_signal_quality.py    # Step 2: Train GBM
    │   ├── train_signal_filter.py     # Step 3: Train PPO + Eval
    │   └── import_forexfactory_csv.py # News calendar manual import
    │
    ├── models/                        # 🧠 Trained AI checkpoints
    │   ├── ppo_signal_filter.zip      # Final PPO agent
    │   ├── ppo_signal_filter_p1.zip   # Phase 1 only
    │   ├── vec_normalize_sf.pkl       # ⚠️ CRITICAL: obs scale (ห้ามลืม commit)
    │   ├── vec_normalize_sf_p1.pkl
    │   └── checkpoints_sf/            # Mid-train backups
    │
    ├── data/
    │   ├── ohlcv/                     # 27 CSV files (9 symbols × 3 TF)
    │   ├── signal_pool_3000.pkl       # Pre-generated signal pool
    │   └── signal_quality_model.pkl   # GBM (with calibrator inside)
    │
    ├── logs/
    │   ├── ftmo_trades.xlsx           # Live trade + signal log (auto-create)
    │   ├── bot_state.json             # FTMO anchor + DD state (ห้ามลบ mid-challenge)
    │   ├── tb_signal_filter/          # TensorBoard logs
    │   └── news_scheduler_state.json
    │
    └── tests/                         # pytest suite
```

---

## 📥 ติดตั้ง (สำคัญ — pin versions)

### ⚠️ เหตุผลที่ต้อง pin versions

PPO agent ที่ verified ได้ Pass Rate 10.0% **train ด้วย package versions เฉพาะเจาะจง**:

```
torch==2.11.0           gymnasium==1.2.3
stable-baselines3==2.8.0  numpy==2.4.4
scikit-learn==1.8.0     pandas==3.0.2
```

ถ้า version ต่างกัน → neural net forward pass อาจคลาด ~1e-6 (CPU BLAS) → action ส่วนใหญ่เหมือน แต่ **borderline cases (prob 0.49 vs 0.51) อาจ flip** → Pass Rate drift

[`requirements.txt`](ftmo_trading_bot/requirements.txt) **pin versions ครบทุกตัวแล้ว** — ใช้ `pip install -r` ตามนี้

### Step 1: Clone

```bash
git clone <repo-url>
cd BOT-FTMO
```

### Step 2: Python version

ต้องการ **Python 3.11+** (แนะนำ 3.14.4 ให้ตรงกับเครื่อง train)

```bash
python3 --version    # ตรวจ version
```

### Step 3: สร้าง venv

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate           # หรือใช้ .venv/bin/python ตรงๆ
```

**Windows (VPS):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Step 4: ติดตั้ง dependencies

**macOS / Linux:**
```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r ftmo_trading_bot/requirements.txt
```

**Windows (PowerShell / CMD):**
```cmd
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r ftmo_trading_bot\requirements.txt
```

### Step 5: ติดตั้ง MetaTrader5 (เฉพาะ Windows)

```cmd
.venv\Scripts\pip install MetaTrader5
```

⚠️ MetaTrader5 library **ใช้ได้เฉพาะ Windows** — บน macOS/Linux จะเข้า Mock Mode อัตโนมัติ (ใช้สำหรับ training/dev เท่านั้น, ไม่สามารถเทรดจริง)

### Step 6: ตรวจสอบ install สำเร็จ

**macOS / Linux:**
```bash
.venv/bin/pip freeze | grep -E "^(torch|stable|gymnasium|numpy|scikit)"
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\pip freeze | Select-String "torch|stable|gymnasium|numpy|scikit"
```

ผลลัพธ์ที่ถูก (ทั้ง 2 platform):
```
gymnasium==1.2.3
numpy==2.4.4
scikit-learn==1.8.0
stable_baselines3==2.8.0
torch==2.11.0
```

### Step 7: Quick smoke test

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/main.py --status
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\main.py --status
```

ถ้าเห็น banner + version + symbols list → install สำเร็จ

---

## 🚀 Deploy บน VPS — ห้ามผิดเวอร์ชัน

VPS Live trading ต้อง:

1. **ใช้ Windows VPS** (เพื่อให้ MetaTrader5 library ทำงานได้)
2. **Python version match** — แนะนำ 3.11+ (3.14.4 ตรงเครื่อง train ดีที่สุด)
3. **Pin versions match** — ห้าม upgrade packages โดยไม่ retrain

### ขั้นตอน deploy

```cmd
# 1. Clone repo (มี models/ + data/ + logs/ครบ)
git clone <repo-url>
cd BOT-FTMO

# 2. สร้าง venv + install ตาม pinned versions
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r ftmo_trading_bot\requirements.txt
pip install MetaTrader5

# 3. ตรวจ versions ตรงกับเครื่อง train
pip freeze > vps_versions.txt
# เปรียบเทียบกับ dev machine — ถ้าต่างมีปัญหาแน่
```

### ⚠️ ไฟล์ที่ต้องมีบน VPS (commit ขึ้น git ครบ)

| ไฟล์ | ทำไมต้องมี |
|------|-----------|
| `ftmo_trading_bot/models/ppo_signal_filter.zip` | RL agent weights |
| `ftmo_trading_bot/models/vec_normalize_sf.pkl` | Obs normalization stats — **ขาดไม่ได้** ไม่งั้น obs scale ผิด → agent พัง |
| `ftmo_trading_bot/data/signal_quality_model.pkl` | GBM + calibrator |
| `ftmo_trading_bot/config/settings.py` | Risk/symbols config |

ถ้าใน `.gitignore` มี `*.zip` หรือ `*.pkl` → ต้องลบออก หรือใช้ `git lfs` แทน

### รัน live bot บน VPS

```cmd
.venv\Scripts\python ftmo_trading_bot\main.py
```

จะรัน loop ทุก 5 วินาที + scan signals ทุก 1 นาที + log ทุก trade ลง Excel

แนะนำให้รันใน **Windows Task Scheduler** หรือ **NSSM** เพื่อ auto-restart เวลา crash

---

## 📊 เตรียมข้อมูล (เฉพาะตอน train ใหม่)

⚠️ ถ้า**ไม่ retrain** ข้ามไปได้เลย — repo มี models เทรนเสร็จแล้ว (Pass Rate 10%)

### วิธี 1: ดึงจาก MT5 อัตโนมัติ (Windows + MT5 connected)

```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\fetch_mt5_data.py
```

(macOS/Linux ใช้ MT5 ไม่ได้ — Mock Mode เท่านั้น)

ดึง OHLCV ของ 9 symbols × 3 TF (M15, H1, H4, D1) → save ลง [`data/ohlcv/`](ftmo_trading_bot/data/ohlcv/)

### วิธี 2: ใช้ CSV ของตัวเอง

วาง CSV ใน [`data/ohlcv/`](ftmo_trading_bot/data/ohlcv/) ตามรูปแบบ:

- ชื่อไฟล์: `EURUSD_M15.csv`, `EURUSD_H1.csv`, ..., `XAUUSD_D1.csv`
- คอลัมน์: `time, open, high, low, close, tick_volume, real_volume, spread`
- แนะนำ ≥ 6 ปีย้อนหลัง สำหรับ pool ขนาด 3000 episodes

---

## 🏋️ Training Pipeline

3 ขั้นตอน — **ห้ามข้าม ห้ามสลับลำดับ**

### Step 1 — Build Signal Pool

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py \
    --pool_size 3000 --workers 8 --max_days 45
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\build_signal_pool.py ^
    --pool_size 3000 --workers 8 --max_days 45
```

**ทำอะไร:** simulate SMC strategy run บน historical data → save 3000 episodes (แต่ละ ep = 45 วัน, มี 0–N signals พร้อม outcome) ลง `data/signal_pool_3000.pkl`

**เวลา:** ~30–45 นาทีบน CPU 8 cores

### Step 2 — Train ML Quality Model (GBM + Calibrator)

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\train_signal_quality.py
```

**ทำอะไร:**
1. Load `signal_pool_3000.pkl`
2. Train GBM ด้วย `GroupKFold cross_val_predict` (group = episode_id) → OOF predictions
3. Fit `IsotonicRegression` calibrator บน OOF
4. Save `data/signal_quality_model.pkl` (มี GBM + calibrator)
5. Re-score pool → update `ml_score` ของทุก signal

**เวลา:** ~5 นาที

### Step 3 — Train PPO Agent (with Auxiliary Task)

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py \
    --pool_size 3000 \
    --ml_threshold 0.36 \
    --risk_per_trade 0.007 \
    --timesteps_p1 300000 \
    --timesteps_p2 200000
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\train_signal_filter.py ^
    --pool_size 3000 ^
    --ml_threshold 0.36 ^
    --risk_per_trade 0.007 ^
    --timesteps_p1 300000 ^
    --timesteps_p2 200000
```

**Curriculum 2-phase:**

- **P1 (300K steps):** Quality-first reward — เน้น win rate/profit, no DD penalty
- **P2 (200K steps):** Risk-aware — เพิ่ม DD penalty + activity floor + early-stop on value loss

**Auxiliary Task:** policy ทำนาย `outcome_pnl_ratio` (`aux_head`) เพิ่ม MSE loss weight=0.5 — บังคับให้ representation รู้จัก signal quality (จาก paper "Auxiliary task helps PPO converge")

**เวลา:** ~2–3 ชม. บน CPU (CPU เพียงพอ — ไม่ต้องใช้ GPU)

**Output:**
- `models/ppo_signal_filter.zip` — final agent
- `models/vec_normalize_sf.pkl` — obs/reward normalization stats
- `models/checkpoints_sf/` — backup ทุก 50K steps

---

## 🧪 Evaluation

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py \
    --eval_only \
    --pool_size 3000 \
    --ml_threshold 0.36 \
    --risk_per_trade 0.007
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\train_signal_filter.py ^
    --eval_only ^
    --pool_size 3000 ^
    --ml_threshold 0.36 ^
    --risk_per_trade 0.007
```

รัน 5000 episodes (default) แล้ว report:

```
 Eval Result (5000 episodes)
   Pass (hit 10%):     500  (10.0%)    ← target
   Breach (DD limit):   25  ( 0.5%)    ← FTMO breach (ต้อง < 5%)
   Survive, no target: 4475 (89.5%)
```

⚠️ ใช้ venv interpreter เสมอ (`.venv/bin/python` บน macOS/Linux, `.venv\Scripts\python` บน Windows) — หรือ activate venv ก่อน. ถ้าใช้ Python คนละ env, RNG seed + package versions จะต่าง → Pass Rate drift

---

## 💰 Live Trading

### Step 1 — ตั้งค่า .env (Windows VPS)

```bash
# ftmo_trading_bot/.env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=FTMO-Demo

# Discord webhook (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/.../...
```

### Step 2 — Sync config

ตรวจ [`ftmo_trading_bot/config/settings.py`](ftmo_trading_bot/config/settings.py) ให้ตรงกับ challenge:

| Field | Value | คำอธิบาย |
|-------|-------|---------|
| `DEFAULT_RISK_PER_TRADE_PCT` | 0.007 | 0.7% (verified optimal) |
| `ML_FILTER_THRESHOLD` (v6.12) | 0.36 | live ↔ training sync — signals ที่ ML score < 0.36 ถูก reject ก่อน agent |
| `MAX_DRAWDOWN_HARD_STOP_PCT` | 0.08 | 8% (buffer 2% จาก FTMO 10%) |
| `DAILY_LOSS_HARD_STOP_PCT` | 0.04 | 4% (buffer 1% จาก FTMO 5%) |
| `PROFIT_TARGET_PCT` | 0.10 | 10% target |
| `CONSISTENCY_RULE_THRESHOLD` | 1.0 | 2-step Standard ไม่มีกฎนี้ |
| `MAX_OPEN_POSITIONS` | 3 | สูงสุด 3 positions พร้อมกัน |
| `CONSECUTIVE_LOSS_PAUSE_COUNT` (v6.13) | 3 | แพ้ติด 3 ครั้ง → pause 60 นาที (เดิม 2 — DD trigger เร็วเกินจริง) |
| `CONSECUTIVE_LOSS_HALT_COUNT` (v6.13) | 4 | แพ้ติด 4 ครั้ง → halt ทั้งวัน (เดิม 3 — sync กับ pause count) |
| `XAUUSD.sl_atr_multiplier` (v6.13) | 1.8 | XAU SL = 1.8×ATR (เดิม 1.5× global; XAU wick noise สูง) |
| `XAUUSD.tp_atr_multiplier` (v6.13) | 3.6 | XAU TP = 3.6×ATR (รักษา RR 1:2) |

### Step 3 — รัน bot

**macOS / Linux (Mock Mode — dev only):**
```bash
.venv/bin/python ftmo_trading_bot/main.py
```

**Windows VPS (live trading):**
```cmd
.venv\Scripts\python ftmo_trading_bot\main.py
```

**Console output (quiet mode — v6.9):**
- Banner + init logs (ครั้งเดียวตอน start)
- `📡 [Agent] TAKE ...` — ทุกครั้งที่ agent ตัดสินใจเปิดเทรด
- `✅ [Bot] เปิดเทรดสำเร็จ: Ticket xxx`
- `🟢/🔴 [Logger] บันทึกเทรดปิด: ... P/L=$xxx`
- `🔒 [Bot] Daily Halt — รอวันถัดไป` (ครั้งเดียวตอน entry — announce-once)
- `🌙 [Bot] Weekend Sleep` (ครั้งเดียวตอน entry)
- Errors / FTMO breach alerts

**ไม่ปริ้นแล้ว:** per-signal SKIP, per-loop status repeat (เก็บใน Excel + Discord แทน)

### Step 4 — Stop bot

`Ctrl+C` — bot จะ graceful shutdown (ปิด Excel, save state)

---

## 📝 Live Logging (Excel)

ทุก trade + signal scan → log ลง [`ftmo_trading_bot/logs/ftmo_trades.xlsx`](ftmo_trading_bot/logs/) (auto-create เมื่อรันครั้งแรก)

### 4 Sheets

#### 1. **Trades** — 63 columns

| Group | Columns |
|-------|---------|
| Core (1–19) | Ticket, Symbol, Type, Entry/SL/TP, Lot, Risk%, RR, Confluence, ATR, Times, P/L, Reasons |
| ML features (20–32) | Session, DoW, Hour, Spread, HTF Bias, Vol Regime, ConsecLoss, DD%, MAE, MFE, TimeInTrade, ExitPath |
| Decision (33–37) | ML Score (cal/raw), Agent Action, Decision, Threshold |
| Confluence breakdown (38–42) | HTF/MTF/OB/FVG/Sweep pts |
| Trade mgmt (43–46) | BE Moved, Partial Closed, Trailing, Final SL |
| Live exec (47–51) | Bid/Ask @Entry/Exit, Spread (pips) |
| Market (52–55) | ADX H1/H4, MTF/D1 Bias |
| Account (56–58) | Balance@Entry/Close, Equity Peak |
| Overtrading (59–62) | Trades Today, Trades 1h, Sec Since Last Open (any/same-sym) |
| **Retrain** (63) | **Obs27 JSON** ← full 27-dim obs vector → reconstruct state สำหรับ retrain |

#### 2. **Signals** — 20 columns
ทุก scan event (รวม `AGENT_SKIP`, `REJECTED`, `NO_SIGNAL`) — ใช้วิเคราะห์ signal frequency + reject distribution

#### 3. **Daily** — สรุปรายวัน
Date, Trades, Wins, Losses, Win Rate, P/L, DD, Balance EOD

#### 4. **Stats** — สถิติรวม
Win Rate, Sharpe, Profit Factor, etc.

### ⚠️ Schema migration

ถ้าไฟล์ `ftmo_trades.xlsx` เดิมเป็น schema เก่า (62 cols) → bot จะ append ผิดคอลัมน์
**วิธีแก้:** rename ไฟล์เดิมก่อน start bot — `mv logs/ftmo_trades.xlsx logs/ftmo_trades_old.xlsx`

---

## 📅 News Calendar Update

### Auto-Import (Sunday 23:30 EET)

`NewsScheduler.check_and_run()` ตรวจทุกอาทิตย์: ถ้ามี CSV ใน [`config/news_inbox/`](ftmo_trading_bot/config/news_inbox/) → import → save ลง `config/news_calendar.json`

### Manual Import

⚠️ **CSV path เป็น positional argument** (ไม่ใช่ `--csv`)

**macOS / Linux:**
```bash
.venv/bin/python ftmo_trading_bot/scripts/import_forexfactory_csv.py path/to/calendar.csv
```

**Windows:**
```cmd
.venv\Scripts\python ftmo_trading_bot\scripts\import_forexfactory_csv.py path\to\calendar.csv
```

Optional flags:
- `--tz-offset 0.0` — timezone offset ของ CSV (default 0 = UTC). ใช้ `--tz-offset -5` ถ้า export เป็น EST
- `--valid-days 7` — จำนวนวันที่ calendar ยัง valid
- `--output PATH` — override default `config/news_calendar.json`

**ขั้นตอน weekly:**
1. ดาวน์โหลด CSV จาก [forexfactory.com](https://forexfactory.com)
2. วางไฟล์ใน `config/news_inbox/`
3. รอ Sunday 23:30 EET — bot import auto, หรือรัน manual

---

## 🔄 เริ่ม FTMO Challenge ใหม่

⛔ **ห้ามลบ `logs/bot_state.json` ขณะ challenge ยังรัน** — `_initial_balance` (FTMO anchor) จะหายไป → DD calc พัง

### ขั้นตอน reset (เริ่มรอบใหม่)

**macOS / Linux:**
```bash
# 1. Backup state เดิม (กรณีต้องดูประวัติ)
mv ftmo_trading_bot/logs/bot_state.json ftmo_trading_bot/logs/bot_state.json.bak_$(date +%s)

# 2. Backup Excel เดิม
mv ftmo_trading_bot/logs/ftmo_trades.xlsx ftmo_trading_bot/logs/ftmo_trades_challenge1.xlsx

# 3. ตั้งค่า MT5 account ใหม่ใน .env
# 4. รัน bot — bot_state.json + ftmo_trades.xlsx จะ auto-create จาก balance ใหม่
.venv/bin/python ftmo_trading_bot/main.py
```

**Windows VPS:**
```cmd
REM 1. Backup state เดิม
move ftmo_trading_bot\logs\bot_state.json ftmo_trading_bot\logs\bot_state.json.bak

REM 2. Backup Excel เดิม
move ftmo_trading_bot\logs\ftmo_trades.xlsx ftmo_trading_bot\logs\ftmo_trades_challenge1.xlsx

REM 3. ตั้งค่า MT5 account ใหม่ใน .env
REM 4. รัน bot
.venv\Scripts\python ftmo_trading_bot\main.py
```

---

## ⚙️ Configuration Reference

### Symbols (9 forex + 1 metal)

```python
EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD,
EURJPY, GBPJPY,             # JPY pairs (pip = 0.01)
XAUUSD                      # Gold (digits=2, contract=100 oz)
```

แก้ใน `config/settings.py:SymbolConfig.symbols` (ต้อง symbol ตรงกับ broker)

### Risk Settings

```python
DEFAULT_RISK_PER_TRADE_PCT = 0.007   # 0.7% per trade
MIN_RISK_PER_TRADE_PCT = 0.005       # floor
MAX_RISK_PER_TRADE_PCT = 0.008       # cap

# FTMO buffers
MAX_DRAWDOWN_HARD_STOP_PCT = 0.08    # 8% (FTMO rule = 10%)
DAILY_LOSS_HARD_STOP_PCT = 0.04      # 4% (FTMO rule = 5%)
PROFIT_TARGET_PCT = 0.10             # 10%
```

### Anti-Overtrading

```python
MAX_CORRELATED_POSITIONS = 1            # 1 ตำแหน่งต่อกลุ่ม
MIN_CONFLUENCE_SCORE = 70.0             # ขั้นต่ำ
COOLDOWN_AFTER_LOSS_MIN = 60            # 60 นาทีก่อนเปิดคู่เดิมหลัง SL
CONSECUTIVE_LOSS_PAUSE_COUNT = 2        # หยุด 60 นาทีหลังแพ้ติด 2 ครั้ง
CONSECUTIVE_LOSS_HALT_COUNT = 3         # halt ทั้งวันหลังแพ้ติด 3 ครั้ง
```

### Post-TP Lock (anti-FOMO)

หลัง TP hit → block direction เดิมใน symbol นั้นจนกว่า price จะ pullback ≥ 0.3×ATR หรือแตะ EMA20 M15 หรือ TTL 60 นาที

---

## ❓ FAQ

### Q: เครื่อง dev (Mac) ใช้ Mock Mode — ใช้ test ก่อน deploy ได้ไหม?

A: **ใช้ Mock Mode** ทดสอบ logic / training / eval ได้ครบ แต่ **เทรดจริงไม่ได้** — ต้องใช้ Windows VPS + MetaTrader5 เท่านั้น

### Q: ทำไม Pass Rate ต่างกันเวลารัน eval หลายๆ รอบ?

A: env ไม่ได้ pin seed (`signal_filter_env.reset(seed=None)`) → แต่ละรอบ sample episodes คนละชุด → fluctuate ±1–2%. ผลที่ verified (Pass 10%) วัดจาก 5000 eps avg

### Q: รัน eval ด้วย `python` แทน `.venv/bin/python` ได้คะแนนต่างกัน?

A: เป็นไปได้ — ถ้า `python` resolve ไป interpreter อื่น (system Python หรือ Homebrew) ที่ไม่มี packages versions ตรงกัน. **วิธีแก้:**
1. ใช้ `.venv/bin/python` ตรงๆ เสมอ
2. หรือ `source .venv/bin/activate` ก่อน

### Q: VPS ต้องใช้ GPU ไหม?

A: **ไม่ต้อง** — PPO + GBM ทำงาน CPU พอ. Live inference เร็วมาก (<10ms ต่อ signal) ใน VPS 2 core

### Q: Bot ปิดบางส่วน 50% + ย้าย SL มา BE → ชน BE-SL = WIN หรือ LOSS?

A: **WIN** (ถ้า partial profit > 0) — `TradeExecutor.sync_with_mt5` accumulate profit จากทุก deals ของ position. ตัวอย่าง: partial @ 1R = +$50, remainder @ BE = $0 → `ExecutedTrade.profit = +$50` → WIN. ดูรายละเอียดที่ [`wiki/05-invariants.md` FAQ](wiki/05-invariants.md)

### Q: ใช้ FTMO Swing/Pro แทน 2-step Standard ได้ไหม?

A: ได้ แต่ต้องแก้ `CONSISTENCY_RULE_THRESHOLD` เป็น `0.45` (Swing/Pro มีกฎ max day ≤ 50% ของ total profit). ตอนนี้ตั้ง `1.0` = ปิด check (Standard ไม่มีกฎนี้)

### Q: VPS ต้อง sync NTP เวลาไหม?

A: **ต้อง** — Bot ใช้ EET ของ broker เทียบ UTC ของ system. คลาดเคลื่อน > 5s อาจทำให้ session detect ผิด, friday close miss

**Windows Setup (Run as Administrator):**
```cmd
REM 1. Config NTP source + enable manual sync
w32tm /config /manualpeerlist:"time.windows.com" /syncfromflags:manual /reliable:yes /update

REM 2. Force resync ทันที
w32tm /resync

REM 3. Verify — ตรวจ Last Successful Sync Time (ต้องไม่เก่ากว่า 1 ชม.)
w32tm /query /status
```

ถ้า `Last Successful Sync Time` ยังเก่า / error → ลอง restart Windows Time service:
```cmd
net stop w32time && net start w32time
w32tm /resync
```

### Q: Model เก่าใช้ต่อได้ไหมหลัง update repo?

A: ขึ้นกับ:
- ถ้า **obs space (27 dims) ไม่เปลี่ยน** → ใช้ `ppo_signal_filter.zip` ต่อได้
- ถ้า **เปลี่ยน obs** → ต้อง retrain ทั้งหมด (Brain 1 SMC + Brain 2 GBM + Brain 3 RL) เพื่อให้ wiring ตรงกัน

ดู `wiki/05-invariants.md § Hard Invariants § Observation Space Sync`

### Q: Console เงียบไป — ทำไมไม่เห็น scan ของ 9 symbols?

A: v6.9 เปลี่ยนเป็น **quiet mode** — per-signal SKIP/scan logs ไปอยู่ใน `Signals` sheet ของ Excel แทน. ดูได้ใน `logs/ftmo_trades.xlsx` sheet "Signals" — มี SCAN/SKIP/TAKE/REJECT records ครบ

---

## 🔗 ลิงก์อ้างอิง

- **Project Hub:** [`context.md`](context.md) — index ของ wiki + headline numbers
- **Architecture:** [`wiki/01-architecture.md`](wiki/01-architecture.md)
- **Modules ref:** [`wiki/02-modules.md`](wiki/02-modules.md)
- **RL Training:** [`wiki/03-rl-training.md`](wiki/03-rl-training.md)
- **Operations:** [`wiki/04-operations.md`](wiki/04-operations.md)
- **Invariants:** [`wiki/05-invariants.md`](wiki/05-invariants.md) ⛔ ห้ามฝ่าฝืน

---

## 📜 License

ใช้สำหรับ FTMO Challenge ของเจ้าของบัญชีเท่านั้น — ไม่อนุญาตให้แจกจ่ายหรือใช้เชิงพาณิชย์
