# 🤖 FTMO Trading Bot

> ระบบเทรด Forex อัตโนมัติเพื่อผ่าน **FTMO 2-Step Standard Challenge** (10% profit, 4% daily DD, 10% total DD) ใช้ **Mean Reversion + Trend Filter** + ML Quality Filter + RL Agent (PPO with Auxiliary Task)
>
> **Last Updated:** 2026-06-01 (v8.1.1 — เพดานขาดทุนรวม 8%→10% (= กฎ FTMO เต็ม) + แจ้งเตือน Discord แยก MR/TF ชัดเจน)
>
> **⚖️ v8.1.1 status — เพดานขาดทุนรวม + ป้ายกลยุทธ์บน Discord (สรุปภาษาคน)**: (1) **เพดานหยุดฉุกเฉิน (ขาดทุนรวม) ปรับจาก 8% เป็น 10%** ตามที่ผู้ใช้สั่ง — เดิมบอทเผื่อ buffer หยุดที่ 8% (ก่อนถึงเส้นจริง FTMO 10%) ตอนนี้ใช้เส้นจริง 10% เต็ม เพื่อให้บัญชีมีพื้นที่ฟื้นตัวมากขึ้น. ⚠️ **ข้อแลกเปลี่ยน**: ไม่มี buffer แล้ว — ตอนปิดฉุกเฉินถ้าราคากระชาก (slippage) อาจขาดทุนทะลุ 10% นิดหน่อย = ผิดกฎจริง. มีคำเตือนล่วงหน้าที่ 8% ให้รู้ตัวก่อน. (2) **การแจ้งเตือนบน Discord แยก MR / TF ชัดเจน** — ทุกข้อความเปิด/ปิด/ปิดบางส่วน มีป้าย `[MR]`/`[TF]` ที่หัวข้อ + ช่อง "Strategy" (🔄 MR · Mean Reversion / 📈 TF · Trend Following) จะได้รู้ทันทีว่าไม้นั้นมาจากกลยุทธ์ไหน. ดูรายละเอียดที่ [`wiki/05-invariants.md`](wiki/05-invariants.md).
>
> **🛡️ v8.0.80 status — Source-code audit (สรุปภาษาคน)**: ตรวจโค้ดทั้งระบบเจอจุดต้องแก้ 13 จุด (ยืนยันกับโค้ดจริงแล้ว). สำคัญสุด 3 จุด: (1) ตอนบอทเลื่อน SL มาเท่าทุน/ล็อกกำไร ถ้า SL ชิดราคาเกินไป โบรกเกอร์เคย "ปฏิเสธเงียบ ๆ" → SL ไม่ขยับจริง = เสี่ยงเต็ม ๆ ทั้งที่คิดว่าปลอดภัย → ตอนนี้กันระยะ + แจ้งเตือน + ลองใหม่. (2) เพิ่มตัวกันขาดทุนเกิน 4% ของ FTMO ที่ทำงาน "เสมอ" (เดิมมีช่องที่บอทหยุดเทรดแต่ไม้เก่ายังเปิดค้างขาดทุนทะลุ 4% โดยไม่ปิด). (3) ตอนฝึกบอทขาดด่านกรองบางตัวที่ตอนเทรดจริงมี → ฝึกบนสถานการณ์ที่จริง ๆ ไม่ได้เทรด → กำลัง **ฝึกใหม่** ให้ตรงกัน. ดูรายละเอียดเทคนิคที่ [`wiki/05-invariants.md`](wiki/05-invariants.md). Backups: `*.pre_v8080`.
>
> **✅ v8.0.55 status — Live filters KEPT, RL reverted**:
> - **เก็บไว้** (live-only, ไม่ต้อง retrain): Entry Confirmation + Spread Spike + Cluster Cooldown
>   - Entry Confirm — เช็คก่อนยิง: slip / แท่ง M1 ทิศตรงไหม / BB %B ยังอยู่เขต
>   - Spread Spike — เทียบ spread กับค่าเฉลี่ย 30 ค่า, เกิน 2x = ข้าม (auto-calibrate ต่อ broker)
>   - Cluster Cooldown — 5 นาทีทั่วไป / 10 นาที same theme USD/JPY/METAL
> - **คืนกลับ**: RL/GBM/Pool (retrain Pass 68.1% ตก -2.6pp, holdout overfit → ไม่คุ้ม)
> - **ผล**: ได้ Pass 70.7% ของเดิม + filter ใหม่กัน disaster ฟรี ✅
>
> **🎉 v8.0.8 status — Production ready**: pipeline autonomous + self-correct ทำงานสำเร็จในรอบเดียว. ตัวเลขสุดท้าย: Pass Rate 59.30%, Profitable 89.10%, Total DD max 5.80%, Daily DD max 3.00%, Breach 0%, Profit avg +7.23%. Best model อยู่ที่ [`models/mr/best/`](ftmo_trading_bot/models/mr/best/). รัน `python main.py` ได้เลย — ระบบโหลด MR model อัตโนมัติ.
>
> **v8.0.6 cleanup**: SMC code ลบทั้งหมด (8 ไฟล์ ~214 KB), Excel schema ปรับ (Trades 66→58 cols, Signals 23→20 cols, name-based `_COL` lookup, auto-archive legacy file), env DD guards เข้ม (DAILY 3.0% / TOTAL 5.8%).
>
> **v8.0.7-8 fixes**: Windows VPS UTF-8 audit fix + MRSignal backward-compat properties (`rr_ratio`/`tp_distance`/`timestamp`).

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

**Bot ทำหน้าที่ (v8.0+):**

- ใช้ **Mean Reversion + Trend Filter** หา setup (BB %B extreme + RSI confirm + ADX H1 ≤ 30 + reversal-wick)
- ใช้ **ML (GBM + Isotonic Calibration)** กรอง signal ที่มีโอกาสชนะสูง (ml threshold 0.30)
- ใช้ **PPO RL Agent** + Auxiliary Task ตัดสินใจ TAKE/SKIP มองคุณภาพ signal + สถานะบัญชี (DD, progress, trades today)
- ทำงาน **24/5** อัตโนมัติ + buffer ป้องกัน FTMO breach (env DD guards: Daily 3.0% / Total 5.8%, ห่างกฎ FTMO 5%/10%)
- SMC strategy ลบไปทั้งหมดในการ cleanup v8.0.6 (~214 KB)

---

## 🧠 สถาปัตยกรรม 3 สมอง (v8.0+)

### 1. 🎯 Mean Reversion Strategy (Brain 1) — หา setup

ใน [`ftmo_trading_bot/strategy/`](ftmo_trading_bot/strategy/) เหลือแค่ 2 ไฟล์:

- `mean_reversion_strategy.py` — `MeanReversionStrategy` + `LiveMRScanner` (drop-in replacement สำหรับ SMCStrategy เดิม) + `MRSignal` dataclass + `TradeSignal`/`SignalType` aliases
- `indicators.py` — RSI, MACD, ADX, ATR, Bollinger %B, Stochastic, ATR z-score, volatility regime

**กฎเข้าออเดอร์ (M15):**

1. ATR pips ≥ floor (per-symbol) — กันตลาดเงียบ
2. ADX H1 ≤ 30 — ห้ามเทรดสวน trend แรง
3. BB %B ≤ 0.30 (BUY) หรือ ≥ 0.70 (SELL) — ราคาแตะขอบ band
4. RSI ≤ 40 (BUY) หรือ ≥ 60 (SELL) — ยืนยัน momentum
5. Reversal wick ratio ≥ 0.4 — มีแท่งกลับตัว
6. Confluence score ≥ 30 — รวมทุก factor

**SL / TP:** SL = 1.0×ATR (tight), TP = 1.0×SL (RR 1:1, quick TP)

**Output:** `MRSignal` (symbol, BUY/SELL, entry, sl_price, tp_price, confluence_score, atr_value, mr_setup_score, bb_extreme, bb_band_width_atr, reversal_wick_ratio)

### 2. 🔬 ML Quality Filter (Brain 2) — กลั่นกรอง

ใน [`ftmo_trading_bot/ml/signal_quality.py`](ftmo_trading_bot/ml/signal_quality.py):

- **GBM Classifier** (sklearn `GradientBoostingClassifier`) เรียนทำนาย P(win) ของแต่ละ signal
- **Isotonic Calibration** บน OOF (`GroupKFold cross_val_predict`) → กัน overconfident probabilities
- ใช้ **28 features** (v8.0.6) — 17 SMC-compat + 7 temporal/regime + 4 MR extras (`bb_extreme`, `bb_band_width_atr`, `mr_setup_score`, `reversal_wick_ratio`)
- **Threshold = 0.30** (`bot_config.ftmo.ML_FILTER_THRESHOLD`, v8.0.3) — signal ที่ score < 0.30 ถูก reject ทันที (ก่อนถึง RL agent) ทั้งใน live + train. Reject log เป็น `Result = "ML_FILTERED"` ใน Signals sheet
- **Model file:** `data/mr_signal_quality_model.pkl` (1.2 MB)

**Output:** `ml_score ∈ [0, 1]` (calibrated)

### 3. 🎓 RL Agent — PPO + Auxiliary Task (Brain 3)

ใน [`ftmo_trading_bot/ml/`](ftmo_trading_bot/ml/):

- `aux_aware_policy.py` — PPO policy พร้อม `aux_head: nn.Linear(latent_dim_pi, 1)` ทำนาย `outcome_pnl_ratio`
- `aux_aware_ppo.py` — PPO subclass เพิ่ม MSE aux_loss (weight = 0.5)
- `aux_rollout_buffer.py` — RolloutBuffer พร้อม `aux_targets`
- `signal_filter_env.py` — Gymnasium env (32-dim obs, action = TAKE/SKIP, env guards Daily 3.0% / Total 5.8%)
- `mean_reversion_env.py` — `MeanReversionFilterEnv` (subclass ของ FTMOSignalFilterEnv) reward shaping ใหม่: quick-TP +0.50R, slow-win +0.20R, base-loss -0.10R, duration-fine 0.02/แท่ง (cap -0.30R), prolonged-loss -0.40R, ADX violation -0.30R
- `mean_reversion_backtester.py` — pool generator (subclass ของ StrategyBacktester) — scan 48 ครั้ง/วัน + dedup 4 bar
- `rl_agent.py` — `SelfLearningAgent` — inference wrapper, path-aware (โหลด `models/mr/best/` ก่อน fallback `models/`)
- `chronos_forecaster.py` — Amazon Chronos 2 zero-shot forecaster (optional, ใส่ obs[27,28])

**Obs 35 dims** (v8.0+ — ต้อง sync 3 จุด — ดู [`wiki/05-invariants.md`](wiki/05-invariants.md)):

```text
[0]  confluence_norm       [12] stoch_norm        [24] spread_pct_of_atr
[1]  rr_norm               [13] bb_pctb           [25] has_opposite_recently_closed
[2]  direction             [14] atr_chg           [26] adx_inverse_norm   ⭐ MR
[3]  atr_norm              [15] price_roc         [27] chronos_alignment
[4]  bb_extreme    ⭐ MR    [16] ml_score_norm     [28] chronos_uncertainty_norm
[5]  bias_align            [17] total_dd_n        [29] floating_pnl_norm (zeroed v7.2.1)
[6]  sl_atr                [18] daily_dd_n        [30] open_losing_count_norm (zeroed)
[7]  rsi_norm              [19] progress_n        [31] mins_since_session_norm
[8]  macd_norm             [20] day_progress
[9]  trend_str             [21] trades_today_n
[10] bb_band_width_atr/3   ⭐ MR (was ob_size_atr)
[11] adx_norm              [22] recent_wr_norm
                           [23] consec_norm
```

**MR-specific obs slot reinterpretation:** obs[4]=`bb_extreme`, obs[10]=`bb_band_width_atr/3`, obs[26]=`adx_inverse_norm`

**Reward (MR):** PnL ratio + capital-preservation shaping + DD penalty + activity floor + auxiliary signal

**Verified Pass Rate (v8.0.5):** **59.30%** (5000-eps eval, ทะลุ gate 8% ไป 51 pp), Profitable 89.10%, Breach 0.00%, Total DD max 5.80%, Daily DD max 3.00%, Profit avg +7.23%

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
    ├── strategy/                      # 🎯 Brain 1: Mean Reversion (v8.0+)
    │   ├── mean_reversion_strategy.py # MeanReversionStrategy + LiveMRScanner + MRSignal/TradeSignal aliases
    │   └── indicators.py              # RSI, MACD, ADX, ATR, Bollinger, Stochastic
    │   # SMC files (smc_strategy.py + 5 detectors) ลบใน v8.0.6
    │
    ├── ml/                            # 🔬🎓 Brain 2 + 3 + Forecaster
    │   ├── signal_quality.py          # GBM wrapper (with calibrator) — 28 features (v8.0.6)
    │   ├── rl_agent.py                # PPO inference wrapper (path-aware: models/mr/ first)
    │   ├── signal_filter_env.py       # Gymnasium env base (32 dims, env guards 3.0/5.8)
    │   ├── mean_reversion_env.py      # MR env subclass — quick-TP/loss-duration shaping
    │   ├── mean_reversion_backtester.py # MR pool generator (subclass of StrategyBacktester)
    │   ├── strategy_backtester.py     # Data infra base class (post-v8.0.6 cleanup, ~327 lines)
    │   ├── chronos_forecaster.py      # Amazon Chronos 2 zero-shot forecaster (optional)
    │   ├── aux_aware_policy.py        # PPO + aux head
    │   ├── aux_aware_ppo.py           # PPO subclass + aux loss
    │   └── aux_rollout_buffer.py      # RolloutBuffer + aux_targets
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
    │   ├── build_mr_signal_pool.py    # Step 1: Generate MR pool
    │   ├── train_mr_signal_quality.py # Step 2: Train GBM (28 features, MR-specific)
    │   ├── train_mr_signal_filter.py  # Step 3: Train PPO + Eval (MR env)
    │   ├── auto_train_pipeline.py     # ⭐ Autonomous orchestrator (Build→GBM→RL→Eval→Self-correct)
    │   ├── leakage_audit.py           # Mandatory: no future fields in obs/x
    │   ├── parity_audit.py            # Mandatory: train↔live aligned
    │   ├── pipeline_status.sh         # Quick status check during pipeline run
    │   └── import_forexfactory_csv.py # News calendar manual import
    │
    ├── models/
    │   └── mr/                        # 🧠 v8.0+ MR model artifacts
    │       ├── best/
    │       │   ├── ppo_mr_filter.zip      # ⭐ Production RL agent (~1 MB)
    │       │   ├── vec_normalize_mr.pkl   # ⚠️ CRITICAL: obs scale stats
    │       │   └── best_meta.json         # Best snapshot metadata
    │       ├── ppo_mr_filter.zip          # Top-level mirror
    │       └── vec_normalize_mr.pkl       # Top-level mirror
    │   # checkpoints_mr/ + Phase 1 intermediates + .bak_* files = gitignored
    │
    ├── data/
    │   ├── ohlcv/                     # 30 CSV files (10 symbols × 3 TF) — gitignored
    │   ├── mr_signal_pool_3000.pkl    # MR pool (~309 MB, gitignored — rebuild ได้)
    │   └── mr_signal_quality_model.pkl # GBM (with calibrator inside, ~1.2 MB)
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

### Step 6 (v7): หมายเหตุเรื่อง Chronos 2

ตั้งแต่ v7.0 (2026-05-01) บอทใช้ **Amazon Chronos 2** (zero-shot time-series foundation model) เพื่อทำนายราคา M15 ใน 8 แท่งข้างหน้า — เป็น obs feature ตัวที่ 27-28 ของ RL agent.

- `chronos-forecasting==1.5.2` + `transformers==4.46.3` + `accelerate==1.2.1` ถูก pin ใน [`requirements.txt`](ftmo_trading_bot/requirements.txt) แล้ว — ติดตั้งอัตโนมัติใน Step 4
- **ครั้งแรกที่รัน** บอทจะ download โมเดล `amazon/chronos-bolt-small` ขนาด ~200 MB จาก Hugging Face Hub มาเก็บที่ `~/.cache/huggingface/hub/` (ไม่ต้องตั้งค่า token)
- **Disable Chronos** (ถ้าต้องการ fallback ไป v6.14 behavior): ตั้ง `BOT_DISABLE_CHRONOS=1` หรือแก้ `bot_config.ml.CHRONOS_ENABLED = False` ใน `config/settings.py` → obs[27,28] = 0.0 (neutral)
- ⛔ ⚠️ การเปลี่ยน `CHRONOS_MODEL_NAME` / `CHRONOS_PREDICTION_LENGTH` / `CHRONOS_CONTEXT_LENGTH` หลัง train เสร็จ → obs distribution shift → **ต้อง retrain ใหม่ทั้ง pool + GBM + RL**

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
| `ftmo_trading_bot/models/mr/best/ppo_mr_filter.zip` | **RL agent weights** (~1 MB) — โหลดอัตโนมัติโดย `SelfLearningAgent` |
| `ftmo_trading_bot/models/mr/best/vec_normalize_mr.pkl` | Obs normalization stats — **ขาดไม่ได้** ไม่งั้น obs scale ผิด → agent พัง |
| `ftmo_trading_bot/models/mr/best/best_meta.json` | Metadata ของ best snapshot |
| `ftmo_trading_bot/models/mr/ppo_mr_filter.zip` | Top-level mirror (auto-load fallback) |
| `ftmo_trading_bot/models/mr/vec_normalize_mr.pkl` | Top-level mirror |
| `ftmo_trading_bot/data/mr_signal_quality_model.pkl` | GBM + calibrator (~1.2 MB) |
| `ftmo_trading_bot/config/settings.py` | Risk/symbols config |

> **Pool ไม่ต้องอัพ VPS** — `data/mr_signal_pool_3000.pkl` (309 MB) ใช้แค่ตอน retrain เท่านั้น. Live ดึง OHLCV จาก MT5 real-time. Pool ถูก gitignored แล้ว.
>
> รวมที่ live VPS ใช้: **~2.2 MB** ของ artifacts

### รัน live bot บน VPS

```cmd
git pull
.venv\Scripts\python ftmo_trading_bot\scripts\leakage_audit.py
.venv\Scripts\python ftmo_trading_bot\scripts\parity_audit.py
.venv\Scripts\python ftmo_trading_bot\main.py
```

ครั้งแรกที่รัน v8.0.6 บน VPS — ถ้ามี `ftmo_trades.xlsx` legacy schema (66/23 cols) จะถูก auto-archive เป็น `ftmo_trades.bak_pre_v8_<ts>.xlsx` + สร้างไฟล์ใหม่ 58/20 cols ให้อัตโนมัติ

จะรัน loop ทุก 5 วินาที (เร็วขึ้นเป็น **1 วินาที** อัตโนมัติเมื่อมีออเดอร์เปิดอยู่ — เพื่อเลื่อน BE/SL/TP ให้แม่นยำ, v8.0.70) + scan signals ใหม่ทุก 1 นาที (คงที่ wall-clock) + log ทุก trade ลง Excel

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
- แนะนำ ≥ 6 ปีย้อนหลัง สำหรับ pool ขนาด 10000 episodes

---

## 🏋️ Training Pipeline (v8.0+ MR pipeline)

**แนะนำ — Autonomous loop** (Build → GBM → RL → Eval → Self-correct):

```bash
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 10 --max_hours 60 \
    --pool_size 3000 --timesteps_p1 5000000 --timesteps_p2 2000000 \
    --target_pass_rate 0.08 --target_dd_max 0.06 \
    --target_daily_dd_max 0.035 --target_profitable 0.55
```

orchestrator จะรัน 3 step ด้านล่างให้เอง + ปรับ hyperparams เองเมื่อ gates ไม่ผ่าน

**Manual — 3 ขั้นตอน** (ถ้าอยาก control เอง):

### Step 1 — Build MR Signal Pool

```bash
.venv/bin/python ftmo_trading_bot/scripts/build_mr_signal_pool.py \
    --pool_size 3000 --workers 8 --max_days 45
```

**ทำอะไร:** simulate `MeanReversionStrategy` บน historical data → save 3000 episodes (45 วัน/ep, scan 48 ครั้ง/วัน, dedup 4 bars) ลง `data/mr_signal_pool_3000.pkl` (~309 MB, gitignored)

**เวลา:** ~11 min บน CPU 8 cores

### Step 2 — Train MR ML Quality Model (GBM + Calibrator)

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_quality.py
```

**ทำอะไร:**
1. Load `mr_signal_pool_3000.pkl`
2. Train GBM ด้วย `GroupKFold cross_val_predict` (group = episode_id) → OOF predictions
3. Fit `IsotonicRegression` calibrator บน OOF
4. Save `data/mr_signal_quality_model.pkl` (GBM + calibrator + drift baseline)
5. Re-score pool → update `ml_score` ของทุก signal

**เวลา:** ~2 min

**Features (28):** core (confluence, atr, rsi, adx, macd, stoch, bb, ...) + temporal/regime (hour, dow, atr_zscore, vol_regime) + MR extras (`mr_setup_score`, `bb_extreme`, `bb_band_width_atr`, `reversal_wick_ratio`)

### Step 3 — Train MR PPO Agent (with Auxiliary Task)

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py --fresh \
    --pool_size 3000 \
    --ml_threshold 0.30 \
    --risk_per_trade 0.007 \
    --n_envs 8 \
    --outcome_noise 0.05 \
    --timesteps_p1 5000000 \
    --timesteps_p2 2000000
```

**Curriculum 2-phase:**

- **P1 (5M steps):** Alpha — เรียนเลือก signal คุณภาพ, no DD penalty
- **P2 (2M steps):** Risk — เพิ่ม DD penalty + activity floor + early-stop on value loss

**MR-specific reward shaping:** quick-TP +0.50R (≤5 bars), slow-win +0.20R, base-loss -0.10R, duration-fine 0.02R/bar (cap -0.30R), prolonged-loss -0.40R, ADX violation -0.30R

**Auxiliary Task:** policy ทำนาย `outcome_pnl_ratio` (`aux_head`) + MSE loss weight=0.5 — บังคับ representation ให้เรียนรู้คุณภาพ signal

**เวลา:** ~10-12 min บน CPU @ n_envs=8 (รวมทั้ง pipeline ~15-30 min/iteration)

**Output:**
- `models/mr/ppo_mr_filter.zip` — final RL agent
- `models/mr/vec_normalize_mr.pkl` — obs/reward normalization stats
- `models/mr/best/` — best snapshot (auto-copied โดย `auto_train_pipeline.py` เมื่อมี Pass Rate สูงกว่าเก่า)
- `models/mr/checkpoints_mr/` — backup ทุก 50K steps (gitignored)

---

## 🧪 Evaluation

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py \
    --eval_only \
    --pool_size 3000 \
    --ml_threshold 0.30 \
    --risk_per_trade 0.007
```

---

## 🚀 v8.0 Mean Reversion Pipeline (Active)

กลยุทธ์ปัจจุบัน — เน้น win rate สูง + DD ต่ำ. SMC ลบทั้งหมดแล้วใน v8.0.6

### หลักการเทรด (v8.0.5 production)

- **เข้าเมื่อราคาผิดปกติ:** Bollinger %B แตะขอบล่าง (≤ 0.30) + RSI < 40 = ซื้อ / Bollinger %B แตะขอบบน (≥ 0.70) + RSI > 60 = ขาย
- **ต้องมี wick rejection:** ไส้เทียนกลับตัว ≥ 0.4 เท่าของตัวเทียน
- **กรอง trend แรง:** ถ้า ADX H1 > 30 = ตลาด trend แรง ห้าม MR
- **SL แน่น TP เร็ว:** SL = 1.0×ATR, TP = 1.0×ATR (RR 1:1)

### Reward Shaping (สอนบอท preserve capital)

| สถานการณ์ | Reward |
|-----------|--------|
| ✅ ชนะเร็ว ≤ 5 แท่ง | +0.50 R bonus |
| 🟢 ชนะช้า | +0.20 R bonus |
| ❌ แพ้ทุกตัว | -0.10 R baseline |
| ⏳ ลอย-ขาดทุนทุกแท่ง | -0.02 R/แท่ง (cap -0.30) |
| 🔴 ลอยแดง ≥ 12 แท่งแล้วโดน SL | -0.40 R extra |
| ⚠️ เปิดสวน trend (ADX > 25) | -0.30 R |

### Auto Training Pipeline (เปิดทิ้งไว้แล้วเดินจากโต๊ะได้)

ผมเขียน script `auto_train_pipeline.py` ให้รันเองอัตโนมัติ — Build pool → Train GBM → Train RL → Eval → ถ้าไม่ผ่าน gate ก็ปรับ hyperparams เองแล้วลูปต่อ จนกว่าจะผ่านหรือชนเพดาน iterations/ชั่วโมง.

**คำสั่งรัน (รันครั้งเดียวแล้วทิ้งได้):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 6 \
    --max_hours 60 \
    --pool_size 5000 \
    --timesteps_p1 5000000 \
    --timesteps_p2 2000000 \
    --target_pass_rate 0.08 \
    --target_dd_max 0.06 \
    --target_profitable 0.55
```

**Eval gates (ค่า default):**

| เกณฑ์ | ค่าเริ่มต้น | หมายเหตุ |
|-------|-----------|---------|
| Pass Rate | ≥ 8 % | conservative buffer (verified MR pass rate = 59.30%) |
| Total DD max | ≤ 6 % | training eval gate (live hard stop = 10% ตั้งแต่ v8.1.1) |
| Daily DD max | ≤ 3.5 % | ห่างจาก FTMO 4% limit |
| Profitable Rate | ≥ 55 % | กว่าครึ่งของ episodes ปิดบวก |
| Breach Rate | ≤ 5 % | breach น้อย |

**Auto-tune logic** (อ่านได้ละเอียดใน `tune_hyperparams()` ของ script):

- 🔴 Breach > 5% → ลด `risk_per_trade` ครึ่งนึง + เพิ่ม loss penalty (กัน DD)
- 🟡 DD เกิน → เพิ่ม `prolonged_loss_penalty` + `duration_fine_coef` + ลด risk
- 🟠 Pass Rate ต่ำ → เพิ่ม `quick_tp_bonus` + ลด `ml_threshold` (push agent ให้ TAKE มากขึ้น)
- 🟢 Profitable ต่ำ → เพิ่ม `slow_win_bonus` + ลด duration fine (ให้รางวัล winner ที่ใช้เวลา)

### Logs ที่จะเห็นเมื่อกลับมา

| ไฟล์ | ข้างใน |
|------|--------|
| `logs/auto_train_pipeline.log` | log แบบอ่านเข้าใจง่าย — ทุก iteration เขียนเหตุผลที่ tune + metrics |
| `logs/auto_train_pipeline.jsonl` | JSON line ต่อ event (ใช้ analyze ภายหลังได้) |
| `logs/auto_train_pipeline_state.json` | snapshot สถานะ + best metrics |
| `models/mr/best/` | best model + meta (iteration ที่ดีที่สุด) |
| `logs/tb_mr_filter/` | TensorBoard ดู training curve |

### Live deploy (ทำงานออโต้แล้วใน v8.0+)

`main.py` รันแล้วโหลด MR model + LiveMRScanner อัตโนมัติ — ไม่ต้องตั้งค่าเพิ่มเติม:

```cmd
.venv\Scripts\python ftmo_trading_bot\main.py
```

`SelfLearningAgent` หา model ตามลำดับ:
1. `models/mr/best/ppo_mr_filter.zip` (ผ่าน auto pipeline best snapshot)
2. `models/mr/ppo_mr_filter.zip` (top-level mirror)
3. `models/ppo_signal_filter.zip` (legacy fallback)

เลือกอันแรกที่เจอ. ถ้าโหลดสำเร็จ → console prints `✅ RL agent loaded from <path>`

---

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
| `DEFAULT_RISK_PER_TRADE_PCT` | 0.007 | 0.7% (v8.0.43 — paired with Option X trail) |
| `ML_FILTER_THRESHOLD` (v8.0.3) | 0.30 | live ↔ training sync — signals ที่ ML score < 0.30 ถูก reject ก่อน agent |
| `MAX_DRAWDOWN_HARD_STOP_PCT` | 0.10 | 10% (v8.1.1 — = กฎ FTMO เต็ม ไม่มี buffer; เตือนล่วงหน้าที่ 8%) |
| `DAILY_LOSS_HARD_STOP_PCT` | 0.04 | 4% (buffer 1% จาก FTMO 5%) |
| `PROFIT_TARGET_PCT` | 0.10 | 10% target |
| `CONSISTENCY_RULE_THRESHOLD` | 1.0 | 2-step Standard ไม่มีกฎนี้ |
| `MAX_OPEN_POSITIONS` | 3 | สูงสุด 3 positions พร้อมกัน |
| `bot_config.mr.bb_oversold/overbought` | 0.30 / 0.70 | MR entry threshold |
| `bot_config.mr.rsi_oversold/overbought` | 40 / 60 | MR entry threshold |
| `bot_config.mr.adx_trend_block` | 30 | ADX H1 > 30 → ห้าม MR |
| `bot_config.mr.sl_atr_mult` | 1.0 | tight SL |
| `bot_config.mr.rr_ratio` | 1.0 | RR 1:1 quick TP |
| **env `DAILY_DD_GUARD`** (v8.0.4) | 0.030 | clamp eval ที่ 3.0% (ใต้ gate 3.5%) |
| **env `TOTAL_DD_GUARD`** (v8.0.5) | 0.058 | clamp eval ที่ 5.8% (ใต้ gate 6.0%) |

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

### 4 Sheets (v8.0.6 schema)

#### 1. **Trades** — 58 columns

| Group | Columns |
|-------|---------|
| Core (1–19) | Ticket, Symbol, Type, Entry/SL/TP, Lot, Risk%, RR, Confluence, ATR, Times, P/L, Reasons |
| Time + spread (20–24) | Session, DoW, Hour, Spread@Entry, Slippage |
| Risk + outcome (25–31) | Volatility Regime, ConsecLoss, DD%, MAE, MFE, TimeInTrade, ExitPath |
| Decision (32–36) | ML Score (cal/raw), Agent Action, Decision, Threshold |
| Trade mgmt (37–40) | BE Moved, Partial Closed, Trailing, Final SL |
| Live exec (41–45) | Bid/Ask @Entry/Exit, Spread (pips) |
| Market (46–47) | ADX H1, ADX H4 |
| Account (48–50) | Balance@Entry/Close, Equity Peak |
| Overtrading (51–54) | Trades Today, Trades 1h, Sec Since Last Open (any/same-sym) |
| Skipped flag (55) | Partial Skipped |
| **Retrain** (56) | **Obs JSON** ← full 32-dim obs vector → reconstruct state สำหรับ retrain |
| Chronos (57–58) | Chronos Align, Chronos Unc |

> v8.0.6 ลบ 8 SMC cols (HTF Bias, HTF/MTF/OB/FVG/Sweep pts, MTF Bias, D1 Bias) — ใช้ name-based `_COL` lookup ป้องกัน off-by-one

#### 2. **Signals** — 20 columns

ทุก scan event (`AGENT_TAKE`, `AGENT_SKIP`, `AGENT_TAKE_FAIL`, `ML_FILTERED`) — ใช้วิเคราะห์ signal frequency + reject distribution. **สร้าง lazy** ตอน scan event แรก — ถ้าบอท run แล้วยังไม่มี signal เลย sheet นี้จะยังไม่ถูกสร้าง

> v8.0.6 ลบ 3 SMC cols (HTF Bias, MTF Bias, D1 Bias)

#### 3. **Daily** — สรุปรายวัน

Date, Trades, Wins, Losses, Win Rate, P/L, DD, Balance EOD

#### 4. **Stats** — สถิติรวม

Win Rate, Sharpe, Profit Factor, etc.

### ⚠️ Schema migration (v8.0.6 auto-archive)

ถ้าไฟล์ `ftmo_trades.xlsx` เดิมเป็น schema เก่า (66/23 cols) → bot จะ **auto-archive เป็น `ftmo_trades.bak_pre_v8_<timestamp>.xlsx`** + สร้างไฟล์ใหม่ schema 58/20 cols ให้อัตโนมัติ. ไม่ต้อง rename เอง — ตรวจ console message ตอน start bot ครั้งแรกหลัง update

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

### News Filter พฤติกรรม (v7.1.10)

บอทกัน 2 ระดับ:

1. **Block สัญญาณใหม่** — ก่อนข่าว 30 นาที + หลังข่าว 15 นาที (ปรับใน `config.no_trade_before_news_minutes` / `no_trade_after_news_minutes`) → MR strategy ไม่ scan signal ของ symbol ที่กระทบ
2. **ปิด position ที่เปิดอยู่** (v7.1.10) — ก่อนข่าว 30 นาที (sync กับข้อ 1) → `TradeManager.check_news_close()` ปิดทุก position ของ symbol ที่กระทบ ด้วย reason `"Pre-news close"`

**Currency mapping** — USD news (NFP/CPI/FOMC) กระทบ: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, **XAUUSD** (ทอง spike แรงตาม USD strength โดยตรง)

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
MAX_DRAWDOWN_HARD_STOP_PCT = 0.10    # 10% (v8.1.1 = FTMO rule เต็ม, no buffer; warn @ 8%)
DAILY_LOSS_HARD_STOP_PCT = 0.04      # 4% (FTMO rule = 5%)
PROFIT_TARGET_PCT = 0.10             # 10%
```

### Anti-Overtrading

```python
MAX_CORRELATED_POSITIONS = 99           # per-group guard ปิดไว้ (ใช้ 2 ตัวล่างแทน)
MAX_USD_THEME_POSITIONS = 2             # จำกัดฝั่ง USD เดียวกันรวมทุกคู่ ≤ 2 (v7.1)
MAX_SAME_CURRENCY_LEG_POSITIONS = 1     # 2026-06-03: กลับเป็น 1 (กันเบิ้ลทิศเดียว) หลัง challenge สอบตก —
                                        #          cap=2 เปิดทาง GBPJPY BUY 2 ไม้ + USDJPY BUY แพ้ยกก้อน. ตั้ง 2 = ผ่อน
ML_FILTER_THRESHOLD = 0.30              # signal ต้อง ML score ≥ 0.30 ก่อน agent (v8.0.3)
COOLDOWN_AFTER_LOSS_MIN = 60            # 60 นาทีก่อนเปิดคู่เดิมหลัง SL
CONSECUTIVE_LOSS_PAUSE_COUNT = 3        # หยุด 60 นาทีหลังแพ้ติด 3 ครั้ง (v6.13)
CONSECUTIVE_LOSS_HALT_COUNT = 4         # halt ทั้งวันหลังแพ้ติด 4 ครั้ง (v6.13)
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

- ถ้า **obs space (32 dims, v8.0+) ไม่เปลี่ยน** → ใช้ `models/mr/best/ppo_mr_filter.zip` ต่อได้ (พร้อม `vec_normalize_mr.pkl` ที่ตรงกัน)
- ถ้า **เปลี่ยน obs / strategy params / env guards** → ต้อง retrain ทั้งหมด (Brain 1 MR + Brain 2 GBM + Brain 3 RL) เพื่อให้ wiring ตรงกัน

ดู `wiki/05-invariants.md § Hard Invariants § Observation Space Sync` + § Invariant 0a (TRAIN ↔ LIVE PARITY)

### Q: Console เงียบไป — ทำไมไม่เห็น scan ของ 10 symbols?

A: v6.9 เปลี่ยนเป็น **quiet mode** — per-signal SKIP/scan logs ไปอยู่ใน `Signals` sheet ของ Excel แทน. ดูได้ใน `logs/ftmo_trades.xlsx` sheet "Signals" — มี TAKE/SKIP/ML_FILTERED/AGENT_TAKE_FAIL records ครบ. **Note:** Signals sheet ถูกสร้างแบบ lazy ตอน scan event แรก — ถ้า MR ยังไม่ปล่อย signal เลย sheet นี้จะยังไม่มี (ปกติของ MR strategy ที่ selective)

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
