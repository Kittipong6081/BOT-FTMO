# CONTEXT — FTMO Trading Bot (LLM Wiki Hub)
> Last Updated: 2026-05-28 (v8.0.75 — Dynamic Slope cap, live-only) | Scope: Hub / Index — read this first, then drill into wiki/*

## TL;DR (LLM read first — 30-second scan)

- **v8.0.75 (2026-05-28) — 🎚️ Dynamic Slope cap (ATR-adaptive, live-only, no retrain)**: v8.0.74 ใช้ `KC_SLOPE_THRESHOLD = 0.15` แบบ fixed cap → volatile regime (ข่าว NFP/CPI) SKIP เกินไป, quiet regime (Asian) slope เล็กๆ ก็ยังหลุดผ่าน. Fix: ใช้ `MRSignal.atr_zscore_30bars` (มี helper อยู่แล้วจาก v7.1) ปรับเพดานเป็น `base × clip(1 + atr_z × 0.15, 0.7, 1.6)` → cap หดเป็น 0.105 ตอน atr_z=-2, ขยายเป็น 0.195 ตอน atr_z=+2 (cap สูงสุด 0.240 ที่ atr_z≥+4). Affected: `FTMOConfig` (4 keys NEW), `MRSignal` (1 field NEW: `atr_zscore_30bars`), `TradeExecutor._check_entry_confirmation` (slope block). **Live-only** — `MeanReversionBacktester` ไม่ใช้ filter นี้อยู่แล้ว → ไม่ต้อง retrain. Rollback 1-line: `KC_SLOPE_ATR_ADAPTIVE = False`. ดู [`wiki/05-invariants.md` v8.0.75](wiki/05-invariants.md#-version-log-entry--v8075-2026-05-28--dynamic-slope-cap-atr-adaptive-live-only).
- **v8.0.74 (2026-05-27) — 📏 Keltner Channel / ATR Band anti-whipsaw (obs 32→35, RETRAIN REQUIRED)**: ปัญหา BE-Whipsaw 64% (9/14 ไม้) เพราะบอทเข้ากลางทางในเทรนด์แรง. Fix: **(1)** `TechnicalIndicators.calculate_keltner` (NEW) — EMA(21) ± 2.5×ATR(14) bands + 4 derived features (`kc_distance_norm`, `ema_slope_norm`, `consecutive_outside`, `band_squeeze_ratio`). **(2)** Live Filters 3 ชั้นใน `TradeExecutor._check_entry_confirmation`: KC entry (ราคาต้องอยู่นอก band), EMA slope (ห้ามชันเกิน 0.15), consecutive outside (≥3 แท่ง = เทรนด์). **(3)** Obs 32→35 dims: obs[32]=`kc_distance_norm`, obs[33]=`ema_slope_norm`, obs[34]=`band_squeeze_ratio`. **(4)** `MRSignal` + pool builder + MR env + main obs builder ทั้งหมด sync แล้ว. ⚠️ **ต้อง retrain ทั้ง pipeline** (pool + GBM + RL) ก่อน deploy — OBS_DIM mismatch จะ crash ทันที. Config: `FTMOConfig.KC_ENTRY_FILTER_ENABLED`, `KC_SLOPE_*`, `KC_CONSEC_*` + `MeanReversionConfig.kc_*`. Backup: `ppo_mr_filter.zip.pre_v8074`. ดู [`wiki/05-invariants.md` v8.0.74](wiki/05-invariants.md#-version-log-entry--v8074-2026-05-27--keltner-channel-atr-band-anti-whipsaw).
- **v8.0.73 (2026-05-26) — ✂️ Stage 2 TP extension 1.5R removed (keep SL lock 0.5R only)**: 2-day Micro-RCA (25-26 พ.ค., n=13) พบ **0/13 ไม้ hit TP** (Stage 2 ขยับ TP→1.5R แต่ continuation rate 0% ในตัวอย่างนี้, ตรงกับ v8.0.61 M1 replay = 26.9%). Fix: `TradeManager.TP_STEP_NEW_TP_RR` **1.5 → 1.0** (TP คงเดิม) + drop TP-invariant check ใน `_tp_step()` ให้ SL lock @ 0.5R ยังทำงาน. Stage 3 trail ยังอยู่ในโค้ดเป็น safety net (gap/slippage) แต่ rarely fires เพราะ TP fill ที่ 1.0R ก่อน. คาดผล: +0.07R/2-day (noise level) — real benefit = ความเรียบง่าย (ตัด dead code). ไม่ retrain. ดู [`wiki/05-invariants.md` v8.0.73](wiki/05-invariants.md#-version-log-entry--v8073-2026-05-26--stage-2-tp-extension-removed-keep-sl-lock-only).
- **v8.0.72 (2026-05-26) — 🔇 Silence ❌ JPYUSD log via root-cause silent probe**: User เจอ log "❌ [MT5] ดึงราคา JPYUSD ล้มเหลว" ทุก order JPY-cross (GBPJPY/EURJPY). ไม่ใช่ bug — เป็น fallback logic ปกติของ `PositionSizer._calculate_pip_value` ที่ลอง direct `JPYUSD` ก่อน (broker ไม่มี) แล้ว fall through ไป inverse `USDJPY` (สำเร็จ). แค่ noisy. Fix ที่ต้นเหตุ: เพิ่ม `MT5Connector.symbol_exists(symbol)` (silent probe via `mt5.symbol_info()` → bool, cached) แล้ว position_sizer pre-check ก่อน `get_current_price` → ข้าม direct probe เลย ไม่ปริ้น ❌. Verified: GBPJPY pip value ยัง = $6.689/lot (เดิม) ผ่าน USDJPY fallback ปกติ. ไม่ต้อง retrain. Affected: `core/mt5_connector.py` (new method) + `core/position_sizer.py` (pre-check). ดู [`wiki/05-invariants.md` v8.0.72](wiki/05-invariants.md#-version-log-entry--v8072-2026-05-26--silent-probe-for-nonexistent-conversion-symbols).
- **v8.0.71 (2026-05-26) — 🐛 Remove broken GBPJPY `max_trades_per_day: 3` (latent bug)**: User เจอ block message "GBPJPY: เทรดครบ 3 ครั้งวันนี้แล้ว" ทั้งที่ Excel โชว์ GBPJPY = 0 ไม้ปิดวันนั้น. Root cause: `RiskManager.can_open_trade` อ่าน per-symbol cap (`get_symbol_config(symbol, "max_trades_per_day", ...) → 3`) แต่ compare กับ `self._daily_trades_count` ซึ่งเป็น **GLOBAL counter ทุกคู่รวมกัน** → GBPJPY ถูก block ทันทีหลัง total trades = 3 (แม้ยังไม่เคยเทรด). Comment ในโค้ดยอมรับเอง ("track เป็น total count"). Fix: ลบ `max_trades_per_day: 3` ออกจาก GBPJPY overrides (ตั้ง None implicit). Safety: RL + cooldown 60min + spread/ATR + entry-confirm + cluster cooldown กรองคุณภาพหนักอยู่แล้ว (Signals sheet 2026-05-26: 37 GBPJPY signals → AGENT_SKIP 12 + AGENT_TAKE_FAIL 25 = เปิดจริง 0). ไม่ต้อง retrain. ดู [`wiki/05-invariants.md` v8.0.71](wiki/05-invariants.md#-version-log-entry--v8071-2026-05-26--remove-broken-gbpjpy-per-day-cap).
- **v8.0.70 (2026-05-26) — ⚡ Full-lifecycle 1s adaptive loop + wall-clock signal scan (user request, live-only)**: User เจอปัญหา "order ถึงจุดที่ต้องเลื่อน BE แล้วบอทไม่เลื่อนเพราะไม่ถึงเวลาสแกน". เดิม v8.0.46 ใช้ 1s loop เฉพาะตอน `profit_r ≥ 0.5R` — แต่ BE trigger ตอนนี้ที่ **0.3R** (v8.0.56) → loop ยังเป็น 5s ตอนที่ต้อง modify SL → price อาจ spike ผ่าน 0.3R + revert ใน 5s window = BE หลุด. Fix 2 จุด: **(1)** Adaptive condition: `if open_positions: sleep_interval = 1` (ทุก position open ไม่ว่ากำไรเท่าไหร่). **(2)** Signal scan แยก wall-clock 60s gate (`_last_signal_scan_ts`) แทน `% 12` loops — ถ้าไม่แยก loop 1s × 12 = scan ทุก 12s (เร็วเกิน 5× → ML/RL spam + อาจซ้ำสัญญาณ). ไม่ต้อง retrain (execution-only). Affected: `main.py` (3 จุด: `__init__` `_last_signal_scan_ts`, scan section, adaptive section). ดู [`wiki/05-invariants.md` v8.0.70](wiki/05-invariants.md#-version-log-entry--v8070-2026-05-26--full-lifecycle-1s-adaptive-loop).
- **v8.0.69 (2026-05-26) — 🔧 MAX_OPEN_POSITIONS 2 → 3 (user request, live-only)**: User ขอให้บอทเปิดได้สูงสุด 3 ไม้พร้อมกัน. v8.0.56 เคยลด 3→2 เพราะ audit 111 ไม้ ($-460) แต่หลังจากนั้นมี (1) `SymbolConfig.blocked_symbols` ตัด AUDUSD/USDCAD/USDCHF (-$1,297) (2) v8.0.55 cluster cooldown + entry-confirm + spread-spike gates (3) v8.0.68 OPPOSING_THEME_BLOCK → quality ไม้ที่ผ่านเข้ามาดีขึ้นชัด → ปลอดภัยพอที่จะขยาย concurrent slot กลับเป็น 3. **Risk math**: 3 × 0.70% = 2.1% concurrent risk (vs DAILY_LOSS_CAP 2.5% / FTMO 4%) ยังมี buffer 0.4pp ก่อน soft cap, 1.9pp ก่อน FTMO breach. ไม่ต้อง retrain (live-only config). Single-line change: `FTMOConfig.MAX_OPEN_POSITIONS = 3`.
- **v8.0.68 (2026-05-26) — 🚫 Opposing-Theme Block (live-only filter, no retrain)**: User feedback หลังเห็น loss วันนี้ — "อย่าเปิดออเดอร์สวนทิศ เช่น GBPUSD SELL + EURUSD BUY (ทั้งคู่เดิมพัน USD ตรงข้ามกัน)". ก่อนหน้านี้ `CLUSTER_COOLDOWN_SAME_THEME_SEC` (v8.0.55) block ธีมเดียวกัน 10 นาที แต่ **ไม่ block ธีมตรงข้าม** หลัง cooldown หมด → 1 ไม้กำไร + 1 ไม้ขาดทุน = Net EV ≈ 0 − spread × 2 = ขาดทุนแน่นอน. Fix: `RiskManager._check_opposing_theme()` วน `connector.get_open_positions()` แล้ว block ถ้าเจอ USD_LONG ↔ USD_SHORT / JPY_LONG ↔ JPY_SHORT / METAL_LONG ↔ METAL_SHORT. Wired ใน `can_open_trade` ที่ check #2.5. Config: `FTMOConfig.OPPOSING_THEME_BLOCK_ENABLED = True`. Unit test ยืนยัน scenario GBPUSD SELL + EURUSD BUY → BLOCKED ✓. ดู [`wiki/05-invariants.md` v8.0.68](wiki/05-invariants.md#-version-log-entry--v8068-2026-05-26--opposing-theme-block).
- **v8.0.67 (2026-05-25) — 🔴 XAU-only morning block (data-driven by CSV audit 8,058 entries)**: Multi-TF audit (M15+M1 OHLCV, Feb-May 2026) ของ 7 symbols พบ morning EET 00-04 (= ICT 04-08) มี EV +$7.64/trade (slightly BETTER than others +$6.15). **6/7 symbols morning positive** (+$6 ถึง +$12). แต่ **XAUUSD เท่านั้น -$6.06/trade** (continuation rate ต่ำ 44.8%, Asian quiet time). Decision: 🟢 KEEP general unblocked, 🔴 BLOCK XAU only morning. Config: `XAU_WEEKDAY_DELAY_ENABLED = True`. Caveat: Excel small-sample (23 trades) ก่อนหน้านี้ผิด — dominated by May 12 outlier (4 trades = -$432). ดู [`wiki/05-invariants.md` v8.0.67](wiki/05-invariants.md#-version-log-entry--v8067-2026-05-25--xau-only-morning-block-data-driven).
- **v8.0.66 (2026-05-25) — 🐛 Fix news CSV parser: Holiday impact ถูก hardcode เป็น high**: `config/news_csv_parser.py:100` HARDCODE `"impact": "high"` แม้ filter accept "holiday" → JSON บันทึก Holiday events เป็น "high" ผิดทุกครั้ง. Fix: ใช้ `impact` variable ที่ parse จาก CSV (preserve original). Behavior block path คงเดิม (ทั้ง high+holiday block). Test verified: 2 high + 2 holiday → JSON ออกถูก. ดู [`wiki/05-invariants.md` v8.0.66](wiki/05-invariants.md#-version-log-entry--v8066-2026-05-25--fix-news-csv-parser-preserve-holiday-impact).
- **v8.0.65 (2026-05-25) — 🇹🇭 Thai plain-language explanation on Discord**: เพิ่ม field `🇹🇭 อธิบายภาษาไทย` ใน Discord trade_open notification — แปลศัพท์เทคนิค (RSI, ADX, BB %B, ML) เป็นภาษาคนเข้าใจง่าย พร้อม analogy (ยางยืด, เครื่องวัดไข้). Files: `core/thai_explainer.py` (NEW), `core/notifier.py` (wire +field), `config/settings.py` (toggle `thai_explain_enabled`). 🟢 ZERO impact trading (output-layer only, fail-silent try/except, Discord-only). Toggle via env `BOT_THAI_EXPLAIN=0`. ดู [`wiki/05-invariants.md` v8.0.65](wiki/05-invariants.md#-version-log-entry--v8065-2026-05-25--thai-plain-language-explanation-on-discord).
- **v8.0.64 (2026-05-25) — 🚨 HOTFIX: TZ-aware vs naive datetime compare**: `⚠️ [Bot] Trade Manager error: can't compare offset-naive and offset-aware datetimes` spam ทุก scan loop. Root: v8.0.63 migrate `datetime.utcnow()` (naive) → `datetime.now(timezone.utc)` (aware) ทั้งระบบ แต่ `news_events.py` ออกแบบให้ใช้ naive ทั่วไฟล์ (strip tzinfo ทุกที่) → compare aware vs naive = TypeError. Fix: line 153 + 174 ใส่ `.replace(tzinfo=None)` หลัง `datetime.now(timezone.utc)` → preserve naive convention. ดู [`wiki/05-invariants.md` v8.0.64](wiki/05-invariants.md#-version-log-entry--v8064-2026-05-25--hotfix-tz-aware-vs-naive-comparison).
- **v8.0.63 (2026-05-23) — 🔇 Drift throttle + datetime.utcnow deprecation fix**: GBM Drift warning print ทุก 1 ชม. (เดิม 24 ครั้ง/วัน) → throttle เฉพาะ count เปลี่ยน ≥3 หรือ 6h elapsed (~4 ครั้ง/วัน). Drift window 100→200 ลด false-positive KS test. แก้ `datetime.utcnow()` deprecated → `datetime.now(timezone.utc)` ที่ 12 จุดใน 4 ไฟล์ (main + notifier + news_events + signal_quality + trade_executor). Python 3.12+/3.13 clean. File log `logs/gbm_drift.log` ยัง audit trail ครบ. ไม่ต้อง retrain. ดู [`wiki/05-invariants.md` v8.0.63](wiki/05-invariants.md#-version-log-entry--v8063-2026-05-23--drift-throttle--datetimeutcnow-fix).
- **v8.0.62 (2026-05-23) — 🔇 Friday Force Close throttle (anti-spam)**: `TradeManager.check_session_close` print `🚨 FRIDAY FORCE CLOSE` ทุก 5s scan หลัง 20:45 EET วันศุกร์ (เดิม 2000+ ครั้ง/3 ชม.) → spam log. Fix: เพิ่ม `_friday_close_announced_date` flag (mirror v8.0.16/v8.0.49 pattern) → announce 1 ครั้ง/วัน. Close attempt ยัง fire ทุก tick (orphan recovery). ไม่ต้อง retrain — execution-only cosmetic fix. ดู [`wiki/05-invariants.md` v8.0.62](wiki/05-invariants.md#-version-log-entry--v8062-2026-05-23--friday-force-close-throttle-anti-spam).
- **v8.0.61 (2026-05-23) — 🔴 REVERT RR 1.5 → 1.0 (data-driven via M1 OHLCV replay)**: เขียน [scripts_local/m1_replay_validity_check.py](scripts_local/m1_replay_validity_check.py) walk M1 high/low bars จริงของ 106/113 ไม้ → **Continuation Rate (1.0R→1.5R) = 26.9% << 40% gate**. รายคู่: USDJPY 45.5% (ผ่าน), อื่นๆ 0-50% (ส่วนใหญ่ MR เด้งกลับทันที). หลังหัก -$10 spread: Phase 1 Only = **+$109** (PF 1.08) vs Phase 1+2 = **-$627** (PF 0.57). พบ Polled MFE underestimate (TRUE max 6.72R vs polled cap 1.50R, 44% trades poll พลาด peak). Revert: `MRConfig.rr_ratio` 1.5→1.0, restore pool/GBM/RL จาก `.pre_v8060` (= v8.0.52 baseline Pass 70.7%). **Phase 1 fixes ยังคงไว้ทั้งหมด** (block + BE 0.3R + Partial 0.8R/33% + Risk 0.7% + Daily Cap 2.5%). ดู [`wiki/05-invariants.md` v8.0.61](wiki/05-invariants.md#-version-log-entry--v8061-2026-05-23--revert-rr-15--10-data-driven).
- **v8.0.60 (2026-05-22, SUPERSEDED by v8.0.61) — Phase 2 RR 1.5 retrain**: build pool RR 1.5 (4900 ep, WR 49.31%) + train GBM (AUC 0.6140 ผ่าน gate 0.6135) + train RL (Pass 64.5%, gate ≥60% ผ่าน). แต่ M1 OHLCV replay พบว่า MR strategy ไม่ continuation ถึง 1.5R → revert.
- **v8.0.56 (2026-05-22) — 🎯 Phase 1 EV fix (live-only, no retrain)**: Audit 111 trades — WR 59.46% แต่ Net **-$460**, EV **-$4.15/trade**. Realized RR 0.589 (vs design 1.0) — "ตัดกำไรเร็ว ปล่อยขาดทุนเต็ม". Fixes: **(1)** `SymbolConfig.blocked_symbols` (NEW) — block AUDUSD/USDCAD/USDCHF (-$1,297 รวม, WR 27-46%). **(2)** `TradeManager` BE 0.5→0.3 + Partial 0.5→0.8 + PartialPct 0.5→0.33 — ปล่อยไม้ชนะวิ่งก่อนหั่น, จับไม้แพ้ revert ทัน. **(3)** Risk reduction: `DEFAULT_RISK_PER_TRADE_PCT` 0.85%→0.70%, `MAX_OPEN_POSITIONS` 3→2, `DAILY_LOSS_CAP_ENABLED` True @2.5% (1.5pp buffer ก่อน FTMO 4%). RL pool/reward เป็น R-multiple → ไม่ต้อง retrain (Pass Rate คาดลดเพียง 5-8pp). Expected EV: -$4.15 → +$5-8/trade. Backup: `logs/bot_state.json.pre_v8056`. ดู [`wiki/05-invariants.md` v8.0.56](wiki/05-invariants.md#-version-log-entry--v8056-2026-05-22--phase-1-ev-fix-live-only-no-retrain).
- **v8.0.55 (2026-05-22) — ✅ 3 live filters KEPT, ⏪ RL/Pool/GBM reverted to v8.0.52**: After 2026-05-21 16-trade analysis ($-80 net, 6 trades MFE ≤ 4 = -$472), built 3 pre-execution gates + retrain attempt. **Retrain result: Train Pass 68.1% (-2.6pp vs v8.0.52 70.7%), Holdout 48.5% (mild overfit Δ +5.3pp) → REVERT decision**. Training proxy (M15 first-bar slip ≤ 0.6R) didn't transfer well to live (live uses M1 backward + slip + BB). However, **3 live filters work without RL retraining** (RL-blind gates, same pattern as v7.1.10 news / v8.0.21 pre-news / v8.0.26 bulk guard) — kept in code, reduce TAKE rate ~10-20% in exchange for gap/cluster disaster protection. **(1)** `TradeExecutor._check_entry_confirmation` — slip 0.30R + M1 last bar direction + BB %B still extreme. **(2)** `TradeExecutor._check_spread_spike` — rolling median 30 bars, > 2x → SKIP (broker-agnostic). **(3)** `RiskManager` cluster cooldown — 300s global / 600s same-theme USD/JPY/METAL. State schema v8 (`_last_open_theme` persisted). RL/GBM/Pool restored from `models/mr/best_v8052_pass707/` + `data/*.pre_v8055`. Backups still on disk. ดู [`wiki/05-invariants.md` v8.0.55](wiki/05-invariants.md#-version-log-entry--v8055-2026-05-22--3-live-filters-kept--rlpool-reverted).
- **v8.0.54 (2026-05-21) — Revert v8.0.53 (Pass 70.4% ≈ 70.7%, no improvement)**: Stage 2/3 trail already extends RR, raising base RR has no effect. Reverted MR config + restored from `.v8052_backup`.
- **v8.0.53 (2026-05-21) — RR 1.0→1.2 + Stage 2 TP 1.8R (REVERTED in v8.0.54)**: Test higher RR for bigger wins. `MRConfig.rr_ratio` 1.0→1.2, `MeanReversionStrategy.RR_RATIO` 1.0→1.2, `TradeManager.TP_STEP_NEW_TP_RR` 1.5→1.8. Hypothesis: WR 70% × $85 win - 30% × $90 loss = +$32.5/trade EV (vs +$0.4 current). Risk: WR may drop 5-8pp. Backup .v8052_backup saved for rollback. ดู [`wiki/05-invariants.md` v8.0.53](wiki/05-invariants.md#-version-log-entry--v8053-2026-05-21--retrain-in-progress).
- **v8.0.52 (2026-05-21) — Pass 70.7% deployed (best ever)**: Retrain v8.0.48b code → +1.9pp vs v8.0.48b, +$179 profit. Pool/GBM identical (AUC 0.6135, Brier 0.2363). Also added `scripts/validate_live_xlsx.py` blind validation script (95 trades alignment 51%, today 70%).
- **v8.0.51 (2026-05-21) — 🎯 ADX threshold 30→25 (Wilder default; data-driven, NOT YET RETRAINED)**: 87-trade analysis revealed ADX H1 25-30 = "killer zone" (16 trades, WR 25%, -$679). Original 25 was correct; "30 = only block extreme" was intuition not data. Setting changes (train + live unified at 25): `MRConfig.adx_trend_block`, `adx_trend_block_xau`, `_training` variants, plus `MeanReversionStrategy` class defaults. Retrain pending. ดู [`wiki/05-invariants.md` v8.0.51](wiki/05-invariants.md#-version-log-entry--v8051-2026-05-21--not-yet-retrained).
- **v8.0.50 (2026-05-21) — 🎯 Train-live parity: remove all Asian Delays**: Discovery — training simulator has NO Asian Delay logic, but live blocked Mon-Fri 00-07 EET (non-XAU) + Mon 00-04 (XAU 00-05). Pass 68.8% includes Asian Early in training. Disabled: `MONDAY_DELAY_ENABLED`, `WEEKDAY_DELAY_ENABLED`, `XAU_WEEKDAY_DELAY_ENABLED` (all True → False). RL now decides TAKE/SKIP for Asian signals (~124 historical blocks → flow to RL). Safety: `DAILY_LOSS_HARD_STOP_PCT 4%` ยังป้องกัน FTMO breach. Watch: 5-day Asian P/L < -$100/day average → revert flags. ดู [`wiki/05-invariants.md` v8.0.50](wiki/05-invariants.md#-version-log-entry--v8050-2026-05-21).
- **v8.0.49 (2026-05-20) — 🔇 Throttle Daily Loss approach-limit spam**: `RiskManager.check_daily_loss` print `⚠️ Daily Loss: X%` ทุก 5s ตอน loss > 3% = spam. Fix mirrors v8.0.16 Give-back throttle: new state `_last_daily_loss_alert_pct`, reset วันใหม่, print เฉพาะข้าม 0.5pp milestone (3.0%, 3.5%, 4.0%). ไม่ต้อง retrain. ดู [`wiki/05-invariants.md` v8.0.49](wiki/05-invariants.md#-version-log-entry--v8049-2026-05-20).
- **v8.0.48b (2026-05-20) — 🏆 Pass 68.8% (best ever), Profitable 94.3%, Profit avg $8,454**: Stepwise Trail (Stage 2 @ 0.8R: TP→1.5R, SL→0.5R; Stage 3 @ 1.0R: SL→1.0R + trail floor) + sim hit_sl bug fix (was hardcoded -1R, now computes from sl_price) + caps simplified (disabled DAILY_PROFIT_CAP 1.6% + DAILY_LOSS_CAP 3% — overlap with HARD_STOP 4%). Mean outcome pool +0.0070 (first time positive). GBM AUC 0.6134 best ever. value explained_variance 0.409 (+68% vs v8.0.47). ดู [`wiki/05-invariants.md` v8.0.48b](wiki/05-invariants.md#-version-log-entry--v8048b-2026-05-20--deployed-pass-688).
- **v8.0.48 (2026-05-20) — 🪜 Stepwise Trail (user-requested: Stage 2 @ 0.8R + Stage 3 @ 1.0R with SL floor)**: User-proposed 3-stage logic to lock more profit at clear thresholds. Stage 1 @ 0.5R (existing partial + BE), Stage 2 @ 0.8R (NEW: TP→1.5R, SL→0.5R), Stage 3 @ 1.0R (NEW: SL→1.0R + trail chase with `max(entry+1R, best-0.5R)` floor). Mirrored in `_resolve_trade()` sim. Gate: Pass ≥ v8.0.47 (61.5%) → push, else revert. **Status: training in progress.** ดู [`wiki/05-invariants.md` v8.0.48](wiki/05-invariants.md#-version-log-entry--v8048-2026-05-20--in-training).
- **v8.0.47 (2026-05-20) — 🔁 Trail 0.9R + removed reward cap (fallback from v8.0.45 Pass 52%)**: v8.0.45 trail @ 0.8R + reward cap eval Pass **52%** (regression -13.7pp from v8.0.43e 65.7%). Hypothesis: 0.8R too aggressive → SL trail catches retrace early, cap suppresses big-runner reward → agent stops chasing. Fix: `TRAIL_ACTIVATION_RR` 0.8→**0.9** (compromise) + removed `outcome > 1.5` cap in `signal_filter_env.py` step(). Live: restored v8.0.43e models from `.bak_1779255651` (Pass 65.7%) during retrain. v8.0.46 adaptive 1s loop kept (orthogonal). Gate: Pass ≥ 60% → push, else fallback Option A. ดู [`wiki/05-invariants.md` v8.0.47](wiki/05-invariants.md#-version-log-entry--v8047-2026-05-20).
- **v8.0.45/46 (DEPRECATED)**: v8.0.45 pre-emptive 0.8R failed (Pass 52%). v8.0.46 adaptive loop interval kept (live TP race fix).
- **v8.0.29 (2026-05-15) — 🔀 แยก training/live config (กัน retrain ทำให้ pool หด)**: v8.0.27 (XAU ADX 27) + v8.0.28 (confluence 70) เป็น live filter แต่ class-level `MeanReversionStrategy` ใช้ค่าเดียวกันทั้ง live + training → ถ้า retrain pool จะหดเหลือครึ่ง. Fix: เพิ่ม `MRConfig.min_confluence_score_training=30.0` + `adx_trend_block_training=30.0` + `adx_trend_block_xau_training=30.0` + `MeanReversionBacktester.__init__` override 3 ค่า. ผล: Live = 70/30/27, Training = 30/30/30 (กว้างเท่าเดิม). ไม่ต้อง retrain. ดู [`wiki/05-invariants.md` v8.0.29](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.28 (2026-05-15) — 🎯 Confluence floor enforced 70 (was unused, strategy used 30)**: Audit 48 live trades พบ confluence < 70 = WR 33-50% loss -$428, ≥ 70 = WR 76% +$425. `FTMOConfig.MIN_CONFLUENCE_SCORE=70` ตั้งไว้นานแต่ไม่ enforce — strategy class default 30 ทำให้ไม้คุณภาพต่ำผ่าน. Fix: เพิ่ม `MRConfig.min_confluence_score=70.0` + bump `MeanReversionStrategy.MIN_CONFLUENCE_SCORE` 30→70 + wire ผ่าน `__init__`. ผลคาดการณ์: 25/48 ไม้ผ่าน (block 48%), WR 58%→76%, P/L -$4→+$425. ไม่ต้อง retrain. ดู [`wiki/05-invariants.md` v8.0.28](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.27 (2026-05-15) — 🎯 Per-symbol ADX threshold for XAUUSD (27 vs default 30)**: วันนี้ทอง 2 ไม้ SL hit -$212 (Trade #2 ADX H1=28.5). Live data 14 XAU trades — block threshold ADX>27 จะ block แค่ 1/14 ไม้ (เป็นไม้แพ้ของวันนี้) ไม่ตัดไม้กำไรเลย. เพิ่ม `MRConfig.adx_trend_block_xau=27.0` + `MeanReversionStrategy.ADX_TREND_BLOCK_XAU=27.0` + per-symbol routing ใน `scan_signal()` (XAU=27, อื่นๆ=30). Default ADX block 30 ของ symbol อื่นไม่กระทบ. ไม่ต้อง retrain (strategy-level filter). ดู [`wiki/05-invariants.md` v8.0.27](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.26 (2026-05-14) — ⏱️ Bulk-trading guard (anti The5ers flag)**: Excel audit พบ 2 ไม้เปิดห่างกัน 0.00s/0.01s (12 พ.ค., pre-news bug ที่ fix แล้ว v8.0.21). The5ers ห้าม "bulk trading" = หลายไม้เปิดในวินาทีเดียว → flag bot ตอน withdrawal review. Fix: เพิ่ม `MIN_SECONDS_BETWEEN_OPENS_ENABLED=True` + `MIN_SECONDS_BETWEEN_OPENS_SEC=60` ใน FTMOConfig + state `_last_open_time_iso` ใน RiskManager + gate ก่อนทุก check + `record_trade_open()` เรียกหลัง executor สำเร็จ. Live median gap = 18.4 min อยู่แล้ว — gate เป็น backstop กัน edge case. ดู [`wiki/05-invariants.md` v8.0.26](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.25 (2026-05-14) — 🌅 Weekday Delay ขยายรวม Monday**: User request — Mon non-XAU เริ่มเทรด 11:00 ICT (07:00 EET) เหมือน Tue-Fri, Mon XAU เริ่ม 08:00 ICT (04:00 EET) ตาม v8.0.19 Monday delay. Fix: เปลี่ยน weekday filter ใน `RiskManager.can_open_trade` จาก `(1,2,3,4)` → `(0,1,2,3,4)`. Layered: Mon XAU ใช้ v8.0.19 อย่างเดียว, Mon non-XAU ใช้ทั้ง v8.0.19 + v8.0.25 (cap ที่ 07 EET = 11 ICT). Tue-Fri ไม่กระทบ. ดู [`wiki/05-invariants.md` v8.0.25](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.24 (2026-05-13) — 🌅 Weekday Asian Early Delay (Tue-Fri, XAU exception)**: ข้อมูลจริง 3 วัน, 42 trades — Non-XAU Asian early (00-07 EET) ขาดทุน -$300, แต่ XAUUSD Asian early ชนะ 80% WR +$159. Fix: block trade Tue-Fri 00:00-06:59 EET (= 04:00-10:59 ICT) ยกเว้น XAUUSD. Layered กับ v8.0.19 Monday delay (Mon 00:00-03:59 EET). Config: `WEEKDAY_DELAY_ENABLED=True`, `WEEKDAY_DELAY_END_HOUR_EET=7`, `WEEKDAY_DELAY_EXCEPT_SYMBOLS=("XAUUSD",)`. ดู [`wiki/05-invariants.md` v8.0.24](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.22 (2026-05-12) — 🛑 Daily Loss Cap -3% (Option D mirror)**: Production deploy ระหว่าง user สอบ The5ers จริง — วันนี้ peak DD 3.45% ใกล้ FTMO breach 4%. เพิ่ม `DAILY_LOSS_CAP_ENABLED=True` + `DAILY_LOSS_CAP_PCT=0.030` ใน FTMOConfig. Mirror ของ v8.0.17 profit cap — ครบ -$300 (3% × $10k) → ปิดทุก position + block trade ใหม่ทั้งวัน → reset วันใหม่ EET midnight. Buffer ก่อน FTMO breach = 1% = $100. Symmetric กับ profit cap (+1.6%). ดู [`wiki/05-invariants.md` v8.0.22](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.21 (2026-05-12) — 📰 Pre-news block ใน can_open_trade**: Excel จริงโชว์ 6 ไม้เปิด-ปิดใน 4-8s ใกล้ข่าว 15:30 EET → เสียค่า spread $11.79 รวม. Root: `is_near_high_impact_news` ถูกใช้แค่ใน `TradeManager.check_news_close` (ปิด open) ไม่มีใน `can_open_trade`. Fix: เพิ่ม news gate ก่อน check อื่นๆ ใน `can_open_trade`. ใช้ window เดียวกัน (30 นาทีก่อน, 15 นาทีหลัง). ดู [`wiki/05-invariants.md` v8.0.21](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.20 (2026-05-12) — 🔌 Auto-detect filling mode (multi-broker)**: The5ers MT5 reject order ด้วย retcode=10030 "Unsupported filling mode" — บอท hardcode IOC แต่ The5ers รองรับเฉพาะ FOK. Fix: `MT5Connector._get_filling_type(symbol)` อ่าน `symbol_info.filling_mode` bitmask แล้ว pick mode ที่ broker รองรับ (IOC > FOK > RETURN). ใช้ทั้งที่ open + close + partial close. รองรับทุก broker. ดู [`wiki/05-invariants.md` v8.0.20](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.19 (2026-05-11) — 🌅 Monday Morning Delay (4 ชม.)**: ข้อมูลจริง Monday 11 พ.ค.: 3 ไม้แรก (01:06, 03:50, 03:54 EET) เสีย $300+ ก่อนตลาดสงบ. Fix: เพิ่ม `MONDAY_DELAY_ENABLED=True` + `MONDAY_DELAY_END_HOUR_EET=4` — block trade ใหม่ Monday 00:00-03:59 EET (04:00-07:59 ICT). Tue-Fri ไม่กระทบ. Open positions ค้างจาก Friday ยัง managed ปกติ. ดู [`wiki/05-invariants.md` v8.0.19](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.18 (2026-05-10) — 🔄 Flip-lock TTL (กัน lock ค้างข้าม weekend)**: Friday ปิด SELL XAUUSD @ 4746 → Monday ราคา 4673 ไม่ retrace → flip-lock BUY ค้างถาวร. Fix: เพิ่ม `FLIP_LOCK_MAX_MINUTES=240` (4 ชม. TTL) — auto-expire ไม่ว่าราคาจะ retrace หรือไม่. `_load_state` ล้าง legacy locks. ดู [`wiki/05-invariants.md` v8.0.18](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.17 (2026-05-09) — 🎯 Daily Profit Cap (Option D Hard Stop)**: User-requested feature เพื่อ lock กำไรวันนี้ + หยุดเทรด เมื่อ daily P/L (closed + floating) ถึง **1.6% ของ initial balance** ($10k→$160, $100k→$1600 — มี 0.1pp buffer สำหรับ slippage/spread vs target 1.5%). Anchor = `_initial_balance` (คงที่ตลอด challenge ไม่ scale daily). Reset = broker EET midnight. Action = ปิดทุก position + block new trades. Feature flag `DAILY_PROFIT_CAP_ENABLED=True` ใน `FTMOConfig`. ใช้สำหรับ FTMO Phase 1 challenge — ผ่าน +10% ใน 7-10 วัน. ดู [`wiki/05-invariants.md` v8.0.17](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.16 (2026-05-08) — 🔇 Throttle Give-back-from-peak alert (anti-spam)**: `RiskManager.check_risk()` print "Give-back X%" ทุก 5s = 17,000+ ครั้ง/วัน. Fix: เพิ่ม `_last_give_back_alert_pct` แล้ว print เฉพาะเมื่อข้าม 1% milestone ใหม่ (2.0%→3.0%→...). Reset เมื่อ peak ใหม่ + วันใหม่. ลดเหลือ ~2-3 ครั้ง/วัน. ดู [`wiki/05-invariants.md` v8.0.16](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.15 (2026-05-08) — 🔧 RR FP-precision fix v2 (XAUUSD regression)**: User เจอ `Risk:Reward (1.00) ต่ำกว่าขั้นต่ำ (1.0)` กลับมาอีกครั้งบน XAUUSD. v8.0.11 ใช้ tolerance 1e-4 (พอสำหรับ FX 5-digit) แต่ **XAUUSD digits=2 → drift ถึง ~0.4%** (40× over budget). Fix 2 layers: (1) tolerance เป็น **relative 1%** ใน RiskManager + PositionSizer, (2) `MRSignal.rr_ratio` snap-to-half-multiple ภายใน 1% → คืนค่าตรง design (1.0) ทุก symbol. Verified 4 symbols (XAU/NZD/GBP/JPY) → ผ่านหมด. ดู [`wiki/05-invariants.md` v8.0.15 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.14 (2026-05-07) — 🔧 Partial-first fix (close 50% before BE move)**: User เจอ bug บอทเลื่อน SL ไป BE โดยไม่ปิด 50% ก่อน. Root cause: v8.0.12 ตั้ง BE 0.5R + Partial 0.7R + code order BE ก่อน — ถ้า MFE peak 0.6R แล้ว revert จะได้ BE-only ไม่มี partial → revert ปิดที่ entry = 0 บาท. Fix: `PARTIAL_TRIGGER_RR` 0.7 → **0.5** (เท่า BE) + reverse code order **Partial ก่อน BE**. ผลลัพธ์: MFE 0.5R → Partial 50% (lock 0.25R) → BE → revert ได้ +0.25R, ถึง TP ได้ +0.75R. ดู [`wiki/05-invariants.md` v8.0.14 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.13 (2026-05-07) — ♻️ Orphan position recovery + trail_states persistence**: Bot restart ทำให้เปิด order ซ้ำ (เห็นจริง: NZDUSD SELL 2x ในรูป user) เพราะ `sync_with_mt5` ทำซิงค์แค่ทางเดียว (active→MT5 หา closed trades) แต่ไม่ทำ MT5→active (import orphan). Fix 3 จุด: (1) `_rebuild_executed_trade_from_mt5` import orphan ตอน sync, (2) `main.connect` เรียก `sync_with_mt5` ก่อน scan signal, (3) `TradeManager._save/_load_trail_states` persist BE/Partial/best_price ผ่าน restart (`logs/trail_states.json`). หลัง restart บอท re-attach orphans + จำสถานะ BE/Partial ได้ → ไม่เปิดซ้ำ + ไม่ trigger BE/Partial ซ้ำ. ดู [`wiki/05-invariants.md` v8.0.13 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.12 (2026-05-07) — 🔧 TradeManager BE/Partial/Trail tuned for MR RR=1:1**: Constants ค้างจากยุค SMC RR=1:2.5 → ภายใต้ MR RR=1:1 = dead code ทุกตัว (TP ปิดที่ 1.0R ก่อน BE/Partial ทำงาน, Trail 1.5R เป็นไปไม่ได้เลย). Live ผลกระทบ: GBPUSD trade MFE ≈ 0.95R revert มา full SL — ถ้า BE ทำงานที่ 0.5R จะปิดที่ entry ไม่เสีย. Tune ใหม่: BE 1.0R → **0.5R**, Partial 1.0R → **0.7R**, Trail 1.5R → **99.0** (disabled). ไม่ต้อง retrain (TradeManager เป็น live-only). ดู [`wiki/05-invariants.md` v8.0.12 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.11 (2026-05-07) — 🔧 RR FP-precision fix (RiskManager + PositionSizer)**: VPS log แสดง `Risk:Reward (1.00) ต่ำกว่าขั้นต่ำ (1.0)` — display พูด 1.00 แต่ค่าจริงเป็น 0.99999... จาก floating-point drift หลัง `round(entry - sl_distance, 5)`. `MRSignal.rr_ratio` คำนวณจาก rounded price → strict `<` comparison reject ทุก MR signal ที่ RR=1:1 พอดี. Fix: เพิ่ม `1e-4` epsilon tolerance ที่ 2 จุด (`RiskManager.can_open_trade` + `PositionSizer.calculate_sl_tp_prices`). ดู [`wiki/05-invariants.md` v8.0.11 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.10 (2026-05-07) — 🔬 Anti-overfit retrain pipeline + holdout eval gate**: v8.0.5 model showed Pass Rate 59.30 % on the same 3000-pool it trained on — no independent generalization test. Built holdout pool (`data/mr_signal_pool_holdout.pkl`, seed=999, 783 valid eps) and `scripts/holdout_eval.py` to measure Δ Pass Rate (train pool vs holdout). Retrain settings: pool 3000 → **5000**, outcome_noise 0.05 → **0.08**, P2 timesteps 2M → **5M**. Backup of v8.0.5 best at `models/mr/best_v8.0.5_pass59pct/`. New invariant: holdout Δ Pass Rate ≤ 10 pp gate before promoting any model to live. ดู [`wiki/05-invariants.md` v8.0.10 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0.5 (2026-05-07) — 🎉 MR pipeline ALL GATES PASSED in iter 1, live wired**: After 4 sub-iterations of fixes (leakage audit, parity audit, daily/total DD guards), the autonomous pipeline converged in a single training iter. Final eval (5000 eps): **Pass Rate 59.30%**, Profitable Rate 89.10%, Total DD max 5.80% (≤ 6% gate), Daily DD max 3.00% (≤ 3.5% gate), Breach Rate 0%, Profit avg +$7,229.59 (+7.23%). Best model snapshotted at `models/mr/best/`. Live `main.py` already refactored to use `LiveMRScanner` (drop-in for SMCStrategy) and `bot_config.mr.strategy_mode = "mean_reversion"`. Run `python main.py` to deploy. SMC code preserved as deprecated reference (`strategy/smc_strategy.py` still exists, no longer imported by live). Two new invariants enforced: **NO LEAKAGE** (`scripts/leakage_audit.py`) + **TRAIN/LIVE PARITY** (`scripts/parity_audit.py`) — both must pass before any commit touching `ml/`, `strategy/`, or `main._build_signal_observation`. ดู [`wiki/05-invariants.md` v8.0.5 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v8.0 (2026-05-06) — Mean Reversion strategy pivot (initial spec)**: Parallel pivot to a high-win-rate Mean Reversion + Trend Filter strategy. New modules: `MeanReversionStrategy` (BB %B extreme + RSI confirm + ADX H1 trend block + ATR-based tight SL + RR 1:1), `MeanReversionBacktester` (pool builder), `MeanReversionFilterEnv` (RL env with quick-TP bonus + prolonged-loss penalty + duration fine), and `auto_train_pipeline.py` (autonomous Build pool → GBM → RL → Eval → Self-correct loop). Pool path: `data/mr_signal_pool_<N>.pkl`, model path: `models/mr/ppo_mr_filter.zip`. Reward shaping focuses on capital preservation: quick-TP (≤5 bars) bonus +0.5R, slow win bonus +0.2R, base loss −0.1R, duration fine 0.02R/bar (capped 0.3R), prolonged-loss (≥12 bars red) penalty −0.4R, ADX > 25 violation penalty −0.3R.
- **v7.2.1 (2026-05-06) — obs[29/30] leak fix (audit-driven)**: Audit หา data leakage หลัง v7.2 retrain Pass Rate **6.3%** (ต่ำกว่า baseline v6.13 = 9.7%). พบ leak ใน `FTMOSignalFilterEnv` — `_open_positions` เก็บ `outcome_partial` (final outcome × decay) → `_floating_pnl_norm` (obs[29]) + `_open_losing_count_norm` (obs[30]) **เห็น sign ของ outcome ตั้งแต่ position เปิด**. Live ใช้ true unrealized PnL จาก `RiskManager` ที่ oscillate กับ price path → distribution mismatch + future leak. Fix: zero-out obs[29/30] ใน training env (ลบ block calculation + `outcome_partial` field). 5 surface อื่น (GBM features/HTF align/training OOF/Chronos/Aux) ผ่าน clean. Pool/GBM ไม่ต้อง rebuild — leak อยู่ที่ env เท่านั้น. Retrain RL อย่างเดียว.
- **v7.2 (2026-05-06) — Chronos un-flip + Session log fix (audit-driven)**: Live audit ของ `logs/ftmo_trades.xlsx` (524 signals 2026-05-05, strong DOWN-trend day, 100% SELL) พบ `chronos_alignment = -1` ติดทุก signal → reward penalty `-0.30` ลงทุก TAKE ระหว่าง train → agent learned skip-default (TAKE max ML 0.598 vs SKIP max ML 0.693 = discrimination ผิดทาง). Root cause = v7.0.2 flip-sign semantics ออกแบบสำหรับ contrarian/reversal strategy แต่ SMC จริง ๆ เป็น trend-following (ใช้ HTF Tier 1 hard veto Counter-D1) → mismatch. **Patch B**: un-flip formula ใน `ChronosForecaster.compute_features` → `delta = median_h - last_close` (alignment +1 = SMC+Chronos agree on direction). **Patch A**: `FTMOTradingBot._log_signal_scan` Session column ที่ค้าง None ใน Signals sheet → เพิ่ม `_compute_current_session()` helper. **Pipeline**: rebuild pool + retrain GBM + retrain RL (~36 ชม.). Backups `*.bak_v7.1` พร้อม rollback.
- **v7.1.10 (2026-05-06) — Pre-news close fix**: ก่อนหน้านี้ news filter block สัญญาณใหม่อย่างเดียว แต่ position ที่เปิดอยู่ถูกถือผ่านข่าว = ผิดกฎ FTMO. เพิ่ม `TradeManager.check_news_close()` ปิด position ที่ symbol จะชนข่าวแรงใน 30 นาที (sync กับ `no_trade_before_news_minutes`). เรียกใน main loop ก่อน `check_session_close`. Priority: Friday/Daily Overnight > Pre-News > Trailing/BE/Partial. เพิ่ม `XAUUSD` เข้า `_CURRENCY_TO_SYMBOLS["USD"]` (ทอง spike แรงตอน NFP/CPI/FOMC). ไม่ต้อง retrain — fix อยู่ใน execution path เท่านั้น
- **Goal**: pass the FTMO 2-step Standard Challenge (10 % profit, 4 % daily DD, 8 % total DD).
- **3 brains + 1 forecaster (v8.0+)**: `LiveMRScanner` (Mean Reversion rules — BB+RSI+ADX) → `SignalQualityModel` (GBM + Isotonic calibrator) → **`ChronosForecaster`** (Amazon Chronos 2 zero-shot, optional) → `SelfLearningAgent` (PPO + Auxiliary Task — TAKE/SKIP).
- **Live entry**: `python main.py` → `FTMOTradingBot.run` loops every 5 s. Console runs in **quiet mode** (announce-once for idle states; per-signal SKIP/NO_AGENT logged to Excel `Signals` sheet, not console).
- **Obs = 26 dims** (v8.0.74b, 2026-05-27 — trimmed 9 dead/duplicate dims from 35). Must stay in sync across three places: `FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM`.
- **Runs on**: macOS/Linux (train + backtest), Windows + MT5 (live).
- **Live logging**: `TradeLogger` (re-enabled v6.9, schema bumped v6.10) writes Excel — Trades 64 cols (incl. `Obs27 JSON` for retrain), Signals 21 cols (per-scan log), Daily, Stats.
- **v6.11 SMC overhaul (2026-04-29)**: Counter-D1 hard veto + Sweep within 8 bars + Fresh M15 BOS within 6 bars + ADX H4 ≥ 22 + Quiet-vol × off-overlap blocker + IDM detector + OB grading (Extreme/Decisional/Internal). BE trigger ใช้ `best_price` (rolling MFE). Per-component pts populate ใน TradeSignal → Trades sheet เห็น HTF/MTF/OB/FVG/Sweep pts จริง. ดู [`wiki/05-invariants.md` v6.11 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological).
- **v6.11.1 (2026-04-29 evening)**: Post-impl audit fix — `StrategyBacktester._init_strategy` เพิ่ม `_idm_detector` init ให้ตรงกับ `SMCStrategy.__init__` (กัน AttributeError ตอน rebuild pool). Eval Pass Rate 11.2 % ที่ได้จาก cached pool (Apr 25, pre-v6.11) — **invalid สำหรับ v6.11**. ต้อง rebuild pool + GBM ก่อน eval ใหม่.
- **v6.11.2 (2026-04-29 evening)**: Partial rollback — Tier 2.2 (Sweep prereq) + Tier 2.3 (Fresh M15 BOS prereq) hard gates → soft bonuses. Pool หาย 99 % ภายใต้ v6.11 hard gates → 0.0 % Pass Rate. หย่อน 2 จุด → pool 78k signals, Pass Rate 2.7 %, WR 65.6 %.
- **v6.11.3 (2026-04-29 evening)**: Mild tune — IDM penalty 5→2 + ADX H4 floor 22→20. Pool 90k (+15 %), Pass Rate 2.7→**3.4 %**, WR 65.6→**68.8 %**, DD max 4.46→**3.23 %** (safer). Pure improvement ทุกมิติ. **KEEP + deploy demo**. Backups `*.bak_v6.11.2` พร้อมสำหรับ rollback.
- **v6.12 (2026-04-29 night)**: Live-vs-train sync fix — `FTMOConfig.ML_FILTER_THRESHOLD = 0.36` + ML gate ใน `FTMOTradingBot.run` ก่อน agent. ปิด silent regression จากการที่ live ไม่บังคับ ML threshold (training บังคับ). Logging fix — `ML Threshold` column ใน `Signals` sheet ตอนนี้แสดง 0.36 จริง (เดิม 0.0 เพราะ getattr ผิด attribute). ไม่ต้อง retrain.
- **v7.1 (2026-05-04) — Code staged, awaiting retrain ⚠️ ห้ามรัน live จนกว่าจะ retrain (OBS_DIM 29→32 mismatch)**:
  - **RCA**: 5 ออเดอร์โดน SL hit 100% ใน 3 ชม. (Daily DD 2.85% / Max DD 3.97%) แม้ HTF ตรง 4/5 + ML cal ≥ 0.44 + RL TAKE ทั้งหมด → 3 brains "ไม่รู้เวลา + ไม่รู้ portfolio risk" — ไม่ใช่ bug จุดเดียว
  - **Runtime guards (no retrain)**: `RiskManager.check_unrealized_circuit_breaker` (floating ≤ −1.5% × open ≥ 2 → pause), `MAX_USD_THEME_POSITIONS=2` (cross-group cap), `SPREAD_ATR_RATIO_LIMIT=0.20` (thin liquidity), Stats sheet limits จาก config (เลิก hardcode 10%/5%), close payload populate (Bid@Exit/Balance@Close/Equity Peak)
  - **SMC pre-filters ใหม่ (G/H/I)**: HTF=Neutral + ADX H1 < 25 hard veto, vol_regime ∈ {high, explosive} block, spread/ATR > 20% block, session warmup + post-weekend → required confluence +5/+10
  - **Volatility regime classifier**: `TechnicalIndicators.classify_volatility_regime` (quiet/normal/high/explosive) + `compute_atr_zscore_30bars`
  - **Dynamic SL multiplier**: `_compute_dynamic_sl_multiplier` (per-symbol base × clip(1+atr_z×0.3, 0.8, 1.8))
  - **GBM features 17→24**: เพิ่ม `hour_of_day_sin/cos`, `day_of_week`, `minutes_since_session_start`, `is_post_weekend_first_hour`, `volatility_regime_score`, `atr_zscore_30bars` + `compute_temporal_features` helper + `detect_drift` (KS test live vs train)
  - **Chronos formula**: linear `(q90-q10)/(atr×√8)` → `log1p(...)/2` (กัน saturation ที่ 3.0 ทุก signal)
  - **RL obs 29→32**: เพิ่ม `floating_pnl_norm`, `open_losing_count_norm`, `mins_since_session_norm`
  - **Reward shaping**: Chronos disagreement penalty (-0.40 ถ้า align<0 + ml<0.55, -0.15 อื่น), concurrent loss penalty (-0.25 ถ้า floating<-1%), missed-winner softened P2 -0.90→-0.65
  - **Pipeline**: ต้อง rebuild pool + retrain GBM + retrain RL ทั้งหมด ~36 ชม. compute. Backups `*.bak_v7.0.7` ก่อนเริ่ม
  - **Eval gate**: Pass Rate ≥ 9% + DD max ≤ 4.5% + Profitable ≥ 70% → keep. < 7% → restore. ดู [`wiki/05-invariants.md` v7.1 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.7 (2026-05-02) — Revert threshold 30 → 20 (v7.0.5 retrain after backup loss)**:
  - **Why** — v7.0.6 (threshold 30) eval Pass Rate **7.4%** (regression −3.3 pp จาก v7.0.5 = 10.7%). User skip backup step → v7.0.5 model หาย → ต้อง retrain ใหม่
  - **Lesson learned** — Early stop ที่ value_loss=20.45 = **inflection point detector** (ไม่ใช่ false positive). Phase 2 หลังจุดนี้ทำให้ agent over-tune toward safety: WR ขึ้น (65→68%), Profitable ขึ้น (86→88%), แต่ **passes น้อยลง** (selective เกินไม่ aggressive พอจะถึง 10% ใน 45 วัน)
  - **Fix** — single-line revert: `threshold=20.0` (กลับ v7.0.5 config) — engineered sweet spot @ Phase 2 ~70%
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Risk**: stochastic — RNG อาจให้ Pass 9.5-11.5% (ไม่เป๊ะ 10.7%). v7.0.3 backup (10.0%) เป็น final safety net
  - **Backup discipline** — เพิ่ม emphasis: ต้อง execute backup commands ก่อนทุก retrain ห้ามข้าม
  - ดู [`wiki/05-invariants.md` v7.0.7 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.6 (2026-05-02) — Phase 2 EarlyStop threshold 20 → 30, Pass Rate 7.4% (regressed)**:
  - **Why** — v7.0.5 trigger ที่ `value_loss = 20.45` (just 1.02× threshold) = borderline. หลัง LR fix (5e-5) value_loss variance ต่ำลง → threshold 20 อาจ aggressive เกิน. Phase 2 รัน 70% (3.5M / 5M) ก่อนหยุด — มี potential ที่ยังไม่ได้
  - **Fix** — single-line: `EarlyStopOnValueLoss(threshold=30.0, ...)` (จาก 20.0). 30 = 1.5× = ตรงกับ "natural transient spike range" ของ v6.13/v7.0.3 historical (31)
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Probability**: 40% Pass > 10.7%, 35% equivalent, 20% mild regress, 5% catastrophic. Backup `*.bak_v7.0.5` (10.7%) เป็น safety net
  - **Watch** — ถ้า value_loss > 30 = true divergence (early stop trigger). ถ้า Pass Rate trajectory ลด 2 snapshots ติด = kill + restore v7.0.5
  - **Gate** ≥ 11% = win, 10-11% = equivalent (keep), < 9% = restore v7.0.5
  - ดู [`wiki/05-invariants.md` v7.0.6 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.5 (2026-05-02) — Phase 2 LR schedule proper fix (latent bug จาก v6.x), Pass Rate 10.7%**:
  - **Why** — v7.0.4 retrain Phase 2 รันเต็ม 5M (warmup ปลด early-stop) → Pass Rate **2.8%** (regression −7.2 pp). Trajectory ลดลงตอนปลาย Phase 2 = over-training
  - **Root cause Layer 1 (latent bug)** — SB3 PPO `_setup_model()` wrap `lr_schedule` ตอน init เท่านั้น. การ set `model.learning_rate = 5e-5` หลัง `AuxAwarePPO.load()` ไม่ rebuild schedule → optimizer ใช้ Phase 1 default 3e-4 (= **6× สูงกว่า intended** ตลอด v6.x → v7.0.4)
  - **Root cause Layer 2** — High LR (3e-4) × Long Phase 2 (5M, ก่อนหน้านี้ early-stop @ ~32k) = over-train. v6.13/v7.0.3 "lucky" เพราะ early-stop ปกป้อง by accident
  - **Fix** — rebuild lr_schedule + update optimizer 3 ชั้น:
    ```python
    from stable_baselines3.common.utils import FloatSchedule
    model_p2.learning_rate = 5e-5
    model_p2.lr_schedule = FloatSchedule(5e-5)
    for _pg in model_p2.policy.optimizer.param_groups:
        _pg['lr'] = 5e-5
    ```
  - **No rebuild** pool/GBM. Retrain RL อย่างเดียว ~30 ชม.
  - **Watch** — Phase 2 step 0 TensorBoard `learning_rate = 0.00005` (ถ้าเป็น 0.0003 = fix fail)
  - **Gate** ≥ 10% = keep, < 7% = restore `*.bak_v7.0.3`
  - Backups `*.bak_v7.0.3` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.5 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.4 (2026-05-02) — Phase 2 EarlyStopOnValueLoss warmup grace, Pass Rate 2.8% (regressed)**:
  - **Why** — v7.0.3 retrain Phase 2 trigger early stop ที่ step 32k / 5M (0.6% ของ target). value_loss spike 31 > threshold 20 ตอน Phase 2 start เพราะ reward distribution shift (Phase 1 [-2,5] → Phase 2 [-15,5] หลังเปิด DD penalty + activity floor). Eval Pass Rate 10.0% (ผ่าน gate baseline) แต่ Phase 2 "เกือบไม่ได้ทำงาน" → potential ที่ขาด
  - **Fix** — เพิ่ม `warmup_steps` parameter ใน `EarlyStopOnValueLoss`:
    - Phase 2: `warmup_steps=50_000` (1% ของ 5M target) — ปล่อยให้ value head re-fit reward distribution ใหม่ก่อน enable check
    - Phase 1: `warmup_steps=0` (default) — backward compat, ไม่กระทบ
  - **No rebuild** pool / GBM. Retrain RL อย่างเดียว ~25-30 ชม. Backup `*.bak_v7.0.3` เผื่อ rollback
  - **Gate** > 10.0% = win (warmup ช่วย push), 9-10% = equivalent (no harm), < 9% = restore `*.bak_v7.0.3`
  - Backups พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.4 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.3 (2026-05-02) — Correlation simulator HOLD=0 (over-block fix), Pass Rate 10.0%**:
  - **Why** — v7.0.2 eval Pass Rate ตก 0.6% (จาก 9.7% baseline). Root cause: time-scale mismatch ระหว่าง pool (4 scans/day = 1 slot ≈ 6h) กับ live (avg position hold 75 นาที) → `HOLD_SIGNALS_APPROX = 1` block correlation 6h ใน live time = over-block 4-5×. Agent skip-all (Take Rate 50% → 32%, Orders/ep 7.7 → 4.8)
  - **Fix** — single-line: `HOLD_SIGNALS_APPROX = 1 → 0` ใน `ml/signal_filter_env.py`. Drop-stale logic clear `_open_positions` ทันทีทุก step → effective correlation simulator off, infrastructure คงไว้ (เผื่อ tune กลับ)
  - **No rebuild** pool/GBM (ไม่ขึ้นกับ HOLD). Retrain RL อย่างเดียว ~25-30 ชม.
  - **Gate** ≥ 9.7% = keep, < 4% = revert. Test isolation: ถ้าผ่าน = Chronos formula fix ใช้งานได้ จริง correlation simulator ต้อง parameter ที่ถูกต้อง. ถ้าไม่ผ่าน = revert + consider Trend Following (Plan B)
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.3 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0.2 (2026-05-01) — Chronos formula fix + correlation training-live sync**:
  - **Why** — v7.0 retrain Pass Rate ตก 9.7% → 4.0%. Root cause analysis (pool diagnostics + ftmo_trades audit) พบ 3 จุด: (1) `chronos_uncertainty_norm` saturated 96.2% ที่ค่า max, (2) `chronos_alignment` corr ติดลบ (−0.0178 anti-signal), (3) live correlation block 96.3% ของ TAKE แต่ training env ไม่มี check
  - **Fix Chronos formula** (`ml/chronos_forecaster.py`): flip alignment sign (`direction × sign(close − median)`) + Brownian-scaled uncertainty (`/ atr × √8`) → distribution กระจาย ไม่ saturate, sign ตรงกับ outcome
  - **Add correlation simulator** (`ml/signal_filter_env.py`): mirror `TradeExecutor.CORRELATION_GROUPS` (USD_WEAK/STRONG, JPY_CROSS, EUR/GBP_PAIRS, SAFE_HAVEN), `_open_positions` virtual list, `_is_correlation_blocked()` method, forced SKIP ใน `step()` ถ้า block — training distribution ตรงกับ live
  - **No config change** — ห้ามแตะ Daily DD logic, OBS_DIM ยังคง 29, live source of truth (`TradeExecutor`) ไม่แก้
  - **Pipeline**: rebuild pool → retrain GBM → retrain RL ทั้งหมด. Eval gate ≥ 9.7% = keep, < 7% = revert จาก `*.bak_v6.14`
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0.2 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v7.0 (2026-05-01) — Amazon Chronos 2 zero-shot forecast features added (obs 27 → 29)**:
  - **New module** — `ftmo_trading_bot/ml/chronos_forecaster.py` (`ChronosForecaster`). โหลด `amazon/chronos-bolt-small` ผ่าน `BaseChronosPipeline.from_pretrained` ครั้งเดียว, cache forecast per `(symbol, last_bar_ts)`, deterministic via `torch.manual_seed(0)`.
  - **2 new obs features** — `obs[27] chronos_alignment` (`direction × sign(median_h+8 − close)`, ±1) + `obs[28] chronos_uncertainty_norm` (`(q90−q10)/atr`, [0, 3]). Sync ทั้ง 3 ที่ (`FTMOSignalFilterEnv._get_obs` / `FTMOTradingBot._build_signal_observation` / `SelfLearningAgent.OBS_DIM = 29`).
  - **Pool inject** — `StrategyBacktester.generate_episode_signals` คำนวณ chronos features จาก `ltf_slice` (closed bars only, ห้าม leak future) → เก็บใน signal dict สำหรับ `FTMOSignalFilterEnv._get_obs` อ่าน.
  - **Live path** — `FTMOTradingBot.__init__` instantiate `ChronosForecaster`. `_build_signal_observation` + `_build_live_context` ส่งค่าเข้า obs + Excel log.
  - **Excel schema bump** — `Signals` sheet 21 → 23 cols (+ `Chronos Align`, `Chronos Unc`); `Trades` sheet 64 → 66 cols (+ chronos @ entry). Pre-v7 Excel ต้อง `mv logs/ftmo_trades.xlsx logs/ftmo_trades_pre_v7.xlsx` ก่อน first run.
  - **Config single source of truth** — `bot_config.ml.CHRONOS_MODEL_NAME` (default `"amazon/chronos-bolt-small"`), `CHRONOS_PREDICTION_LENGTH=8`, `CHRONOS_CONTEXT_LENGTH=512`, `CHRONOS_ENABLED=True`. Disable via env `BOT_DISABLE_CHRONOS=1` (unit tests).
  - **ห้ามแตะ Daily DD** — `RiskManager.update_daily_pnl` / `_initial_balance` / `_daily_start_balance` ไม่ได้รับ touch ใดๆ. Chronos อยู่ที่ pre-trade gate เท่านั้น.
  - **Pipeline**: ต้อง rebuild pool + retrain GBM + retrain RL (Pass Rate gate ≥ 9.7 % vs v6.13 baseline = keep, ไม่ผ่าน = revert จาก `*.bak_v6.14`).
  - Backups `*.bak_v6.14` พร้อม rollback. ดู [`wiki/05-invariants.md` v7.0 Version Log](wiki/05-invariants.md#-version-log-reverse-chronological)
- **v6.14 (2026-04-30) — Live demo log audit, 4 bug fixes** (silent regression vs v6.13 spec):
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
| Strategy (v8.0+) | **Mean Reversion** + ADX trend filter (SMC removed v8.0.6) | `LiveMRScanner`, `MeanReversionStrategy` |
| Symbols (configured) | **10** (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, XAUUSD) | `SymbolConfig.symbols` |
| **Symbols (LIVE, v8.0.56)** | **7** (blocked: AUDUSD/USDCAD/USDCHF — audit Net -$1,297) | `SymbolConfig.blocked_symbols` |
| Timeframes | M15 (entry) / H1 (ADX trend filter) | only M15 + H1 used in v8.0 (H4 ignored by MR) |
| Profit target | 10 % | `FTMOConfig.PROFIT_TARGET_PCT` |
| FTMO Daily DD limit | 5 % (we cap at 4%) | `FTMOConfig.DAILY_LOSS_HARD_STOP_PCT` |
| FTMO Total DD limit | 10 % (we cap at 8%) | `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` |
| **MR RR (v8.0.61)** | **1.0** (reverted from v8.0.60 RR 1.5 — M1 replay showed continuation 26.9% < 40% gate) | `MRConfig.rr_ratio` / `MeanReversionStrategy.RR_RATIO` |
| Default risk per trade | **0.70 %** (v8.0.56 — Phase 1 EV fix, was 0.85%) | `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT` |
| **BE / Partial (v8.0.56)** | BE 0.3R / Partial 0.8R @ 33% close (was 0.5/0.5/50%) — "ปล่อยไม้ชนะวิ่งก่อน, จับไม้แพ้ revert" | `TradeManager.BE_TRIGGER_RR/PARTIAL_TRIGGER_RR/PARTIAL_CLOSE_PCT` |
| **Stepwise Trail (v8.0.73)** | Stage 1@0.5R (BE only after v8.0.56), Stage 2@0.8R (**SL→0.5R lock, TP unchanged 1.0R** per v8.0.73), Stage 3@1.0R (SL→1.0R + chase — safety-net only, rarely fires after v8.0.73) | `TradeManager.TP_STEP_*` / `TRAIL_*` |
| **Daily caps (v8.0.56)** | DAILY_LOSS_HARD_STOP_PCT 4% + **DAILY_LOSS_CAP_PCT 2.5%** (re-enabled, 1.5pp buffer) | `FTMOConfig.DAILY_LOSS_CAP_*` |
| Main loop interval (v8.0.70) | 5s default, **1s when ANY position open** (was: only profit ≥ 0.5R) | `bot_config.main_loop_interval`, adaptive in `FTMOTradingBot.run` |
| Signal scan cadence (v8.0.70) | **wall-clock 60s** (decoupled from loop count — keeps 1/min even when loop = 1s) | `FTMOTradingBot._last_signal_scan_ts` |
| Trail reward (v8.0.47) | Raw outcome (no cap — let big runners flow) | `FTMOSignalFilterEnv.step()` |
| ML threshold (v8.0.3) | **0.30** (was 0.36 SMC era) | `FTMOConfig.ML_FILTER_THRESHOLD` |
| Max open positions | **3** (v8.0.69 — bumped from 2; blocked_symbols + opposing-theme block keep quality) | `FTMOConfig.MAX_OPEN_POSITIONS` |
| **Env DAILY_DD_GUARD (v8.0.4)** | **3.0 %** (was 4.0 %) | `FTMOSignalFilterEnv.DAILY_DD_GUARD` |
| **Env TOTAL_DD_GUARD (v8.0.5)** | **5.8 %** (was 8.5 %) | `FTMOSignalFilterEnv.TOTAL_DD_GUARD` |
| **MR BB oversold/overbought** | 0.30 / 0.70 | `bot_config.mr.bb_oversold/overbought` |
| **MR RSI oversold/overbought** | 40 / 60 | `bot_config.mr.rsi_oversold/overbought` |
| **MR ADX H1 trend block** | > 30 → no entry | `bot_config.mr.adx_trend_block` |
| **MR SL / RR** | 1.0 × ATR / 1:1 (quick TP) | `bot_config.mr.sl_atr_mult/rr_ratio` |
| **MR scan cadence** | every 30 min (48/day) + dedup 4 bars | `MeanReversionBacktester.MR_SCAN_POINTS_PER_DAY` |
| **Obs dims (v8)** | **32** (production model trained at 32 dims) | `SelfLearningAgent.OBS_DIM` |
| GBM features | **28** (v8.0.6 — added MR-specific: bb_extreme, bb_band_width, mr_setup_score, reversal_wick_ratio) | `train_mr_signal_quality.FEATURE_KEYS` |
| Chronos model (v7+) | `amazon/chronos-bolt-small` (disable via env `BOT_DISABLE_CHRONOS=1`) | `bot_config.ml.CHRONOS_MODEL_NAME` |
| **RL model (v8)** | `models/mr/best/ppo_mr_filter.zip` + `vec_normalize_mr.pkl` (auto fallback to `models/mr/`) | `SelfLearningAgent` (v8.0 path-aware) |
| **ML model (v8)** | `data/mr_signal_quality_model.pkl` (auto fallback to legacy SMC) | `SignalQualityModel` |
| **Pool (v8, training only)** | `data/mr_signal_pool_3000.pkl` (~309 MB, gitignored) | `MeanReversionBacktester` |
| FTMO program | 2-step Standard (no Consistency Rule → threshold = 1.0) | `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` |
| **Excel schema (v8.0.6)** | Trades 58 cols / Signals 20 cols (was 66/23) | `TradeLogger.TRADE_HEADERS/SIGNAL_HEADERS` |

## Verified Performance (v8.0.5 — MR pipeline, all gates passed, 5000 eps eval, 2026-05-07)

| Metric | v6.13 SMC baseline | **v8.0.5 MR** | Δ | Gate |
|--------|------------------:|--------------:|----|----:|
| **Pass Rate** | 9.7 % | **59.30 %** | **+49.6 pp** ⭐ | ≥ 8 % |
| Profitable Rate | n/a | **89.10 %** | — | ≥ 55 % |
| Breach Rate | 0 % | **0.00 %** | same | ≤ 5 % |
| Win Rate | 64.8 % | 61.55 % | -3 pp | (info) |
| Take Rate | n/a | 46.35 % | — | (info) |
| Total DD max | 4.40 % | **5.80 %** | + (under env 5.8% guard) | ≤ 6 % |
| Daily DD max | 2.15 % | **3.00 %** | + (under env 3.0% guard) | ≤ 3.5 % |
| Profit avg | +3.89 % | **+7.23 %** (+$7,229) | +3.34 pp | (info) |

**v8 progression** (5 sub-iterations from v8.0 → v8.0.5): pilot yield 4 sig/ep → relaxed BB/RSI to 14 sig/ep → ml_threshold 0.40→0.30 → DAILY_DD_GUARD 0.04→0.03 → TOTAL_DD_GUARD 0.085→0.058 → **all gates passed iter 1, 12 min**. Details in [wiki/05-invariants.md § Version Log](wiki/05-invariants.md).

**Audit certification (mandatory)**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py   # ✅ all clean
.venv/bin/python ftmo_trading_bot/scripts/parity_audit.py    # ✅ all aligned
```

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
├── strategy/                ← MeanReversionStrategy + LiveMRScanner + Indicators (SMC removed v8.0.6)
├── ml/                      ← SignalQualityModel, SelfLearningAgent, MeanReversionFilterEnv, MeanReversionBacktester, ChronosForecaster, AuxAware{PPO,Policy,Buffer}
├── core/                    ← RiskManager, MT5Connector, TimeManager, PositionSizer, NewsCalendarScheduler, DiscordNotifier
├── execution/               ← TradeExecutor, TradeManager
├── analytics/               ← PerformanceAnalyzer + TradeLogger (Excel: Trades 58 cols / Signals 20 cols / Daily / Stats, v8.0.6)
├── scripts/                 ← build_mr_signal_pool, train_mr_signal_quality, train_mr_signal_filter, auto_train_pipeline, leakage_audit, parity_audit, holdout_eval (v8.0.10), pipeline_status.sh
├── data/                    ← OHLCV CSVs + mr_signal_pool_3000.pkl (gitignored) + mr_signal_quality_model.pkl
├── models/mr/               ← ppo_mr_filter.zip + vec_normalize_mr.pkl + best/ snapshot (v8.0+)
└── logs/                    ← bot_state.json + ftmo_trades.xlsx + tensorboard + auto_train_pipeline.log
```

Full module details → [wiki/02-modules.md](wiki/02-modules.md).

---

## Quick Commands (v8.0+)

**Autonomous training (recommended)** — single command runs Build → GBM → RL → Eval → Self-correct loop:

```bash
.venv/bin/python ftmo_trading_bot/scripts/auto_train_pipeline.py \
    --max_iterations 10 --max_hours 60 \
    --pool_size 3000 --timesteps_p1 5000000 --timesteps_p2 2000000 \
    --target_pass_rate 0.08 --target_dd_max 0.06 \
    --target_daily_dd_max 0.035 --target_profitable 0.55
```

**Manual training (3 steps, in order)**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/build_mr_signal_pool.py --pool_size 3000 --workers 8
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_quality.py
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py --fresh \
    --timesteps_p1 5000000 --timesteps_p2 2000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.05 \
    --ml_threshold 0.30 --risk_per_trade 0.0099
```

Trainer auto-uses `AuxAwarePPO` + `AuxAwareACPolicy` (aux loss weight = 0.5).

**Evaluation**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_mr_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.30 --risk_per_trade 0.0099
```

**Anti-overfit holdout eval (v8.0.10, mandatory before promoting to live)**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/holdout_eval.py \
    --train_pool ftmo_trading_bot/data/mr_signal_pool_5000.pkl \
    --holdout_pool ftmo_trading_bot/data/mr_signal_pool_holdout.pkl \
    --n_episodes 2000
# Verdict: Δ Pass Rate ≤ 5 pp = HEALTHY, > 10 pp = OVERFIT (exit 1)
```

**Audits** (mandatory before commit):

```bash
.venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py    # exit 0 = no leakage
.venv/bin/python ftmo_trading_bot/scripts/parity_audit.py     # exit 0 = train↔live aligned
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
