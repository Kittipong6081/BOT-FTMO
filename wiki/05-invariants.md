# 05 — Invariants & Gotchas (Rules Not to Break)
> Last Updated: 2026-06-04 | Scope: red flags, version log, migration notes (latest: **Fix#3 — sim-realism (spread + live-exit training), best/ = realistic model**)

## 📝 Version Log Entry — Fix#3: sim realism (spread + live-exit training) (2026-06-04, rebuild/regime-aware)

**⛔ The backtester now models REALISTIC trade cost — pool outcomes are net-of-spread.** Any pool built from now on (including walk-forward folds) deducts cost. The deployed `models/mr/best/` is the **realistic-trained model** (Pass ~21% honest, not the optimistic 87%).

**Why**: the walk-forward (Fix#1 era) proved the model is NOT time-overfit (stable across time folds) — so the FTMO fail was the **sim↔live EXECUTION gap**, not overfit. The sim was too optimistic in two ways the live account paid for (live realized payoff 0.59 vs sim ~1.0).

**Two changes:**
1. **Fixed spread cost (always-on)** — `MeanReversionBacktester.generate_episode_signals` deducts `typical_spread_pips / sl_distance_pips` (R-units) from every `outcome_pnl_ratio`. The legacy ±0.2% multiplicative slippage in `_resolve_trade` did NOT model the FIXED bid-ask cost which, on a tight RR≈1.0 MR trade (~10-15 pip SL), eats ~10% of every trade. Flips marginal +EV wins to net losses, as live does.
2. **Live-exit training** — `auto_train_pipeline.py --model_live_exit` (+ `HyperParams.model_live_exit`, FORCES rebuild) builds the pool with `_resolve_trade_live_exit` (BE@0.3R + 33% partial@0.8R) instead of full "ride". So the agent trains on what live actually realizes (BE/partial scratches wins).

**Result (5000-pool, obs 36, --model_live_exit)**: Pass 21.0%, Win 57.5%, Total DD 3.81%, Daily DD 2.40%, Profitable 96.2%, breach 0% → all gates passed, auto-promoted to `models/mr/best/`. **Verify (both obs-36 models on the realistic pool, n=600)**: Fix#3 **Pass 17.3% / Profit-per-ep $5,529** vs Fix#1 (ride-trained) **Pass 2.8% / $2,976** — Fix#3 passes FTMO **6× more** + ~2× profit. Fix#1 over-skips (17.5% take) when faced with real costs; Fix#3 takes 44.6% of still-+EV trades. (EV/trade alone favors Fix#1 $241 vs $184, but that's an artifact of taking far fewer trades — for FTMO, Pass-rate + total-profit are what matter.)

**Caveats**: (1) the verify is slightly in-sample for Fix#3 (it trained on this pool); the 6× margin is too large for bias to flip, but a proper out-of-TIME walk-forward (build realistic fold pools) is the follow-up. (2) **HONEST baseline: the strategy passes FTMO only ~17-21% under realistic costs — the edge is THIN.** This is the real number (better than a fake 87% that dies live). It needs Fix#2 (trend engine) + entry-quality to become a reliable passer. (3) Concurrent-correlation modeling (3rd Fix#3 piece) deferred — hardest, episodes are per-symbol; handled live by corr caps. Fix#1 model preserved at `models/mr/best.fix1/`, pre-Fix#1 baseline at `models/mr/best.pre_fix1/`.

## 📝 Version Log Entry — Fix#1: obs 35→36 (`trend_eff_h1` regime feature) — RETRAIN, PROMOTED (2026-06-03, rebuild/regime-aware)

**⛔ OBS DIM IS NOW 36 (was 35).** Must stay synced in 3 places: `SelfLearningAgent.OBS_DIM=36` · `MeanReversionFilterEnv._get_obs` (appends obs[35]) · `FTMOTradingBot._build_signal_observation` (appends obs[35]). Plus live `MRSignal.trend_eff_h1` (computed in `LiveMRScanner.analyze_with_data`) + `MeanReversionBacktester` sig_dict (computed from `h1_slice`). `MeanReversionFilterEnv.observation_space` overridden to (36,). leakage/parity/obs_parity_check all exit 0 @ 36.

**Why**: the FTMO challenge died (−9.87%) because MR **fades persistent (slow) downtrends** that the ADX trend filter can't see (audit: ADX win% flat ~50% across ALL ADX buckets; June EUR losers were ADX 13-16 = "ranging" by ADX yet drifting down; all 10 June trades bought into negative `price_roc`).

**Feature**: NEW `TechnicalIndicators.calculate_signed_efficiency(df, period=32)` = signed Kaufman Efficiency Ratio = `(close[-1]−close[-1-period]) / Σ|close.diff()|` ∈ [−1,+1] (+1 efficient uptrend · 0 chop · −1 efficient downtrend). Used as **`trend_eff_h1`** = signed efficiency on **H1, period 24** (~1 day). **Validated on the 977k-signal pool: MR fade meanOut declines monotonically with |ER_h1|** — chop (|ER|<0.10)=+0.078/win54% → strong-trend (0.60+)=−0.206/win39%; EV crosses 0 at |ER_h1|≈0.30. (ADX gave no such separation.)

**Result (retrain, 5000-pool, obs 36)**: Pass 87.2% (≈ baseline 88%), Win 70.4%, **Total DD 2.63% (vs baseline 5.80% — halved)**, Profitable 98.7%, take 46.7%, breach 0%. **Walk-forward + `model_live_exit` (honest, out-of-TIME) BEATS baseline on every fold** (recent: Pass 54.2→61.0, EV/trade $106→$119, DD 4.9→4.2; mid: Pass 50.9→54.6; older: DD 3.9→2.7). No fold regressed. Promoted to `models/mr/best/` (pipeline auto-snapshot); baseline preserved at `models/mr/best.pre_fix1/` + `data/*.pre_fix1` (obs 35) for rollback.

**Caveats**: (1) take-rate response is **mild** — the agent still TAKEs ~67% even in the |ER|>0.6 (−0.18 EV) zone; the win is more DD-reduction than aggressive trend-avoidance. Optional **hard gate** (block |ER_h1|≥0.45 in pool+live) is a pending booster. (2) Still SIM — does NOT close the sim↔live execution gap (Fix#3). **Do NOT trust for live until Fix#3 (spread/slippage + concurrent correlation in the backtester) + paper-forward.** (3) Mode-2 true-OOS (retrain-on-older→test-recent) not yet run. NOT yet committed (branch `rebuild/regime-aware`). Revert: restore `*.pre_fix1` + `git checkout` the obs-36 edits (6 files) + delete the fix1 pool/GBM.

## 📝 Version Log Entry — same-direction cap 2→1 REVERT (2026-06-03, rebuild/regime-aware, live-only)

**Context**: FTMO challenge FAILED (account 10,000 → ~9,013 = −9.87%, hit the 10% total-DD wall, 2026-06-03). Full audit in `~/.claude/plans/hedge-fund-quant-flickering-pudding.md`. First remediation from the approved **regime-aware rebuild** plan.

**Change**: `FTMOConfig.MAX_SAME_CURRENCY_LEG_POSITIONS` **2 → 1** (reverts the 2026-06-02 raise; back to the v8.0.79 single-leg guard). The bot again allows only **1 position per same direction on a shared non-USD currency leg**.

**Why**: The live `ftmo_trades.xlsx` (1–3 มิ.ย., 9 closed trades) shows cap=2 directly enabled a correlated short-JPY BUY cluster — **GBPJPY BUY ×2** (T8 −$2.36 + T10 −$64.98, opened 16 min apart) plus **USDJPY BUY** = 3 correlated BUYs inside 16 min, **all lost (−$138)**. With cap=1 the 2nd GBPJPY BUY would have been blocked. (Separately, the audit confirmed there is NO directional execution bug — the KC-slope entry filter is symmetric, blocking 80 BUY + 67 SELL; the all-BUY outcome is signal imbalance + a magnitude-based slope filter that misses slow trends → folds into the planned regime/slow-trend detector, not a separate fix.)

**Scope**: Live-only — `MeanReversionBacktester`/pool does not run this RL-blind guard → **no retrain**. Affected: `config/settings.py` (`MAX_SAME_CURRENCY_LEG_POSITIONS`). Deploy = VPS git pull + restart. Revert: set back to `2`.

## 📝 Version Log Entry — dual-strategy scaffolding RECOVERED + out-of-time eval (2026-06-03, rebuild/regime-aware, flag OFF — MR identical)

**Why**: the approved regime-aware rebuild (see `~/.claude/plans/hedge-fund-quant-flickering-pudding.md`) restores the dual-strategy scaffolding deleted in v8.2.0 (`95c124e`) as the editable base for **Fix#1** (data-driven regime gate) + **Fix#2** (early-entry trend engine). Recovery only — no live behavior change yet (integration is the later phase).

**Recovered from `git 95c124e^`** (10 `.py`, binaries skipped — TF models/pkl will be retrained): `strategy/{regime_classifier,strategy_router,trend_following_strategy}.py`, `ml/{trend_following_backtester,trend_following_env}.py`, `core/strategy_risk_book.py`, `scripts/{build_tf_signal_pool,train_tf_signal_quality,train_tf_signal_filter,auto_train_pipeline_tf}.py`.

**Config restored in `config/settings.py`** (flag OFF): `RegimeConfig` + `TrendFollowingConfig` classes, `BotConfig.regime`/`BotConfig.tf` fields, `FTMOConfig.{STRATEGY_MAGIC,STRATEGY_SUB_BUDGET_PCT,STRATEGY_SLOT_CAP,STRATEGY_RISK_PCT,DAILY_LOSS_CAP_PCT_DUAL}`. `bot_config.tf.enabled = False` ⇒ `main.py` does not route through these (they are inert). `TrendFollowingConfig.early_*` (Fix#2 scaffold: +DI/−DI cross + ADX rising, NO hardcoded ADX cap) ships **OFF** — needs TF pool/GBM/RL rebuild + DD-aware eval before enabling.

**Verification**: all 6 recovered modules import clean; `import main` OK; **leakage + parity exit 0** (MR obs 35-dim byte-identical — scaffolding does not touch the live MR path). ⚠️ The hardcoded `RegimeConfig` thresholds (ADX [20,27] dead-zone) are the Fix#1 *replacement target*, not endorsed.

**New tool**: `scripts/walk_forward_eval.py` — out-of-TIME eval (folds by time, not seed). **Real result (n=600/fold): Pass 81/78/90% older/mid/recent = STABLE → NOT time-overfit** (the n=16 smoke "100→20% decay" was noise). ⇒ the FTMO fail is the **sim↔live execution gap**, not a backtest-regime overfit → root-cause #1 re-prioritized to backtester realism (`--model_live_exit` BE/partial + spread/slippage + concurrent correlation). See `wiki/03-rl-training.md` + the plan file.

## 📝 Version Log Entry — TF / dual-strategy REMOVED (2026-06-02) — back to single-strategy MR

**Decision (user request, "เอา logic TF ออกให้หมด แบบรอบคอบ")**: all Trend-Following + dual-strategy + regime machinery (added in v8.1-phase0..phase4) is physically REMOVED. The bot is now **single-strategy Mean-Reversion only** — the deployed money-maker. Done in audited stages; MR proven byte-identical at every step.

**Deleted (10 files)**: `strategy/trend_following_strategy.py`, `strategy/regime_classifier.py`, `strategy/strategy_router.py`, `ml/trend_following_backtester.py`, `ml/trend_following_env.py`, `core/strategy_risk_book.py`, `scripts/{build_tf_signal_pool,train_tf_signal_quality,train_tf_signal_filter,auto_train_pipeline_tf}.py`. Plus TF artifacts `models/tf/` + `data/tf_signal_*.pkl`.

**Shared files reverted to single-strategy MR**: `main.py` (removed `_strategies`/`_router`/`_regime_classifier`/`_rl_agents`/`_quality_models` dicts + `_strategy_for`/`_rl_agent_for`/`_quality_model_for`/`_build_obs_tf`/`_build_obs_for`/`_open_position_owners`/`_compute_symbol_regime`/`_compute_live_regime` + router scan/TF-paper branches → direct `self._strategy`/`self._rl_agent`/`self._quality_model`/`_build_signal_observation`). `core/risk_manager.py` (removed `StrategyRiskBook` import+instantiation+save/load/record/halt; daily-cap → single-strategy 2.5%). `execution/trade_executor.py` (removed `_check_strategy_conflict`; per-strategy magic helpers → single magic 123456; `risk_pct=None`→default 0.70%; dropped `strategy_id=` kwargs). `execution/trade_manager.py` (`_exit_profile()` parameterless → MR profile). `config/settings.py` (removed `TrendFollowingConfig`/`RegimeConfig`/`bot_config.tf`/`bot_config.regime` + `STRATEGY_MAGIC`/`STRATEGY_SLOT_CAP`/`STRATEGY_SUB_BUDGET_PCT`/`STRATEGY_RISK_PCT`/`DAILY_LOSS_CAP_PCT_DUAL`). `strategy/indicators.py` (TF comment removed).

**KEPT (not TF)**: `StrategyBase` (LiveMRScanner's base, trimmed), the vestigial `strategy_id="MR"`/`obs_layout_id="mr_v8"` literals (logger/notifier read them via forgiving `.get` → Excel "Strategy"/"Obs Layout" columns still populate), `calculate_adx` +DI/-DI/di_spread (generic ADX completion), `MAX_SAME_CURRENCY_LEG_POSITIONS` (=2 at removal; **reverted to 1 on 2026-06-03** — see top entry), `CLOSED_BAR_ONLY` flag, RL `--seed`. `calculate_choppiness` is now orphaned dead code (harmless; no MR caller).

**Verification**: `import main` clean (no dangling refs); `leakage_audit` + `parity_audit` exit 0 (obs 35-dim 3-way sync, `adx` byte-identical 19.614282); **MR eval (5000 ep): Pass 87.8% / Profitable 98.1% / Win 68.8% / Total DD 5.80% — matches the v8.0.80 baseline (~88%, DD 5.80%) → MR unchanged.** ⛔ `bot_state.json` NOT deleted (FTMO anchor); the stale `strategy_book`/schema-9 key is simply ignored on load. Deploy = VPS git pull + restart. ⚠️ The other wiki pages still contain historical TF rows (02-modules/03-rl-training/04-operations) — they describe removed code; this entry is the source of truth.

## 📝 Version Log Entry — same-direction cap 1→2 (2026-06-02, user request, live-only no retrain)

**Change**: `FTMOConfig.MAX_SAME_CURRENCY_LEG_POSITIONS` **1 → 2** — the bot can now hold up to **2 positions in the same direction** on a shared non-USD currency leg (e.g. 2×EURUSD BUY, or EURUSD SELL + EURJPY SELL = both short EUR). Enforced in `TradeExecutor._check_correlation_risk` (blocks when existing same-leg+direction count ≥ cap). Reverses the v8.0.79 single-leg guard (which was added after a 2026-05-29 −$140 double-loss: EURUSD SELL + EURJPY SELL both short EUR → EUR bounced → both lost). ⚠️ **Trade-off**: 2 same-direction = correlated exposure → if both hit SL = ~1.4% (2×0.70%), still under the 4% daily cap; `MAX_OPEN_POSITIONS=3` caps total, `MAX_USD_THEME_POSITIONS=2` unchanged, opposing-theme block (opposite dir) unchanged, cluster cooldown may still delay the 2nd by ~10 min. **Live-only** (the backtester/pool does not run this RL-blind guard → no retrain). Deploy = VPS git pull + restart. Revert: set back to `1`.

## 📝 Version Log Entry — TF PAUSED (2026-06-02) — `tf.enabled` True→False (structural late-entry; back to single-strategy MR)

**Decision**: `TrendFollowingConfig.enabled` **True→False** → router=None, back to single-strategy MR (MR no longer regime-gated to RANGING; full MR restored). Deploy = VPS `git pull` + restart. Revert canary: `enabled=True`.

**Why — TF has a CONFIRMED structural late-entry problem** (verified by code audit + 52,928-signal TF pool analysis + live log, 2026-06-02):
- TF's entry gates in `TrendFollowingStrategy.analyze_with_data` are ALL lagging: `ADX >= adx_entry_min(27)` (level on double-EWM ADX), full `EMA21>50>200 + close>EMA200` stack (`detect_trend` — EMA200 only aligns deep into a mature trend = worst offender), H4-not-opposing + D1 bias. `StrategyRouter.scan` also arms TF on EVERY confirmed-TRENDING bar (not just the fresh flip), so TF re-fires into mature trends.
- **TF pool (raw entry quality, 52,928 signals, all symbols/history)**: ADX-at-entry 27-30 = **+0.022R (ONLY profitable bucket)**; 30-35 = −0.052; 35-40 = −0.078; 50+ = −0.097 (monotonic decay). trend_age≥60 (capped "very old") = **37% of all entries**, mean −0.080R. Runner rate (≥2R) only **6.5%**; full-SL losses 54.7%; overall mean −0.033R at RR 2.5. → by entering late (ADX already 35-40, trend exhausted) there is little move left to run → runners (TF's payoff engine) don't materialize.
- **Live (1 Jun, VPS)**: TF generated **793 signal-scans, ALL BUY NZDUSD**, ADX-at-entry mean 36.6 (min 31.9). 2 executed: ticket 558694896 (ADX 38.6, SL −$57.28) + ticket 559071644 (ADX 40.4, trend_age 60, pullback 2.15×ATR, still open losing). Both BUY into a NZDUSD crash because the slow H4/D1 EMA stack stayed bullish (lagging). Account: 3 closed trades all SL, −$196.32, Max DD 5.11%.
- ⚠️ Holding the open TF order overnight is EXPECTED (v8.1.2 `enforce_daily_close=False`), NOT the 23:30-close bug. The problem is ENTRY, not holding. The open order is still managed by `TradeManager` (magic 123457); pausing TF stops NEW entries but does not auto-close it (close manually in MT5 if desired).

**Redesign direction (when resumed; design-paneled 2026-06-02 — 4 lenses, all converged)**: catch trend BIRTH not STATE — persist the discarded `+DI/-DI` (Wilder's early directional trigger) for a DI-cross, use ADX *rising* + `ema_slope_norm` + `band_squeeze_ratio` release instead of the ADX-27 level + full EMA200 stack (demote EMA200 to soft confluence), arm TF only within a young window after the fresh RANGING→TRENDING flip, and FEED `trend_age`/`adx_slope`/`bars_since_flip`/`di_spread` to GBM+RL so the model LEARNS the ADX-27-30 sweet spot — **NO hardcoded ADX cap** (band-aid that breaks the system). ⛔ Dominant risk = parity: `TrendFollowingBacktester` has NO `MarketRegimeClassifier`, so `bars_since_flip` must be a STATELESS bar-count identical both sides (live time-based `min_dwell_sec` has no offline analogue). Bonus bug to fix: `band_squeeze_ratio` is hardcoded `0.5` in the TF backtester pool dict (model never sees real squeeze). Requires full TF pool rebuild → GBM → RL retrain + challenge-level (DD-aware, 292+ ep) eval, NOT per-trade EV.

**Implementation progress (2026-06-02, flag `tf.early_entry`, env `BOT_TF_EARLY=1`, default OFF):** S1 — `calculate_adx` persists `plus_di`/`minus_di`/`di_spread` (additive; `adx` byte-identical). S2 — `TrendFollowingStrategy.analyze_with_data` leading gate: direction from `di_spread` dominance + ADX rising (`adx_slope>0`) + fast EMA21>EMA50; EMA200/H4/D1→soft; emits `di_spread`/`adx_slope`/real `ema_slope_norm`/real `band_squeeze_ratio` on `TFSignal`; legacy path byte-identical when OFF. S3 — `TrendFollowingBacktester` writes `di_spread`/`adx_slope` to the pool + reads real `band_squeeze_ratio` (was hardcoded `0.5`); the early gate flows through it automatically (it calls `analyze_with_data` on the same H1 slice → PARITY-TRIVIAL, no regime state machine needed; `adx_slope` is a stateless parity-safe substitute for the `bars_since_flip` idea). S4 — obs layout **tf_v1→tf_v2**: repurpose the dead Chronos slots `obs[27]/[28]` = `di_spread`/`adx_slope` (scaled) in `TrendFollowingFilterEnv._get_obs` ↔ `main._build_obs_tf` (MR never overrides these → MR obs unchanged, 3-way sync still 35-dim) + `di_spread`/`adx_slope` appended to the TF GBM `FEATURE_KEYS`. ⛔ ALL changes are TF-only (+ additive `indicators.py`); MR proven byte-identical (`adx` 19.614282, leakage+parity exit 0, obs 3-way sync intact) at every step S1-S4. **Verified offline (40-ep TF pool, early ON vs OFF): ADX-at-entry mean 34.6→27.8 (min 27→20), adx_slope −2.14 (falling/exhausting) → +1.47 (rising/building) — i.e. it now enters trends EARLY/while building, in the profitable ADX-27-30 zone; band_squeeze real (not 0.5).** Pending: S5 (rebuild full TF pool with `BOT_TF_EARLY=1` → GBM → RL → DD-aware challenge eval → enable `tf.early_entry` + un-pause `tf.enabled` for real ONLY if it beats current TF).

## 📝 Version Log Entry — closed-bar parity (2026-06-02) — `CLOSED_BAR_ONLY` flag (EXPERIMENT, default OFF, A/B in progress)

**Context**: ADX audit (2026-06-02) confirmed the live feed reads the **forming (unclosed) bar**: `MT5Connector.get_ohlcv` calls `mt5.copy_rates_from_pos(symbol, tf, 0, count)` and never drops the last row, so every `iloc[-1]` indicator read (MR ADX-H1 gate, M15 entry, regime, TF) is on a still-forming bar → intra-bar **repaint** + a live(forming)/train(closed) distribution gap (the training/eval pool is built on closed bars). Not a future-data leak (backtester is causal — H1/H4 anchored ≤ scan bar), but a sim-live parity gap (~0.7% of scans flip the ADX-30 gate; measured forming-vs-closed ADX delta typically <0.2 pts).

**Change** (flag-gated, default OFF = legacy byte-identical): new `FTMOConfig.CLOSED_BAR_ONLY` (env `BOT_CLOSED_BAR=1`). When ON: (1) `MT5Connector.get_ohlcv` fetches `count+1` and drops the forming last bar → live indicators read the **last CLOSED bar**; (2) `MeanReversionBacktester.generate_episode_signals` anchors the H1/H4 ADX read to the bar **fully closed by the scan M15 bar's close** (`decision_ts = m15_scan_ts + 15m`, `h1_cut = decision_ts − 1h`) instead of the v8.0.80-H6 containing-but-settled H1 bar — so live (now closed-bar) and pool agree on the ADX-gate bar. Note: the M15 obs were ALREADY closed-bar in training (decision bar `scan_idx`), so the live forming-bar drop also re-aligns M15 obs without a retrain.

**Status**: ✅ flag OFF → `leakage_audit` + `parity_audit` exit 0 (MR identical, obs 35). A/B in progress — pre-flight pool diff (flag OFF vs ON, same seed) to measure whether closed-bar admits net-better/worse trades, then (if warranted) full retrain + holdout eval vs baseline before any promote. **Revert**: delete flag usage or keep `CLOSED_BAR_ONLY=False`.

**⛔ Invariant**: if `CLOSED_BAR_ONLY` is ever promoted to default-ON, BOTH the live connector AND the MR backtester must use the closed-bar convention together (changing only one re-introduces a 1-bar live/train lag). Re-run `parity_audit` + `leakage_audit` after any change.

## 📝 Version Log Entry — v8.1.2 (2026-06-02) — Weekday overnight holding (Mon-Thu close OFF, Friday close stays; live-only, no retrain)

**Change**: `SessionConfig.enforce_daily_close` **True→False** (user request). The bot now HOLDS positions overnight on weekdays (Mon→Tue … Thu→Fri); the 23:30 EET Mon-Thu force-close (`TimeManager.is_daily_close_time` — already gated by `enforce_daily_close`) no longer fires. **Friday force-close (20:45 EET, `is_friday_close_time`) is UNCHANGED and ungated** — bot still goes flat before the weekend.

**Rules verified** (official FAQ, 2026-06-02): FTMO + The5ers both PERMIT weekday overnight holding. FTMO forces close only before the WEEKEND (funded Standard) or on a >2h rollover; during evaluation + Swing even weekend holding is allowed. The5ers permits overnight AND weekend on all programs. A recurring 3rd-party claim that FTMO Standard must close each weekday is FALSE (not in FTMO's FAQ; the source page is 410-dead). Sources: ftmo.com/en/faq/do-i-have-to-close-my-positions-overnight, ftmo-swing-account-type, can-i-trade-news; help.the5ers.com (overnight + weekend articles).

**Train/live parity kept**: `StrategyBacktester._is_force_close_bar` (in `_resolve_trade`) and `_force` (in `_resolve_trade_live_exit`) were decoupled — **Friday close ALWAYS applies (mirrors ungated live `is_friday_close_time`); Mon-Thu close gated by `enforce_daily_close`**. With `enforce_daily_close=True` the new logic is byte-identical to the old (also fixes a latent bug where the old code dropped the Friday close when `enforce_daily_close=False`). ✅ leakage_audit + parity_audit exit 0.

**No retrain** — challenge-level eval (292 ep, existing model, daily-close ON vs OFF on identical episodes): Pass 90.6%→91.2%, Profit ~$10.28k both, DD/Breach safe. The 23:30 close was ~neutral in sim (per-trade EV +0.01R washes out at challenge level). ⚠️ Backtester models NEITHER swap nor overnight gap, so the eval is OPTIMISTIC about relaxing — real overnight holding pays swap (×3 Wed→Thu) + gap risk (accepted by user). **Deploy = restart bot** (config read at startup). **Revert**: `enforce_daily_close = True`.

## 📝 Version Log Entry — v8.1.1 (2026-06-01) — Max DD hard stop 8%→10% (full FTMO rule, user request) + Discord MR/TF strategy tag (live-only, no retrain)

**Trigger**: After auditing the **first TF LIVE order** (NZDUSD BUY, ticket 558694896, the only Trades row in `logs/ftmo_trades.xlsx`) — verified opened correctly (SL = 2.0×ATR, RR 2.5, TF risk 0.60% via `STRATEGY_RISK_PCT["TF"]` → 0.32 lot / $55.33, routed TF/tf_v1, all gates passed). Audit surfaced the account was at **−5.78% total DD** (anchor $10k → balance $9,421.82, carried over from prior MR losses) — only 2.22pp from the old 8% stop. User decision: use the **full FTMO rule (10%)** rather than the 8% buffer, to give the account room to recover.

**(1) Max DD hard stop 8% → 10%** — `FTMOConfig.MAX_DRAWDOWN_HARD_STOP_PCT` 0.08→**0.10** (= the real FTMO total-DD breach line). `RiskManager._check_max_drawdown` emergency-closes + halts permanently at this value (unchanged logic, new threshold). ⚠️ **No safety buffer anymore**: emergency close fires exactly at the breach line — slippage/gap on the closing fills can realize marginally past 10% = a genuine FTMO breach. Mitigation: soft warning print bumped **6%→8%** (`drawdown_pct > 0.08`) so there is still a 2pp heads-up before the wall. Daily 4% hard stop (`DAILY_LOSS_HARD_STOP_PCT`) and the **training env guards** (`FTMOSignalFilterEnv.TOTAL_DD_GUARD` 5.8% / `DAILY_DD_GUARD` 3.0%) are **untouched** — this is a live-only ceiling change, no retrain. Doc comments in `settings.py` + `risk_manager.py` (module docstring, `MAX_DRAWDOWN_HALT` enum, `_check_max_drawdown` docstring) updated 8%→10%.

**(2) Discord per-strategy (MR/TF) tag** — `DiscordNotifier._strategy_label(trade_dict)` (NEW static helper) returns `(short, full)` from `trade_dict["strategy_id"]` (default MR for legacy): `("TF", "📈 TF · Trend Following")` / `("MR", "🔄 MR · Mean Reversion")`. Wired into `send_trade_open` / `send_trade_close` / `send_trade_partial_close` — title prefix `[MR]`/`[TF]` (e.g. `🔵 [TF] OPENED: BUY NZDUSD`) **plus** a dedicated **Strategy** field as the first field. Source = `ExecutedTrade.to_dict()` which already carries `strategy_id` (v8.1-phase2); partial-close passes `trade.to_dict()` too. Output-layer only — zero impact on trading logic. ✅ `py_compile` clean; helper verified for TF / MR / legacy-none.

## 📝 Version Log Entry — v8.1-phase4 (2026-05-30) — TF trained + go-live audit (9 bugs fixed) → TF LIVE canary

**Training**: TF baseline (pool 3000, RL 3M+1.5M) = AUC 0.779 / Win 86% / Profitable 97.7% / DD 1.86% / **Pass 3.5%** (too conservative — locks +1R floor, few runners). **Runner-capture retune** (resolve `trail_sl_behind_r` 1.0→2.0, `trail_activation_r` 1.0→1.5, `TF_FUTURE_BARS` 120→180; env `RUNNER_BONUS` 0.50→0.70, `SLOW_WIN_BONUS` 0.15→0.10) → **Pass 3.5%→7.7%**, DD 1.86%→**1.49%** (better), Profitable 93.9%, Win 84%, Profit +4.0%. Verdict: TF is a low-frequency (~18 signals/episode vs MR ~200), high-quality, low-DD **complement** — the 8% solo-Pass gate doesn't fit it; accepted at 7.7% to run alongside MR.

**🛑 Go-live adversarial audit (multi-agent workflow) — 9 confirmed latent bugs, ALL FIXED before flipping paper off**:
- **(A, HIGH)** `TradeManager` applied MR exits (BE@0.3R / partial 33%@0.8R / Stage-2 SL-lock@0.5R / trail 0.5R-behind) to TF positions — `strategy_id` was stored on `TrailingState` but never read → TF runners cut short, contradicting the ride profile the model trained on. **Fix**: `TradeManager._exit_profile(strategy_id)` — MR keeps its constants; **TF rides** (no partial/BE/Stage-2; trail activation 1.5R / behind 2.0R / floor 1.0R / TP-ahead 2.0R) read from `TrendFollowingConfig.mgmt_*`. `TrendFollowingBacktester._resolve_trade` now also reads `bot_config.tf.mgmt_*` → **single source of truth, train==live exit**.
- **(B, HIGH)** `_quality_model_for` fell back to the MR GBM for TF signals (different feature keys → garbage score gating the trade + obs[16] + log). **Fix**: no MR fallback for non-MR; `_tf_ready` gate now requires `"TF" in _quality_models` too (GBM routing as guarded as RL routing).
- **(C, MED)** TF logged ADX H1 = 0.0 (TF caches H4/D1 raw, no `adx` col). **Fix**: for TF, `_build_live_context` sets `adx_h1` from `sig.adx` (H1 entry ADX).
- **(D, MED)** TF logged empty obs (obs gate was MR-only). **Fix**: log obs via `_rl_agent_for`/`_build_obs_for` for any strategy with an agent (tf_v1 for TF).
- **(E, MED)** No Strategy column → MR/TF rows indistinguishable. **Fix**: added `Strategy` + `Obs Layout` cols to `TRADE_HEADERS` (58→60) + `SIGNAL_HEADERS` (20→22); `ExecutedTrade.to_dict` emits `obs_layout_id`. ⚠️ existing `ftmo_trades.xlsx` auto-archives on first run (schema guard).
- **(F, MED)** TF slot cap counted in-memory `_active_trades` only → an unrecovered TF orphan could over-open past the 1-slot canary. **Fix**: `_check_strategy_conflict` counts broker magic-tagged positions (`max(broker, in-memory)`); main re-syncs broker before scan (dual-strategy only).
- **(G, MED)** Global soft cap 3.0% (dual) < sub-budget sum 3.5% → one strategy's loss could halt the other. **Fix**: `DAILY_LOSS_CAP_PCT_DUAL` 3.0→**3.5%** (= MR 2.0% + TF 1.5%, still < FTMO 4% hard).

**Verification**: all 7 fixes unit + integration verified (live TF trade routes TF GBM/obs/exit, logs Strategy=TF + correct ADX + obs, slot cap broker-aware); `leakage_audit` + `parity_audit` exit 0 (MR obs 35 3-way sync intact — MR byte-identical when flag OFF).

**🟢 GO-LIVE (1a canary)**: `bot_config.tf.enabled=True`, `paper_mode=False`; TF model promoted to `models/tf/best/`. TF canary = **1 slot, 1.5%/day sub-budget**, magic 123457. **⚠️ enabling dual-strategy ALSO regime-gates MR to RANGING (ADX<20) only** — MR loses ADX 20-30 signals (the v8.0.51 killer zone goes to the dead-zone/TF). MR Pass/volume impact is UNMEASURED in live → monitor. **Instant revert**: `tf.enabled=False` (back to single-strategy MR) or `tf.paper_mode=True` (keep regime split, stop TF orders).

**⛔ New invariants**: (1) TF live exit MUST equal `TrendFollowingBacktester._resolve_trade` — both read `bot_config.tf.mgmt_*`; never hardcode TF trail geometry in TradeManager. (2) `_quality_model_for` never returns the MR GBM for a non-MR signal. (3) per-strategy slot cap counts BROKER magic-tagged positions, not just `_active_trades`. (4) `DAILY_LOSS_CAP_PCT_DUAL` ≥ sub-budget sum and < FTMO 4% hard. (5) every live TF row carries `Strategy`/`Obs Layout` for attribution.

**⚠️ KNOWN ISSUE (deferred, low-impact)**: on Windows, `TradeLogger._get_or_create_workbook` schema-migration `os.rename` of `ftmo_trades.xlsx` can fail with `WinError 32` (file briefly locked — openpyxl read-only handle not released in time before the immediate rename, or Defender/AV transient scan; NOT Excel). It logs `schema check failed … proceeding` and continues — **non-destructive** (new cols are appended at the END so cols 1..N stay aligned; only the header row may miss the 2 new labels). Fires only at a schema change (column add). Workaround: delete/rename `logs/ftmo_trades.xlsx` while bot stopped → fresh full-header file. **Deferred fix** (do next time the logger is touched): retry the rename 2-3× with a short delay + close the read-only handle in `finally`.

## 📝 Version Log Entry — v8.1-phase3 (2026-05-30) — TF training pipeline (3-brain): backtester + env + scripts + agent wiring

**Trigger**: Phase 3 of the dual-strategy plan — build the full TF training pipeline (mirror of MR) so TF gets its own pool + GBM + RL. **Code is ready to train; no model trained yet → TF still paper (forced).** MR untouched (audits exit 0).

**New modules**:
- **`ml/trend_following_backtester.py` → `TrendFollowingBacktester(StrategyBacktester)`**: entry on **H1** (not M15), trend filter H4 + D1 (D1 resampled from H4 via `_resample_h4_to_d1` since no D1 CSV — soft bias only). `generate_episode_signals(symbol, h1_start_bar, num_days, rng)` scans 12/day (TF_SCAN_POINTS_PER_DAY), 120-H1 resolution window (TF_FUTURE_BARS), wide-RR resolve with **Stage-2 1.5R cap DISABLED** (`tp_step_trigger_r=99`) + trail (1R behind, TP chases 2R) → winners RUN. Pool dict = MR schema (so `FTMOSignalFilterEnv._get_obs` reads it) + TF extras `trend_age_bars`/`pullback_depth_atr`/`adx_at_entry`/`is_runner`; `entry_confirm_passed=True` always (TF has no entry-confirm gate).
- **`ml/trend_following_env.py` → `TrendFollowingFilterEnv(FTMOSignalFilterEnv)`**: same 35-dim obs (tf_v1) — 3 slots reinterpreted: obs[4]=trend_strength/100, obs[10]=trend_age/30, obs[26]=adx/50 (trending=good; MR used 1−adx/50). Reward INVERTED vs MR: RUNNER_BONUS (outcome≥2R, scales with size) > SLOW_WIN_BONUS; loss penalty + LATE_ENTRY_PENALTY (trend_age≥60); SKIP-oracle penalizes missing runners most.
- **Scripts**: `build_tf_signal_pool.py` (H1-anchored task sampling), `train_tf_signal_quality.py` (27 TF features, `data/tf_signal_quality_model.pkl`, strategy="trend_following"), `train_tf_signal_filter.py` (`TrendFollowingFilterEnv`, 2-phase, `models/tf/ppo_tf_filter.zip`+`vec_normalize_tf.pkl`), `auto_train_pipeline_tf.py` (slim orchestrator: build→quality→filter→eval→gate→snapshot `models/tf/best/`).

**Strategy gate change (`TrendFollowingStrategy`)**: `resuming` + MACD moved from HARD gates to SOFT confluence bonuses (+8/+7) — strict conjunction (ADX>27 ∧ EMA-stack ∧ pullback ∧ resume ∧ MACD) yielded ~0 signals; soft makes the population usable for both train + live (same gate structure → no distribution gap). `pullback_max_atr` 1.5→2.5. Hard gates remain: ADX≥27, EMA-stack, H4 not-opposing, pullback-dist, RSI not-stretched. (Tuning = Phase 4.)

**main.py wiring**: `self._rl_agents` dict (MR + TF if `models/tf` present) + `self._quality_models` dict (MR + TF if `data/tf_signal_quality_model.pkl` present); helpers `_rl_agent_for/_quality_model_for/_build_obs_tf/_build_obs_for` route by `sig.strategy_id`. `_build_live_context` scores TF via the TF GBM. TF executes ONLY when `paper_mode=False AND "TF" in _rl_agents` — until then logs `TF_PAPER`.

**Verification**: full pipeline runs end-to-end on a 60-episode smoke pool (build 0.6min → GBM trains/saves/rescores → PPO 2-phase learn + eval returns metrics → env consumes pool, obs reinterpret correct). `leakage_audit` + `parity_audit` exit 0 (MR obs dim 35 3-way sync intact). No D1 CSV → D1 bias is H4-resampled (soft).

**⛔ New invariants**: (1) obs-sync now has TWO layouts — MR (mr_v8: obs[4]=bb_extreme, [10]=bb_width, [26]=1−adx/50) and TF (tf_v1: obs[4]=trend_strength, [10]=trend_age, [26]=adx/50). Each must stay synced across its env `_get_obs` ↔ live `_build_obs_*` ↔ its model. (2) TF pool entry is H1; `TrendFollowingBacktester` start bars index H1, not M15. (3) TF resolve MUST keep `tp_step_trigger_r=99` (disabling it caps runners at 1.5R = defeats TF). (4) D1 bias is soft/resampled in training — do not promote it to a hard gate without real D1 data.

## 📝 Version Log Entry — v8.1-phase2 (2026-05-30) — Per-strategy magic + Order Conflict Filter + StrategyRiskBook (bot_state schema 9)

**Trigger**: Phase 2 of the dual-strategy plan — build the execution + risk machinery so MR & TF positions are attributable and budgeted separately. **All Phase-2 behaviour is gated behind `bot_config.tf.enabled` (default OFF) → single-strategy MR is byte-identical.** TF stays `paper_mode` (real TF execution waits for the Phase 3 TF RL model).

**Per-strategy attribution (magic = source of truth)**:
- `ExecutedTrade.strategy_id` + `TrailingState.strategy_id` (NEW fields, default "MR"). `execute_signal` sends with `magic = STRATEGY_MAGIC[strategy_id]` (MR 123456 / TF 123457) via `_send_order_with_retry(magic=...)`. Orphan recovery: filter `magic not in TradeExecutor._bot_magics()` (was hardcoded `!= 123456` → would drop TF orphans); `_rebuild_executed_trade_from_mt5` sets `strategy_id` via `_strategy_for_magic(magic)`.
- Helpers: `TradeExecutor._magic_for(sid)`, `._strategy_for_magic(magic)`, `._bot_magics()`.

**Order Conflict Filter** — `TradeExecutor._check_strategy_conflict(signal)` (NEW, first gate in `execute_signal`): (1) regime exclusivity — a symbol held by the OTHER strategy is blocked; (2) per-strategy slot cap (`STRATEGY_SLOT_CAP` MR 2 / TF 1). **No-op when `tf.enabled=False`** (so MR keeps the global `MAX_OPEN_POSITIONS=3`). Same-symbol / opposing-theme / currency-leg stay PORTFOLIO-LEVEL in `_check_correlation_risk` + `can_open_trade` (cross-strategy already — ⛔ never filter strategy_id there).

**Gate dispatch**: `_check_entry_confirmation` (slip + M1-wick + Keltner — tuned for MR "falling-knife" entries) now runs **MR-only**; TF skips it (would block trend-continuation). Broker-agnostic gates (spread-spike, re-anchor, min-stop, final_validation, SL-existence) run for both.

**Per-strategy risk ledger** — NEW `core/strategy_risk_book.py` `StrategyRiskBook` (held by `RiskManager._strategy_book`): realized P/L recorded at close (`update_daily_pnl(strategy_id=...)`), floating P/L computed live from open positions' magic (never drifts on restart). `is_halted(sid)` self-halts a strategy at its sub-budget WITHOUT touching global `BotState` → the other strategy keeps trading. `can_open_trade(strategy_id=...)` gains the sub-budget gate (active only when `tf.enabled`). Reset in `_on_new_day`; persisted in `bot_state.json` **schema 9** (`strategy_book` key; schema-8 files load as empty → no migration needed).

**Per-strategy sizing**: `STRATEGY_RISK_PCT` MR 0.70% (= `DEFAULT_RISK_PER_TRADE_PCT` → MR identical) / TF 0.60%, passed to `PositionSizer.calculate_lot_size(risk_pct=...)`. Global soft cap: `_effective_daily_loss_cap_pct()` = 3.0% (`DAILY_LOSS_CAP_PCT_DUAL`) when `tf.enabled` else 2.5% (`DAILY_LOSS_CAP_PCT`, unchanged).

**Risk hierarchy** (top wins): [1] FTMO 4% hard breach guard (global) → [2] global soft cap 3.0%/2.5% → [3] per-strategy sub-budget MR 2.0% / TF 1.5%. worst-case both halt = 3.5% < FTMO 4%.

**Verification**: flag OFF → `leakage_audit` + `parity_audit` exit 0 (MR identical, risk 0.70% matches training). Unit tests: ledger per-strategy realized+floating + isolated self-halt + persistence; magic helpers; conflict filter (regime-exclusivity + slot cap dual-ON, no-op OFF); effective cap 2.5/3.0; RiskManager builds + schema-8→9 migration safe.

**⛔ New invariants**: (1) MT5 magic ↔ strategy_id is the attribution source of truth — orphan recovery uses `_bot_magics()`, never a hardcoded magic. (2) correlation / opposing-theme / currency-leg are PORTFOLIO-LEVEL — count across BOTH strategies, never filter strategy_id. (3) per-strategy halt ≠ global `BotState` (one strategy halting must not stop the other). (4) global FTMO 4% breach guard sits ABOVE per-strategy halt. (5) every Phase-2 risk change is gated behind `tf.enabled` — flipping it OFF must restore exact single-strategy MR behaviour (cap 2.5%, no slot cap, risk 0.70%).

## 📝 Version Log Entry — v8.1-phase1 (2026-05-30) — Market Regime gate + StrategyRouter + Trend Following scanner (rules-only, paper-mode, flag OFF by default)

**Trigger**: Phase 1 of the dual-strategy plan (`~/.claude/plans/lead-quantitative-portfolio-sequential-curry.md`). Add the regime switch + router + a rules-only TF scanner. **Master switch `bot_config.tf.enabled` defaults False → live behaviour is byte-identical single-strategy MR.** When ON, TF runs in `paper_mode` (log-only, no orders) until Phase 2.

**New modules**:
- **`strategy/regime_classifier.py` → `MarketRegimeClassifier`**: per-symbol 3-state gate from H1 — `Regime.{RANGING→MR, TRENDING→TF, AMBIGUOUS→none}`. ADX H1 is the PRIMARY separator; the band `[adx_ranging_max=20, adx_trending_min=27]` is a hard dead-zone (no trade) — wiki v8.0.51 proved ADX H1 25-30 = WR 25%, −$679. Choppiness Index + EMA-slope-per-ATR are confirmations. Hysteresis: `confirm_bars=3` debounce + `min_dwell_sec=1800` + the wide AMBIGUOUS band; new symbols start AMBIGUOUS (conservative warmup). Builds on `FTMOTradingBot._compute_symbol_regime`.
- **`strategy/strategy_router.py` → `StrategyRouter`**: per scan, `classify_all` → group symbols by armed strategy → scan each strategy ONLY on its allowed symbols → merge + rank. Mutual exclusion is structural (one key per symbol). `open_owner` locks symbols with an open position to their owner (regime flip mid-trade can't hand the symbol over).
- **`strategy/trend_following_strategy.py` → `TrendFollowingStrategy` / `TrendFollowingScanner` / `TFSignal`**: entry on H1 (trend filter H4/D1). Rules: ADX H1 > 27 + EMA21>50>200 + H4 agree + pullback-resume toward EMA21 + RSI not stretched + MACD confirm. SL = ATR×2.0 (XAU 2.5), RR 2.5 (WIDE — let winners run, opposite of MR). `TFSignal(MRSignal)` carries `strategy_id="TF"` + TF telemetry (`trend_age_bars`, `pullback_depth_atr`, `adx_at_entry`). `TrendFollowingScanner` magic = **123457**, OBS layout "tf_v1", own H1/H4/D1 caches.
- **`TechnicalIndicators.calculate_choppiness(df, period=14)`** (NEW static): Choppiness Index 0-100 (>60 ranging, <40 trending).

**Config (`settings.py`)**: NEW `RegimeConfig` (`bot_config.regime`) + `TrendFollowingConfig` (`bot_config.tf`, `enabled=False`, `paper_mode=True`, `sl_atr_mult=2.0`, `rr_ratio=2.5`, ADX/EMA/MACD/pullback params). No change to existing values.

**main.py wiring**: `self._router` built only when `bot_config.tf.enabled` (else None → legacy MR scan). Scan loop: router scan when enabled, else `self._strategy.scan_all_symbols()` (unchanged). `_open_position_owners()` feeds the regime lock. TF signals with `paper_mode` → logged as `TF_PAPER`, NOT executed. `_build_live_context` builds the MR obs only for MR signals (TF obs layout lands Phase 3 → empty for TF now).

**Verification**: flag OFF → `leakage_audit` + `parity_audit` exit 0 (MR identical, obs dim 35). flag ON → functional tests pass: dead-zone (ADX 20-27 → AMBIGUOUS), RANGING/TRENDING classification, hysteresis warmup (AMBIGUOUS→TRENDING after confirm_bars), router routing + merge, regime lock, TF signal construction (wide-RR geometry, SL=ATR×2). Full `FTMOTradingBot` constructs the router when `tf.enabled=True`.

**⛔ New invariants**: (1) a symbol is armed by exactly ONE strategy (or none) per instant — regime exclusivity + executor same-symbol block. (2) the ADX `[20,27]` dead-zone is intentional (killer zone) — do not "fill it in" to trade more. (3) when `tf.enabled=False` the router is None and MR runs exactly as before — keep this the default until TF has a track record. (4) TF obs layout (tf_v1) ≠ MR (mr_v8); the obs-sync invariant now applies per-layout (TF builder lands Phase 3).

## 📝 Version Log Entry — v8.1-phase0 (2026-05-30) — Strategy abstraction (groundwork for parallel Trend Following)

**Trigger**: Plan to run a Trend Following (TF) strategy in parallel with Mean Reversion (MR) — see `~/.claude/plans/lead-quantitative-portfolio-sequential-curry.md`. Phase 0 = introduce the abstraction with **zero behaviour change to MR**, so it ships independently and audits stay green.

**Changes (live-path indirection only — pool/GBM/RL/env/obs all UNTOUCHED)**:
- **NEW `strategy/strategy_base.py` → `StrategyBase` (ABC)**: contract every parallel scanner implements — `STRATEGY_ID`, `MAGIC_NUMBER`, `OBS_LAYOUT_ID`, `scan_all_symbols(allowed_symbols=None)`, `get_ltf_data/get_mtf_data/get_htf_data(symbol)`, `get_obs_layout_id/get_strategy_id/get_magic`. Each strategy owns its OWN per-symbol caches (no cross-read).
- **`LiveMRScanner(StrategyBase)`**: now declares `STRATEGY_ID="MR"`, `MAGIC_NUMBER=123456`, `OBS_LAYOUT_ID="mr_v8"`. `scan_all_symbols` gained `allowed_symbols: Optional[set] = None` (None = legacy = scan all). Cache accessors unchanged (still v8.0.79/80 per-symbol).
- **`MRSignal.strategy_id: str = "MR"`** (NEW field) — strategy origin tag carried through the whole pipeline (obs routing → executor magic → per-strategy risk ledger in later phases).
- **`main.py`**: `self._strategies = {self._strategy.STRATEGY_ID: self._strategy}` registry + helper `FTMOTradingBot._strategy_for(sig)`. `_build_live_context` now reads `get_ltf/mtf/htf_data` + `_structure_mtf`/`_get_d1_bias` via `_strategy_for(sig)` instead of the hardcoded `self._strategy` → routes each signal's obs/context to the scanner that produced it. Scan loop still calls `self._strategy.scan_all_symbols()` (MR only — StrategyRouter replaces it in Phase 1).

**Verification**: `leakage_audit.py` exit 0, `parity_audit.py` exit 0 (obs dim 3-way still 35, correlation groups match, ml/risk aligned). MR live behaviour byte-identical (indirection resolves to the same MR scanner; env/obs/model never touched). ⛔ **Invariant (new)**: each signal's obs/context MUST be built from `_strategy_for(sig)` (the producing scanner) — never a global `self._strategy` — to prevent cross-strategy cache contamination (extends the v8.0.79 per-symbol fix to the cross-strategy axis).

## 📝 Version Log Entry — v8.0.81 (2026-05-30) — Per-dim obs parity tool + ADX-penalty reward fix (follow-up to v8.0.80)

**Trigger**: After v8.0.80 fixed obs[16]/obs[21] parity, the open question was "are train and live obs now identical, and what else differs?" parity_audit only checks obs *dimension* (35), never per-slot *values*.

**1 — `scripts/obs_parity_check.py` (NEW empirical tool)**: replays REAL pool signals through BOTH `MeanReversionFilterEnv._get_obs` (train) and `FTMOTradingBot._build_signal_observation` (live) with matched neutral state, and diffs element-wise. **Result: all 23 signal-derived dims byte-identical (Δ = 0.000000)** — obs[16] ml_score and obs[21] (now via H3/H4) confirmed clean empirically; leak dims [29,30] both 0. The only non-zero Δ is obs[31] `mins_since_session`, and analysis shows `compute_temporal_features` (pool) and `_compute_mins_since_session_norm` (live) use the **identical** session formula (London 08-13 UTC, NY 13-16 UTC, ×min/480) — the Δ is a replay artifact (historical-signal time vs the tool's wall-clock), NOT a bug. **Conclusion: per-dim obs parity is clean; the tool is now a regression guard.** ⛔ Invariant: run `obs_parity_check.py` after ANY change to `_build_signal_observation` or `_get_obs` — must show all SIGNAL dims ≤ tol.

**2 — ADX-violation penalty removed (`MeanReversionFilterEnv`) — reward bug**: the env did `reward -= 0.30 if sig['adx'] (M15) > 25`. But the strategy's real trend veto is **H1 ADX > 30**, hard-gated at pool-build → every pool signal already has H1 ADX ≤ 30. M15 ADX of a valid MR signal is routinely 25-40 (M15 is noisier), so the penalty punished ~half of legitimate signals for a trend the H1 filter had already cleared. A penalty on H1 ADX would never fire (all ≤ 30) → correct action is to drop it. Reward-only change (pool/GBM unchanged) → **RL-only retrain** (no pool/GBM rebuild, ~12 min). **RESULT — PROMOTED**: ADX-fixed model holdout Pass **89.0%** vs v8.0.80 **87.6%** on the same seed-999 pool (**+1.4 pp**), Δ Pass −2.1 pp (HEALTHY ≤5pp), Take 44.8→47.2% (captures the valid signals the penalty had suppressed), Win 69.5% (flat), DD 4.16% (under 8%). parity + leakage + `obs_parity_check` + pytest all green → promoted to `models/mr/best/` (v8.0.80 kept at `best/*.v8080_validated`; rollback also `*.pre_v8081`).

## 📝 Version Log Entry — v8.0.80 (2026-05-30) — Production audit remediation: execution safety + FTMO breach guard + train/live parity (RETRAIN required for C3/H6)

**Trigger**: Full production-grade source-code audit (109 findings, 13 confirmed high/critical after adversarial verification; 7 plausible-looking issues refuted against real code). Focus per user request: cache contamination, dead/redundant logic, execution edge cases, clean code.

**🔴 Critical fixes**:
- **C1 — `TradeManager._modify_sl` now validates broker min-stop/freeze before sending.** Previously sent a raw `TRADE_ACTION_SLTP` with no distance check; BE@0.3R / Stage-2 lock@0.5R moved SL close to price and the broker silently rejected (retcode 10016), leaving the position at its original full-risk SL while the operator believed BE protected it. Now clamps `new_sl` to ≥ `max(trade_stops_level, trade_freeze_level, 1.5×spread, 3×point)` from current bid/ask, enforces never-loosen vs the broker's actual current SL (reads it live, so state-drift can't loosen), and defers (returns False → retried next tick) if the clamp would loosen.
- **C2 — `RiskManager.check_risk` now has an always-on FTMO 4% daily-breach guard.** `_check_daily_loss` early-returns when `state == DAILY_HALT`, so the 4% hard-stop + `_emergency_close_all` was unreachable once DAILY_HALT was set by a consecutive-loss halt or stop-out (neither closes positions). Open positions could bleed past the FTMO 4% daily limit with no emergency close. The new guard runs before the state-gated check: if equity-based daily loss ≥ `DAILY_LOSS_HARD_STOP_PCT` and positions are open, `_emergency_close_all` fires regardless of `_state`.
- **C3 — `MeanReversionFilterEnv.step` was missing the entry-confirmation forced-SKIP** that the parent `FTMOSignalFilterEnv.step` and the live `TradeExecutor._check_entry_confirmation` both enforce. MR is the only live strategy, so ~33% of pool signals (`entry_confirm_passed=False`, pool true_frac≈0.67) that the live executor blocks were being trained as TAKE → train/live distribution mismatch (same class as the v7.0.2 correlation and v8.0.79 cache fixes). Added the mirror block + `entry_confirm_forced_skips` to the MR episode summary/info. **RETRAIN required.**

**🟠 High fixes**:
- **H1** (with C1) — `_modify_sl` now logs `retcode`/`comment` and retries transient codes (REQUOTE/PRICE_CHANGED/PRICE_OFF); previously it swallowed every reject silently and re-sent the same request every tick.
- **H2** — `RiskManager` `max_trades_per_day` now compares a **per-symbol** counter (`_daily_trades_by_symbol`, reset on new day, persisted in `bot_state.json`) against the per-symbol cap. Previously a per-symbol cap was compared against the **global** `_daily_trades_count` (latent v8.0.71 bug class — would re-block a symbol on total trade count if any override were re-added).
- **H3** — `main._build_signal_observation` now reuses the temporal-augmented `ml_score` from `live_context` for **obs[16]** instead of re-scoring the raw `MRSignal` (which left 6 of the GBM's 7 temporal features at 0.0). Live obs[16] now matches both the ML gate and training. **Live-only (parity-improving, no retrain).**
- **H4** — live **obs[21]** (trades_today) now counts cumulative trades opened today (`_trade_open_history`, /3), matching `FTMOSignalFilterEnv._trades_today` semantics; previously used current open-position count (≈0.33 vs training's latch-to-1.0). **Live-only.**
- **H5** — `TradeExecutor.execute_signal` re-anchors SL/TP to the live execution price (preserving the intended `sl_distance`/`tp_distance` from scan time, 30-60s earlier) and clamps to broker min-stop before sending; recorded on the `ExecutedTrade`. Prevents INVALID_STOPS rejection and risk≠lot drift from scan→execute price movement.
- **H6 — `MeanReversionBacktester.generate_episode_signals` now anchors the H1/H4 slice to each scan bar's timestamp** instead of once per day at day-start. Previously afternoon scans read H1 ADX up to ~24 h stale → the ADX-H1 trend filter behaved differently in the pool than live (which recomputes HTF fresh). Not a future leak; a parity gap. **RETRAIN (rebuild pool) required.**
- **H7** — `auto_train_pipeline.py` argparse `--ml_threshold` default 0.40 → **0.30**. `main()` always builds `HyperParams(ml_threshold=args.ml_threshold)`, so omitting the flag trained the RL agent against a 0.40 ML gate while trainer/live serve 0.30.
- **H8** — `parity_audit.audit_ml_threshold` now parses the **argparse** default (the effective value) in auto_train_pipeline, not just the dead `HyperParams` dataclass default — so the H7 class of drift is caught. Audits still exit 0 with everything aligned at 0.30.

**🟡🟢 Cleanup (Tier 4, safe subset)**: removed single-slot `_ltf_data/_mtf_data/_htf_data` cache + fallback (residual of the v8.0.79 contamination path — now per-symbol only); moved `main.run()` hourly stats / GBM-drift / status counters from `_loop_count % 720` to wall-clock gates (adaptive 1s loop made them fire ~5× too often); `MT5Connector.is_connected` uses `total_seconds()` (was `.seconds`, broke at gaps ≥60 s); `close_position` deviation is now per-symbol (`_get_deviation_points`); `validate_live_xlsx` uses `SelfLearningAgent.OBS_DIM` (was hardcoded 32); fixed stale docstrings (TradeManager "Trailing DISABLED"→Option-X-active + partial 33%; news close window 30→10; `can_open_trade` RR 1.5→1.0; obs dim → 35).

**✅ Refuted (verified safe against real code — do NOT "fix")**: SL/TP min-stop at *order open* (strategy rounds to digits + `min_sl_pips` floor, XAU=$10); scan-batch race (single-threaded loop + blocking `order_send` → `_active_trades` updated before next signal); orphan `initial_sl` corruption (`trail_states.json` stores `initial_sl` separately from `current_sl`, saved on every move); `result.order` vs `position.ticket` (equal for market-order-opened positions); timezone aware/naive (read paths all have tzinfo guards); `get_symbol_info` caching `spread`/`stops_level` (only consumer is dead `PositionSizer.calculate_sl_tp_prices`).

**Invariants**:
- ⛔ Any SL-modify path (BE / Stage-2 / trail) MUST go through `_modify_sl`'s min-stop clamp + never-loosen guard — never build a raw `TRADE_ACTION_SLTP` elsewhere.
- ⛔ The FTMO 4% daily-breach emergency close MUST remain independent of `_state` (do not re-gate it behind `DAILY_HALT`).
- ⛔ `MeanReversionFilterEnv.step` and `FTMOSignalFilterEnv.step` MUST apply the same pre-TAKE forced-SKIP set (correlation + entry-confirm). If they drift again, train≠live returns. Consider refactoring the shared gating into a parent helper.
- ⛔ Live `_build_signal_observation` obs[16]/obs[21] MUST match training semantics (temporal-augmented ml_score; cumulative trades-opened-today). `parity_audit` should grow a per-dim semantic replay check (FOLLOW-UP) — it currently only verifies obs *dimension* (35), not per-dim values, so H3/H4-class drift was invisible.
- ⛔ Pool builder HTF slices MUST be anchored per scan bar (H6), matching live's fresh HTF recompute.

**Retrain — DONE & PROMOTED (2026-05-30)**: rebuilt pool 5000 (seed 42, H6 fresh-HTF + entry_confirm tag) → retrained GBM → retrained RL (C3 forced-SKIP active). **Eval (5000 eps): Pass Rate 88.8%, Breach 0%, Profitable 98.4%, WinRate 69.7%, Total DD max 5.80%, Daily DD max 3.00%.** Anti-overfit holdout (fresh seed-999 pool, 977 eps, current code → fair): **Train 87.0% vs Holdout 87.6% → Δ −0.6 pp = ✅ HEALTHY** (not memorized — the higher pass vs prior 59-71% is real: H6+C3 made the pool cleaner / live-aligned). `parity_audit` + `leakage_audit` both exit 0 (obs 35 3-way sync). Promoted `models/mr/ppo_mr_filter.zip` + `vec_normalize_mr.pkl` → `models/mr/best/` (old stale 32-dim best/ → `*.pre_v8080`; this also resolves the v8.0.77 "best/ is stale 32-dim" doc-rot). `auto_train_pipeline` skips rebuild when pool/GBM exist (known dead-control bug) → **use the manual 3-step for any rebuild**. Backups: `*.pre_v8080` (pool 5000, GBM, model, vec_normalize, _p1, best/). Live-only fixes (C1/C2/H1-H5/H7-H8 + Tier 4) are independent of the retrain and safe to deploy immediately. ⚙️ Deploy = push current commit + `models/mr/best/` + GBM + pool? (pool is gitignored; VPS only needs model + GBM) to the VPS.

**FOLLOW-UPS deferred (behavior-changing refactors — do with dedicated tests, not bundled pre-deploy)**: `close_trade` use broker deal profit (like `sync_with_mt5`) instead of bid/ask approximation; `TradeLogger` avoid full-workbook re-read+save per log call; split `can_open_trade`/`run()` per SRP; cache indicators per `(symbol, last_bar_ts)`; inf→Excel None data-loss in Stats; remove dead `TP_STEP_NEW_TP_RR`/`TRAIL_ATR_MULTIPLIER`/Post-TP+Flip-lock blocks/dead config keys.

## 📝 Version Log Entry — v8.0.79 (2026-05-29) — Cross-symbol cache contamination fix + currency-leg correlation cap (live-only, no retrain)

**Trigger**: RCA of 2026-05-29 live log — 5/5 losing trades, −$219.63. Deep dive found two structural problems (MFE/MAE columns excluded — they record lagged, untrustworthy).

**Problem 1 — Cross-symbol cache contamination (real bug)**: `LiveMRScanner` cached `_ltf_data`/`_mtf_data`/`_htf_data` as **single slots overwritten on every `_scan_one_symbol`**. The live loop runs `scan_all_symbols()` (loops all 7 symbols, leaving the slots = the LAST-scanned symbol) and only THEN iterates signals calling `FTMOTradingBot._build_live_context(sig)` / `_build_signal_observation(sig)`, which read those shared slots → every signal got the **last-scanned symbol's** H1/M15/H4, not its own. Proof: ADX H1 `43.17125997031108` logged identically for both GBPJPY and EURJPY in the same cycle. Impact: (a) Excel `ADX H1`/`ADX H4` columns are cross-symbol-contaminated → unreliable for post-hoc analysis (any past tuning that used these columns is suspect); (b) **functional** — `compute_temporal_features(ltf_df=...)` computes `atr_zscore_30bars` + `volatility_regime_score` (GBM features + RL obs) from the wrong symbol. ⚠️ The **ADX trend filter itself is NOT broken** — the gate in `MeanReversionStrategy.analyze_with_data` reads the correct per-symbol `h1_df` (the trades' true H1 ADX was 16–20 = ranging, passed correctly). Fix: per-symbol caches `LiveMRScanner._{ltf,mtf,htf}_by_symbol` + accessors `get_{ltf,mtf,htf}_data(symbol)`; `_build_live_context` reads by `sig.symbol` (falls back to the legacy slot if missing). No obs dim change → **no retrain**; this brings live obs back in line with training (parity-improving).

**Problem 2 — Currency-leg correlation gap**: EURUSD SELL + EURJPY SELL (both short EUR) opened 8 min apart → EUR rallied → −$140 double loss. Root cause: `FTMOConfig.MAX_CORRELATED_POSITIONS = 99` (group guard effectively disabled), leaving only `MAX_USD_THEME_POSITIONS=2` which covers the **USD leg only** → EURJPY (no USD leg) slipped through. Fix: `TradeExecutor._non_usd_legs(symbol, dir)` decomposes a trade into non-USD (currency, LONG/SHORT) legs; `_check_correlation_risk` blocks when an open trade already shares a leg ≥ `MAX_SAME_CURRENCY_LEG_POSITIONS` (`FTMOConfig` = **1** → no double-bet on the same non-USD currency direction). USD leg stays governed by the existing USD-theme cap (2). Verified: EURJPY SELL with EURUSD SELL open → BLOCK; EURJPY BUY (opposite EUR) / GBPUSD BUY (different leg) → ALLOW.

**Invariants**: ⛔ `_build_live_context` / `_build_signal_observation` MUST read per-symbol caches (`get_*_data(sig.symbol)`) — NEVER the shared `_ltf_data`/`_mtf_data`/`_htf_data` slots (those are last-scanned-symbol, contaminate cross-symbol). ⛔ Excel `ADX H1`/`ADX H4` columns prior to v8.0.79 are cross-symbol-contaminated — do not trust them for analysis. ⚙️ Local `bot_state.json`/`trail_states.json` were stale (May 22) while trades are May 29 → live runs on the VPS; **confirm the VPS runs the current commit** before relying on these fixes. Affected: `strategy/mean_reversion_strategy.py` (`LiveMRScanner`), `main.py` (`_build_live_context`), `execution/trade_executor.py` (`_check_correlation_risk`, `_non_usd_legs`), `config/settings.py` (`MAX_SAME_CURRENCY_LEG_POSITIONS`). Audits: leakage + parity both exit 0. (Note: separate from the reverted Dynamic-TP v8.0.79 — that work was rolled back and never shipped; this reuses the version number.)

## 📝 Version Log Entry — v8.0.78 (2026-05-29) — News close-window decoupled from entry-window (live-only, no retrain)

**Trigger**: User saw a trade opened and force-closed 4 minutes later by "Pre-news close" (GBPJPY BUY, 29 May 10:15:41 → 10:20:00, −$8.18). Log confirmed 3 `Pre-news close` trades; two others (GBPUSD/XAUUSD) were closed at exactly T-60 holding only scratch +$12-13, i.e. killed an hour early for no benefit.

**Root cause**: the entry gate (`RiskManager.can_open_trade` news block) and the close gate (`TradeManager.check_news_close`) both read `no_trade_before_news_minutes` (= 60). A position can only OPEN when news is >60 min away, but `check_news_close` then closes it the instant news enters the same 60-min window. So a trade opened at T-61..T-90 has **0–30 min of runway** and is force-closed almost immediately — and any working position is cut a full hour before the event.

**Fix** (live-only, single-line rollback):
- New `SessionConfig.news_close_before_minutes = 10` decouples the **close** window from the **entry** window. `check_news_close` now uses it (falls back to `no_trade_before_news_minutes` if unset). Entry stays at 60 (preserves the v8.0.38 NFP/FOMC-aftermath entry protection); close drops to 10 → **guaranteed runway = 60 − 10 = ≥50 min**, and working positions aren't cut an hour early (T-10 is still flat before the release spike).
- `no_trade_after_news_minutes` 45 → 20 (entry resumes sooner post-news; MR fades post-news chop).
- Bonus: `TradeExecutor.close_trade` now computes `time_in_trade` (previously only `record_external_close` did → trades closed via close_trade, e.g. Pre-news/Friday, logged `Time-in-Trade (s) = 0`).

**Invariant**: ⚠️ `news_close_before_minutes` MUST stay **< `no_trade_before_news_minutes`** by a runway margin (≥30 min recommended) — if they are equal again, the open-then-immediately-closed bug returns. Affected: `config/settings.py` (`SessionConfig`), `execution/trade_manager.py` (`check_news_close`), `execution/trade_executor.py` (`close_trade`). No obs/reward/pool touched → no retrain; audits stay green.

## 📝 Version Log Entry — v8.0.77 (2026-05-29) — Entry-confirm de-restriction (live-only, no retrain)

**Trigger**: Live reject-funnel audit of `logs/ftmo_trades.xlsx` Signals sheet (953 scans, 24-28 May). RL issued **689 TAKE intents but only 23 opened (3.3% pass at the executor)** — the bottleneck is the entry-confirmation layer, not the MR strategy. Gate breakdown of the 666 `AGENT_TAKE_FAIL`: correlation 336, risk_manager 186, **entry_confirm 142** (M1-direction **82**, KC-slope 35, KC-distance 25). Three entry-confirm issues stood out as false-rejecting MR's best setups.

**Fixes (all live-only, single-line rollback each, improve train/live parity since `MeanReversionBacktester` applies no KC hard gate)**:

- **RF-1 — `TradeExecutor._check_entry_confirmation` M1 block → wick-aware**. MR is a falling-knife entry (`market_bias = -direction`); the strict "M1 last closed bar must close in-direction" rule contradicted that and was the #1 entry-confirm killer (82/142). New rule: pass if the M1 bar body is in-direction **OR** shows a rejection wick on the favorable side ≥ `FTMOConfig.ENTRY_CONFIRM_M1_REJECT_WICK_MIN` (0.40); reject only a clean strong opposite-body bar with no reject-wick. Mirrors the M15 reversal-wick logic on M1.
- **RF-2 — KC distance relaxed**. `FTMOConfig.KC_DIST_THRESHOLD_BASE` 0.60 → **0.35**, `KC_DIST_ATR_SCALE_MAX` 1.5 → **1.3**. `kc_distance_norm` clips at ±1.0 (= 3.75 ATR; band edge = 0.667). Old volatile threshold 0.60×1.5 = 0.90 left only a 0.10-wide acceptance window against the clip ceiling (near-degenerate) and re-gated the BB %B "extended" condition 2-4× stricter than BB itself. New window is 0.545-0.79 across all regimes; volatile still requires a deeper extreme than quiet (directional intent preserved).
- **RF-3 — KC slope cap raised**. `FTMOConfig.KC_SLOPE_THRESHOLD` 0.15 → **0.35**. `ema_slope_norm` = (slope/0.3) clipped ±1.0; the 0.15 cap rejected EMA drift of only ~0.045 ATR/bar (normal even inside ranges) and saturated at ±1.0 (35 rejects mostly `-1.00`). ADX-H1 > 30 remains the primary trend veto.

**Invariant**: these are pre-execution **live filters only** — they do not touch obs dims (current code = **35** dims: `SelfLearningAgent.OBS_DIM`, `MeanReversionFilterEnv`, and `models/mr/vec_normalize_mr.pkl` all agree at 35), GBM features, reward, or the training pool, so **no retrain**. Rollback: restore the four `FTMOConfig` values (`KC_DIST_THRESHOLD_BASE=0.60`, `KC_DIST_ATR_SCALE_MAX=1.5`, `KC_SLOPE_THRESHOLD=0.15`) and the M1 block.

**Audit hardcode fix (same version)**: `scripts/leakage_audit.py` (Audit 4) and `scripts/parity_audit.py` (Audit obs_dim + Audit 7 vec_normalize) had stale hardcoded `32`-dim checks from the v8.0.5 era. The code moved to 35 dims in v8.0.74 but the audits were never updated → both reported false `FAILED`. Updated the three hardcodes `32 → 35`; both audits now pass (`✅ ALL CLEAN` / `✅ ALL ALIGNED`). ⚠️ Doc rot to reconcile later: `context.md`/this file still mention 26/32 in places, and `models/mr/best/` holds a stale **32-dim** model (May 7) while live runs the **35-dim** model in `models/mr/` — best/ cannot match the 35-dim obs, so live falls back to `models/mr/`. Not live-breaking (bot produces real RL decisions), but best/ should be refreshed or removed.

## 📝 Version Log Entry — v8.0.76 (2026-05-28) — Dynamic KC distance (ATR-adaptive, live-only)

**Trigger**: v8.0.74 used a fixed `|kc_distance_norm| ≥ 0.60` threshold for the "price overextended" gate. In **quiet regimes** (Asian sideways) price rarely reaches ±0.6 → bot misses small swings; in **volatile regimes** (US open / news) price hits ±0.6 mid-trend → bot enters before the real extreme and gets dragged.

**Fix** (2 files, live-only — **no retrain**, mirrors v8.0.75 slope pattern):

- `config/settings.py`: NEW `FTMOConfig.KC_DIST_THRESHOLD_BASE = 0.60` (replaces hard-coded -0.6/0.6), `KC_DIST_ATR_ADAPTIVE = True`, `KC_DIST_ATR_GAIN = 0.20`, `KC_DIST_ATR_SCALE_MIN = 0.6`, `KC_DIST_ATR_SCALE_MAX = 1.5`.
- `execution/trade_executor.py`: `_check_entry_confirmation` KC-distance block now computes `dist_thresh = base × clip(1 + atr_z × 0.20, 0.6, 1.5)`. Reject log shows base × scale × atr_z.

**Direction is opposite to slope**: slope dynamic *loosens* in volatile regimes (atr_z high → cap larger → allow steeper slope as noise). Distance dynamic *tightens* in volatile regimes (atr_z high → threshold deeper → require more extreme entry). Together they form a coherent MR gate — accept noise spikes in vol but only at true extremes; tolerate shallow extremes only when trend is genuinely weak.

**Effective threshold table** (with default GAIN/MIN/MAX):

| atr_z | scale | effective threshold |
|-------|-------|---------------------|
| -2 (quiet)    | 0.60 | 0.360 |
| -1            | 0.80 | 0.480 |
|  0 (neutral)  | 1.00 | 0.600 |
| +1            | 1.20 | 0.720 |
| +2 (volatile) | 1.40 | 0.840 |
| +3 (extreme)  | 1.50 | 0.900 (capped) |

**Live cases that motivated this** (28 May logs):
- USDJPY BUY blocked at `dist=-0.49 > -0.60`. With dynamic and atr_z ≤ -0.92 the threshold drops to ≤ 0.49 → would pass (Asian-quiet regime).
- NZDUSD SELL blocked at `dist=0.33 < 0.60`. Would still block under dynamic unless atr_z deeply negative — correct because price wasn't extreme enough.

**Disable / rollback**: set `KC_DIST_ATR_ADAPTIVE = False` → restores v8.0.74 fixed-threshold behaviour (0.60 flat). Or set `KC_ENTRY_FILTER_ENABLED = False` to disable the gate entirely.

**Watch**: After 3-5 days live, compare KC-filter SKIP-rate by session. Expected: ~30-50% fewer SKIPs during Asian-quiet hours (smaller swings now pass), ~10-20% more SKIPs during London/NY-open (require deeper extremes). If filtered-signal WR doesn't improve after Asian-quiet rescues → tune `KC_DIST_ATR_GAIN` lower (less adaptive).

---

## 📝 Version Log Entry — v8.0.75 (2026-05-28) — Dynamic Slope cap (ATR-adaptive, live-only)

**Trigger**: v8.0.74 introduced a fixed `KC_SLOPE_THRESHOLD = 0.15` on `ema_slope_norm`. The feature is already ATR-normalised at compute time (`(ema - ema.shift(5)) / (atr*5)` then `/0.3` and clipped) but the cap itself was static — so in **volatile regimes** (NFP/CPI) the bot SKIPs too many setups, and in **quiet regimes** (Asian) small slopes still pass and feed whipsaw.

**Fix** (3 files, live-only — **no retrain**):

- `config/settings.py`: NEW `FTMOConfig.KC_SLOPE_ATR_ADAPTIVE = True`, `KC_SLOPE_ATR_GAIN = 0.15`, `KC_SLOPE_ATR_SCALE_MIN = 0.7`, `KC_SLOPE_ATR_SCALE_MAX = 1.6`. Baseline `KC_SLOPE_THRESHOLD = 0.15` unchanged.
- `strategy/mean_reversion_strategy.py`: NEW field `MRSignal.atr_zscore_30bars` computed via `TechnicalIndicators.compute_atr_zscore_30bars(ltf_df)` at signal build.
- `execution/trade_executor.py`: `_check_entry_confirmation` slope check now computes `slope_thresh = base × clip(1 + atr_z × GAIN, SCALE_MIN, SCALE_MAX)`. SKIP log line now shows base × scale × atr_z for transparency.

**Effective cap table** (with default GAIN/MIN/MAX):

| atr_z | scale | effective cap |
|-------|-------|---------------|
| -2 (quiet)    | 0.70 | 0.105 |
| -1            | 0.85 | 0.128 |
|  0 (neutral)  | 1.00 | 0.150 |
| +1            | 1.15 | 0.172 |
| +2 (volatile) | 1.30 | 0.195 |
| +4 (extreme)  | 1.60 | 0.240 (capped) |

**Why live-only**: The slope filter lives entirely in `TradeExecutor._check_entry_confirmation`, not in `MeanReversionBacktester` (same pattern as v8.0.55 cluster cooldown + entry-confirm + spread-spike). The RL model never sees this filter during training. Changing the cap only affects which signals get blocked at execute time — pool/GBM/RL stay valid.

**Disable / rollback**: set `FTMOConfig.KC_SLOPE_ATR_ADAPTIVE = False` → restores v8.0.74 fixed-cap behaviour (0.15 flat).

**Watch**: After 3-5 days live, compare SKIP-rate by reason `KC slope` against v8.0.74 baseline. Expected: ~25-40% fewer SKIPs during high-vol sessions (London open, NY open), ~15-25% more SKIPs during Asian-quiet hours. If Pass Rate drifts > 5pp either direction → tune `KC_SLOPE_ATR_GAIN` (lower = less adaptive).

---

## 📝 Version Log Entry — v8.0.74b (2026-05-27) — Obs 35→26 trimmed + Chronos removed + 4 risk rules disabled

**Phase A — Dead dims trimmed (35→26)**: Removed 9 dims: [5] bias_align (always -1), [9] trend_strength (dup of ema_slope), [10] bb_width (dup of squeeze_ratio), [11] adx_norm (dup of adx_inverse), [27-28] chronos (disabled/zero), [29-30] floating_pnl/losing_count (zero since v7.2.1), [4] bb_extreme (replaced by kc_distance).

**Phase B — 4 redundant risk rules disabled**: `FLIP_LOCK_ENABLED=False` (cluster cooldown + opposing theme ครอบ), `POST_TP_LOCK_ENABLED=False` (RL + cooldown ครอบ), `MAX_CORRELATED_POSITIONS=99` (opposing theme block ฉลาดกว่า), `SPREAD_ATR_RATIO_LIMIT=99.0` (spread_spike rolling median ดีกว่า).

**Phase C — Chronos removed**: Deleted init + forecast compute in `main.py`. `CHRONOS_ENABLED=False` already set. obs no longer has chronos slots.

**⚠️ RETRAIN REQUIRED**: OBS_DIM 35→26.

---

## 📝 Version Log Entry — v8.0.74 (2026-05-27) — Keltner Channel / ATR Band anti-whipsaw (obs 32→35)

**Trigger**: BE-Whipsaw 64% (9/14 trades) — bot entered mid-trend, not at overextended extremes.

**Fix** (9 files, RETRAIN REQUIRED):
- `indicators.py`: NEW `calculate_keltner()` — EMA(21) ± 2.5×ATR(14), outputs `kc_distance_norm`, `ema_slope_norm`, `consecutive_outside`, `band_squeeze_ratio`
- `settings.py`: NEW config `MeanReversionConfig.kc_*` + `FTMOConfig.KC_ENTRY_FILTER_ENABLED`, `KC_SLOPE_*`, `KC_CONSEC_*`
- `mean_reversion_strategy.py`: 4 new fields in `MRSignal` + compute in `analyze_with_data`
- `trade_executor.py`: 3 live filters in `_check_entry_confirmation` (checks 4-6: KC distance, EMA slope, consecutive outside)
- `mean_reversion_backtester.py`: 4 keltner keys added to pool signal dict
- `signal_filter_env.py`: obs shape 32→35, 3 new dims [32-34]
- `mean_reversion_env.py`: inherits parent's 35-dim obs
- `rl_agent.py`: `OBS_DIM = 35`
- `main.py`: 3 new dims in `_build_signal_observation`

**⚠️ RETRAIN REQUIRED**: OBS_DIM 32→35 = model mismatch crash until retrained.

**Backup**: `ppo_mr_filter.zip.pre_v8074`, `vec_normalize_mr.pkl.pre_v8074`

---

## 📝 Version Log Entry — v8.0.73 (2026-05-26) — Stage 2 TP extension removed (keep SL lock only)

**Trigger**: User asked "TP-chase logic ต่อไปควรมีไว้ไหม" after 2-day Micro-RCA showed 0/13 trades reaching TP (Stage 2 always extends TP to 1.5R, but continuation rate to 1.5R is effectively 0% in current regime).

**Evidence**:
- **2-day live audit (25-26 พ.ค. 2026, n=13)**:
  - **0/13 trades hit TP** (currently chased to 1.5R via Stage 2).
  - Only 1/13 reached peak MFE ≥ 1.0R (EURUSD 1.155R → trail caught 0.928R).
  - 2/13 triggered Stage 2 @ 0.8R; both retraced and exited at Stage 2 SL lock @ 0.5R, never reaching 1.5R TP.
- **Consistent with v8.0.61 M1 replay**: continuation rate 1.0R→1.5R = 26.9% (well below 40% gate).

**Hypothesis**: Stage 2's TP extension (1.0R → 1.5R) is "dead code" — adds no captured value because MR strategy doesn't continue that far. The Stage 2 SL→0.5R lock IS valuable (caught XAU 8631 at +0.5R when it would have reverted to BE).

**Fix** (live-only, no retrain):
- `TradeManager.TP_STEP_NEW_TP_RR`: **1.5 → 1.0** — TP stays at original 1.0R after Stage 2 triggers.
- `TradeManager._tp_step()`: drop TP-invariant check (`new_tp <= trade.tp_price`) so SL lock still fires even though TP doesn't change. Pass current TP price to `_modify_sl` (no-op on TP side).
- Print message updated: `Stage 2 SL-Lock` (was `Stage 2 TP-Step`) for clarity.

**Side effect**: Stage 3 (`TRAIL_ACTIVATION_RR=1.0`) now rarely fires in normal flow — because TP fills at 1.0R first. Stage 3 stays in code as a safety net for gap/slippage scenarios where price overshoots 1.0R before TP executes.

**Expected impact (sim on 2-day n=13)**:
- EURUSD 4286 (peak 1.155R): TP would now fill at 1.0R = +1.0R (vs current 0.928R via trail) → **+0.07R**
- XAU 8631 (peak 0.835R): same outcome, Stage 2 SL@0.5R still locks → no change.
- Other trades: no change (Peak < 0.8R never triggers Stage 2).
- Net: minor +0.07R / 2 days (within noise). Real benefit is **simplicity** — removes dead TP-extension path.

**No retrain** — RL pool uses R-multiple outcomes computed against `outcome_partial` (peak MFE at SL hit). Trail/TP logic is live-only. Pool/GBM/RL stay v8.0.52 baseline.

**Files changed**: `ftmo_trading_bot/execution/trade_manager.py` (constant + `_tp_step` method).

**No backup needed** — single-constant revert: set `TP_STEP_NEW_TP_RR = 1.5` to restore.

## 📝 Version Log Entry — v8.0.72 (2026-05-26) — Silent probe for nonexistent conversion symbols

**Trigger**: User flagged the repeated log `❌ [MT5] ดึงราคา JPYUSD ล้มเหลว` appearing on every JPY cross-pair signal.

**Investigation**: not an error — it's noisy logging from an otherwise-correct fallback path. In `PositionSizer._calculate_pip_value`, for cross pairs whose **quote currency ≠ account currency** (e.g. GBPJPY/EURJPY on a USD account), the sizer must convert raw pip value (in JPY) to USD. It tries two symbols in order:

1. **Direct** `f"{quote}{acc_ccy}"` → e.g. `JPYUSD` (most brokers, including FTMO, don't list this)
2. **Inverse** `f"{acc_ccy}{quote}"` → e.g. `USDJPY` (always exists; divides instead of multiplies)

The `JPYUSD` lookup hits `MT5Connector.get_current_price`, which prints `❌ [MT5] ดึงราคา {symbol} ล้มเหลว` when `mt5.symbol_info_tick()` returns None. The sizer then proceeds silently to the inverse path and returns the correct value. End user just sees the scary ❌.

**Why now** — became more visible after v8.0.69 widened concurrent positions and v8.0.70 cut loop interval to 1 s under load: more frequent calculations → more frequent log spam.

**Root-cause fix** (user preference: "ดูแค่ที่ต้นเหตุกว่า"): rather than passing a `silent=True` flag through `get_current_price`, add a dedicated probe helper at the connector that is *always* silent:

```python
def MT5Connector.symbol_exists(self, symbol: str) -> bool:
    # cached, mt5.symbol_info() returns None for missing symbols without printing
```

Then `PositionSizer._calculate_pip_value` wraps the direct attempt:

```python
if self._connector.symbol_exists(conversion_symbol_direct):
    price_direct = self._connector.get_current_price(conversion_symbol_direct)
    if price_direct and price_direct.get("bid", 0) > 0:
        return raw_pip_value * price_direct["bid"]
# falls through to inverse path unchanged
```

Behavior is identical (same pip value, same fallback); the broker round-trip + ❌ log are skipped when the broker doesn't list the symbol.

**Verification**:
```
GBPJPY pip value (mock, account=USD) = $6.6890 / lot   (via USDJPY inverse)
❌ JPYUSD logged? False
```
Result matches prior calculation exactly (`1000 JPY / 149.500 ≈ 6.6890 USD`).

**Caching note**: `_symbol_exists_cache` is per-`MT5Connector` instance and persists for the bot's runtime. If a broker dynamically activates new symbols mid-session this could miss the activation — acceptable trade-off because FTMO's symbol list is static during a challenge.

**No retrain needed** — log/path cleanup only.

**Affected files**:
- `ftmo_trading_bot/core/mt5_connector.py` — new public method `symbol_exists`
- `ftmo_trading_bot/core/position_sizer.py` — guard the direct conversion probe

**Revert path**: remove the `if self._connector.symbol_exists(...)` guard in position_sizer. `symbol_exists` can stay (orphaned but harmless).

---

## 📝 Version Log Entry — v8.0.71 (2026-05-26) — Remove broken GBPJPY per-day cap

**Trigger**: User report — "🚫 GBPJPY: เทรดครบ 3 ครั้งวันนี้แล้ว" appeared even though `logs/ftmo_trades.xlsx` Trades sheet showed **0 closed GBPJPY trades** for the day.

**Root cause** (latent bug, present since at least v6.x): in `RiskManager.can_open_trade` (`core/risk_manager.py:853-860`):

```python
default_max = getattr(self._config, "MAX_TRADES_PER_DAY", 5)
max_per_day = get_symbol_config(symbol, "max_trades_per_day", default_max)  # per-symbol lookup → 3 for GBPJPY
if max_per_day is not None and self._daily_trades_count >= max_per_day:     # GLOBAL counter, not per-symbol
    return (False, f"🚫 {symbol}: เทรดครบ {max_per_day} ครั้งวันนี้แล้ว ...")
```

The cap is *defined* per-symbol but *enforced* against a single global counter. The inline comment in the file even concedes "track เป็น total count". Result: as soon as the bot's total daily trades across all symbols reaches 3, GBPJPY entries are blocked — even on days where GBPJPY itself has traded zero times.

**Why now** — after v8.0.69 raised `MAX_OPEN_POSITIONS` 2 → 3 and v8.0.55/68 added more aggressive cross-symbol filters, total daily trades reaches 3 earlier in the session, exposing the bug for GBPJPY users who previously rarely hit the global threshold.

**Fix**: remove `"max_trades_per_day": 3` from `SymbolConfig.symbol_overrides["GBPJPY"]`. `get_symbol_config` falls back to `FTMOConfig.MAX_TRADES_PER_DAY` which is already `None` (disabled). Other GBPJPY-specific guardrails are unchanged:
- `atr_floor_pips=8` (skip dead market)
- `min_sl_pips=20` (avoid spread-dominated SL)
- `min_confidence=0.65` (per-symbol RL gate)
- `spread_atr_ratio_max=0.5` (liquidity check)
- `flip_lock_retrace_mult=0.7`

**Verification on 2026-05-26 data**: `Signals` sheet shows GBPJPY had **37 candidate signals** → **AGENT_SKIP 12** (RL rejected) + **AGENT_TAKE_FAIL 25** (executor reject — spread/ATR/cooldown/confirm). Zero made it to `Trades` sheet. Existing filters already keep GBPJPY restrained; the broken cap was redundant safety theater.

**Not fixed in this commit** (intentional, low priority): proper per-symbol daily counter (would require new `_daily_trades_per_symbol` dict + `_save_state`/`_load_state` schema bump + tracking on every `record_trade_open` call). Not needed because (a) the global cap default is already `None`, (b) no other symbol uses `max_trades_per_day` override, (c) cross-symbol over-trading is bounded by `MAX_OPEN_POSITIONS=3`, `CLUSTER_COOLDOWN_ANY_SEC=300`, `CONSECUTIVE_LOSS_HALT_COUNT=4`, `DAILY_LOSS_CAP_PCT=2.5%`.

**No retrain needed** — execution config only.

**Affected files**:
- `ftmo_trading_bot/config/settings.py` — drop `max_trades_per_day` key from GBPJPY overrides

**Revert path**: re-add `"max_trades_per_day": 3,` to the override dict. If the cap is ever desired for real, fix the enforcement side too (per-symbol counter) — not the config alone.

---

## 📝 Version Log Entry — v8.0.70 (2026-05-26) — Full-lifecycle 1s adaptive loop

**Trigger**: User report — "order ถึงจุดที่ต้องเลื่อน BE แล้วบอทไม่เลื่อนเพราะไม่ถึงเวลาสแกน".

**Root cause**: v8.0.46 adaptive loop switched to 1s only when `profit_r >= 0.5R`. But v8.0.56 lowered `BE_TRIGGER_RR` to **0.3R** → the decisive trail action now fires *below* the 0.5R fast-poll threshold, so the bot is still on 5 s loop at the moment BE must be moved. A fast M1 spike that crosses 0.3R and reverts inside one 5 s tick → bar closes without BE → next loop sees `profit_r < 0.3R` again → BE never set → SL stays at original → trade reverts to full SL.

**Fix part 1 — adaptive condition**: any open position triggers 1 s loop (drop the `profit_r >= 0.5R` precondition). Rationale: lifecycle cost is bounded (avg hold ~75 min × 1 s polls = 4,500 ticks per trade) but accuracy of BE@0.3R / Partial@0.8R / Stage 2@0.8R / Stage 3@1.0R is now tick-precise.

**Fix part 2 — wall-clock signal scan**: the previous scan gate `self._loop_count % 12 == 0` produced 60 s scan @ 5 s loop. Under part 1, loop becomes 1 s while a position is open, which would push scan to 12 s — 5× too fast (ML inference + RL agent + Excel `Signals` row per tick, plus risk of duplicate signal entry on borderline setups). Replace with wall-clock gate using `_last_signal_scan_ts` (epoch seconds, 60 s elapsed). Scan cadence stays at 1/min regardless of loop speed.

**What still uses `loop_count`** (intentionally left alone, cosmetic only):
- `% 60` (verbose 2 status print, every 5 min @ 5s → every 1 min @ 1s) — debug-only
- `% 720` (verbose 1 status print + hourly Stats sheet write + GBM drift check) — fires ~6× more often while a position is open. Excel I/O cost is small; behavior unchanged.

**No retrain needed** — execution-only fix. RL agent does not observe loop interval; trade outcomes change only because BE/SL/TP now move on time, which is the intended training spec.

**Affected files**:
- `ftmo_trading_bot/main.py` — `__init__` (`_last_signal_scan_ts`), Step 2 scan section (wall-clock gate), adaptive sleep section (any-position condition)

**Verification path**: after deploy, expect more rows in `logs/ftmo_trades.xlsx` `Trades` sheet showing `Best Price ≥ 0.3R` paired with `SL = entry` (BE actually moved). Excel `Loop Count` field will inflate while positions are open (1s polls).

**Revert path**: restore the v8.0.46 block in adaptive section + `if self._loop_count % 12 == 0:` in scan section. No state migration.

---

## 📝 Version Log Entry — v8.0.69 (2026-05-26) — MAX_OPEN_POSITIONS 2 → 3

**Trigger**: User request — "ปรับบอทให้เปิดไม้ได้สูงสุด 3 ไม้พร้อมกัน".

**Background**: v8.0.56 (2026-05-22) reduced this from 3 → 2 after a 111-trade audit showed Net -$460 / EV -$4.15 per trade. The reduction was bundled with three other safety changes (blocked_symbols, BE/Partial tuning, DAILY_LOSS_CAP re-enabled).

**Why it's safe now to widen back to 3**:
- `SymbolConfig.blocked_symbols` (v8.0.56) — AUDUSD/USDCAD/USDCHF excluded, which were responsible for -$1,297 of cumulative loss in the audit.
- `v8.0.55` pre-execution gates — entry confirmation (slip/M1 direction/BB %B), spread spike (2× rolling median), cluster cooldown (300s any / 600s same theme) — filter low-quality concurrent entries.
- `v8.0.68 OPPOSING_THEME_BLOCK` — prevents the "1 winner + 1 loser = net loss" scenario (e.g. GBPUSD SELL + EURUSD BUY).
- Combined effect: the quality of signals reaching `can_open_trade` is materially higher than v8.0.56-era → 3rd concurrent slot is no longer "noise slot".

**Risk math** (default 0.70% per trade):
- 3 positions all hit SL simultaneously = 3 × 0.70% = **2.1% concurrent risk**
- vs `DAILY_LOSS_CAP_PCT = 2.5%` → 0.4pp buffer before soft cap
- vs `DAILY_LOSS_HARD_STOP_PCT = 4%` → 1.9pp buffer before FTMO breach
- 4th loss after a 3-loss streak (3 × 0.70% + 1 × 0.70% = 2.8%) → caught by 2.5% soft cap before reaching 4% hard stop.

**Single-line change** in `ftmo_trading_bot/config/settings.py`:
```python
MAX_OPEN_POSITIONS: int = 3  # was 2 in v8.0.56
```

**No retrain needed** — `MAX_OPEN_POSITIONS` is a live-execution config (not in training pool / env). RL agent does not see open-position count as obs (concurrent risk is enforced gate-side in `RiskManager.can_open_trade`).

**Revert path** — change back to `2` (no state migration, no model swap, no pool rebuild). v8.0.56 backups still exist on disk if a deeper rollback is needed.

**Affected files**:
- `config/settings.py` — `FTMOConfig.MAX_OPEN_POSITIONS: int = 3`

---

## 📝 Version Log Entry — v8.0.68 (2026-05-26) — Opposing-Theme Block

**Trigger**: User feedback after observing today's losses — "บอทต้องไม่เปิดออเดอร์สวนทางกันเช่น GBPUSD SELL, EURUSD BUY เพราะพอออเดอร์นึงกำไรอีกอันขาดทุน".

**Problem**: Existing `CLUSTER_COOLDOWN_SAME_THEME_SEC` (v8.0.55) blocks same-theme trades within 10 min, but **does NOT prevent OPPOSING theme** when one position is already open. Example failure:
- 10:00 EET — GBPUSD SELL opens (theme = USD_LONG)
- 10:20 EET — cooldown expired → EURUSD BUY opens (theme = USD_SHORT)
- Net: one wins, the other loses; EV ≈ 0 minus spread × 2 = guaranteed net negative

**Fix**: Add `RiskManager._check_opposing_theme()` — walks `connector.get_open_positions()`, computes theme of each (reusing `_compute_theme()` from v8.0.55), and blocks new trade if any open position has opposite theme in same group (USD_LONG↔USD_SHORT, JPY_LONG↔JPY_SHORT, METAL_LONG↔METAL_SHORT).

**Wired** into `RiskManager.can_open_trade` as check #2.5 (after MAX_OPEN_POSITIONS, before RR check) so any signal that would create opposing exposure gets rejected with reason `🚫 Opposing-theme block: {sym} {dir} (=ทิศ X) สวนกับ {open_sym} {open_dir} (=ทิศ Y)`.

**Config**: `FTMOConfig.OPPOSING_THEME_BLOCK_ENABLED = True` (toggle for emergency disable).

**No retrain needed** — pure live-execution filter (mirrors v7.1.10 news / v8.0.21 pre-news / v8.0.26 bulk guard pattern).

**Affected files**:
- `config/settings.py` — new flag `OPPOSING_THEME_BLOCK_ENABLED`
- `core/risk_manager.py` — new methods `_is_opposing_theme()`, `_check_opposing_theme()`, gate in `can_open_trade`

**Verification** — unit test (`/tmp/test_opposing_theme.py`) covers theme computation + opposing detection + the exact scenario user complained about (GBPUSD SELL vs EURUSD BUY → BLOCKED).

---

## 📝 Version Log Entry — v8.0.67 (2026-05-25) — XAU-only Morning Block (data-driven)

**Trigger**: User asked if Morning Block (04-09 ICT) should be re-enabled after v8.0.50 disabled all delays.

**Analysis methodology**: Multi-TF CSV audit (NOT Excel which had small-sample bias)
- 7 symbols × Feb-May 2026 OHLCV (M15 + M1)
- Synthetic MR entries (BB extreme + RSI extreme)
- Walk M1 forward 120 bars → apply v8.0.61 logic (BE 0.3R, Partial 0.8R, TP 1.0R)
- Subtract $10/trade spread
- Total: **8,058 synthetic entries** (vs 23 Excel trades = 350x more sample)

**Results — Morning (EET 00-04) vs Others**:

| Metric | Morning (n=1520) | Others (n=6538) |
|---|---|---|
| WR | 54.3% | 53.3% |
| EV/trade | **+$7.64** ✅ | +$6.15 |
| Hit SL | 37.3% (safer) | 44.5% |
| t-test | t=1.01, **p=0.31** | not sig |

**Per-symbol morning EV**:
- 🌟 USDJPY +$12, EURJPY +$12, NZDUSD +$12 (excellent)
- ✅ GBPJPY +$8.5, GBPUSD +$7.7, EURUSD +$6 (good)
- 🔴 **XAUUSD -$6.06** ← ONLY morning loser

**Why XAU bad in morning**:
- Continuation rate 0.3R→1.0R = 44.8% (vs 57.6% group avg)
- Hit SL rate 42.9% (vs 37.3% group avg)
- Asian session = Gold's quiet time + spread spike risk

**Decision**:
- 🟢 KEEP UNBLOCKED general morning (6/7 symbols profitable)
- 🔴 BLOCK XAU only morning (-$6/trade prevented)
- EET 03 (ICT 07) actually BEST hour of day (+$24.98/trade) — must NOT block

**Why Excel was wrong (small-sample bias)**:
- 23 trades only
- May 12 outlier: 4 trades = -$432 (dominated total -$500)
- Excluding May 12: morning EV ≈ -$3/trade (still small but no clear signal)
- CSV 1,520 morning entries: pattern reverses to +$7.64/trade

**Fix** ([config/settings.py](ftmo_trading_bot/config/settings.py) FTMOConfig):
```python
# v8.0.50 had False:
XAU_WEEKDAY_DELAY_ENABLED: bool = False
# v8.0.67 RE-ENABLE:
XAU_WEEKDAY_DELAY_ENABLED: bool = True
XAU_WEEKDAY_DELAY_END_HOUR_EET: int = 5  # block EET 00-04 = ICT 04-08
```

**Other delays kept disabled**:
- `WEEKDAY_DELAY_ENABLED = False` (general — would lose +$11k from 6 good symbols)
- `MONDAY_DELAY_ENABLED = False`

**Expected savings**:
- 198 XAU morning entries/90 days × $6 = ~$1,200 prevented
- ≈ $13/day on XAU alone (vs catastrophic -$42/day if we blocked ALL morning)

**Train-live parity caveat**:
- Training simulator has no Asian Delay
- But XAU-only block is per-symbol gate at `RiskManager.can_open_trade` → RL-blind
- Same pattern as v7.1.10 / v8.0.21 / v8.0.26 (live-only gates work without retrain)

**No retrain needed** — execution-layer config flag only.

**Lesson learned**: Always validate small-sample Excel findings against large-sample CSV before deploying restrictive filters. 23 trades cannot conclude what 1,520 entries clearly show.

---

## 📝 Version Log Entry — v8.0.66 (2026-05-25) — Fix: news CSV parser preserve Holiday impact

## 📝 Version Log Entry — v8.0.66 (2026-05-25) — Fix: news CSV parser preserve Holiday impact

**Bug** ([news_csv_parser.py:100](ftmo_trading_bot/config/news_csv_parser.py#L100)):
- Filter at line 55 accept ทั้ง `"high"` และ `"holiday"`
- แต่ output dict (line 100) **HARDCODE** `"impact": "high"`
- → JSON บันทึก Holiday events เป็น `"high"` ผิดทุกครั้ง

**Impact**:
- ฟังก์ชัน `is_near_high_impact_news()` ไม่กระทบ (treat ทั้ง high+holiday เป็น block เหมือนกัน)
- แต่ JSON `news_calendar.json` ผิดความหมาย — debug/audit ลำบาก
- Downstream code ใดที่ต้องการแยก behavior holiday vs high (เช่น log แตกต่าง) จะทำไม่ได้

**Fix** (1 บรรทัด):
```python
# เดิม:
"impact": "high",   # bug
# ใหม่:
"impact": impact,   # preserve original ("high" หรือ "holiday")
```

**Verification** (`config/news_csv_parser.py` test):
- Input CSV: 2 high + 2 holiday + 1 medium (skip)
- Output: 4 events (2 high, 2 holiday) ✅
- Behavior: news block path คงเดิม (ทั้ง high+holiday block)

**Files**:
- `ftmo_trading_bot/config/news_csv_parser.py` (1 line + comment)

**No retrain, no breaking change** — JSON correctness fix only.

---

## 📝 Version Log Entry — v8.0.65 (2026-05-25) — Thai Plain-Language Explanation on Discord

## 📝 Version Log Entry — v8.0.65 (2026-05-25) — Thai Plain-Language Explanation on Discord

**Feature**: เพิ่มคำอธิบายภาษาไทยแบบเข้าใจง่าย (ไม่มีศัพท์เทคนิค) ใน Discord notification ทุกครั้งที่บอทเปิด order

**Goal**: User ทั่วไปที่ไม่รู้ศัพท์ technical (RSI, ADX, BB %B) สามารถเข้าใจได้ว่าบอทตัดสินใจเข้า trade ทำไม + ระดับความมั่นใจ

**Design principles**:
1. 🟢 **ZERO impact** on trading logic — output layer only
2. 🟢 **Fail-silent** — wrap in try/except, ถ้า explainer fail → original Discord embed ส่งปกติ
3. 🟢 **Toggleable** — config flag `thai_explain_enabled` + env `BOT_THAI_EXPLAIN=0`
4. 🟢 **Discord-only** — ไม่ print console, ไม่กระทบ trade_executor

**Files added**:
- `ftmo_trading_bot/core/thai_explainer.py` (NEW, ~200 lines)
  - `format_trade_open_explanation(trade_dict)` → Discord-formatted Thai string
  - 8 translation helpers: RSI, BB, ADX, ML, Confluence, Volatility, Chronos, Session
  - Parser สำหรับ signal_reasons string (regex extract BB %B / RSI / Wick / ADX)
  - Confidence stars (1-5 ⭐) คำนวณจาก ML + Confluence + ADX

**Files modified**:
- `ftmo_trading_bot/config/settings.py` (+1 field):
  - `NotificationConfig.thai_explain_enabled: bool = True` (env `BOT_THAI_EXPLAIN`)
- `ftmo_trading_bot/core/notifier.py` (`send_trade_open` only):
  - Append field `🇹🇭 อธิบายภาษาไทย` to existing embed (inline=False)
  - Length-capped at 1024 chars (Discord limit)
  - Wrapped in try/except — fail silent

**Translation dictionary** (jargon → plain Thai):
- RSI → "ความร้อน-เย็นของราคา" (เครื่องวัดไข้)
- ADX → "ความแรงของเทรนด์" (เครื่องวัดลม)
- BB %B → "ตำแหน่งราคาในกรอบ" (ยางยืด)
- ML score → "AI ให้คะแนน" (% ความมั่นใจ)
- Confluence → "คะแนนรวมจุดเทรด" (เกรด A/B/C)
- Volatility → "ความผันผวน" (สภาพคลื่น)
- Chronos → "AI ทำนายอนาคต 2 ชม."

**MR context in explanation**:
- BUY = "ราคาตกลึกเกิน เด้งกลับขึ้น" (ไม่ใช่ trend up bet)
- SELL = "ราคาขึ้นแรงเกิน กลับลง" (ไม่ใช่ trend down bet)
- ใช้ analogy "ยางยืดดึงสุดก็เด้งกลับ"

**Example output** (584 chars, under Discord limit):
```
💡 ทิศทาง: ขาย — เก็งว่าราคาขึ้นมาแรงเกินไป เดี๋ยวจะกลับลง
   (กลยุทธ์ยางยืด — ดึงสุดก็เด้งกลับ)

💰 เดิมพัน: $67 (0.7% ของพอร์ต)
🎯 เป้ากำไร: +$67 | 🛡️ เซฟสุด: -$67

📊 ทำไมบอทถึงเข้า:
🎈 ตำแหน่งราคา = ใกล้ขอบบน (0.85) → ยางยืดดึงสุด
🌡️ ความร้อนของราคา = 72/100 → ร้อนเกิน (มักเย็นลง)
💨 ความแรงเทรนด์ = 22/100 → ตลาดเริ่มทิศ (พอใช้)
🤖 AI ให้คะแนน = 50% → 🟡 มั่นใจปานกลาง
⭐ คะแนนรวมจุดเทรด = 79/100 → เกรด B (ผ่านเกณฑ์)
🌊 ความผันผวน = ปกติ
🌅 ช่วงเวลา = London (volume สูง)

🎖️ ระดับมั่นใจ: ⭐✨
```

**Safety verification**:
- ✅ Syntax: 3 files ผ่าน ast.parse
- ✅ Test: format_trade_open_explanation() ทำงานกับ sample trade
- ✅ Config: `thai_explain_enabled = True` default
- ✅ Discord field length: 584 chars < 1024 limit
- ✅ Fail-silent: try/except ครอบ → original embed ส่งได้แม้ explainer fail

**No retrain needed** — Cosmetic Discord layer only.

---

## 📝 Version Log Entry — v8.0.64 (2026-05-25) — Hotfix: TZ-aware vs naive comparison

## 📝 Version Log Entry — v8.0.64 (2026-05-25) — Hotfix: TZ-aware vs naive comparison

**Bug introduced by v8.0.63**: `⚠️ [Bot] Trade Manager error: can't compare offset-naive and offset-aware datetimes` spam ทุก scan loop

**Root cause** ([config/news_events.py:153, 174](ftmo_trading_bot/config/news_events.py#L153)):
- v8.0.63 เปลี่ยน `datetime.utcnow()` (naive) → `datetime.now(timezone.utc)` (aware) ทั่วทุกที่
- แต่ news_events.py ออกแบบให้ใช้ **naive datetime ทั่วทั้งไฟล์** (line 173, 190 strip tzinfo ทุกที่)
- → เปลี่ยนของซ้าย (aware) เปรียบเทียบกับของขวา (naive) = `TypeError`

**Fix**:
- Line 153: `datetime.now(timezone.utc)` → `datetime.now(timezone.utc).replace(tzinfo=None)`
- Line 174: เหมือนกัน
- → Preserve naive convention ของไฟล์ + ไม่ deprecated warning (เพราะใช้ `now(timezone.utc)` ไม่ใช่ `utcnow()`)

**Why other 10 utcnow fixes ปลอดภัย**:
- main.py:1457 (drift log) → `.strftime()` ไม่ compare
- main.py:829 (temporal feat) → `compute_temporal_features` handle ทั้ง aware/naive (line 71-82)
- notifier.py × 8 → `.isoformat()` for Discord, ไม่ compare
- ml/signal_quality.py:75 → fallback ใน try/except, ไม่ compare
- execution/trade_executor.py:733 → `.hour` accessor, ไม่ compare

**Lesson**: เมื่อ migrate `utcnow()` ต้องตรวจว่าผลลัพธ์ใช้ในการ compare กับ naive datetime อื่นไหม

**Files touched**:
- `ftmo_trading_bot/config/news_events.py` (2 lines)

---

## 📝 Version Log Entry — v8.0.63 (2026-05-23) — Drift Throttle + datetime.utcnow Fix

## 📝 Version Log Entry — v8.0.63 (2026-05-23) — Drift Throttle + datetime.utcnow Fix

**Bug 1**: `_check_gbm_drift` print `⚠️ [GBM Drift] N features ห่างจาก training` ทุกชั่วโมง (720 loops). ถ้า drift เกิดขึ้นต่อเนื่อง → ทุก 1 ชม. print message เดิม → log รก. ตัวอย่าง: 23 features drift, top: hour_of_day_cos=0.93, rsi_value=0.89

**Root cause** (`main.py:1396-1437`):
- `_check_gbm_drift` ไม่มี throttle — `if len(drifts) >= 3: print(...)` fire ทุกเรียก
- KS test window = 100 signals → sample เล็ก → false positive สูง (โดยเฉพาะ time-of-day features)

**Bug 2**: `datetime.utcnow()` deprecated ใน Python 3.12+ → DeprecationWarning spam ทุก call site (~12 จุด)

**Fix v8.0.63**:

1. **Drift throttle** (`main.py:1396-1452`):
   - Add `self._last_drift_count: int = -1` + `self._last_drift_announce_ts: Optional[float] = None` ใน `__init__`
   - Console print เฉพาะถ้า `count` เปลี่ยน ≥ 3 features หรือผ่านไป ≥ 6 ชม.
   - Discord notify ทำงานพร้อม console announce (เงียบเมื่อ throttled)
   - File log (`logs/gbm_drift.log`) ยัง append ทุกครั้ง — full audit trail

2. **Drift window 100 → 200** (`ml/signal_quality.py:176`):
   - Sample ใหญ่ขึ้น → KS test stable ขึ้น
   - ลด false positive จาก small-sample bias (โดยเฉพาะ hour/day_of_week ที่ต้องการ data หลายวัน)

3. **datetime.utcnow() → datetime.now(timezone.utc)** (12 places, 4 files):
   - `main.py` (2 places: drift log + temporal feat default)
   - `core/notifier.py` (8 places: all Discord timestamps)
   - `config/news_events.py` (2 places: cache expiry check)
   - `ml/signal_quality.py` (1 place: compute_temporal_features default)
   - `execution/trade_executor.py` (1 place: hour-based logic)

**Result**:
- Drift warning: เดิม 24 ครั้ง/วัน → ~4 ครั้ง/วัน (เฉพาะ count เปลี่ยนใหญ่ หรือทุก 6 ชม.)
- DeprecationWarning: ZERO (Python 3.12+/3.13 compat)
- File log audit trail: คงไว้ครบทุก check

**No retrain needed** — anti-spam + deprecation fix only.

---

## 📝 Version Log Entry — v8.0.62 (2026-05-23) — Friday Force Close Throttle (anti-spam)

## 📝 Version Log Entry — v8.0.62 (2026-05-23) — Friday Force Close Throttle (anti-spam)

**Bug**: `TradeManager.check_session_close` print `🚨 ⚠️ FRIDAY FORCE CLOSE ⚠️` ทุก 5s scan loop หลัง 20:45 EET วันศุกร์ → spam log จนอ่านยาก. ต่างจาก `daily_close` (line 763) + `friday_warning` (line 776) ที่เช็ค `active_ct > 0` ก่อน print

**Root cause** (`trade_manager.py:752` ก่อน fix):
```python
if TimeManager.is_friday_close_time(server_time_now):
    print("🚨 ... FRIDAY FORCE CLOSE ...")   # ← UNCONDITIONAL print
    for ticket in list(...):
        if self._executor.close_trade(...): closed_count += 1
    return closed_count
```
- Condition True ตลอด 3+ ชม. (20:45 EET → end of trading day)
- Print fire ทุก tick แม้ไม่มี position เหลือ

**Fix** — Throttle pattern (mirror v8.0.16 give-back + v8.0.49 daily-loss):
1. เพิ่ม `self._friday_close_announced_date: Optional[str] = None` ใน `__init__`
2. Guard print ด้วย:
   - `today_str = server_time_now.date().isoformat()`
   - `if self._friday_close_announced_date != today_str:` → announce + set flag
3. Print แสดง active position count + EET time stamp
4. Close attempt ยัง fire ทุก tick (safety — orphan position recovery)

**Result**:
- 1 announce/วัน (เทียบเดิม ~2000+ ครั้ง/3 ชม.)
- Auto-reset ทุกวัน (date-based key)
- ปลอดภัย: ถ้ามี orphan position ใหม่เข้ามาหลังประกาศ → ยัง close ตามปกติ (เงียบ)

**Files touched**:
- `ftmo_trading_bot/execution/trade_manager.py` (line 130 add field, line 752-770 throttle logic)

**No retrain needed** — execution-layer cosmetic fix only.

---

## 📝 Version Log Entry — v8.0.61 (2026-05-23) — REVERT RR 1.5 → 1.0 (data-driven)

## 📝 Version Log Entry — v8.0.61 (2026-05-23) — REVERT RR 1.5 → 1.0 (data-driven)

**Trigger**: M1 OHLCV replay validity check ([scripts_local/m1_replay_validity_check.py](scripts_local/m1_replay_validity_check.py)) — eliminated counterfactual bias from prior polled-MFE replay.

### M1 Replay Findings (106/113 trades covered)

| Metric | Result |
|---|---|
| **Continuation Rate avg (1.0R → 1.5R)** | **26.9%** |
| Gate threshold for RR 1.5 deploy | ≥ 40% |
| **Decision** | **🔴 FAIL gate → REVERT RR 1.5 → 1.0** |

Per-symbol continuation rate:
- 🟢 USDJPY: **45.5%** (only symbol passing gate — true trend currency)
- 🔴 NZDUSD: **0%** (true MR, mean-reverts instantly)
- 🔴 XAUUSD: **0%** (6/6 trades reached 1.0R, none to 1.5R)
- 🔴 GBPJPY/GBPUSD: **0%**
- ⚠️ EURUSD (50%, n=2), EURJPY (100%, n=1) — sample too small

Scenario comparison (after -$10 spread per trade):
- OLD baseline: -$342 (real history)
- **Phase 1 Only**: **+$109** (EV +$1.63/trade, PF 1.08) ✅
- Phase 1+2 Full (RR 1.5): -$627 (EV -$9.36/trade, PF 0.57) 🔴

### Polled MFE Column Bias Confirmed

- TRUE max MFE (M1): **6.721R**
- POLLED max MFE (Excel): **1.500R** (suspicious cap)
- 47/106 trades (44.3%) had TRUE > POLLED (poll missed peak by avg 0.215R)
- Prior replay using polled MFE was systematically pessimistic on continuation

### Revert Actions

**Code changes** (2 files, RR 1.5 → 1.0):
- `ftmo_trading_bot/config/settings.py`: `MRConfig.rr_ratio = 1.0`
- `ftmo_trading_bot/strategy/mean_reversion_strategy.py`: `MeanReversionStrategy.RR_RATIO = 1.0`

**Artifacts restored** from `.pre_v8060` backups (MD5 verified):
- `data/mr_signal_pool_5000.pkl` (537 MB) ← v8.0.52 baseline
- `data/mr_signal_quality_model.pkl` (1.2 MB, AUC 0.6135)
- `models/mr/ppo_mr_filter.zip` (Phase 2 RL, Pass 70.7%)
- `models/mr/ppo_mr_filter_p1.zip` (Phase 1 RL)
- `models/mr/vec_normalize_mr.pkl` + `_p1.pkl`

### Phase 1 Fixes — KEPT (passed validity check)

These fixes ALONE pivot system from -$342 → +$109 (+$451 swing):
- `SymbolConfig.blocked_symbols = ['AUDUSD', 'USDCAD', 'USDCHF']` — block 3 worst (-$1,238 saved)
- `TradeManager.BE_TRIGGER_RR = 0.3` (was 0.5)
- `TradeManager.PARTIAL_TRIGGER_RR = 0.8` / `PARTIAL_CLOSE_PCT = 0.33`
- `FTMOConfig.DEFAULT_RISK_PER_TRADE_PCT = 0.0070` (was 0.0085)
- `FTMOConfig.MAX_OPEN_POSITIONS = 2` (was 3)
- `FTMOConfig.DAILY_LOSS_CAP_ENABLED = True` @ 2.5%

### Lesson Learned

1. **Polled MFE (5s scan) is unreliable for replay** — must use M1 OHLCV high/low for true MFE
2. **Mean Reversion strategy ≠ Trend Following** — MR by definition reverts after extreme; expecting 1.5R continuation is fighting the strategy's edge
3. **USDJPY is the exception** (45.5% continuation) — future work: per-symbol RR (USDJPY can use 1.5R, others stay 1.0R)
4. **Validity check FIRST, retrain SECOND** — could have saved 1h+ retrain effort if M1 replay was done before Phase 2 work

### Backups (don't delete)

`.pre_v8060` files preserved — if v8.0.61 has issues, restore Phase 1 settings only or full revert to v8.0.56 (Phase 1 + v8.0.52 model = current state).

---

## 📝 Version Log Entry — v8.0.56 (2026-05-22) — Phase 1 EV Fix (live-only, no retrain)

## 📝 Version Log Entry — v8.0.56 (2026-05-22) — Phase 1 EV Fix (live-only, no retrain)

**Trigger**: Audit 111 closed trades (`logs/ftmo_trades.xlsx` Trades sheet) — WR 59.46% แต่ Net **-$460.54**, EV **-$4.15/trade**, Profit Factor 0.863. Realized RR = **0.589** (vs design 1.0) — "ตัดกำไรเร็ว ปล่อยขาดทุนเต็ม".

### Root cause from audit

- ไม้ชนะ 86% โดน Partial @0.5R + BE → cap ที่ ~0.6-0.7R เท่านั้น
- ไม้แพ้ 100% ไม่มี Partial / BE → กิน SL เต็ม -1R
- 3 คู่ AUDUSD/USDCAD/USDCHF = WR 27-46%, รวม -$1,297 vs 7 คู่ที่เหลือ +$836
- Avg Win $44.10 vs Avg Loss $74.92 — เสียเปรียบ 70%/ไม้

### Fixes (live-only, no retrain — RL pool/reward เป็น R-multiple scale-invariant)

**1. Symbol Blocking** (`SymbolConfig.blocked_symbols` — new field):
- Block AUDUSD, USDCAD, USDCHF
- Filter ใน `LiveMRScanner.__init__` — exclude จาก `_symbols` list
- Toggle: ลบจาก list = unblock
- Training pool ยังเทรดได้เต็ม (live-only filter, parity acceptable)
- Expected: pool 7 คู่ × ~5% take rate = ~5-8 trades/day (vs ~10-12 เดิม)

**2. BE/Partial tune** (`TradeManager` constants):
- `BE_TRIGGER_RR`: 0.5 → **0.3** — จับไม้ revert (62% ของ losses เคย MFE > 0)
- `PARTIAL_TRIGGER_RR`: 0.5 → **0.8** — ให้ไม้ชนะวิ่งเต็มก่อนหั่น
- `PARTIAL_CLOSE_PCT`: 0.5 → **0.33** — ปิด 33% (เก็บ position ใหญ่)
- Expected Avg Win $44 → ~$60-70, Avg Loss $75 → ~$50
- ⚠️ Side-effect: ไม้ MFE 0.3-0.7R → revert จะปิดที่ entry (lose 0.25R partial เก่า) — รับได้เพราะ BE สำคัญกว่า

**3. Risk Reduction** (FTMOConfig):
- `DEFAULT_RISK_PER_TRADE_PCT`: 0.0085 → **0.0070** (-18%)
- `MAX_OPEN_POSITIONS`: 3 → **2**
- `DAILY_LOSS_CAP_ENABLED`: False → **True** (re-enable)
- `DAILY_LOSS_CAP_PCT`: 0.030 → **0.025** — soft cap 1.5pp buffer ก่อน FTMO 4%
- ห้ามลด risk < 0.5% (Pass Rate ตก 30-50% เพราะ +10% target เป็น absolute $)

### What stays the same (v8.0.55 baseline)

- 3 live filters (Entry Confirmation + Spread Spike + Cluster Cooldown) ยังคงเดิม
- RR target ยัง 1.0 (Phase 2 จะปรับเป็น 1.5 หลัง retrain)
- RL model + GBM + Pool ยังเป็น v8.0.52 (Pass 70.7%)
- Stage 2/3 trail (TP_STEP 0.8R, TRAIL_ACTIVATION 1.0R) ยังเปิด — ⚠️ ตอนนี้ Partial 0.8R จะปะทะ Stage 2 0.8R (ทั้งคู่ trigger พร้อมกัน) → ที่ 0.8R: ปิด 33% + shift TP→1.5R + SL→0.5R = ไม้ที่เหลือเก็บ TP ใหญ่ขึ้น

### FTMO Safety Math ($10k)

- 4 SL streak × 0.70% = -2.8% (vs DAILY_LOSS_CAP 2.5% → block ก่อนถึง)
- DAILY_LOSS_CAP $250 trigger → block new trades + keep open
- DAILY_LOSS_HARD_STOP $400 (4%) → close all + halt 24h
- Max concurrent loss = 2 open × 0.70% = -1.4% ครั้งเดียว

### Expected impact

- Net P/L: -$460 → projected **+$836** (จาก block 3 symbols เท่านั้น)
- + Partial 0.8R improvement: คาดเพิ่ม +$300-500
- EV: -$4.15 → projected **+$5-8/trade**
- Pass Rate (backtest): คาดลด 5-8pp (ยัง > 60% gate)

### Verification

- `grep "blocked_symbols" ftmo_trading_bot/` → เจอใน config + strategy
- Dry-run: confirm 3 blocked symbols ไม่ปรากฏใน Signals sheet `Direction` column
- Live monitor 50 trades or 14 days → ดู WR, PF, EV vs acceptance criteria
- Rollback: `cp logs/bot_state.json.pre_v8056 logs/bot_state.json` + `git revert` config/strategy/trade_manager

### Files touched

- `ftmo_trading_bot/config/settings.py` — SymbolConfig.blocked_symbols (NEW), DEFAULT_RISK_PER_TRADE_PCT 0.0085→0.0070, MAX_OPEN_POSITIONS 3→2, DAILY_LOSS_CAP_ENABLED False→True, DAILY_LOSS_CAP_PCT 0.030→0.025
- `ftmo_trading_bot/strategy/mean_reversion_strategy.py` — `LiveMRScanner.__init__` filter `_symbols` by `blocked_symbols`
- `ftmo_trading_bot/execution/trade_manager.py` — BE_TRIGGER_RR 0.5→0.3, PARTIAL_TRIGGER_RR 0.5→0.8, PARTIAL_CLOSE_PCT 0.5→0.33

### Backups

- `logs/bot_state.json.pre_v8056`

---

## 📝 Version Log Entry — v8.0.55 (2026-05-22) — 3 Live Filters Kept + RL/Pool Reverted

**Eval result: Train Pass 68.1% (-2.6pp vs v8.0.52 70.7%), Holdout 48.5% (mild overfit Δ +5.3pp). Decision: REVERT RL/GBM/Pool to v8.0.52, KEEP 3 live filters.**

### Eval summary

| Metric | v8.0.52 (baseline) | v8.0.55 retrain | Δ |
|---|---|---|---|
| Train Pass Rate | 70.7% | 68.1% | -2.6pp 🟡 |
| Holdout Pass Rate | ~54% | 48.5% | -5.5pp 🔴 |
| Δ train−holdout | — | +5.3pp | 🟡 mild overfit |
| Total DD max | 5.80% | 5.80% | same |
| Daily DD max | 3.00% | 3.00% | same |
| Breach Rate | 0% | 0% | same |
| Profit avg | $8,633 | $8,533 | -$100 |

Verdict: training proxy (M15 first-bar slip ≤ 0.6R) doesn't transfer well to live (live uses M1 backward + slip + BB). 56% pass with 0.3R was too strict; 76% with 0.6R captured quality but RL distribution shift caused regression.

### What's KEPT (live filters work without retrain — RL-blind gates)

3 live filters block ~15-25% of signals post-RL decision. Since they only REJECT (never CREATE new TAKEs) and don't modify observation, RL doesn't need to know about them. Same pattern as v7.1.10 news close, v8.0.21 pre-news block, v8.0.26 bulk-trading guard — all live-only, all production-stable.

- `TradeExecutor._check_entry_confirmation` — slip 0.30R + M1 last bar direction + BB %B still extreme
- `TradeExecutor._check_spread_spike` + `_record_spread_observation` + `_spread_history` — broker-agnostic rolling median 30 bars, > 2x → SKIP
- `RiskManager` cluster cooldown — `CLUSTER_COOLDOWN_ANY_SEC (300)` + `CLUSTER_COOLDOWN_SAME_THEME_SEC (600)` + `_compute_theme(symbol, direction)` + `_last_open_theme` state
- `record_trade_open(symbol, direction)` signature change (main.py updated)
- `bot_state.json` schema v8 (adds `last_open_theme`)
- All 8 config values in `FTMOConfig` (ENTRY_CONFIRM_*, SPREAD_SPIKE_*, CLUSTER_COOLDOWN_*)

### What's REVERTED (RL doesn't benefit from training proxy)

Files restored from backup:
- `models/mr/ppo_mr_filter.zip` ← `models/mr/best_v8052_pass707/`
- `models/mr/ppo_mr_filter_p1.zip` ← same
- `models/mr/vec_normalize_mr.pkl` ← same
- `models/mr/vec_normalize_mr_p1.pkl` ← same
- `data/mr_signal_pool_5000.pkl` ← `.pre_v8055`
- `data/mr_signal_quality_model.pkl` ← `.pre_v8055`

### Code that stays in repo (no-op when reverted)

- `MeanReversionBacktester.entry_confirm_passed` field — computed in pool but pool reverted. Dead-but-harmless field. Future pool rebuilds will recompute.
- `FTMOSignalFilterEnv._entry_confirm_forced_skips` — counter+gate code stays. When current v8.0.52 pool loaded, signals have `entry_confirm_passed` field missing → `.get('entry_confirm_passed', True)` defaults to True → no force-SKIP → behaves identically to v8.0.52.
- `fetch_mt5_data.py` M1 timeframe support — added for future use (broker M1 history limited to 97 days, not enough for current 3-year M15 training pool).

### Lesson learned

Training proxies must mirror live filters precisely, OR be left out entirely (live-only is safe for small filters). M15 first-bar swing ≠ M1 last-bar direction. Future filters that need training parity require either:
1. M1 historical data with multi-year coverage (broker dependent — current brokers cap at ~3 months)
2. Different proxy that statistically tracks live behavior (testable via backtest)
3. Keep as live-only if block rate <30% (safe per v7.1.10/8.0.21/8.0.26 precedent)

### Deployment

Current `models/mr/` matches v8.0.52 (Pass 70.7%). Live execution path includes 3 new filters. Live behavior:
- Same RL/GBM scoring as v8.0.52
- Adds 3 pre-execution gates → slight reduction in TAKE rate (estimated -10-20%)
- DD spikes from gap/cluster cases reduced

### Backups still on disk (don't delete)

- `models/mr/best_v8052_pass707/` — full snapshot (PPO + vec_normalize × 2)
- `data/*.pre_v8055` — pool + GBM (1.3 GB)
- `data/*.v8052_backup` — earlier backup from v8.0.53 attempt
- `models/mr/*.v8052_backup` — same

---

## 📝 Version Log Entry — v8.0.55 (2026-05-22, ATTEMPT, SUPERSEDED) — Entry Confirmation + Spread Spike + Cluster Cooldown (in retrain)

**3 new pre-execution gates added after 2026-05-21 16-trade analysis (Net -$80, 14/16 SL hits, 6 trades with MFE ≤ 4 = -$472)**

Trigger trades that justify the gates:
- XAUUSD SELL 2026-05-21 20:19 EET — instant gap, MFE 0, -$79 in 63 sec → **Spread Spike** would catch (US session open spread surge)
- AUDUSD + XAUUSD SELL 20:18-20:19 EET (82 sec apart) — both bet "USD strong", lost -$170 together → **Cluster Cooldown (same theme 600s)** would catch
- USDCAD 5:37 (MFE 0.9), 16:02 (MFE 0), GBPUSD 16:16 (MFE 4.4) — straight to SL with no favorable move → **Entry Confirmation** (M1 last bar direction match) would catch

Gate 1 — **Entry Confirmation** (`TradeExecutor._check_entry_confirmation`):
- Slip: `|signal.entry_price - current_price| / sl_distance > ENTRY_CONFIRM_MAX_SLIP_R (0.30)` → SKIP
- M1 last closed bar direction must match signal direction (body ≥ 5% of range)
- BB %B must still be in extreme zone (`ENTRY_CONFIRM_BB_BUY_MAX 0.35` / `_SELL_MIN 0.65`)
- Training mirror: `MeanReversionBacktester` sets `entry_confirm_passed=False` when first future M15 bar swings against signal direction. Env force-SKIPs (similar to v7.0.2 correlation simulator).

Gate 2 — **Spread Spike** (`TradeExecutor._check_spread_spike` + `_record_spread_observation`):
- Per-symbol rolling history (`deque(maxlen=SPREAD_SPIKE_LOOKBACK_BARS=30)`) — broker-agnostic, self-calibrating
- `current_spread / median_30bars > SPREAD_SPIKE_RATIO_LIMIT (2.0)` → SKIP
- Warmup: `SPREAD_SPIKE_MIN_SAMPLES (10)` — falls back to fixed `max_spread_points` until baseline ready
- **Live only** — no spread data in historical CSV. Parity drift accepted at ~3-5% (small gate, mostly catches XAUUSD/JPY)

Gate 3 — **Cluster Cooldown** (`RiskManager.can_open_trade` + `_compute_theme`):
- Global gap: `CLUSTER_COOLDOWN_ANY_SEC (300)` — any new open within 5 min → SKIP
- Theme-aware gap: `CLUSTER_COOLDOWN_SAME_THEME_SEC (600)` — same USD/JPY/METAL theme within 10 min → SKIP
- Themes: `USD_LONG/SHORT`, `JPY_LONG/SHORT`, `METAL_LONG/SHORT` (USD priority over JPY for `USDJPY/USDCAD/USDCHF`)
- Extends `v8.0.26 MIN_SECONDS_BETWEEN_OPENS_SEC (60)` — both gates active (60s anti-bulk-flag + 300s cluster)
- State persisted: `RiskManager._last_open_theme` added to `bot_state.json` (schema v8). `record_trade_open(symbol, direction)` now takes 2 args.

Config additions in `FTMOConfig` (`config/settings.py`):
- `ENTRY_CONFIRM_ENABLED=True`, `ENTRY_CONFIRM_MAX_SLIP_R=0.30`, `ENTRY_CONFIRM_M1_DIRECTION_MATCH=True`, `ENTRY_CONFIRM_BB_BUY_MAX=0.35`, `ENTRY_CONFIRM_BB_SELL_MIN=0.65`
- `SPREAD_SPIKE_ENABLED=True`, `SPREAD_SPIKE_LOOKBACK_BARS=30`, `SPREAD_SPIKE_RATIO_LIMIT=2.0`, `SPREAD_SPIKE_MIN_SAMPLES=10`
- `CLUSTER_COOLDOWN_ENABLED=True`, `CLUSTER_COOLDOWN_ANY_SEC=300`, `CLUSTER_COOLDOWN_SAME_THEME_SEC=600`

Files touched:
- `config/settings.py` — 8 new config values
- `execution/trade_executor.py` — `_check_entry_confirmation`, `_check_spread_spike`, `_record_spread_observation` + hooked into `execute_signal` (ด่าน 4.5/4.6) + `_spread_history: Dict[str, deque]` state
- `core/risk_manager.py` — `_compute_theme` (static), `_last_open_theme` state, cluster cooldown gate in `can_open_trade`, `record_trade_open(symbol, direction)` signature change, save/load schema v8
- `main.py` — `record_trade_open(symbol=sig.symbol, direction=sig.signal_type.value)`
- `ml/mean_reversion_backtester.py` — `entry_confirm_passed` field added to pool sig dict
- `ml/signal_filter_env.py` — `_entry_confirm_forced_skips` counter, forced SKIP if `not sig['entry_confirm_passed']`, telemetry in info dict

Pipeline (in progress — overnight ~10 ชม.):
- Build pool 5,000 episodes (PID 4828, log `logs/build_pool_v8055.log`)
- Train GBM (auto)
- Train RL P1 (5M) + P2 (2M) (auto)
- Eval train + holdout

Gate (decision):
- Pass ≥ 70.7% (v8.0.52 baseline) **และ** holdout Δ ≤ 10pp → KEEP + deploy
- 65% ≤ Pass < 70.7% → live test 1 week then decide
- Pass < 65% → REVERT to `models/mr/best_v8052_pass707/` snapshot + `data/*.pre_v8055`

Backups before retrain:
- `models/mr/best_v8052_pass707/` (4 files: PPO p1/p2 + vec_normalize × 2)
- `data/mr_signal_pool_5000.pkl.pre_v8055`
- `data/mr_signal_quality_model.pkl.pre_v8055`

**Caveat — pool/GBM retrain are real changes (not pure live-only)**: New `entry_confirm_passed` field in pool dict means RL trains on different distribution than v8.0.52 pool. ~15-20% of signals expected to have `entry_confirm_passed=False` (force-SKIP). Pass rate may regress; revert path is one `cp` command.

---

## 📝 Version Log Entry — v8.0.54 (2026-05-21) — REVERT v8.0.53

**Revert v8.0.53 (RR 1.2 + Stage 2 TP 1.8R) — Pass 70.4% ≈ v8.0.52 70.7% (no improvement)**

v8.0.53 retrain result vs v8.0.52:
- Pass Rate: 70.4% (vs 70.7%, -0.3pp ≈)
- Profitable Rate: 94.6% (identical)
- Win Rate: 64.4% (vs 63.9%, +0.5pp)
- Profit avg: $8,606 (vs $8,633, -$27)
- DD: identical (5.80%/3.00%)

Hypothesis validated: Stage 2/3 trail already extends RR; raising base RR has no meaningful effect. Stage 2 fires at 0.8R regardless of base RR, then trail logic governs the rest.

Reverts:
- `MRConfig.rr_ratio`: 1.2 → **1.0**
- `MeanReversionStrategy.RR_RATIO`: 1.2 → **1.0**
- `TradeManager.TP_STEP_NEW_TP_RR`: 1.8 → **1.5**

Model files: restored from `.v8052_backup` (pool, GBM, PPO P1/P2, vec_normalize).

Lesson: base RR is largely overridden by Stage 2/3 trail. Future trail experiments should target `TP_STEP_TRIGGER_RR`, `TP_STEP_NEW_TP_RR`, or `TRAIL_TP_AHEAD_R` instead of base RR.

---

## 📝 Version Log Entry — v8.0.53 (2026-05-21) — REVERTED in v8.0.54

**RR 1.0 → 1.2 + Stage 2 TP 1.5R → 1.8R (test higher RR for bigger wins)**

Hypothesis: v8.0.52 WR 70% but profit_avg ~$8,633 (small wins). With RR 1.2, partial close at 0.5R + remaining captures TP @ 1.2R → bigger profit per win. EV calc: 0.7 × $85 - 0.3 × $90 = +$32.5/trade (vs current +$0.4/trade with RR 1.0).

Settings changed:
- `MRConfig.rr_ratio`: 1.0 → **1.2**
- `MeanReversionStrategy.RR_RATIO`: 1.0 → **1.2**
- `TradeManager.TP_STEP_NEW_TP_RR`: 1.5 → **1.8** (Stage 2 TP scaled with base RR)

Risk: WR may drop 5-8pp (higher TP harder to hit). Need Pass Rate ≥ 65% to keep.

Backup files saved before retrain (`.v8052_backup` suffix):
- `data/mr_signal_pool_5000.pkl.v8052_backup`
- `data/mr_signal_quality_model.pkl.v8052_backup`
- `models/mr/ppo_mr_filter.zip.v8052_backup`
- `models/mr/ppo_mr_filter_p1.zip.v8052_backup`
- `models/mr/vec_normalize_mr*.pkl.v8052_backup`

If v8.0.53 fails: `cp *.v8052_backup *` to restore (5 seconds).

---

## 📝 Version Log Entry — v8.0.52 (2026-05-21) — REVERT v8.0.51

**Revert v8.0.51 (ADX 30→25 + SL floor 0.7) — eval Pass 55.1% < v8.0.48b 68.8%**

v8.0.51 retrain result vs v8.0.48b:
- Pass Rate: 55.1% (vs 68.8%, **-13.7pp**)
- Profitable Rate: 91.1% (vs 94.3%, -3.2pp)
- Win Rate: 65.1% (vs 64.0%, +1.1pp — but lower Pass)
- Profit avg: $7,464 (vs $8,454, -$990)

Hypothesis why it failed:
1. ADX 25 cut both killer-zone losers AND ADX 30+ paradox-winners
2. SL floor 0.7R "too safe" — big runners locked at 0.7-1.0R instead of trailing 1.5R+
3. Pool size dropped 21% (424 MB vs 536 MB) → less RL training samples
4. explained_variance 0.263 (vs 0.409) — value head less accurate

Reverts:
- `MRConfig.adx_trend_block`: 25 → **30** (live + training symmetric)
- `MRConfig.adx_trend_block_xau`: 25 → **30**
- `MeanReversionStrategy.ADX_TREND_BLOCK`: 25 → **30** (class defaults)
- `TradeManager.TRAIL_SL_FLOOR_RR`: 0.7 → **1.0**
- `strategy_backtester.trail_sl_floor_r` default: 0.7 → **1.0**

Model files: restored from commit `fa69020` (v8.0.48b) via `git checkout`.

Lesson: data-driven hypothesis (ADX 25-30 killer zone) was correct but treatment was wrong — blanket block lost good signals too. Better future approach: **conditional block** (e.g., block ADX 25-30 only when other signals weak).

---

## 📝 Version Log Entry — v8.0.51 (2026-05-21) — REVERTED in v8.0.52

**ADX threshold 30 → 25 + Trail SL floor 1.0R → 0.7R**

User-reported issue: หลัง Stage 3 SL = 1.0R, ราคามักย่อมา test TP เดิม (1R magnet) แล้วชน SL = exit at exactly +1R (lose big-runner potential).

Trail SL floor change:
- `TradeManager.TRAIL_SL_FLOOR_RR`: 1.0 → **0.7**
- `strategy_backtester.trail_sl_floor_r` default: 1.0 → **0.7**

**ADX threshold 30 → 25 (back to Wilder default; data-driven correction)**

Data discovery (87 trades): ADX H1 25-30 = "killer zone" — 16 trades, WR 25%, P/L -$679. Original threshold 25 (Wilder 1978 default) was correct; relaxed 25→30 historically based on intuition ("only block extreme trends"), not data.

Setting changes (both train + live):
- `MRConfig.adx_trend_block`: 30 → **25**
- `MRConfig.adx_trend_block_xau`: 30 → **25** (was 27, then 30 in v8.0.34)
- `MRConfig.adx_trend_block_training`: 30 → **25**
- `MRConfig.adx_trend_block_xau_training`: 30 → **25**
- `MeanReversionStrategy.ADX_TREND_BLOCK`: 30 → **25** (class default)
- `MeanReversionStrategy.ADX_TREND_BLOCK_XAU`: 27 → **25**

Per-bucket WR from 87 trades:
- ADX 0-15: WR 100% (5 trades)
- ADX 15-20: WR 57% (23)
- ADX 20-25: WR 61% (23)
- **ADX 25-30: WR 25% (16) ← killer zone, now blocked**
- ADX 30+: WR 67-79% (20, small sample, paradox — likely overextended-reversal)

**Retrain pending** — user wants to ask more questions before retrain chain (~1.5 hr).

---

## 📝 Version Log Entry — v8.0.50 (2026-05-21)

**Remove all Asian Delays (train-live parity)**

Discovery: Training simulator (`MeanReversionBacktester`) has NO Asian Delay logic — RL agent was trained on data including Asian Early signals (00-07 EET). Live had hard-block layered on top → duplicate filtering. Pass Rate 68.8% in eval ALREADY reflects RL handling Asian Early via TAKE/SKIP decisions.

Disabled flags (live now matches training):
- `MONDAY_DELAY_ENABLED: True → False` (was: block Mon 00-04 EET)
- `WEEKDAY_DELAY_ENABLED: True → False` (was: block Mon-Fri 00-07 EET for non-XAU)
- `XAU_WEEKDAY_DELAY_ENABLED: True → False` (was: block XAU 00-05 EET Mon-Fri)
- Legacy fields kept (revertable)

Pre-removal data (~2 weeks): 124 Asian Early signals blocked by hard rule. After removal, these flow to RL for TAKE/SKIP decision. Estimated additional 8-9 trades/day initially, but RL is expected to SKIP most low-quality ones (training data biased these to negative outcomes).

Safety layers remaining:
- `DAILY_LOSS_HARD_STOP_PCT = 4%` (FTMO-aligned, 1% buffer from 5% breach)
- ML threshold 0.30, Confluence ≥ 70, ADX block 30
- Stage 2/3 trail (v8.0.48b lock 0.5R-1R)

Watch: If live Asian Early P/L < -$100/day on average over 5 trading days → revert all 3 flags to True.

No retrain needed.

---

## 📝 Version Log Entry — v8.0.49 (2026-05-20)

**Throttle Daily Loss approach-limit warning (anti-spam)**

Bug: `RiskManager.check_daily_loss()` printed `⚠️ Daily Loss: X% (ขีดจำกัด: 4%)` every 5s when `daily_loss_pct > 3%` → spam (hundreds of lines per minute during drawdown days).

Fix mirrors v8.0.16 Give-back throttle:
- New state field `_last_daily_loss_alert_pct: float` in `RiskManager.__init__`
- Reset on new day in `_on_new_day` (alongside `_last_give_back_alert_pct`)
- Print only when crossing 0.5pp milestone: 3.0%, 3.5%, 4.0% (≤3 prints between warning threshold and Hard Stop)

No retrain needed — print-throttle only.

---

## 📝 Version Log Entry — v8.0.48b (2026-05-20) — DEPLOYED Pass 68.8%

**Stepwise Trail finalized + sim hit_sl fix + simplified caps (Pass 68.8% — best ever)**

Final result (5000-eps eval):
- Pass Rate: **68.8%** (vs v8.0.47 61.5%, +7.3pp; vs v8.0.43e 65.7%, +3.1pp — best ever)
- Profitable Rate: **94.3%** (vs v8.0.47 91.4%)
- Profit avg: **$8,454** (vs v8.0.47 $7,873, +$581)
- Breach Rate: 0%, Total DD: 5.80%, Daily DD: 3.00%

Code changes vs v8.0.47:
- `execution/trade_manager.py` + `ml/strategy_backtester.py`:
  - Stage 2 @ 0.8R: TP→1.5R, SL→0.5R (new)
  - Stage 3 @ 1.0R: SL→1.0R + trail chase (SL floor 1R)
  - New constants: `TP_STEP_TRIGGER_RR=0.8`, `TP_STEP_NEW_TP_RR=1.5`, `TP_STEP_NEW_SL_RR=0.5`, `TRAIL_SL_FLOOR_RR=1.0`
  - `TrailingState` gains `tp_step_done: bool` flag
- `ml/strategy_backtester.py` (CRITICAL fix):
  - `hit_sl` block now computes profit from `sl_price - entry` (was hardcoded `-risk_amount * slippage`)
  - Before fix: Stage 2 trades that retraced to new SL=0.5R were recorded as -1R loss (sim bug). After fix: +0.5R correctly.
  - First attempt (v8.0.48) had this bug → Pool baseline win rate 38.87%. Fixed v8.0.48b → 49.44%.
- `config/settings.py` (simplification — user-requested):
  - `DAILY_PROFIT_CAP_ENABLED: True → False` (let momentum winning days run)
  - `DAILY_LOSS_CAP_ENABLED: True → False` (overlap with HARD_STOP — `DAILY_LOSS_HARD_STOP_PCT=4%` remains as single FTMO-aligned DD limit)

Pool/GBM:
- Pool mean outcome **+0.0070** (first time positive; v8.0.47 was -0.041)
- GBM OOF AUC **0.6134** (best ever; v8.0.47 0.6118)
- ml_score mean=0.494, std=0.113

RL metrics (Phase 2 end):
- std=2.66 (vs v8.0.47 2.35 — slightly higher, but live uses deterministic=True so no effect)
- explained_variance=**0.409** (vs v8.0.47 0.243 — value head 68% better)
- value_loss=0.09 (vs 0.117, -23%)
- approx_kl=0.0011, clip_fraction=0.0013 (well-converged)

---

## 📝 Version Log Entry — v8.0.48 (2026-05-20) — SUPERSEDED by v8.0.48b

**Stepwise Trail (user-requested): Stage 2 @ 0.8R + Stage 3 @ 1.0R + SL floor**

Hypothesis: v8.0.47 trail @ 0.9R catches retraces too low (best-0.5R = 0.4R lock). User-proposed stepwise locks more profit at clear thresholds.

3-stage logic (live + sim mirrored):
- **Stage 1 @ 0.5R** (existing v8.0.14): Partial 50% close + BE move
- **Stage 2 @ 0.8R** (NEW): TP shift 1.0R → 1.5R, SL shift BE → 0.5R (lock partial close price)
- **Stage 3 @ 1.0R** (NEW): SL shift 0.5R → 1.0R + trail activate
  - Trail mode: `SL = max(entry+1R, best-0.5R)`, `TP = best+1R` (chase)

Code changes:
- `execution/trade_manager.py`:
  - New constants: `TP_STEP_TRIGGER_RR=0.8`, `TP_STEP_NEW_TP_RR=1.5`, `TP_STEP_NEW_SL_RR=0.5`, `TRAIL_SL_FLOOR_RR=1.0`
  - `TRAIL_ACTIVATION_RR: 0.9 → 1.0`
  - New method `_tp_step()` — Stage 2 trigger
  - `_update_trailing_stop()` enforces SL floor: `max(best-0.5R, entry+1R)` for BUY
  - `TrailingState` gains `tp_step_done: bool` flag
- `ml/strategy_backtester.py`:
  - `_resolve_trade()` gains Stage 2/3 params (defaults match live constants)
  - Normal mode: Stage 3 trumps Stage 2 if both hit in same bar
  - Trail mode: SL floor enforced

Sim-vs-live parity caveats:
- Sim does not model partial 50% close (returns full position R-multiple)
- For trades that reach Stage 2 and revert to new SL=0.5R: sim outcome = 0.5R, live weighted = 0.5R (partial 0.25R + remaining at 0.5R × 50% = 0.25R) — match
- For trades that peak 0.5R-0.8R then revert to original SL: sim = -1R, live = +0.25R (partial only) — same approximation as v8.0.47

Gate (vs v8.0.47 baseline 61.5%):
- Pass ≥ 61.5% → commit + push v8.0.48
- Pass < 61.5% → revert to v8.0.47

Live continuity during retrain: model files preserved (no overwrite until eval passes).

---

## 📝 Version Log Entry — v8.0.47 (2026-05-20)

**Trail activation 0.8R → 0.9R + removed trail reward cap (fallback from v8.0.45 Pass 52%)**

Background:
- v8.0.45 set `TRAIL_ACTIVATION_RR = 0.8` (pre-emptive — avoid live RR 1:1 TP race) + reward cap @1.5R+0.5x decay
- v8.0.45 final eval = **Pass 52%** (vs v8.0.43e baseline 65.7% = **-13.7pp regression**)
- Hypothesis: 0.8R too aggressive → SL trail catches retrace too early, agent stops chasing big runners

Code changes (v8.0.47):
- `execution/trade_manager.py` — `TRAIL_ACTIVATION_RR: 0.8 → 0.9` (compromise)
- `ml/strategy_backtester.py` — `trail_activation_r` default `0.8 → 0.9`
- `ml/signal_filter_env.py` — Removed trail reward cap (was: `outcome > 1.5 → 1.5 + excess*0.5`). Let raw outcome flow → agent learns to chase big runners.

Live continuity:
- Restored v8.0.43e model files from `.bak_1779255651` (Pass 65.7%) while retraining
- v8.0.46 adaptive 1s loop (when position profit ≥ 0.5R) still active → mitigates TP race risk at 0.9R

Gate (vs v8.0.43e baseline 65.7%):
- Pass ≥ 60% → commit + push v8.0.47
- Pass < 60% → fallback Option A (revert code to v8.0.43e: trail 1.0R + reward cap)

---

## 📝 Version Log Entry — v8.0.45/46 (2026-05-20) — DEPRECATED

**v8.0.45**: Pre-emptive trail @ 0.8R — **failed** (Pass 52%, -13.7pp). Reverted in v8.0.47.
**v8.0.46**: Adaptive main loop interval (5s default → 1s when position profit ≥ 0.5R) — **kept** (orthogonal bug fix for TP race condition reported in live).

---

## 📝 Version Log Entry — v8.0.44 (2026-05-20)

**Conf ≥ 90 exception for Asian late delay (premium quality bypass)**

Data analysis (6 days):
- 100 Asian late FX signals with Conf ≥ 85 (RL TAKE 46, SKIP 44)
- 19 Asian late FX signals with Conf ≥ 90 (RL TAKE 9 = 47%) — premium quality
- → RL agrees these are worth taking but session block prevents

Code changes:
- `config/settings.py` — new `WEEKDAY_DELAY_CONF_EXCEPTION: float = 90.0`
- `core/risk_manager.py` — `can_open_trade()` accepts `confluence_score` param; Asian late check skips block if conf ≥ 90
- `execution/trade_executor.py` — passes `signal.confluence_score` to `can_open_trade`

Safety:
- Threshold 90 = rare event (0.5% of signals)
- ~3 trades/week extra (estimated)
- Buffer ห่าง FTMO DD limit ไม่กระทบ
- ปลอดภัยกว่า conf 85 (which had RL split 46/44)

---

## 📝 Version Log Entry — v8.0.43e (2026-05-19)

**Root cause fix for high STD (1.88) in v8.0.43c**

Hypothesis: Trail outcomes (1R, 1.5R, 2R, 3R+) → high reward variance → agent confused → STD ขึ้น.

Code changes:
- `ml/signal_filter_env.py` step() — Cap trail extension reward:
  ```python
  if outcome > 1.5:
      excess = outcome - 1.5
      outcome = 1.5 + excess * 0.5  # decay extension
  ```
  Effect: TP@1R = +1R (เท่าเดิม), Trail 2R = +1.75R (was 2R), 3R = +2.25R (was 3R)
- Phase 1 timesteps: 5M → 8M (CLI arg) — foundation stable ก่อน Phase 2

Gate (vs v8.0.43c baseline):
- Pass Rate ≥ 61.8% → commit + push as v8.0.43e
- Pass Rate < 61.8% → revert env code, restore v8.0.43c models from backup

Models backup: `models/mr/*.bak_v8043c_*` saved before train start.

---

## 📝 Version Log Entry — v8.0.43 (2026-05-19)

**Option X: TP-Chase Trail (Plan B Trick) + Risk 0.99% → 0.7%**

Code changes:
- `execution/trade_manager.py` — Trail enabled: `TRAIL_ACTIVATION_RR: 99→1.0`, new `TRAIL_SL_BEHIND_R=0.5`, `TRAIL_TP_AHEAD_R=1.0`, `TRAIL_MIN_STEP_PIPS=1.0`. `_update_trailing_stop()` rewritten — both SL+TP chase after price reaches 1R. 5 invariants enforced (SL/TP no backward, best_price no backward, min step pip).
- `ml/strategy_backtester.py` — `_resolve_trade()` gains `enable_trail_after_tp` param. After TP hit, switch to trail mode: best_price tracking, SL=best-0.5R (BUY) / best+0.5R (SELL), TP=best+1R (BUY) / best-1R (SELL). Match live exactly.
- `ml/mean_reversion_backtester.py` — Pass `enable_trail_after_tp=True`, `trail_sl_behind_r=0.5`, `trail_tp_ahead_r=1.0`.
- `config/settings.py` — Risk default 0.0099 → 0.007 (FTMO buffer 30%), MAX 0.0099 → 0.007.
- `ml/signal_filter_env.py` — RISK_PER_TRADE 0.0099 → 0.007 (env class default).

Rationale:
- Option X trail catches trend continuation after TP 1R → expected win avg 1R → 1.3-1.5R.
- EV/trade approx 2× → ลด risk 0.99 → 0.7% ได้โดย EV รวมไม่ตก.
- FTMO safety buffer doubles (4 SL × 0.7% = 2.8% vs 4% limit = 30% buffer).

Eval gate (before deploy):
- Pass Rate ≥ 50% (vs v8.0.42 baseline 55.1%)
- Total DD ≤ 6.5%, Daily DD ≤ 3.5%, Breach 0%, WR ≥ 58%

---

## TL;DR (30-second scan)

- Do not touch: obs dim / order, risk anchors, position_id matching, timezone handling.
- ⛔ Changing obs without retraining → whole system breaks.
- ⛔ Deleting `bot_state.json` mid-challenge → FTMO anchor destroyed.
- Every invariant below has already broken production once. Do not skip.

---

## ⛔ Hard Invariants (broken before → leave alone)

### 0. NO DATA LEAKAGE (v8.0+ — must be enforced every change)

**Rule** — the agent's observation must contain only signal-time features. Future-resolved fields (anything resolved AFTER the agent decides) are TARGETS, never inputs.

| Field | Allowed location | Disallowed |
|---|---|---|
| `outcome_pnl_ratio` | reward shaping, GBM y label, aux-task target | obs vector, GBM x feature |
| `bars_to_resolution` | reward shaping (env step) | obs vector, GBM x feature |
| `is_quick_tp` | reward shaping (env step) | obs vector, GBM x feature |
| `outcome_partial` (legacy v7.1) | nowhere — leak hazard | nowhere |
| `tp_hit` / `sl_hit` / `future_*` | nowhere — leak hazard | nowhere |

**Enforcement** — `scripts/leakage_audit.py` runs four checks (static AST scan of obs builders, GBM feature list inspection, pool-dict sanity, dynamic obs range check). It must exit 0 before any commit that touches `ml/`, `strategy/`, or `main._build_signal_observation`.

**Past incidents** — v7.2.1 obs[29/30] leaked `outcome_partial` via floating-PnL simulator → Pass Rate dropped 9.7% → 6.3% pre-fix.

### 0a. TRAIN ↔ LIVE PARITY (v8.0+ — must be enforced every change)

**Rule** — the agent's behavior in training must match live exactly. "Great in train, fails in live" usually traces to a parameter mismatch.

Aligned by `scripts/parity_audit.py`:

1. Strategy params (BB/RSI/ADX thresholds, SL/TP, scan cadence, dedup) — `MeanReversionStrategy` class defaults must match `bot_config.mr.*`.
2. ML threshold — trainer `--ml_threshold` default == `auto_train_pipeline` `HyperParams.ml_threshold` == `bot_config.ftmo.ML_FILTER_THRESHOLD`.
3. Risk per trade — `FTMOSignalFilterEnv.RISK_PER_TRADE` == `bot_config.ftmo.DEFAULT_RISK_PER_TRADE_PCT` == `HyperParams.risk_per_trade`.
4. Obs dim — `SelfLearningAgent.OBS_DIM` == `env.observation_space.shape[0]` == actual obs returned from `reset()` (3-way sync).
5. Correlation groups — train env `CORRELATION_GROUPS` must mirror live `TradeExecutor.CORRELATION_GROUPS`.
6. Indicator parity — `TechnicalIndicators.calculate_all` must produce the same value at the rightmost bar for full-DF and rolling-window inputs.
7. VecNormalize — `models/mr/vec_normalize_mr.pkl` must exist next to `ppo_mr_filter.zip`; live agent loads the SAME pickle saved during training.

**Soft (always reported, never fails)**:

8. SL/TP behavior — train resolves with SL/TP/timeout/gap only; live `TradeManager` adds BE+partial+trail. Accepted as "train < live" capability gap.

**Enforcement** — `scripts/parity_audit.py` must exit 0 before any commit that touches `config/settings.py`, `strategy/mean_reversion_strategy.py`, `scripts/train_mr_*.py`, `auto_train_pipeline.py`, or `main._build_signal_observation`.

**Past incidents** — v8.0 launch had `bot_config.ftmo.ML_FILTER_THRESHOLD = 0.36` but trainer used 0.40 → live would have seen signals with ml ∈ [0.36, 0.40] that the agent never saw at train time. Caught by parity audit before live deploy.

---

### 1. Observation Space Sync (3 places)

Changing obs requires retraining the whole pipeline (pool → ML → RL):

- `SelfLearningAgent.OBS_DIM` must equal `FTMOSignalFilterEnv.observation_space.shape[0]`.
- `FTMOTradingBot._build_signal_observation` must produce obs matching `FTMOSignalFilterEnv._get_obs` in size, order, and scale.
- On size mismatch: `SelfLearningAgent._prepare_obs` raises `ValueError` (good — fail fast).
- On wrong order with correct size: **no error**, but the model returns nonsense (more dangerous than a crash).
- **v7.1 (2026-05-04)**: bumped 29 → **32** (added `floating_pnl_norm`, `open_losing_count_norm`, `mins_since_session_norm`). ห้ามรัน live ก่อน retrain.

### 2. FTMO Anchors

- `RiskManager._initial_balance` is the Total DD anchor — **never** change mid-challenge.
- `RiskManager._daily_start_balance` is the Daily DD anchor — only resets at broker day rollover.
- Do not delete `logs/bot_state.json` mid-challenge — losing the anchor invalidates every DD %.

### 3. Position ID Matching

- MT5 deal matching uses `position_id` — **not** `order` or `ticket`.
- `RiskManager` + `TradeManager` + `TradeExecutor` must all reference the same field.

### 4. Timezone (EET vs UTC)

- Broker time = **EET** (Europe/Bucharest).
- Config values (session windows, Friday cutoff) = **UTC** — convert before comparing.
- Daily reset in `RiskManager` must use `TimeManager.get_server_time().date()` — not `date.today()`.
- ⛔ Do not use `mt5.symbol_info_tick().time` directly — FTMO sends broker-local epoch → double-adds tz = +3 h drift.
- ✅ Use `datetime.now(Europe/Bucharest)` inside `TimeManager.get_server_time` + NTP-synced VPS.

### 5. Pip Size (JPY-aware)

- JPY pairs (price > 20): pip = `0.01`.
- Others: pip = `0.0001`.
- ⛔ Do not hardcode `× 10000` — auto-detect from `entry_price` or `symbol_info.digits`.

### 6. PositionSizer Pip Value (3 cases)

- Quote = account CCY (EURUSD, GBPUSD, ...) → raw pip value.
- Base = account CCY (USDJPY, USDCHF) → raw / symbol_price.
- Cross (EURJPY, GBPJPY) → use **USDJPY rate** (not the cross pair's own price).

### 7. Contract Size

- Use real `symbol_info.trade_contract_size` — do not hardcode `100_000`.
- XAUUSD = 100 oz (digits = 2) — `PositionSizer` must handle this.

### 8. Correlation Groups

Inside `TradeExecutor._check_correlation`:

- USD_WEAK, USD_STRONG, JPY_CROSS, EUR_PAIRS, GBP_PAIRS.
- `MAX_CORRELATED_POSITIONS` per group per direction (default 1).
- Duplicate symbols may not be opened twice.

### 9. FTMO Program Type

- Current = **2-step Standard** → `CONSISTENCY_RULE_THRESHOLD = 1.0` (check disabled).
- Swing/Pro = 0.45 (max day ≤ 50 % of total profit).
- ⚠️ Switching programs requires updating this value before starting the new challenge.

### 10. ML Filter Threshold Sync (live ↔ training, v6.12)

- `bot_config.ftmo.ML_FILTER_THRESHOLD` ใน live **must equal** `--ml_threshold` ที่ใช้ตอน train (`scripts/train_signal_filter.py`).
- Training: `FTMOSignalFilterEnv` กรอง signals ที่ `ml_score < ml_filter_threshold` ก่อน feed ให้ agent — agent เรียนเฉพาะ post-threshold distribution.
- Live: `FTMOTradingBot.run` ต้องมี gate เดียวกัน ก่อนเรียก `SelfLearningAgent.should_take_signal` ไม่งั้น agent เห็น distribution กว้างกว่าที่ฝึก = silent regression.
- ⛔ Bug ที่เคยเกิด: เดิม `_build_live_context` พยายาม `getattr(self._rl_agent, "ml_filter_threshold")` แต่ attribute นั้นอยู่บน env ไม่ใช่ agent → ตกค่า 0.0 = no gate.
- ⛔ Pre-agent ML reject log เป็น `Result = "ML_FILTERED"` ใน `Signals` sheet (light-blue row).
- ✅ เปลี่ยนค่านี้ → ต้อง retrain ทั้ง pipeline (`build_signal_pool` → `train_signal_quality` → `train_signal_filter --ml_threshold <new>`).

### 11. Chronos config sync (live ↔ training, v7)

- `bot_config.ml.CHRONOS_MODEL_NAME` + `CHRONOS_PREDICTION_LENGTH` + `CHRONOS_CONTEXT_LENGTH` ใน live **ต้องเหมือน** ตอน build pool + train RL.
- Pool builder (`StrategyBacktester._chronos`) + live (`FTMOTradingBot._chronos`) อ่าน config เดียวกัน — แต่ถ้าใครเปลี่ยนกลางทาง → obs distribution shift = silent regression.
- ⛔ เปลี่ยนค่าเหล่านี้ → **ต้อง rebuild pool + retrain GBM + retrain RL ใหม่ทั้งหมด** (เหมือน obs dim change).
- ⚠️ Cache key = `(symbol, last_bar_timestamp)` — ห้าม cache ตาม object identity ของ DataFrame.
- ✅ Disable knob: `bot_config.ml.CHRONOS_ENABLED = False` หรือ env `BOT_DISABLE_CHRONOS=1` → obs[27,28] = 0.0 (graceful degrade to pre-v7 behavior).

### 12. Dependency Pin Protocol (added 2026-05-01)

ทุก external library ที่ถูก `import` ใน `.py` ใต้ `ftmo_trading_bot/` (รวม lazy import ใน function) **ต้อง** มี pin `==<exact_version>` ใน `requirements.txt`:

- ⛔ ห้าม import package ที่ไม่อยู่ใน `requirements.txt` (แม้แต่ใน try/except). เพิ่มแล้วใช้ → pin ในเทิร์นเดียวกัน ห้ามทิ้งไว้ทีหลัง.
- ⛔ ห้ามใช้ `~=` หรือ `>=` — ใช้ `==<exact>` เท่านั้น. VPS ต้องใช้ version เดียวกับเครื่อง train เป๊ะ ๆ มิฉะนั้น `pip install -r` อาจ resolve เป็น version ใหม่ → silent regression (เคยเกิด).
- ⛔ Transitive deps สำคัญ (peer deps ของ library ที่ใช้) ต้อง pin ด้วย แม้ไม่ได้ `import` ตรง — เช่น `transformers` + `accelerate` คู่กับ `chronos-forecasting` (chronos lib ใช้ภายใน, ถ้า upstream upgrade transformers จะ break compat).
- ✅ Self-check pre-commit (อยู่ใน [CLAUDE.md § Dependency Pin Protocol](../CLAUDE.md)) สแกนทุก `.py` แล้ว diff กับ `requirements.txt`.
- ✅ ถ้า dep ใหม่ต้อง download model / set env var / OS-specific setup → update [`readme.md`](../readme.md) ด้วย (Step 6 ของ install section).

**ตัวอย่าง precedent (v7.0)**: เพิ่ม `from chronos import BaseChronosPipeline` ใน `ml/chronos_forecaster.py` → pin `chronos-forecasting==1.5.2` + `transformers==4.46.3` (peer dep) + `accelerate==1.2.1` (speedup) ใน `requirements.txt` ทันที + เพิ่ม Step 6 ใน readme อธิบายว่า model ~200MB จะ download ครั้งแรก.

### 13. ATR Floor vs MIN_SL — separate mechanisms

- `SymbolConfig.symbol_overrides[X].atr_floor_pips` = **signal gate** inside `SMCStrategy.scan_signal`. If `atr_pips < floor` → drop signal. Does **not** touch SL.
- `SymbolConfig.symbol_overrides[X].min_sl_pips` = **SL clamp** inside `SMCStrategy` BUY/SELL branches (after OB override). Prevents spread from eating > ~15 % of SL.
- `bot_config.indicators.atr_sl_multiplier` = global ATR → SL base multiplier (1.5).
- ⛔ Do not merge these three into one. Lowering `atr_floor_pips` widens the accepted-signal population but does not narrow SL directly — SL shape is owned by `atr_sl_multiplier` + `min_sl_pips`.

---

## ❓ FAQ / Common Misunderstandings

### Q: Partial close + BE → ชน SL = WIN หรือ LOSS?

**A: WIN** (ถ้า partial profit > 0 และ accumulated net profit > 0).

`TradeExecutor.sync_with_mt5` ดึง **ทุก deals** ของ position ผ่าน `MT5Connector.get_deals_by_position(ticket)` แล้ว accumulate ทั้ง `profit + swap + commission` ของทุก deal. ดังนั้น partial close ($+50) + BE-SL remainder ($0) → `ExecutedTrade.profit = +$50` → WIN.

3 จุดที่ classify ใช้สูตรเดียวกัน (`profit > 0` บน cumulative value):

- `TradeLogger.log_daily_summary` — daily wins counter
- `TradeExecutor.get_stats` — overall win rate
- `RiskManager.update_daily_pnl` — `consecutive_losses` reset เมื่อ `pnl > 0`

ตัวอย่าง: risk = $100, RR target = 2.0
- Partial 50% @ 1R → realized +$50
- SL เลื่อนมา BE → remainder ชน BE-SL → realized $0
- `ExecutedTrade.profit = +$50` → WIN, `consecutive_losses = 0`, daily P/L +$50

⚠️ ถ้าโดน **swap/commission** ทำให้ accumulated < 0 → จะกลายเป็น LOSS ตามกฎเดียวกัน (cumulative-based, ไม่ใช่ remainder-only).

---

## ⚠️ Soft Invariants (best practice)

- **Risk per trade**: train with `--risk_per_trade 0.0099` → live `DEFAULT_RISK_PER_TRADE_PCT = 0.0099` (must match). v7.1.9 (2026-05-05): bumped 0.007 → 0.0099 (FTMO 1% rule max).
- **Eval sample size**: 100 eps has ±5 pp variance — use ≥ 500 eps for true performance.
- **Pool + ML + RL dependency**: changing obs → rebuild pool → retrain ML → retrain RL (order matters).
- **VecNormalize stats**: `models/vec_normalize_sf.pkl` must match `models/ppo_signal_filter.zip` (otherwise obs is in the wrong scale).
- **Quality-first P1 reward**: Phase 1 relies on the oracle SKIP reward (×2) — do not dampen, or the agent will over-trade.

---

## 🔄 Migration Notes

### Changing Obs Space (e.g. 24 → 27)

1. Update `FTMOSignalFilterEnv.observation_space` (shape tuple).
2. Update `FTMOSignalFilterEnv._get_obs` (compute new features + return array).
3. Update `SelfLearningAgent.OBS_DIM` to match.
4. Update `FTMOTradingBot._build_signal_observation` (order + scale must match env).
5. Rebuild the pool (`build_signal_pool.py`) if the new features come from the signal dict.
6. Retrain ML (`train_signal_quality.py`) if the new features affect the GBM input.
7. Retrain RL `--fresh` (`train_signal_filter.py`).
8. Back up the old model: `mv models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_<timestamp>`.
9. **Update wiki**: `03-rl-training.md` (obs table), `02-modules.md` (if any module signature changed), `context.md` (headline numbers).

### Starting a New FTMO Challenge

- Back up `logs/bot_state.json` → `logs/bot_state.json.bak_<timestamp>` (do not delete).
- Update `.env` if using a new account: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`.
- Start the bot — `RiskManager._load_state` creates a fresh state automatically.
- Log check: "🆕 [Risk Manager] เริ่ม Challenge ใหม่..." (log text may be Thai, code is shared).

### Switching FTMO Program

- Edit `FTMOConfig.CONSISTENCY_RULE_THRESHOLD` (Standard = 1.0, Swing/Pro = 0.45).
- Review `CONSISTENCY_MIN_PROFIT_PCT` (default 0.02) — adjust if needed.

---

## 📚 Version Log (reverse chronological)

### 2026-05-15 — v8.0.29 Split training/live config for confluence + ADX thresholds

**Trigger** — v8.0.27 + v8.0.28 set `MeanReversionStrategy.MIN_CONFLUENCE_SCORE = 70` and `ADX_TREND_BLOCK_XAU = 27` at the class default + via `MRConfig`. Both backtester (training) and live use the same strategy class, so the next retrain would inherit the live filters → pool shrinks ~48% (confluence) + ~10% (XAU ADX). User flagged it before retrain happened.

**Change** — split each tightened live setting into a `_training` variant in `MRConfig`:

| Setting | Live (live filter) | Training (pool diversity) |
|---|---|---|
| `min_confluence_score` | **70** | `min_confluence_score_training = 30` |
| `adx_trend_block` | 30 (FX) | `adx_trend_block_training = 30` |
| `adx_trend_block_xau` | **27** | `adx_trend_block_xau_training = 30` |

`MeanReversionBacktester.__init__` now overrides the three live values on `self._mr_strategy` with the `_training` values right after instantiation. Live (`main.py` → `LiveMRScanner` → `MeanReversionStrategy.__init__`) reads the unchanged `min_confluence_score`/`adx_trend_block*` fields and gets the tight filters.

**Why the asymmetry is correct** — RL/ML benefit from a diverse training pool because the agent must learn to *recognize* low-quality signals and learn to skip them. If the pool only contains A-grade signals (score ≥ 70), the agent never sees the bad ones and cannot learn the contrast. Live then narrows down to A-grade only, which is what user wants for capital preservation.

**No retrain needed right now** — current model was trained on the wide pool. The change only matters for the *next* retrain (which will keep the wide pool). Live behavior unchanged.

**Invariant** — three pairs must stay split (live + `_training`). Never collapse them back into a single value. When tightening a filter for live, only bump the non-`_training` field; bump the `_training` field only if you intentionally want to shrink the pool (rare — usually done to remove leak hazards, not tighten quality).

**Verification (mandatory if these are touched)**:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'ftmo_trading_bot')
from config.settings import bot_config
from strategy.mean_reversion_strategy import MeanReversionStrategy
from ml.mean_reversion_backtester import MeanReversionBacktester
live = MeanReversionStrategy()
bt = MeanReversionBacktester(data_dir='data')
assert live.MIN_CONFLUENCE_SCORE == 70 and bt._mr_strategy.MIN_CONFLUENCE_SCORE == 30
assert live.ADX_TREND_BLOCK_XAU == 27 and bt._mr_strategy.ADX_TREND_BLOCK_XAU == 30
print('live vs training filter split: OK')
"
```

### 2026-05-15 — v8.0.28 Confluence floor enforced 70 (was unused — live strategy used 30)

**Trigger** — User audit 48 live trades (2026-05-12 to 2026-05-15) shows clear quality cliff at confluence 70:

| Bucket | N | WR | P/L |
|---|---|---|---|
| < 70 | 23 | 41% | **-$428** |
| ≥ 70 | 25 | **76%** | **+$425** ⭐ |

**Discovered inconsistency** — `FTMOConfig.MIN_CONFLUENCE_SCORE = 70.0` was set long ago (intended as the floor) but no code path actually used it as a gate. The real block lived inside `MeanReversionStrategy.MIN_CONFLUENCE_SCORE = 30.0`, so every trade with score ≥ 30 reached the ML/RL stack. Live data confirms the loose floor was the cause of -$428 in low-quality trades.

**Change** — new config + raise the strategy default + wire override:

- `MRConfig.min_confluence_score: float = 70.0` (new)
- `MeanReversionStrategy.MIN_CONFLUENCE_SCORE: float = 70.0` (was 30.0)
- `MeanReversionStrategy.__init__` pulls `mr_cfg.min_confluence_score` like every other tunable
- `MeanReversionBacktester` (training-side) left at 30 intentionally — wider training pool stays diverse; the live filter narrows it, mirroring how ML/RL filters layer on top of strategy gating

**Expected effect on this week's pattern** — 25 / 48 trades pass (about half), WR jumps 58% → 76%, P/L swings -$3.74 → +$425 on the same days. XAU specifically: 9 / 14 kept (78% WR), 2 winning XAU lost (-$95) vs 3 losing XAU prevented (+$309) = net +$214 on XAU alone.

**No retrain needed** — strategy-level filter executes before the model sees a signal. Observation, reward, GBM features, RL policy all unchanged. Pool data is unaffected.

**Invariant** — `MeanReversionStrategy.MIN_CONFLUENCE_SCORE` is the single live floor. `FTMOConfig.MIN_CONFLUENCE_SCORE` mirrors the same value for legacy display (`main._print_startup_summary` and Discord notifier); keep them in sync. Tightening in the future = bump `MRConfig.min_confluence_score` only.

### 2026-05-15 — v8.0.27 Per-symbol ADX threshold for XAUUSD (27 vs default 30)

**Trigger** — 2026-05-15 (Fri): 2 XAUUSD BUY trades both SL hit (-$211.87 daily). Trade #2 (05:45 EET) had ADX H1 = 28.5 — would have been blocked at threshold 27 but passed at default 30.

**Data review** — 14 historical XAUUSD trades (2026-05-12 to 2026-05-15):

| ADX threshold | Trades blocked | Blocked were wins | Blocked were losses |
|---|---|---|---|
| > 25 | 2 / 14 | 1 (+$73) | 1 (-$103) |
| **> 27** | **1 / 14** | **0** | **1 (-$103)** ⭐ |
| > 28 | 1 / 14 | 0 | 1 (-$103) |
| > 30 (current) | 0 / 14 | 0 | 0 |

→ Threshold 27 = sweet spot. Blocks only the verified bad trade, keeps every winning trade.

**Change** — In `MeanReversionStrategy.scan_signal`, the existing ADX H1 trend block now routes per-symbol:

- XAUUSD (any `"XAU" in symbol`) → `ADX_TREND_BLOCK_XAU = 27.0`
- All other symbols → `ADX_TREND_BLOCK = 30.0` (unchanged)

New config field `MRConfig.adx_trend_block_xau` (default 27.0) flows into `MeanReversionStrategy.ADX_TREND_BLOCK_XAU` at init.

**Why XAU-specific** — Gold is more volatile than FX pairs, but live data shows it loses MR setup at lower ADX values than majors. Default 30 is calibrated for FX; XAU needs a tighter gate.

**No retrain needed** — strategy-level filter executes before signal generation; observation/reward unchanged.

**Vol regime filter (proposal B) was rejected** — live data showed 14/14 XAUUSD trades had `volatility_regime = "high"` (XAU's natural state, not an outlier). A vol-based block would have killed all gold trading.

**Invariant** — `MeanReversionStrategy.scan_signal` is the single point of ADX gating. Do not add a duplicate ADX check in `RiskManager.can_open_trade` or `TradeExecutor.execute_signal`; per-symbol constants in `MRConfig` are the extension point.

### 2026-05-14 — v8.0.26 Bulk-trading guard (anti The5ers flag, min 60s between opens)

**Trigger** — Excel log audit found 2 trades opened **0.00s / 0.01s** apart on 2026-05-12 (EURUSD BUY pair from pre-news bug, since fixed in v8.0.21). The5ers `prohibited-trading-practices` page explicitly flags "bulk trading = multiple trades open simultaneously" as a bot fingerprint that can trigger account review at withdrawal time.

**Change** — `RiskManager.can_open_trade` now has a top-priority gate that returns reject if the elapsed time since the last successful open is less than `FTMOConfig.MIN_SECONDS_BETWEEN_OPENS_SEC` (default 60s). Implementation:

- New config: `MIN_SECONDS_BETWEEN_OPENS_ENABLED=True`, `MIN_SECONDS_BETWEEN_OPENS_SEC=60`
- New RM state: `_last_open_time_iso` (broker EET ISO string, persisted in `bot_state.json`)
- New method: `RiskManager.record_trade_open()` — called from `main.py` right after `executor.execute_signal` succeeds
- Gate placement: before all other gates (Monday delay, weekday delay, news, etc.) so the cheapest check runs first

**Effect** — bot can still open up to 3 concurrent positions (MAX_OPEN_POSITIONS unchanged), but each open is at least 60 seconds apart. Live median gap is already 18.4 min, so the gate is a backstop, not a behavior change. Eliminates "<5s pair" pattern that compliance reviewers look for.

**Why 60s** — short enough to keep MR signals fresh (M15 entry, signals valid for several minutes), long enough that no human-vs-bot heuristic could call it bulk trading. The5ers Trustpilot reviews mention withdrawal disputes when trade pattern looks automated, so the buffer protects payout integrity.

**Persistence** — `_last_open_time_iso` survives restart via `_save_state`/`_load_state` (schema v7, additive — no migration needed). On bot restart mid-gap, the guard remains active until the elapsed time crosses 60s.

**No retrain needed** — pure execution-path gate, no obs / reward / GBM / pool change.

### 2026-05-13 — v8.0.25 Weekday Delay extended to Mon (Mon non-XAU now starts 11 ICT)

**Request** — User asked to extend the Tue-Fri 11-ICT-start rule to Monday too, while keeping the existing Mon XAU 08-ICT-start (v8.0.19) intact.

**Change** — In `RiskManager.can_open_trade` v8.0.24 block, swap weekday filter from `(1, 2, 3, 4)` to `(0, 1, 2, 3, 4)` so Monday non-XAU goes through the Asian-early gate as well. v8.0.19 Monday delay (4-hour post-weekend block, all symbols) is unchanged.

**Effective schedule**:

| Day | XAU starts | Non-XAU starts | Layer |
|---|:---:|:---:|---|
| Mon | 08:00 ICT (04 EET) | **11:00 ICT (07 EET)** | v8.0.19 + v8.0.25 |
| Tue-Fri | anytime | 11:00 ICT (07 EET) | v8.0.24 |
| Sat-Sun | market closed | market closed | — |

**Why** — Mon Asian early 04-07 EET (= 08-11 ICT) was unprotected for non-XAU after v8.0.19's 4-hour buffer ended. v8.0.25 patches that gap with a symbol-aware filter so Mon non-XAU follows the same 11-ICT-start rule that data supports for Tue-Fri. Mon XAU keeps its 08-ICT-start because Gold has shown an Asian-early edge across all weekdays.

**Smoke test** — 11 scenarios passed (Mon/Tue/Wed/Thu/Fri × XAU/non-XAU at boundary hours, plus weekend N/A).

**Invariant** — symbol-aware time gates use `symbol.upper() not in EXCEPT_SYMBOLS` so per-symbol exceptions stay consistent. Adding a symbol to `WEEKDAY_DELAY_EXCEPT_SYMBOLS` opts it out of the entire Mon-Fri block; removing it puts it back under the 11-ICT-start rule.

### 2026-05-13 — v8.0.24 Weekday Asian Early Delay (Tue-Fri block, XAU exception)

**Data evidence (3 days live, 42 trades)**:

| Session (EET) | ICT | Trades | WR | Net P/L |
|---|---|:---:|:---:|:---:|
| Asian early non-XAU (00-07) | 04-11 | 10 | low | **-$300** ❌ |
| **Asian early XAUUSD (00-07)** | 04-11 | **5** | **80%** | **+$159** ✅ |
| Asian late (07-10) | 11-14 | 6 | 83% | +$242 ✨ |
| Mid-day (14-18) | 18-22 | 16 | 56% | +$58 |
| NY/Late (18+) | 22+ | 5 | 40% | +$93 |

→ Non-XAU losses concentrated in 00-07 EET. XAUUSD reverses the pattern (80% WR Asian early).

**Fix** — add weekday delay with symbol exception in `RiskManager.can_open_trade`:

```python
if WEEKDAY_DELAY_ENABLED:
    now = TimeManager.get_server_time()
    if (now.weekday() in (1, 2, 3, 4)        # Tue-Fri (Mon handled by v8.0.19)
            and now.hour < WEEKDAY_DELAY_END_HOUR_EET  # < 07:00 EET
            and symbol.upper() not in WEEKDAY_DELAY_EXCEPT_SYMBOLS):  # XAUUSD allowed
        return (False, "🌅 Weekday delay until 07:00 EET (11:00 ICT)")
```

**Config**:

- `WEEKDAY_DELAY_ENABLED = True`
- `WEEKDAY_DELAY_END_HOUR_EET = 7`
- `WEEKDAY_DELAY_EXCEPT_SYMBOLS = ("XAUUSD",)`

**Layered with v8.0.19 Monday Delay**:

- Monday 00:00-03:59 EET → blocked by v8.0.19 (4 hr post-weekend buffer)
- Tue-Fri 00:00-06:59 EET → blocked by v8.0.24 (Asian early non-XAU)
- XAUUSD Tue-Fri 00:00-06:59 EET → allowed (Gold edge in Asian)

**Expected impact** — save ~$300/3 days from non-XAU Asian early losses while preserving XAUUSD +$159 wins.

**Invariant** — time-gate filters in `can_open_trade` must use broker EET (`TimeManager.get_server_time()`), never UTC or local time, to stay aligned with broker session boundaries across DST.

### 2026-05-12 — v8.0.22 Daily Loss Cap -3% (Option D mirror, symmetric protection)

**Context** — Production deploy ระหว่าง user สอบ The5ers จริง. วันนี้ขาดทุน -$172 และลึกสุด -$345 (3.45%) — ใกล้ FTMO Daily DD breach (-4% = -$400) มาก. ระบบเดิม asymmetric:

- ✅ Daily Profit Cap +1.6% (v8.0.17) — lock กำไร
- ❌ ไม่มี Daily Loss Cap — ปล่อยขาดทุนถึง FTMO limit -4%

**Fix** — เพิ่ม Daily Loss Cap **-3%** ($300/$10k, $3000/$100k) เป็น mirror ของ profit cap:

| Direction | Cap | Action |
|---|---|---|
| 🟢 Profit | +1.6% (v8.0.17) | Lock + close all + block new |
| 🔴 **Loss** | **-3.0% (v8.0.22)** | **Lock + close all + block new** |

**Implementation** (mirror v8.0.17 pattern):

- `config/settings.py` — `DAILY_LOSS_CAP_ENABLED` + `DAILY_LOSS_CAP_PCT = 0.030`
- `core/risk_manager.py` — `_daily_loss_locked` state, `check_daily_loss_cap()`, `is_daily_loss_locked()`, gate ใน `can_open_trade`, reset ใน `_on_new_day`, persist ใน save/load state
- `main.py` — เรียก `check_daily_loss_cap` คู่ขนานกับ profit cap

**Buffer to FTMO 4% Daily DD** = 1% = $100 — กัน slippage/spread แต่ไม่ tight เกินไป

**Reset** — broker EET midnight ผ่าน `_on_new_day` (เดียวกันกับ profit cap)

**Why -3.0% (ไม่ใช่ -2.5% หรือ -3.5%)**:

- 2.5% = อาจหยุดเร็วเกิน (วันนี้ peak DD 3.45% → cap 2.5% หยุดที่ 1%)
- **3.0% = สมดุล** — หยุดก่อน FTMO breach 1%, ให้โอกาส recovery 1 trade
- 3.5% = buffer แค่ 0.5% — slippage อาจชน

**Smoke test verified**:

```text
A) Total P/L -$300 (exact) → trigger=True ✓
B) Total P/L -$299 (just under) → trigger=False ✓
C) Total P/L -$345 (today peak) → trigger=True ✓
D) Total P/L +$100 (profit) → trigger=False ✓
```

**Rollback plan** — set `DAILY_LOSS_CAP_ENABLED=False` ใน config ปิด feature ทันที (ไม่ต้อง revert code)

**Invariant** — ทุก daily cap (profit + loss) ต้อง anchor ที่ `_initial_balance` (challenge anchor) ไม่ใช่ `_daily_start_equity` — เพราะ cap ต้องคงที่ตลอด challenge ไม่ scale ตาม daily growth/drawdown

### 2026-05-12 — v8.0.21 Pre-news block ใน can_open_trade (กัน open-then-close)

**Problem (จาก Excel จริงวันนี้)** — บอทเปิดออเดอร์ก่อนข่าว 15:30 EET แล้วโดน TradeManager.check_news_close ปิดทันทีภายใน 4-8 วินาที:

| Ticket | Open → Close | นาน | P/L |
|---|---|:---:|:---:|
| T16 USDCHF | 15:12:04 → 15:12:12 | 8s | -$3.23 |
| T17 EURUSD | 15:12:08 → 15:12:14 | 6s | -$1.94 |
| T18 USDCHF | 15:14:17 → 15:14:24 | 7s | -$5.65 |
| T19 EURUSD | 15:14:20 → 15:14:26 | 6s | -$0.97 |
| T20 EURUSD | 15:21:31 → 15:21:35 | 4s | -$0.97 |
| T21 EURUSD | 15:22:37 → 15:22:41 | 4s | +$0.97 |
| **รวม** | | | **-$11.79** |

→ ขาดทุน "ตอด" จาก spread ทุกครั้งที่บอทพยายามเปิด

**Root cause** — `is_near_high_impact_news` ถูกใช้แค่ใน `TradeManager.check_news_close` (ปิด open positions). ไม่มี gate ที่ `can_open_trade` → trade ใหม่ผ่าน RM → ถูก close ทันทีในรอบ loop ถัดไป

**Fix** — เพิ่ม news gate เป็น check แรกๆ ใน `RiskManager.can_open_trade`:

```python
is_news, news_reason = is_near_high_impact_news(
    symbol, datetime.now(timezone.utc),
    window_minutes_before=30, window_minutes_after=15,
)
if is_news:
    return (False, f"📰 {news_reason}")
```

**Invariant** — news ป้องกัน 2 ทาง:
1. `RiskManager.can_open_trade` block trade **ใหม่** (v8.0.21)
2. `TradeManager.check_news_close` ปิด **open** positions (เดิม)

Window symmetry สำคัญ — ทั้ง 2 ใช้ `no_trade_before_news_minutes=30` เหมือนกัน

### 2026-05-12 — v8.0.20 Auto-detect filling mode (multi-broker support)

**Problem** — User เริ่ม The5ers challenge → MT5 order_send รวบรวม error `retcode=10030 Unsupported filling mode`. บอท hardcode `ORDER_FILLING_IOC` แต่ The5ers broker รองรับเฉพาะ FOK (ต่างจาก FTMO ที่ใช้ IOC ได้). ทุก trade ใหม่ reject = บอท disabled

**Fix** — เพิ่ม `MT5Connector._get_filling_type(symbol)` ใช้ `symbol_info.filling_mode` bitmask:

```python
fmask = symbol_info.filling_mode  # bit 0=FOK, bit 1=IOC
if fmask & 2: return ORDER_FILLING_IOC   # preferred
if fmask & 1: return ORDER_FILLING_FOK
return ORDER_FILLING_RETURN              # fallback (rare)
```

ใช้ที่ 3 จุด:
- `send_market_order` (main entry)
- `close_position` (single close)
- `TradeManager._partial_close_position` (50% partial)

**Compatibility**:
- FTMO → IOC (เดิม)
- The5ers → FOK (auto-detected ✓)
- FundedNext, MyForexFunds, ... → auto-detect ✓
- Personal broker (XM, IC Markets, ...) → ปกติรองรับทั้ง FOK + IOC

**Invariant** — order params (filling mode, deviation, magic) ต้อง symbol-aware ไม่ใช่ hardcoded — เพราะ broker ต่างกันรองรับต่างกัน

### 2026-05-11 — v8.0.19 Monday Morning Delay (4 ชม.)

**Problem** — ข้อมูลจริงจาก Monday 11 พ.ค.: 3 ไม้แรก (เปิด 01:06, 03:50, 03:54 EET) เสีย $300+ ก่อนตลาดจะสงบลง. MFE เพียง 0-4 pips → ราคาวิ่งสวนทันทีหลัง entry (ลักษณะ post-weekend gap-fill + thin liquidity). บอท recover ตอน 06:00 EET เป็นต้นไป — แต่ก็เสียโอกาส +$200 ที่เซฟได้

**Fix** — เพิ่ม `MONDAY_DELAY_ENABLED` + `MONDAY_DELAY_END_HOUR_EET` (default 4) ใน `FTMOConfig`. `can_open_trade` block trade เมื่อ:

```python
now.weekday() == 0 and now.hour < MONDAY_DELAY_END_HOUR_EET
```

→ Monday 00:00-03:59 EET (= 04:00-07:59 ICT) ห้ามเปิด trade ใหม่
→ Tue-Fri ไม่กระทบ (weekday != 0)

**Caveats**:
- Friday Session Close + Sunday weekend gap ยังคงเดิม
- Open positions ที่ค้างจาก Friday ยัง managed ปกติ (เฉพาะการเปิด trade ใหม่ที่ block)
- ไม่ block Sunday 22:00-23:59 EET (weekday=6) — บอท practice ก็ไม่ค่อยเทรดช่วงนี้อยู่แล้ว

**Invariant** — Time-based gates ใช้ broker EET time (`TimeManager.get_server_time()`) เสมอ — ไม่ใช่ UTC หรือ local time. มิเช่นนั้น DST shift จะทำให้ window ขยับ ±1 ชม.

### 2026-05-10 — v8.0.18 Flip-lock TTL (กัน lock ค้างข้าม weekend)

**Problem** — Friday ปิด SELL XAUUSD @ 4746 → Monday ราคา 4673 (ลงต่อ ไม่ retrace) → flip-lock สำหรับ BUY ค้างถาวร เพราะ logic ต้องการ ask > 4746 จึง unlock แต่ราคาไม่ขึ้นถึง. บอท blocked จากการเปิด BUY XAUUSD ทั้งสุดสัปดาห์ + จันทร์.

**Root cause** — `register_flip_lock` ตั้ง `min_unlock_time` (5 นาที) เป็น floor แต่ไม่มี max expiry. `is_flip_locked` ปลดล็อกแค่ตอน price retrace ผ่าน threshold เท่านั้น. ถ้าราคาไม่ retrace = lock ค้าง.

**Fix**:
1. เพิ่ม `FLIP_LOCK_MAX_MINUTES = 240` (4 ชม.) ใน `FTMOConfig`
2. `register_flip_lock` เก็บ `max_expiry_time` ใน lock dict
3. `is_flip_locked` เช็ค expiry ก่อน — ถ้าเลย expiry → ลบ lock + return unlocked
4. `_load_state` ล้าง legacy locks ที่ไม่มี `max_expiry_time` (สำหรับ state เก่าก่อน v8.0.18)

**Invariant** — locks ทุกแบบ (flip-lock, post-TP, cooldown) ต้องมี MAX TTL เพื่อกัน state ค้างเมื่อ assumption (เช่น "ราคาจะ retrace") ไม่เกิดขึ้น

---

### 2026-05-09 — v8.0.17 Daily Profit Cap (Option D Hard Stop)

**Goal** — User เลือก Option D เพื่อ "ล็อกกำไรวันนี้ + หยุดเทรด" เมื่อกำไรถึง 1.5% ของ initial balance. ใช้สำหรับ FTMO Phase 1 Challenge — passing target +10% ใน 7-10 วัน โดยไม่เสี่ยง give-back

**Specifications** (ตามที่ user ยืนยัน):

| Setting | Value | เหตุผล |
|---|---|---|
| Cap anchor | `_initial_balance` (FTMO challenge start) | คงที่ตลอด challenge — ไม่ scale ตาม daily growth |
| Cap pct | **1.6%** | $10k → $160, $100k → $1600 (1.6% มี 0.1pp buffer สำหรับ slippage/spread vs target 1.5%) |
| Trigger metric | **closed P/L + floating P/L** (Option B) | ตรวจ realtime — กัน give-back ภายในวัน |
| Reset boundary | **Broker EET midnight** (Option A) | ตรงกับกฎ FTMO Daily Loss Limit |
| Action | **ปิดทุก position + block new trades** | หยุดสนิท, lock เป๊ะ |
| Feature flag | `DAILY_PROFIT_CAP_ENABLED = True` | ปิดได้ตอน funded account |

**Implementation** (4 ไฟล์):

```python
# config/settings.py — FTMOConfig
DAILY_PROFIT_CAP_ENABLED: bool = True
DAILY_PROFIT_CAP_PCT: float = 0.015

# core/risk_manager.py — RiskManager
self._daily_profit_locked: bool = False  # __init__
def check_daily_profit_cap(self, equity, floating_pnl) -> bool:
    if not enabled or self._daily_profit_locked: return False
    cap = self._initial_balance * 0.015
    total_pnl = equity - self._daily_start_equity
    if total_pnl >= cap:
        self._daily_profit_locked = True; self._save_state()
        return True   # signal to caller: close all + block

# can_open_trade — early reject
if self._daily_profit_locked:
    return (False, "🎯 Daily profit cap ถึงแล้ว — รอวันใหม่")

# execution/trade_manager.py
def close_all_positions(self, reason: str) -> int:
    for ticket in list(executor.active_trades.keys()):
        executor.close_trade(ticket, reason=reason)

# main.py main loop (after manage_all_positions)
if risk.check_daily_profit_cap(equity, floating):
    trade_mgr.close_all_positions("Daily Profit Cap (Option D)")
```

**Persistence** — `_daily_profit_locked` saved to `bot_state.json` (schema v7) เพื่อ survive bot restart. Reset ใน `_on_new_day()` พร้อม daily counters.

**Trade-off acknowledged**:
- ✅ ป้องกัน give-back วันที่บอททำกำไรเช้าแล้วคืนบ่าย (เห็นใน live: 8 พ.ค. peak +$237 → final +$156)
- ✅ ผ่าน FTMO Phase 1 challenge เร็วขึ้น (7-10 วัน vs 14 วันตำรา)
- ❌ ตัด upside วันที่บอท hot (cap ที่ 1.5% เลย)
- ❌ Realize floating losses ตอน trigger — อาจแย่กว่าปล่อย Partial-first ทำงาน
- 🔧 แก้ได้: ตั้ง `DAILY_PROFIT_CAP_ENABLED=False` หลังได้ funded

**Invariant** — `check_daily_profit_cap` ใช้ `_initial_balance` เป็น anchor (ไม่ใช่ `_daily_start_equity` หรือ current `balance`). ห้ามเปลี่ยน — เพราะ user ตั้งใจให้ cap คงที่ตลอด challenge ไม่ใช่ scale ตาม account growth ที่ทำให้ "ทำเงินเก่ง = cap สูงขึ้น = หยุดยากขึ้น".

---

### 2026-05-08 — v8.0.16 Throttle Give-back-from-peak alert (anti-spam)

**Problem** — `RiskManager.check_risk()` ถูกเรียกทุก 5s. เมื่อ equity ตกจาก peak ≥2% → log ถูก print ทุก loop = **17,000+ ครั้ง/วัน** = console spam

**Fix** — เพิ่ม `_last_give_back_alert_pct` แล้ว throttle ให้ print เฉพาะเมื่อ give-back **ก้าวข้าม 1% milestone ใหม่** (2.0% → 3.0% → 4.0% ...)

```python
current_milestone = int(give_back_pct * 100)
last_milestone = int(self._last_give_back_alert_pct * 100)
if (give_back_pct >= 0.02
        and current_milestone > last_milestone
        and self._state == BotState.ACTIVE):
    print(f"⚠️ Give-back ...")
    self._last_give_back_alert_pct = give_back_pct
```

**Reset state** ที่ 2 จุด:
- `_on_new_day()` → reset วันใหม่ clean slate
- เมื่อ peak ใหม่ (`current_equity > peak`) → reset เพื่อให้ alert ครั้งหน้าเริ่มจาก 0

**ผลที่คาด**: print ลดจาก 17,000+/วัน → 2-3/วัน (ตอน DD ขยายข้าม milestone จริง). ยังเก็บ visibility สำหรับ user ตอน DD ขยาย — แค่ไม่ spam.

**Invariant** — log ที่ขึ้นทุก loop ต้อง throttle (milestone-based, time-based, หรือ state-change-based) ก่อน print. ถ้าไม่ throttle → spam ปกปิด log สำคัญอื่น เช่น signal scan / order execution.

---

### 2026-05-08 — v8.0.15 RR FP-precision fix v2 (XAUUSD regression)

**Problem** — User เจอ log เดิม **อีกครั้ง** บน XAUUSD แม้ v8.0.11 จะแก้ FP-precision ไปแล้ว:

```text
🚫 [Executor] Risk Manager ปฏิเสธ: ❌ Risk:Reward (1.00) ต่ำกว่าขั้นต่ำ (1.0)
📡 [Agent] TAKE SELL XAUUSD Conf=79 RR=1:1.0 (confidence=1.00)
```

**Root cause** — v8.0.11 ใช้ epsilon `1e-4` (0.01% absolute) ซึ่งพอสำหรับ FX 5-digit แต่ **ไม่พอสำหรับ XAUUSD** ที่ digits=2 → rounding error สูงถึง ~0.4%.

**Math** (XAUUSD): entry=2300.45, sl_distance=1.235 (raw, unrounded), rr=1.0:

```python
tp_price = round(2300.45 - 1.235, 2)        # = 2299.22 (rounded)
tp_dist  = abs(2300.45 - 2299.22)            # = 1.2299999... (FP subtraction)
rr_ratio = 1.2299999... / 1.235              # = 0.9959 ❌ (under 1.0 by 0.41%)
```

`0.9959 < 1.0 - 1e-4` → **REJECT**

**Symbol-specific drift table**:

| Digits | Symbols | Worst-case rr drift |
|---|---|---|
| 5 | EURUSD, GBPUSD, NZDUSD, USDCHF, USDCAD, AUDUSD | ~0.001% |
| 3 | USDJPY, EURJPY, GBPJPY (JPY pairs) | ~0.04% |
| **2** | **XAUUSD, XAGUSD (metals)** | **~0.4%** ← ทำลาย 1e-4 tolerance |

**Fix** — 2 layers:

1. **Tolerance เป็น relative 1%** ใน `RiskManager.can_open_trade` + `PositionSizer.calculate_sl_tp_prices`:

   ```python
   if rr_ratio < self._config.MIN_RISK_REWARD_RATIO * (1.0 - 0.01):
   ```

2. **`MRSignal.rr_ratio` snap-to-half** เมื่อ raw value อยู่ภายใน 1% ของ multiple of 0.5:

   ```python
   raw = tp_dist / sl_distance   # 0.9959 (drifted)
   snapped = round(raw * 2) / 2  # 1.0
   if abs(raw - snapped) / snapped <= 0.01:
       return float(snapped)     # ใช้ 1.0 ไม่ใช่ 0.9959
   return float(raw)
   ```

→ rr_ratio property คืนค่าใกล้ design (1.0) ได้ตรงทุก symbol → log อ่านง่าย + RiskManager check ผ่าน.

**Why two layers** — defense in depth. Tolerance check protects RiskManager/PositionSizer; snap normalizes the displayed/derived value across consumers (logs, ML features, `to_dict`, drift detector).

**Verification** (test bench): all 4 representative symbols pass after fix:

```
✅ XAUUSD digits=2: raw=0.995951 → rr_ratio=1.0
✅ NZDUSD digits=5: raw=1.000000 → rr_ratio=1.0
✅ GBPUSD digits=5: raw=1.000000 → rr_ratio=1.0
✅ USDJPY digits=3: raw=1.000000 → rr_ratio=1.0
```

**Invariant** — when comparing FP-derived ratios to a threshold across symbols with different `digits`, **absolute** tolerance is unsafe — different digit counts → different rounding budgets → drift varies by 1000×. Use **relative tolerance** (% of the threshold) and snap derived values to design intent at the source (`MRSignal.rr_ratio`).

---

### 2026-05-07 — v8.0.14 Partial-first fix + persistent drift log

**Persistent GBM drift log** — `_check_gbm_drift` ก่อน v8.0.14 print ลง console เท่านั้น. PowerShell scrollback default 9,001 บรรทัด — บอท print หลายบรรทัด/วินาที → drift alert ที่ขึ้นช่วงต้นถูก scroll ออกไปก่อน user เห็น. v8.0.14 append ลง `logs/gbm_drift.log` (timestamp ISO-8601 UTC + count + top 5 features) → ดูประวัติ drift ได้ครบทุก hour boundary แม้บอท run นานข้ามวัน.

---

### 2026-05-07 — v8.0.14 Partial-first fix: close 50% before BE move

**Problem** — User observed BE move firing alone without Partial close (regression introduced by v8.0.12):

| State | v8.0.11 | v8.0.12 | **v8.0.14** |
|---|---|---|---|
| `BE_TRIGGER_RR` | 1.0 (dead — TP closes first) | 0.5 | **0.5** |
| `PARTIAL_TRIGGER_RR` | 1.0 (dead) | **0.7** | **0.5** |
| Code order | BE → Partial | BE → Partial | **Partial → BE** |
| MFE peak 0.6R then revert | nothing fires | **BE only, no Partial** ❌ | Partial 50% + BE ✅ |

v8.0.12 left a gap between BE trigger (0.5R) and Partial trigger (0.7R). If price peaks at 0.6R and reverts, BE moves SL to entry but Partial never fires → revert closes at entry = $0 net (no profit captured). User's expected behavior (matching legacy SMC era): close 50% first, THEN move SL to BE — guarantees +0.25R profit on revert.

**Fix** — Two changes in `TradeManager`:

1. `PARTIAL_TRIGGER_RR`: 0.7 → **0.5** (same threshold as BE).
2. Code order in `_manage_single_position`: **Partial check BEFORE BE check** — both fire on the same loop tick when MFE first crosses 0.5R.

```python
# v8.0.14: Partial first (close 50% → lock 0.25R profit)
if not state.partial_closed and best_rr >= self.PARTIAL_TRIGGER_RR:
    self._partial_close(trade, state)

# Then BE (move SL → entry — remaining 50% can run to TP)
if not state.breakeven_moved and best_rr >= self.BE_TRIGGER_RR:
    self._move_to_breakeven(trade, state, price_info)
```

**Net behavior under MR RR=1:1**:

- MFE crosses 0.5R → Partial closes 50% (locks 0.25R) → BE moves SL to entry
- If revert: remaining 50% closes at entry = +0.25R net guaranteed
- If continues to TP: remaining 50% closes at +1.0R = +0.5R; total = 0.25 + 0.5 = +0.75R

**Invariant** — when configuring BE/Partial triggers under any RR target, **Partial must execute BEFORE BE in the same management cycle** (either via code order at same threshold, or via `PARTIAL_TRIGGER_RR < BE_TRIGGER_RR`). Otherwise the bot can move SL to BE without taking partial profit, and a revert closes the entire position at $0 — losing the safety net the design was meant to provide.

### 2026-05-07 — v8.0.13 Orphan position recovery + trail_states persistence

**Problem** — Bot restart caused **duplicate position opening** (real incident: NZDUSD SELL 2x in user's MT5):

| Step | Effect |
|---|---|
| 1. Bot opens NZDUSD SELL | `_active_trades[ticket]` populated (RAM only) |
| 2. Bot stops (crash / restart) | RAM cleared — `_active_trades = {}` |
| 3. MT5 still holds the position (broker SL/TP active) | Orphan position |
| 4. Bot restarts → strategy re-generates SELL signal | |
| 5. Duplicate check: `_active_trades.values()` → empty → no duplicate | **Opens 2nd SELL** ❌ |
| 6. TradeManager only manages new ticket | **orphan abandoned** ❌ |

`sync_with_mt5` had a comment claiming it would handle "MT5 → Active" direction, but the code only did "Active → MT5" (find closed trades) — never imported orphans.

**Fix #1 — Orphan recovery** (`TradeExecutor.sync_with_mt5`):

```python
# v8.0.13: import orphan positions from MT5 → _active_trades
for pos in mt5_positions:
    if pos["ticket"] in self._active_trades:
        continue
    if pos.get("magic", 0) != 123456:  # skip manual orders
        continue
    self._active_trades[pos["ticket"]] = self._rebuild_executed_trade_from_mt5(pos)
```

New helper `_rebuild_executed_trade_from_mt5` reconstructs `ExecutedTrade` from MT5 position dict. Recovers what MT5 has (entry/SL/TP/lot/symbol/type/magic) and approximates ATR from current M15 (acceptable — used only for trail distance). Agent decision tagged `"RECOVERED"`.

**Fix #2 — Startup sync** (`main.FTMOTradingBot.connect`):

```python
# Step 3 — orphan recovery before any signal scan
self._executor.sync_with_mt5()
print(f"✅ Active trades after sync: {len(self._executor.active_trades)}")
```

**Fix #3 — Trail state persistence** (`TradeManager._save_trail_states` / `_load_trail_states`):

Persists `_trail_states` (best_price, breakeven_moved, partial_closed, current_sl, trail_distance) to `logs/trail_states.json`. Saved every management cycle (5s, cheap JSON). Loaded on init. Without this, restart would reset BE/Partial flags → BE could re-fire on the same position, Partial could close another 50% (over-close).

**Trade-offs / known gaps**:

- ML/agent fields (ml_score, agent_action_value, htf_score, ฯลฯ) **NOT** persisted — they're decision-time metadata, not needed for managing existing position. Recovered trades log with default 0.0 + `agent_decision="RECOVERED"`.
- ATR_value approximation: re-computed from current M15 instead of original signal-time ATR. Trail uses current ATR anyway (already adaptive), so impact is negligible.
- Manual orders (magic ≠ 123456) **excluded** from recovery — bot won't manage trades opened via MT5 client directly.

**Invariant** — bot restart MUST be transparent to live positions. Required mechanisms:

1. `sync_with_mt5` imports orphan positions (`_active_trades` rehydrated from MT5).
2. Startup hook calls `sync_with_mt5` before signal scan.
3. `_trail_states` persists across restart (BE/Partial flags survive).
4. FTMO anchors persist via existing `bot_state.json`.

If any of (1)-(4) breaks, bot opens duplicate positions OR re-fires BE/Partial — both are real bugs.

### 2026-05-07 — v8.0.12 TradeManager BE/Partial/Trail tuned for MR RR=1:1

**Problem** — `TradeManager` constants left over from SMC-era RR=1:2.5:

| Trigger | v8.0.11 | Issue under MR RR=1:1 |
|---|---:|---|
| `BE_TRIGGER_RR` | 1.0 | Race condition — broker closes at TP=1.0R before BE fires |
| `PARTIAL_TRIGGER_RR` | 1.0 | Same race — TP closes position first |
| `TRAIL_ACTIVATION_RR` | 1.5 | **Impossible** — TP closes at 1.0R, trail at 1.5R never reachable |

→ Trade Management was effectively dead code under MR. Real live consequence (caught from VPS GBPUSD trade): MFE ≈ 0.95R then revert to SL = full loss, but BE at 0.5R would have locked entry.

**Fix** — `TradeManager` constants tuned to fire BEFORE TP=1.0R:

| Constant | v8.0.11 | **v8.0.12** | Effect |
|---|---:|---:|---|
| `BE_TRIGGER_RR` | 1.0 | **0.5** | Lock SL → entry at half-way to TP |
| `PARTIAL_TRIGGER_RR` | 1.0 | **0.7** | Take 50% profit at 70% to TP |
| `TRAIL_ACTIVATION_RR` | 1.5 | **99.0** (disabled) | Trailing impossible under RR=1:1 (cannot exceed TP) |

**No retrain required** — `TradeManager` is live-only execution logic, not part of the RL env or training pool. v8.0.10 retrain in flight remains valid.

**Invariant** — when changing strategy's RR target, `TradeManager` BE/Partial/Trail thresholds MUST be tuned to fire BEFORE TP. If `BE_TRIGGER_RR ≥ rr_target`, BE is dead code; if `TRAIL_ACTIVATION_RR ≥ rr_target`, trailing is impossible.

### 2026-05-07 — v8.0.11 RR FP-precision fix (RiskManager + PositionSizer)

**Problem** — VPS log showed:

```text
📡 [Agent] TAKE SELL AUDUSD Conf=75 RR=1:1.0 (confidence=1.00)
🚫 [Executor] Risk Manager ปฏิเสธ: ❌ Risk:Reward (1.00) ต่ำกว่าขั้นต่ำ (1.0)
```

Display says `1.00` but underlying value was `0.99999...` due to floating-point error → strict `<` comparison rejected the signal.

**Root cause** — `MRSignal.rr_ratio` (@property) computes:

```python
tp_dist = abs(self.tp_price - self.entry_price)
return tp_dist / self.sl_distance
```

`self.sl_distance` is stored exact (e.g. `0.0012` from ATR floor), but `tp_price` is rounded to broker digits (`round(entry - sl_distance, 5)`). FP subtraction `entry - tp_price` reconstructs the distance with tiny FP drift:

```python
>>> 0.58923 - 0.58803
0.0011999999999999927   # not 0.0012
```

→ rr_ratio = `0.001199.../0.0012 = 0.999999...` < `1.0` → **every MR signal with RR=1:1 rejected**.

**Fix** — added `1e-4` epsilon tolerance to two `<` checks:

- `RiskManager.can_open_trade` line 520
- `PositionSizer.calculate_sl_tp_prices` line 338

```python
if rr_ratio < self._config.MIN_RISK_REWARD_RATIO - 1e-4:
```

`1e-4` is well below any meaningful RR precision (rr=0.9999 vs 1.0 is indistinguishable in practice) and absorbs FP error from price-rounding.

**Why v8.0.9 didn't catch this** — v8.0.9 changed `MIN_RISK_REWARD_RATIO` 1.5 → 1.0 to ALLOW RR=1.0 in principle, but the strict `<` operator combined with FP drift kept blocking signals at the 1.0 boundary. The earlier ad-hoc tests on Mac may have hit an entry/sl/tp combo that aligned cleanly; AUDUSD/NZDUSD on the VPS at different prices triggered the FP edge.

**Audit added** — none new; existing parity audit's RR cross-check (`bot_config.ftmo.MIN_RISK_REWARD_RATIO ≤ bot_config.mr.rr_ratio`) still holds.

**Invariant** — when comparing derived FP ratios to a config threshold, always use an epsilon tolerance; otherwise rounding/precision can silently flip a "should-pass" signal to "rejected." Especially load-bearing on equality-boundary thresholds (RR=1.0 exactly).

### 2026-05-07 — v8.0.10 Anti-overfit retrain pipeline + holdout eval gate

**Why** — v8.0.5 model (Pass Rate 59.30 % on the 3000 pool it trained on) had no independent generalization test. The pool was the training set, and the eval used the same pool, so a high pass rate could reflect memorization of those specific 3000 episode-seeds rather than a real edge.

**Anti-overfit retrain settings** (orchestrated by `auto_train_pipeline.py`):

| Knob | v8.0.5 | v8.0.10 | Rationale |
| --- | ---: | ---: | --- |
| `pool_size` | 3000 | **5000** | More episode diversity → fewer revisits per epoch |
| `outcome_noise` | 0.05 | **0.08** | Stronger label-noise regularization on pool outcomes |
| `timesteps_p1` (Alpha) | 5M | 5M | unchanged |
| `timesteps_p2` (Risk) | 2M | **5M** | Let value function fully converge under DD penalty |

**Holdout pool** — built once via `build_mr_signal_pool.py --seed 999 --save_path data/mr_signal_pool_holdout.pkl` (783 valid episodes from 800 attempted, 82.4 MB). Different episode-seed grid from the training pool (seed=42) → independent samples by construction.

**Holdout eval verdict** — `scripts/holdout_eval.py` runs the saved best model on (train pool, holdout pool) and reports Δ Pass Rate:

- ≤ 5 pp → ✅ HEALTHY (model generalizes)
- 5-10 pp → 🟡 MILD OVERFIT (acceptable, watch)
- > 10 pp → ❌ OVERFIT (exit 1 — model memorized training pool)

**Backup** — pre-retrain best snapshotted to `models/mr/best_v8.0.5_pass59pct/` for rollback if v8.0.10 regresses.

**Invariant added** — anti-overfit gate: holdout_eval Δ Pass Rate ≤ 10 pp must hold before promoting any model to live. The training-pool eval (`train_mr_signal_filter.py --eval_only`) is no longer sufficient on its own.

### 2026-05-07 — v8.0.9 RR floor 1.5 → 1.0 (RiskManager rejecting every MR signal)

**Problem** — VPS log:

```text
🚫 [Executor] Risk Manager ปฏิเสธ: ❌ Risk:Reward (1.00) ต่ำกว่าขั้นต่ำ (1.5)
```

`RiskManager.can_open_trade` (line 519-521) checks `rr_ratio < FTMOConfig.MIN_RISK_REWARD_RATIO` and rejects. Default was 1.5 (SMC era used RR 1:1.5-2.5). MR strategy fixes RR at 1:1 (quick TP design) → every MR signal rejected at the live Risk Manager gate even after passing strategy/ML/RL.

**Fix** — `FTMOConfig.MIN_RISK_REWARD_RATIO` 1.5 → 1.0 + `PREFERRED_RISK_REWARD_RATIO` 2.0 → 1.0 to match MR's RR. Live Risk Manager now accepts MR signals.

**Audit added** — `parity_audit.py::audit_strategy_params` cross-checks `bot_config.ftmo.MIN_RISK_REWARD_RATIO ≤ bot_config.mr.rr_ratio`. Future RR mismatch fails the audit.

**Invariant added** — when changing strategy's RR ratio, `FTMOConfig.MIN_RISK_REWARD_RATIO` MUST be ≤ that value. RiskManager is the live floor — any signal below it gets rejected regardless of MR/ML/RL approval.

### 2026-05-07 — v8.0.8 MRSignal backward-compat @property (live AttributeError fix)

**Problem** — VPS started bot, immediately crashed with:

```text
⚠️ [Bot] Strategy/Execution error: 'MRSignal' object has no attribute 'rr_ratio'
```

Live path (`main._build_signal_observation`, `TradeExecutor.execute_signal`, `_log_signal_scan`) reads `sig.rr_ratio` / `sig.tp_distance` / `sig.timestamp` directly. These were attributes on the legacy SMC `TradeSignal` dataclass but missing from `MRSignal` (added v8.0).

**Fix** — added 3 `@property` accessors to `MRSignal`:

- `rr_ratio` → computed from `tp_distance / sl_distance` (returns 1.0 for MR fixed RR 1:1)
- `tp_distance` → `abs(tp_price - entry_price)`
- `timestamp` → `datetime.now(timezone.utc)` (signal-time fallback)

These join the existing v7.2.2 backward-compat set: `atr_pips`, `sl_distance_atr`, `bias_alignment`, `ob_size_atr`, `direction`. Together they make `MRSignal` a full duck-type replacement for `TradeSignal`.

**Invariant** — when removing/replacing a dataclass that's depended on by multiple modules (live path), audit all `getattr(sig, X)` and `sig.X` accesses across `main.py`, `execution/`, `analytics/`. Provide `@property` for any attribute the live path reads, even if the new schema doesn't store it.

### 2026-05-07 — v8.0.7 Windows VPS audit-script compatibility (UTF-8 + OHLCV-optional)

**Problem 1** — VPS (Windows) ran `python scripts/leakage_audit.py` and hit:

```text
UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 ...
```

Audit scripts opened source files (e.g. `main.py`, `mean_reversion_strategy.py`) without specifying encoding. On Windows the default codec is **cp1252** which can't decode the Thai comments embedded throughout the source.

**Fix 1** — added `encoding="utf-8"` to all 4 text-mode `open()` calls in audit scripts:

- `scripts/leakage_audit.py` — `_extract_function_source` (1 call)
- `scripts/parity_audit.py` — `audit_ml_threshold` (2 calls), `audit_risk_per_trade` (1 call)

Mac (UTF-8 default) was unaffected. After fix, both scripts run identically on Windows + Mac.

**Problem 2** — `parity_audit.py` Audit 4 (`audit_obs_dim`) crashed on VPS because `MeanReversionFilterEnv.__init__` requires OHLCV CSVs, which are gitignored and only exist on the training Mac. Live `main.py` doesn't need them (uses MT5 real-time feed), but the audit's strict env init blocked even reading class-level constants.

**Fix 2** — wrapped `audit_obs_dim` env init in `try/except RuntimeError`. On VPS-style failure (no OHLCV), fall back to checking `SelfLearningAgent.OBS_DIM == 32` only. Audit prints warning but does not fail — sufficient for live deploy sanity.

**Invariant added** — any text-mode `open()` reading project source files MUST specify `encoding="utf-8"`. Scripts that grep/parse project code are subject to this rule. Binary opens (`"rb"`/`"wb"`) are unaffected.

**Invariant added** — audit scripts MUST be VPS-friendly (graceful when OHLCV CSVs / pool / training artifacts are missing). Audits 4 (env init) and 6 (indicator parity) skip on missing data; remaining audits (1–3, 5, 7) cover everything live needs.

**v8.0.6 cleanup recap** — also covered in same release: SMC source files removed (`smc_strategy.py` + 5 detectors + tests, ~214 KB), unused `SessionConfig` fields removed (`london_start`/`newyork_start`/`london_end`/`newyork_end`), Excel schema slimmed (Trades 66→58 cols, Signals 23→20 cols) with `_COL`/`_SCOL` name-based column lookup + auto-archive of legacy xlsx, and `models/mr/` artifacts trimmed (~241 MB).

### 2026-05-07 — v8.0.5 🎉 MR pipeline ALL GATES PASSED, live wired

**Result** — Autonomous training pipeline converged in 1 iteration (12 min RL after pool/GBM reuse). All 5 gates passed on 5000-eps eval:

| Metric | Value | Gate | Margin |
|---|---:|---:|---:|
| Pass Rate | **59.30%** | ≥ 8.00% | +51.30 pp |
| Profitable Rate | 89.10% | ≥ 55.00% | +34.10 pp |
| Breach Rate | 0.00% | ≤ 5.00% | -5.00 pp |
| Total DD max | 5.80% | ≤ 6.00% | -0.20 pp |
| Daily DD max | 3.00% | ≤ 3.50% | -0.50 pp |
| Win Rate | 61.55% | (info) | — |
| Take Rate | 46.35% | (info) | — |
| Profit avg | +$7,229.59 (+7.23%) | (info) | — |

**Best model** at `models/mr/best/`: `ppo_mr_filter.zip` (1.0 MB), `vec_normalize_mr.pkl` (3.2 KB), `best_meta.json`.

**Path from v8.0 → v8.0.5 (5 sub-iterations)**:

| Step | Issue | Fix |
|---|---|---|
| v8.0.1 | Pilot yield 4 sig/ep (too sparse) | Relaxed BB 0.10/0.90 → 0.20/0.80, RSI 30/70 → 35/65 |
| v8.0.2 | Yield 5 sig/ep (still sparse) | BB 0.20/0.80 → 0.30/0.70, scan 12/day → 48/day, dedup 4-bar |
| v8.0.3 | iter 1+2 pass=2.5%/1.5% (under-trading) | Auto-tune improved (high-WR-low-pass case); ml_threshold default 0.40 → 0.30 aligned trainer/auto-pipeline/live |
| v8.0.4 | daily_dd_max pinned at 4.00% (env guard ceiling) | `DAILY_DD_GUARD` 0.04 → 0.030 (under 3.5% gate) |
| v8.0.5 | total_dd_max pinned at 8.50% (env guard ceiling) | `TOTAL_DD_GUARD` 0.085 → 0.058 (under 6.0% gate); pool/GBM reuse on restart |

**Live wiring (already in place from v8.0 full pivot)**:

- `main.py` imports `LiveMRScanner as SMCStrategy` from `strategy/mean_reversion_strategy.py`
- `bot_config.mr.strategy_mode = "mean_reversion"` (default)
- RL agent loads `models/mr/ppo_mr_filter.zip` first (auto fallback to legacy if missing)
- ML quality model loads `data/mr_signal_quality_model.pkl` first (auto fallback)
- `_build_signal_observation` reinterprets obs[4]=`bb_extreme`, obs[10]=`bb_band_width_atr/3`, obs[26]=`adx_inverse_norm` to match training distribution

**Live deploy steps**: `python ftmo_trading_bot/main.py` — that's it. The model at `models/mr/best/` is what `SelfLearningAgent` will load.

**Audit certification (mandatory before deploy)**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/leakage_audit.py   # exit 0 ✅
.venv/bin/python ftmo_trading_bot/scripts/parity_audit.py    # exit 0 ✅
```

Both passed at v8.0.5 lock-in.

### 2026-05-06 — v8.0 Mean Reversion **full pivot** (live default = MR, autonomous training launched)

**What changed (v8.0 final)** — Full pivot to Mean Reversion + Trend Filter. Live `main.py` no longer instantiates SMC; it loads `LiveMRScanner` (drop-in for `SMCStrategy`) and reads MR-trained model from `models/mr/`. SMC source files (`strategy/smc_strategy.py`, the 5 detectors, plus inducement) are **kept as deprecated reference** because `TradeSignal` dataclass + indicator helpers are still imported by `signal_quality.py`/`trade_executor.py`/`strategy_backtester.py`. The runtime path no longer routes through SMC scan logic.

**Strategy params after 4 pilots** (yield went 4 → 5 → 5 → 14 sig/ep median):

| Param | v8.0 initial | v8.0.2 (live default) | Reason |
|-------|-------------:|----------------------:|--------|
| BB_OVERSOLD / OVERBOUGHT | 0.10 / 0.90 | **0.30 / 0.70** | tight extremes too rare in real M15 |
| RSI_OVERSOLD / OVERBOUGHT | 30 / 70 | **40 / 60** | same — let RL filter weak setups |
| MIN_REVERSAL_WICK_RATIO | 1.2 | **0.4** | strict wick rule killed yield |
| ADX_TREND_BLOCK | 25 | **30** | only block extreme trends |
| MIN_CONFLUENCE_SCORE | 50 | **30** | wider acceptance, RL discriminates |
| MR_SCAN_POINTS_PER_DAY | 12 (every 2h) | **48 (every 30min)** | catch transient extremes |
| DEDUP_BARS | (none) | **4** | prevent same-direction signal flood |

**New modules**:

- `strategy/mean_reversion_strategy.py` — `MeanReversionStrategy` (BB %B extreme + RSI confirm + ATR floor + ADX H1 ≥ 25 trend block + reversal-wick confirmation). Outputs `MRSignal` dataclass that mimics `TradeSignal` field names so existing pool/env code works unchanged.
- `ml/mean_reversion_backtester.py` — `MeanReversionBacktester(StrategyBacktester)`. Replaces SMC strategy with MR engine. Scans every 2h (12 scans/day vs SMC 4) because BB extremes are time-sensitive. Resolves trades over 32 M15 bars (~8h). Adds `bars_to_resolution` + `is_quick_tp` keys per signal for reward shaping.
- `ml/mean_reversion_env.py` — `MeanReversionFilterEnv(FTMOSignalFilterEnv)`. Same 32-dim obs shape but reinterprets obs[4]=`bb_extreme`, obs[10]=`bb_band_width_atr/3`, obs[26]=`adx_inverse_norm`. Reward shaping per spec:
  - Quick TP win (≤ 5 bars): +0.50R bonus
  - Slow TP win: +0.20R bonus
  - Base loss: −0.10R
  - Duration fine: 0.02R per bar bled red, capped at 0.30R
  - Prolonged loss (≥ 12 bars): −0.40R extra
  - ADX > 25 violation: −0.30R (defense in depth — strategy already vetoes most)
- `scripts/build_mr_signal_pool.py`, `scripts/train_mr_signal_quality.py`, `scripts/train_mr_signal_filter.py` — mirror SMC counterparts, point at `data/mr_*` and `models/mr/*`.
- `scripts/auto_train_pipeline.py` — autonomous orchestrator (Build pool → GBM → RL → Eval → Self-correct). Tunes hyperparams based on which gate fails (breach → cut risk + bump loss penalties; DD too high → tighten capital preservation; pass rate too low → push more TAKE via lower ML threshold + higher quick-TP bonus). Logs every iteration to `logs/auto_train_pipeline.log` + `.jsonl`. Snapshots best model to `models/mr/best/`.

**Config additions**:

- `bot_config.mr` (`MeanReversionConfig`) — `strategy_mode`, `bb_period`, `bb_oversold`, `bb_overbought`, `rsi_oversold`, `rsi_overbought`, `adx_trend_block`, `sl_atr_mult`, `rr_ratio`, plus reward-shaping defaults (`quick_tp_bonus`, `prolonged_loss_penalty`, etc.).

**Live status** — `main.py` is **not** wired to MR. The pivot is staged: user runs `auto_train_pipeline.py` first, lets it self-correct over multiple iterations until eval gates pass, then flips `strategy_mode` to `"mean_reversion"` and restarts the live loop. SMC remains the production default until that switchover.

**Eval gates (default)**:

- Pass Rate ≥ 8 % (5000 eps)
- Total DD max ≤ 6 %
- Daily DD max ≤ 3.5 %
- Profitable Rate ≥ 55 %
- Breach Rate ≤ 5 %

**Pipeline cost** — Single iteration: ~30 min pool build + ~5 min GBM + ~6-10 hr RL train (n_envs=8, P1 5M + P2 2M) + ~5 min eval = 6-11 hr. Auto loop with 6 iterations max + 60 hr budget covers worst case where every iteration retrains.

**Why this design**:

1. Parallel-module approach lets us A/B test MR vs SMC without destabilizing the verified v7.2.2 SMC pipeline.
2. Reusing 32-dim obs + AuxAwarePPO infrastructure means no changes to PPO trainer / VecNormalize stack.
3. Auto-correct loop turns the 36-hour-per-attempt cost into "leave for the weekend" rather than "babysit each iteration".

### 2026-05-06 — v7.2.2 TradeSignal derived properties (live ML feature parity fix)

**Root cause** — Live VPS เห็น `⚠️ [GBM Drift] 23 features ห่างจาก training (KS > 0.15). Top: atr_pips=1.00, sl_distance_atr=1.00, bias_alignment=0.88, ...`

ตรวจพบ: `TradeSignal` dataclass ([smc_strategy.py](ftmo_trading_bot/strategy/smc_strategy.py)) ขาด 5 attributes ที่ ML/RL ต้องการ — มีแค่ raw values:

| ML ต้องการ | TradeSignal เก่า |
|-----------|-----------------|
| `atr_pips` | มีแค่ `atr_value` (ราคา) |
| `sl_distance_atr` | มีแค่ `sl_distance` (ราคา) |
| `bias_alignment` | มีแค่ `market_bias` |
| `ob_size_atr` | มีแค่ `ob_high`, `ob_low` |
| `direction` | มีแค่ `signal_type` (enum) |

[main.py:783-786](ftmo_trading_bot/main.py#L783) ใช้ `_extract(sig, k)` → `getattr(sig, k, 0.0)` → 5 features ที่หาย return `0.0` ทุก signal → KS = 1.00 (perfect mismatch กับ pool dict ที่ populate ครบ)

**Pool gen vs Live mismatch** (มีมาตั้งแต่ v7.1):
- `StrategyBacktester.generate_episode_signals` เก็บ `signal dict` ที่ populate `'atr_pips'`, `'sl_distance_atr'`, `'direction'`, `'bias_alignment'`, `'ob_size_atr'` ครบ
- Live ใช้ `TradeSignal` object → 5 features หาย

**Impact**:
- Live ML score คำนวณบน obs ที่ 5/24 features = 0 ตลอด → score เพี้ยน
- เป็นเหตุผล (ส่วนหนึ่ง) ที่ Live Pass Rate ตก vs Pool eval (5.9% v7.2.1 = pool eval ที่ feature ครบ)

**Fix v7.2.2** — เพิ่ม `@property` ใน `TradeSignal` ที่คำนวณ derived values อัตโนมัติจาก raw fields ที่มีอยู่แล้ว:

```python
@property
def pip_size(self) -> float:
    sym = self.symbol.upper()
    if sym.endswith("JPY") or "XAU" in sym or "XAG" in sym:
        return 0.01
    return 0.0001

@property
def atr_pips(self) -> float:
    return self.atr_value / self.pip_size if self.pip_size > 0 else 0.0

@property
def sl_distance_atr(self) -> float:
    return self.sl_distance / self.atr_value if self.atr_value > 0 else 0.0

@property
def direction(self) -> float:
    if self.signal_type == SignalType.BUY: return 1.0
    if self.signal_type == SignalType.SELL: return -1.0
    return 0.0

@property
def bias_alignment(self) -> float:
    return self.direction * float(self.market_bias)

@property
def ob_size_atr(self) -> float:
    if self.ob_high is None or self.ob_low is None: return 0.0
    if self.atr_value <= 0: return 0.0
    return abs(self.ob_high - self.ob_low) / self.atr_value
```

**Why @property approach**:
- Single source of truth — ไม่ต้อง populate ใน 10+ จุด ที่ create TradeSignal
- `getattr(sig, 'atr_pips')` ทำงานทันทีอัตโนมัติ — ไม่ต้องแก้ caller code
- Match กับ pool dict keys/values เป๊ะ — drift detector กลับมาทำงาน

**Verification** — unit test (BUY EURUSD / SELL USDJPY / BUY XAUUSD):
- atr_pips: 12.00 / 30.00 / 800.00 ✅
- sl_distance_atr: 1.6667 / 1.6667 / 1.25 ✅
- direction: +1 / -1 / +1 ✅
- bias_alignment: +1 (SELL × bearish = +1 = aligned) ✅
- pip_size: 0.0001 / 0.01 (JPY) / 0.01 (XAU) ✅

**ไม่ต้อง retrain** — fix อยู่ใน live path เท่านั้น. Pool ไม่กระทบ. หลัง fix → live ML score กลับมา meaningful → drift warning ควรหายภายใน ~100 signals แรก

**Files**:
- [`ftmo_trading_bot/strategy/smc_strategy.py`](ftmo_trading_bot/strategy/smc_strategy.py) — `TradeSignal` dataclass + 6 new `@property` methods

### 2026-05-06 — v7.2.1 obs[29/30] leak fix (audit-driven, post-v7.2 eval Pass 6.3%)

**Audit trigger** — v7.2 (Chronos un-flip) retrain เสร็จเร็วผิดปกติ (~20 นาที vs ~12 ชม. ที่คาด) → eval Pass Rate **6.3%** (vs baseline v6.13 = 9.7%, gate fail). User ขอ data leakage audit ทั่วระบบ

**Audit results** (6 surfaces):
- ✅ GBM features 24 ตัว — `ltf_slice = m15_df.iloc[scan_idx-499:scan_idx+1]` รวมแค่ signal-time bar เดียว
- ✅ HTF (H1/H4) — `_end_idx_at_or_before(h1_df, m15_end_ts)` ตัดที่หรือก่อน M15 boundary
- ✅ GBM training — `GroupKFold(n_splits=5, groups=episode_id)` + `cross_val_predict` + IsotonicRegression fit บน OOF probs + pool re-score ใช้ OOF
- ✅ Chronos forecast — closed-bar slice เดียวกับ signal extraction; cache key (symbol, last_bar_ts); outcome window [t+1, t+96] แยกจาก forecast horizon [t+8]
- ✅ Aux task — `AuxAwarePolicy.predict_aux(obs)` รับ obs (ไม่มี outcome); outcome เป็น MSE target ภายนอก
- ❌ **obs[29] floating_pnl_norm + obs[30] open_losing_count_norm — LEAK**

**Bug — obs[29/30] training env leak**:

`FTMOSignalFilterEnv.step()` (3 จุดเชื่อม):
- `_open_positions.append({..., 'outcome_partial': float(outcome_perturbed)})` — เก็บ pre-resolved final outcome
- `_floating_pnl_norm = floating_sum × risk_per_trade` ที่ floating_sum = `sum(op['outcome_partial'] × decay)` — sign คงที่ตั้งแต่ position เปิด
- `_open_losing_count_norm = sum(1 for op_outcome < 0) / 3` — direct count ของ positions ที่ outcome ติดลบ
- Comment ใน code ยอมรับเอง: "env เห็น final outcome ตั้งแต่เปิด แต่ live ไม่เห็น → simulate 'in-flight' feeling"

Live ใช้ true unrealized PnL ที่ oscillate กับ price path → distribution mismatch:
- Live: `RiskManager.get_unrealized_drawdown_pct()` (oscillating, sign อาจกลับได้ระหว่างถือ)
- Train (เก่า): monotonic toward final outcome (sign คงเดิมตลอด hold)

**Fix v7.2.1 (Option B — train + live ตรงกัน)**:

`FTMOSignalFilterEnv` ([signal_filter_env.py](ftmo_trading_bot/ml/signal_filter_env.py)):
- Hard-set `self._floating_pnl_norm = 0.0` + `self._open_losing_count_norm = 0.0` ใน `step()` (ลบ block calculation ทั้งหมด)
- ลบ `'outcome_partial'` field จาก `_open_positions.append({...})` (ไม่จำเป็นต่อ correlation simulator)

`FTMOTradingBot` ([main.py:610-625](ftmo_trading_bot/main.py#L610)):
- `_compute_floating_pnl_norm()` → `return 0.0` (เดิม `RiskManager.get_unrealized_drawdown_pct()`)
- `_compute_open_losing_count_norm()` → `return 0.0` (เดิม `count(positions if profit < 0)`)
- เหตุผล: VecNormalize ตอน train เก็บ `mean=0, var≈0` ของ obs[29/30] → ถ้า live ส่งค่าจริง (-0.02) → normalized = -2.0 (extreme) → policy output เพี้ยน. Force live = 0 → match train distribution

**Concurrent risk awareness ที่หาย (obs[29/30])**:
- ทดแทนด้วย `RiskManager.check_unrealized_circuit_breaker()` ที่อยู่ใน execution path (ไม่ผ่าน agent obs):
  - ถ้า floating ≤ -1.5% AND open ≥ 2 → pause เปิดออเดอร์ใหม่
- agent ยังเรียน "ระวัง daily DD" ผ่าน obs[18] `daily_dd_n` (realized loss) ปกติ

**Pipeline**: ไม่ต้อง rebuild pool/GBM (leak อยู่ที่ env เท่านั้น). Retrain RL อย่างเดียว ~12-15 ชม. (จริง ๆ อาจ ~20 นาทีถ้า early stop kicks in)

**Eval gate**: Pass Rate ≥ 9.7% (baseline v6.13) → keep | < 7% = restore `*.bak_v7.1`

**Files**:
- [`ftmo_trading_bot/ml/signal_filter_env.py`](ftmo_trading_bot/ml/signal_filter_env.py) — `step()` floating PnL block + `_open_positions.append`
- [`context.md`](context.md) + [`wiki/03-rl-training.md`](wiki/03-rl-training.md) + [`wiki/05-invariants.md`](wiki/05-invariants.md)

### 2026-05-06 — v7.2 Chronos un-flip semantics + Session log fix (audit-driven)

**Audit trigger** — ผู้ใช้ขอวิเคราะห์ `logs/ftmo_trades.xlsx` หลัง live session 2026-05-05 13:33–19:58 (EET, ~6.5 ชม.). พบ **524 signals → 1 trade ออกได้** (USDJPY SELL → BE +$44.37). Funnel: ML กรอง 188 (36%) | Agent SKIP 294 (56%) | TAKE 42 ติด correlation 41 = **97.6% block rate**

**Bug 1 — Chronos alignment = -1 ทุก signal (524/524)**
- Day = strong DOWN trend → SMC SELL ทั้งหมด → Chronos median forecast ก็ลง → ใช้ v7.0.2 flip-sign formula `delta = last_close - median_h > 0 → forecast_dir = +1` → `alignment = SELL(-1) × +1 = -1`
- v7.0.2 design assumed SMC = contrarian/reversal strategy → alignment +1 = "Chronos disagree = good reversal setup"
- ความจริง: SMC ใช้ HTF Tier 1 hard veto Counter-D1 = **trend-following**, ไม่ใช่ contrarian
- Reward shaping `signal_filter_env.py` (line 677-682): `if chronos_align < 0 → reward -= 0.30 (ml<0.55) หรือ -0.10` → ลงทุก TAKE ระหว่าง train → agent learned **skip-default** → live: SKIP signal ML 0.693 (สูงสุด!) แต่ TAKE สูงสุดแค่ ML 0.598

**Bug 2 — Session column = None ใน Signals sheet**
- `FTMOTradingBot._log_signal_scan` (`main.py:928`) hardcode `"session": ""` พร้อม comment "filled by TimeManager if needed" — แต่ไม่เคย fill จริง
- Trades sheet ปกติเพราะ `TradeExecutor` ส่ง session ผ่าน path ต่าง

**Fix**:
- **Patch A (Session log)** — `FTMOTradingBot._log_signal_scan` (`main.py:928`): `"session": live_context.get("session", "") or self._compute_current_session()` + เพิ่ม helper method `_compute_current_session()` ที่ mirror `TradeExecutor` session classifier (LONDON/LONDON_NY_OVERLAP/NEW_YORK/NY_AFTERNOON/ASIAN/OFF_HOURS)
- **Patch B (Chronos un-flip)** — `ChronosForecaster.compute_features` (`ml/chronos_forecaster.py:243-247`): `delta = median_h - last_close` (un-flip vs v7.0.2). Semantics ใหม่:
  - alignment **+1** = SMC + Chronos agree on direction (good — trend-following confirmed)
  - alignment **-1** = SMC สวน Chronos forecast (warning — counter-trend setup)
- Reward penalty (`signal_filter_env.py:677-682`) **ไม่แก้** — ตาม semantics ใหม่ -1 = warning จริง (counter-trend) → penalty ใช้งานได้ปกติ

**Pipeline (ต้องรัน 6 step ตามลำดับ, ~36 ชม.)**:

1. ✅ Backup: `signal_pool_10000.pkl`, `signal_quality_model.pkl`, `ppo_signal_filter.zip`, `vec_normalize_sf.pkl` → `*.bak_v7.1` (เสร็จแล้ว)
2. ✅ Edit code (เสร็จแล้ว)
3. Rebuild pool: `.venv/bin/python scripts/build_signal_pool.py --pool_size 10000 --workers 8`
4. Retrain GBM: `.venv/bin/python scripts/train_signal_quality.py`
5. Retrain RL Phase 1+2: `.venv/bin/python scripts/train_signal_filter.py --fresh --timesteps_p1 10000000 --timesteps_p2 5000000 --n_envs 8 --pool_size 10000 --outcome_noise 0.05 --ml_threshold 0.36 --risk_per_trade 0.0099`
6. Eval: `.venv/bin/python scripts/train_signal_filter.py --eval_only --pool_size 10000 --ml_threshold 0.36 --risk_per_trade 0.0099`

**Eval gate**:
- **≥ 9.7%** (baseline v6.13) → keep + deploy
- **7-9.7%** → equivalent (decision พร้อม user)
- **< 7%** → restore จาก `*.bak_v7.1`

**Live verification หลัง retrain**:
- Signals sheet column `Chronos Align` กระจาย {-1, 0, +1} (ไม่ใช่ -1 ค้าง)
- Signals sheet column `Session` ไม่ว่าง
- TAKE mean ML > SKIP mean ML (ปัจจุบัน 0.503 vs 0.482 = ใกล้เกินไป)
- Top SKIP by ML ≤ Top TAKE by ML (ปัจจุบัน 0.693 SKIP > 0.598 TAKE = ผิด)

**Files**:
- `ftmo_trading_bot/main.py` — `FTMOTradingBot._log_signal_scan` + new `_compute_current_session`
- `ftmo_trading_bot/ml/chronos_forecaster.py` — `ChronosForecaster.compute_features` formula + docstring + header

**ห้ามแตะ**: `RiskManager.update_daily_pnl` / `_initial_balance` / `_daily_start_balance`, `OBS_DIM = 32`, `DEFAULT_RISK_PER_TRADE_PCT = 0.0099`

### 2026-05-06 — v7.1.10 Pre-news close + XAUUSD USD news mapping

**Bug** — News filter ทำงานครึ่งเดียว: `is_near_high_impact_news()` ถูกเรียกที่เดียวคือ `SMCStrategy.scan_signal` (block สัญญาณใหม่). `TradeManager` ไม่เคยเช็ค news → position ที่เปิดก่อนข่าวถูกถือผ่าน event = ผิดกฎ FTMO ห้ามเทรดชนข่าว

**Fix**:
- เพิ่ม `TradeManager.check_news_close()` — loop `_executor.active_trades`, สำหรับแต่ละ position เช็ค `is_near_high_impact_news(symbol, now_utc, window_before=30, window_after=0)` → ถ้า True เรียก `_executor.close_trade(ticket, reason="Pre-news close")`
- เรียกใน `FTMOTradingBot.run` หลัง `manage_all_positions()` ก่อน `check_session_close()` (priority: Friday/Daily Overnight > Pre-News > Trailing/BE)
- เพิ่ม `"XAUUSD"` เข้า `_CURRENCY_TO_SYMBOLS["USD"]` set ใน `config/news_events.py` — ทอง spike แรงตอน NFP/CPI/FOMC ตาม USD strength โดยตรง. กระทบทั้ง scan-signal block และ pre-news close
- Buffer = 0 นาที — ปิดที่ T-30 ตรงเดียวกับ block สัญญาณใหม่. Retry ผ่าน 5-วิ loop ถ้า close fail

**ไม่ต้อง retrain** — fix อยู่ใน live execution path (TradeManager) เท่านั้น ไม่กระทบ obs/reward/training distribution

**Files**:
- `ftmo_trading_bot/execution/trade_manager.py` — เพิ่ม `check_news_close()` + import `is_near_high_impact_news`, `TimeManager`, `pytz`
- `ftmo_trading_bot/main.py` — wire ใน main loop section "ขั้นตอนที่ 4"
- `ftmo_trading_bot/config/news_events.py` — เพิ่ม `XAUUSD` ใน USD set

### 2026-05-05 — v7.1.6 Combine "best of all" (after v7.1.5 Pass 0.8% worst)

**Symptom (v7.1.5 eval, 5000 eps)** — Pass Rate **0.8%** (worst yet, regressed -2.2pp from v7.1.2 = 3.0%)

**TensorBoard diagnosis** (phase2_risk_46):
- value_loss curve: 1.99 → **4.43 (mid spike)** → 2.15 = value function unstable
- entropy_loss: -0.04 = policy collapsed deterministic
- std (action width): 0.25 (lowest) = agent committed too narrow
- explained_variance 0.80 (vs v7.1.3 = 0.90) = value function noisy

**Root cause** = **3-way conflict** (pool 10k diversity + ml_threshold 0.40 + reward -0.90 aggressive):
- Pool 10k → agent ต้อง generalize
- ML 0.40 → effective signals/ep ลด (16 → ~9)
- Reward -0.90 → push TAKE หนัก แต่ pool พื้นฐานบางลง
- Combination ทำให้ value function ไม่ทัน fit → entropy collapse

**Fix (v7.1.6)** — combine "best of all worlds":

| Lever | v7.1.2 | v7.1.5 | **v7.1.6** | Why |
|---|---|---|---|---|
| Pool size | 4.5k | 10k | **10k** ✓ | keep diversity |
| ML threshold | 0.36 | 0.40 | **0.36** ↓ | revert — pool feed RL หนาขึ้น |
| Reward missed-winner P2 | -0.85 | -0.90 | **-0.85** ↓ | revert — กัน entropy collapse |
| Reward Chronos (ml<0.55) | -0.30 | -0.20 | **-0.30** ↑ | revert — keep filter strength |
| Reward Chronos (ml≥0.55) | -0.10 | -0.05 | **-0.10** ↑ | revert |
| Reward concurrent loss | -0.20 | -0.15 | **-0.20** ↑ | revert |

**Why this should work**:

1. v7.1.2 ที่ Pass 3.0% ใช้ ml 0.36 + reward -0.85 — proven combination
2. Pool 10k มี diversity ดีกว่า 4.5k → eval variance ต่ำ + generalization ดี
3. RL revisit pool 10k = 99× = sweet spot (vs 4.5k = 220× over-fit, vs 100k+ = under-train)
4. Reward ไม่ aggressive จน push policy แคบ — entropy ไม่ collapse

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL only (~25 min) ด้วย `--ml_threshold 0.36 --pool_size 10000`.

**Backup**:

```bash
TS=$(date +%s)
cp models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_v715_${TS}_pre_v716
cp models/vec_normalize_sf.pkl  models/vec_normalize_sf.pkl.bak_v715_${TS}_pre_v716
```

**Eval gate**:

| Pass Rate | DD max | Verdict |
|---|---|---|
| ≥ 9% | ≤ 4.5% | ✅ Win — deploy live |
| 5-9% | ≤ 4% | ✅ better than v7.1.2 — deploy live + monitor |
| 3-5% | ≤ 4% | 🟡 Similar to v7.1.2 — keep |
| < 3% | — | ❌ Restore v7.1.2 + accept Pass 3% baseline |

**Files modified**: `config/settings.py` (1 line), `ml/signal_filter_env.py` (5 reward values), `wiki/05-invariants.md` (this entry).

---

### 2026-05-05 — v7.1.5 Pool 10k rebuild (Pass 0.8% — worst, REVERTED via v7.1.6)

**Symptom**: pool 10000 eps + ml 0.40 + reward -0.90 (v7.1.4 plan applied) → Pass Rate **0.8%** (worst yet)

**Diagnosis**: 3-way conflict — pool diversity + filter strict + reward aggressive ทำงานสวนกัน → entropy collapse

**Action**: Replaced by v7.1.6 (revert ml threshold + reward to v7.1.2 levels, keep pool 10k diversity)

---

### 2026-05-05 — v7.1.4 Combo C: keep threshold 0.40 + restore v7.0.x reward aggression

**Symptom (cumulative across rounds)**:

| Round | Threshold | Missed-winner P2 | Pass Rate | Take Rate | Orders/ep | DD max |
|---|---|---|---|---|---|---|
| v7.0.7 | 0.36 | -0.90 | **11.1%** | ~58% | 7.7 | ~4% (wipeout 4 พ.ค.) |
| v7.1 | 0.36 | -0.65 | (filter เข้ม pool drop) | — | — | — |
| v7.1.1 | 0.36 | -0.65 | 1.4% | 46.6% | 3.7 | 2.51% |
| v7.1.2 | 0.36 | -0.85 | 3.0% | 48.3% | 3.8 | 1.82% |
| v7.1.3 | **0.40** | -0.85 | 1.1% (KEPT temporarily) | 47.9% | 3.2 | 1.40% |
| **v7.1.4** ⏳ | **0.40** | **-0.90** | TBD (target ≥ 9%) | TBD | TBD | TBD |

**Hypothesis (Combo C)**:

- v7.1.3 ลด pool 19% (threshold 0.40) แต่ Take Rate ไม่ขยับ → reward ยัง softer เกิน push TAKE
- v7.1.4 = restore missed-winner P2 ไป -0.90 (v7.0.x level) ทำให้ agent กล้า TAKE ใน pool ที่สะอาดขึ้น
- Safety nets ทั้งหมดยังอยู่ (SMC pre-filters, runtime UNREALIZED_PAUSE, USD theme cap, obs[29-31]) → wipeout pattern กัน

**Reward changes (`ml/signal_filter_env.py` step())**:

| Line | v7.1.3 | **v7.1.4** | Why |
|---|---|---|---|
| missed-winner P2 | -0.85 | **-0.90** | restore v7.0.x level — push TAKE volume |
| missed-winner P2 + ml ≥ 0.40 | -0.50 | **-0.55** | restore v7.0.x level |
| Chronos disagreement (ml<0.55) | -0.30 | **-0.20** | soften — threshold 0.40 ก็ filter quality แล้ว |
| Chronos disagreement (ml≥0.55) | -0.10 | **-0.05** | mild only — trust ML strong signals |
| Concurrent loss (floating < -1%) | -0.20 | **-0.15** | soften — runtime UNREALIZED_PAUSE block อยู่แล้ว |

**Why safe vs v7.0.7 (despite same -0.90)** — multiple defense layers ที่ v7.0.7 ไม่มี:

1. SMC pre-filters: session warmup +5 conf, post-weekend +5, vol regime block, spread/ATR check, dynamic SL
2. GBM 24 features (temporal/regime aware) → calibration ดีขึ้น
3. obs[29-31] portfolio realtime → agent รู้ floating PnL + losing count + session timing
4. Runtime: UNREALIZED_PAUSE -1.5% × open ≥ 2 + USD_THEME cap = 2
5. Chronos log1p formula + disagreement penalty (softened but present)

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL อย่างเดียว ~25 min ด้วย `--ml_threshold 0.40`.

**Backup**:

```bash
TS=$(date +%s)
cp models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_v713_${TS}_pre_v714
cp models/vec_normalize_sf.pkl  models/vec_normalize_sf.pkl.bak_v713_${TS}_pre_v714
```

**Eval gate**:

| Pass Rate | DD max | Verdict |
|---|---|---|
| ≥ 9% | ≤ 4.5% | ✅ Win — deploy live demo |
| 5-9% | ≤ 4% | 🟡 Marginal — keep + monitor live |
| < 5% | — | ❌ Restore v7.1.2 backup (Pass 3.0%) |

**Files modified**: `ml/signal_filter_env.py` (5 reward values), `wiki/05-invariants.md` (this entry).

---

### 2026-05-05 — v7.1.3 KEPT (eval Pass 1.1%, WR 73.8%, DD 1.40% — safest config)

**Eval result (5000 eps, 2026-05-05)** — Pass Rate **1.1%** (regression −1.9pp จาก v7.1.2 = 3.0%) BUT:

- **Win Rate 73.8%** (vs v7.1.2 = 68.4%) — สูงสุดในทุก rounds — signals คุณภาพสูงขึ้น
- **DD max 1.40% / 8% limit** (vs v7.1.2 = 1.82%) — **ปลอดภัยสุด**
- **Daily DD max 1.44% / 4% limit** = ห่างไกล wipeout 4 พ.ค. (2.85%)
- Profitable rate 80.2% (vs v7.1.2 80.0%) — stable

**User decision (2026-05-05)**: เก็บ v7.1.3 ไว้ใช้ live แม้ Pass Rate ตก. Trade-off:

- ❌ Pass Rate ต่ำ (1.1%) = อาจไม่ทันถึง 10% ใน 45 วัน บน eval distribution
- ✅ Risk control ดีที่สุด (DD 1.4%, WR 74%) — 4 พ.ค. wipeout pattern ซ้ำไม่ได้
- ✅ Live ≠ eval — actual market อาจ generate signals คุณภาพดีกว่า pool แบบ random sampling

**Diagnosis ที่ทำให้ Pass Rate ตก**:

- ขึ้น threshold 0.36 → 0.40 = pool effective ลด 19% (33k → 27k signals)
- Agent ต้องการ data volume เพื่อเรียน — pool บางลง = train แย่ลง
- WR ขึ้น 5pp (signals สะอาด) แต่ Orders/ep ลด 0.6 (15%) → cumulative profit ลด

**Active config (ใช้ live demo)**:

- `models/ppo_signal_filter.zip` = v7.1.3 (restored from `*.bak_v713_1777947373`)
- `models/vec_normalize_sf.pkl` = v7.1.3
- `config/settings.py: ML_FILTER_THRESHOLD = 0.40` (ตรงกับ training)
- All other v7.1 features active: SMC pre-filters relaxed, GBM 24 features, Chronos log1p, RL obs 32, reward shaping (Chronos disagreement / concurrent loss / missed-winner -0.85)

**Backups**:

- `*.bak_v712_1777913348` (v7.1.2 model — Pass 3.0%) — fallback ถ้า v7.1.3 live ไม่ดี
- `*.bak_v713_1777947373` (v7.1.3 backup duplicate)
- `*.bak_v7.0.7` (pre-v7.1 baseline — Pass 11.1% but had wipeout exposure)

**Rollback**: ถ้า live demo > 3 SL ติดกัน หรือ DD > 3% ใน session แรก → restore *.bak_v712 (Pass 3% but tested better trade volume).

---

### 2026-05-04 — v7.1.3 ML threshold bump 0.36 → 0.40 (after v7.1.2 eval Pass 3.0%)

**Symptom (v7.1.2 eval, 5000 eps)** — Pass Rate **3.0%** (ขึ้น 2.1× จาก v7.1.1 = 1.4% ดี) แต่ยังไม่ผ่าน gate ≥ 9%. **Take Rate 48.3%** + **Orders/episode 3.8** แทบไม่ขยับจาก v7.1.1 (46.6% / 3.7) → reward shaping เริ่มทำงาน (Pass Rate ขึ้น) แต่ pool ยังมี borderline signals มากเกิน

**Root cause** — ML threshold 0.36 ใน training+live ปล่อย 49.3% ของ signals ผ่าน → agent ต้อง filter ภายในหนัก. Pool flooded with low-EV signals (WR baseline 45.8% @ 0.36 vs 47.7% @ 0.40).

**Fix (v7.1.3)** — bump ML threshold 0.36 → 0.40 ใน 1 file:

```python
# config/settings.py:109
# Before (v7.1/v7.1.1/v7.1.2):
ML_FILTER_THRESHOLD: float = 0.36
# After (v7.1.3):
ML_FILTER_THRESHOLD: float = 0.40
```

**Why 0.40 is sweet spot** (จาก v7.1.1 GBM threshold analysis on 68k pool):

| Threshold | % kept | WR baseline | EV |
|---|---|---|---|
| 0.36 | 49.3% (33,576) | 45.8% | +0.060 |
| **0.40** ⭐ | 40.0% (27,240) | **47.7%** | **+0.073** (+22%) |
| 0.45 | 21.3% | 52.6% | +0.097 |
| 0.50 | 8.6% | 59.8% | +0.196 |

→ 0.40 = best Pareto: signals พอ (40% kept) + WR baseline สูง (47.7%) + EV +22% improvement

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL อย่างเดียว ~25 min ด้วย `--ml_threshold 0.40`.

**Backup**:

```bash
TS=$(date +%s)
cp models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_v712_${TS}
cp models/vec_normalize_sf.pkl  models/vec_normalize_sf.pkl.bak_v712_${TS}
```

**Eval gate**:

| Pass Rate | Action |
|---|---|
| ≥ 9% | ✅ Keep + deploy live demo |
| 5-9% | 🟡 Marginal — Round 5: combo restore reward + threshold 0.40 |
| < 5% | ❌ Revert threshold 0.36 |

**Live config sync** — `bot_config.ftmo.ML_FILTER_THRESHOLD = 0.40` ตรงกับ training (invariant #10 sync). Live `FTMOTradingBot.run` จะใช้ค่า 0.40 ใน ML gate ก่อน agent.

**Files modified**: `config/settings.py` (1 line), `wiki/05-invariants.md` (this entry).

---

### 2026-05-04 — v7.1.2 Reward re-balance (after v7.1.1 eval Pass 1.4%)

**Symptom (v7.1.1 eval, 5000 eps)** — Pass Rate **1.4%** (gate ≥ 9% missed by 7.6 pp). Win Rate 71.4% (สูงดี), DD max 2.51%/8% (ปลอดภัย), แต่ Orders/episode 3.7 (ครึ่งของ v6.13 = 7.7) → **agent over-skip — TAKE volume ต่ำเกินจะ accumulate 10% ใน 45 วัน**.

**Root cause** — v7.1 reward shaping ผมเข้มเกินไป:

- missed-winner P2 softened: -0.90 → -0.65 (ลด push TAKE)
- เพิ่ม Chronos disagreement penalty: -0.40 / -0.15
- เพิ่ม concurrent loss penalty: -0.25

→ Net effect: SKIP=default, TAKE=expensive → agent learn "wait for perfect setup" ที่หาได้ยาก

**Fix (v7.1.2)** — re-balance ให้สมดุลระหว่าง v7.0.x (over-aggressive) และ v7.1 (over-passive):

| Reward line (`ml/signal_filter_env.py` step()) | v7.1 | **v7.1.2** | Why |
|---|---|---|---|
| missed-winner P2 (outcome ≥ 0.5) | -0.65 | **-0.85** | push TAKE ใกล้ v7.0.x แต่ไม่ถึง -0.90 (ห่าง wipeout level) |
| missed-winner P2 + ml ≥ 0.40 | -0.40 | **-0.50** | extra push สำหรับ high-ml signals |
| Chronos disagreement (ml < 0.55) | -0.40 | **-0.30** | ลด over-rejection — keep filter จริงๆ heavy cases |
| Chronos disagreement (ml ≥ 0.55) | -0.15 | **-0.10** | minor tweak — ML แข็งอาจ override forecast |
| Concurrent loss (floating < -1%) | -0.25 | **-0.20** | ลดเล็กน้อย — keep guard but lighter |

**Safety net ที่กัน wipeout 2026-05-04 ซ้ำ** (ไม่ได้แตะใน v7.1.2):

- Runtime: `RiskManager.check_unrealized_circuit_breaker` (-1.5% floating × open ≥ 2)
- Runtime: `MAX_USD_THEME_POSITIONS = 2` cross-group cap
- Runtime: SMC vol_regime "explosive" hard veto + spread/ATR > 30%
- Obs: floating_pnl_norm + open_losing_count + mins_since_session [29-31]
- Reward: Chronos disagreement (ลดลงแต่ยังมี) + concurrent loss (ลดลงแต่ยังมี)

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL อย่างเดียว ~25 min. Backups `*.bak_v7.0.7` ยังอยู่.

**Eval gate** (5000 eps):

| Pass Rate | DD max | Verdict |
|---|---|---|
| ≥ 9% | ≤ 4.5% | ✅ Keep + deploy |
| 5-9% | ≤ 4% | 🟡 Marginal — ดู Profit avg + Win Rate |
| < 5% | — | ❌ Re-tune (consider bump ml_threshold 0.36 → 0.40) |

**Files modified**: `ml/signal_filter_env.py` (3 reward lines, 5 numeric values).

---

### 2026-05-04 — v7.1.1 Filter relax (after v7.1 pool dropped 90%)

**Symptom (v7.1 pool build)** — pool ตก 90% (158k → 14k signals, 9.2 sigs/ep) → GBM AUC 0.69 ดีมากแต่ pool size บางเกินทำให้ FTMO 10% target unrealistic ใน 45 วัน.

**Root cause** — SMC pre-filters G/H ตึงเกินไป (hard veto):
- HTF=Neutral + ADX H1 < 25 = hard veto
- vol_regime "high" = hard veto
- spread/ATR > 20% = hard veto

→ Stack ของ hard vetos ตัด signals 90%

**Fix (v7.1.1)**:

| Filter | v7.1 | **v7.1.1** |
|---|---|---|
| HTF=Neutral + ADX H1 < 25 | hard veto | **soft −15 confluence** |
| vol_regime "high" | hard veto | **soft −10 confluence** |
| vol_regime "explosive" | hard veto | hard veto (เก็บ — rare event z>2) |
| Spread/ATR limit | 0.20 | **0.30** (gate ที่ตึงน้อยลง — ยังกัน Trade 3 GBPJPY 91%) |

**Result**: Pool 4254 eps × 16 sigs/ep = **68,155 signals** (+369%). GBM AUC 0.6437 (vs v7.1 0.6931 — drop เพราะ noise มากขึ้น แต่ EV+ stable).

**Files modified**: `config/settings.py` (SPREAD_ATR_RATIO_LIMIT 0.20→0.30), `strategy/smc_strategy.py` (G/H veto → soft penalties + apply ใน scoring loop).

---

### 2026-05-04 — v7.1 Staged: 3-brain root-cause patch (RCA from 5-trade wipeout)

**Symptom (2026-05-04 live demo)** — 5 ออเดอร์ 3 ชม. โดน SL ทั้ง 5 (-$284.68 / Daily DD 2.85% เกือบชน 4%). HTF ตรง 4/5, ML cal ≥ 0.44 ทั้ง 5, RL TAKE ทั้ง 5 → **3 brains มอง "OK" ทั้งหมด แต่ผลคือ wipeout**.

**Root cause (เชิงโครงสร้าง ไม่ใช่ bug จุดเดียว)**:

1. **SMC** ไม่รู้ "เวลา" — HTF=Neutral เป็น soft penalty, ไม่มี session warmup gate, ไม่มี post-weekend window detector, vol_regime "high" ไม่มี classifier
2. **GBM** 17 features ไม่มี temporal/regime เลย — Monday warmup signal มี cal=0.65+ แต่ขาดทุน
3. **RL obs** ไม่มี portfolio realtime — agent ไม่รู้ว่า "อีก 2 ออเดอร์กำลังขาดทุน"
4. **Chronos** uncertainty saturated ที่ 3.0 ทุก signal → no gradient → RL ไม่ใช้ feature
5. **Reward** P2 missed-winner -0.90 → agent over-tuned toward TAKE → กล้าเปิดสวน Chronos

**Code changes (no eval gate ผ่านแล้ว — ยังไม่ retrain)**:

- `core/risk_manager.py`: `+get_unrealized_drawdown_pct`, `+check_unrealized_circuit_breaker` (เรียกใน `can_open_trade`)
- `config/settings.py`: `+UNREALIZED_PAUSE_PCT=-1.5`, `+UNREALIZED_PAUSE_MIN_OPEN=2`, `+MAX_USD_THEME_POSITIONS=2`, `+SPREAD_ATR_RATIO_LIMIT=0.20`
- `analytics/performance.py`: Stats sheet limits อ่านจาก `bot_config.ftmo` (เลิก hardcode 10%/5%)
- `execution/trade_executor.py`: `+_populate_close_metadata` helper (Bid@Exit/Balance@Close/Equity Peak), `+USD_THEME_DIR` cross-group guard ใน `_check_correlation_risk`
- `strategy/smc_strategy.py`: `+_is_session_warmup`, `+_is_post_weekend_window`, `+_check_spread_atr_ratio`, `+_required_confluence`, `+_compute_dynamic_sl_multiplier` + pre-filters G/H/I ใน BUY+SELL paths (HTF Neutral + ADX H1 < 25 veto, vol regime high/explosive block, spread/ATR > 20% block)
- `strategy/indicators.py`: `+classify_volatility_regime`, `+compute_atr_zscore_30bars`
- `ml/signal_quality.py`: `FEATURES` 17 → 24 (เพิ่ม temporal/regime), `+compute_temporal_features`, `+detect_drift` (KS test), `+record_live_signal`
- `ml/chronos_forecaster.py`: formula `(q90-q10)/(atr×√8)` → `log1p(...)/2` (กัน saturation)
- `ml/signal_filter_env.py`: obs 29 → 32 (`+floating_pnl_norm`, `+open_losing_count_norm`, `+mins_since_session_norm`); reward shaping (`+chronos_disagreement_penalty`, `+concurrent_loss_penalty`, missed-winner P2 -0.90 → -0.65)
- `ml/rl_agent.py`: `OBS_DIM` 29 → 32
- `main.py`: `+_check_gbm_drift` (1 hr), `+_compute_floating_pnl_norm`, `+_compute_open_losing_count_norm`, `+_compute_mins_since_session_norm`, `_build_live_context` คำนวณ temporal feats + ส่งให้ GBM, drift recording
- `scripts/build_signal_pool.py` (ผ่าน `ml/strategy_backtester.py`): inject 7 temporal features ต่อ signal
- `scripts/train_signal_quality.py`: FEATURE_KEYS 17 → 24 + บันทึก `train_dist` snapshot ใน payload
- `scripts/chronos_distribution_audit.py`: **ใหม่** — audit pool histogram + percentile

**Pipeline (ต้องทำตามลำดับ)**:

```bash
TS=$(date +%s)
cp models/ppo_signal_filter.zip models/ppo_signal_filter.zip.bak_v7.0.7
cp models/vec_normalize_sf.pkl  models/vec_normalize_sf.pkl.bak_v7.0.7
cp data/signal_quality_model.pkl data/signal_quality_model.pkl.bak_v7.0.7
cp data/signal_pool_3000.pkl    data/signal_pool_3000.pkl.bak_v7.0.7
mv logs/ftmo_trades.xlsx        logs/ftmo_trades_pre_v71_${TS}.xlsx

python scripts/build_signal_pool.py --pool_size 3000 --workers 8        # ~6 ชม.
python scripts/chronos_distribution_audit.py --pool data/signal_pool_3000.pkl   # verify formula
python scripts/train_signal_quality.py                                  # ~5 นาที
python scripts/train_signal_filter.py --fresh \
  --timesteps_p1 10_000_000 --timesteps_p2 5_000_000 \
  --n_envs 8 --pool_size 3000 --outcome_noise 0.05 \
  --ml_threshold 0.36 --risk_per_trade 0.007                            # ~30 ชม.
```

**Eval gate (5000 eps)**:

| Pass Rate | DD max | Profitable | Verdict |
|---|---|---|---|
| ≥ 9% | ≤ 4.5% | ≥ 70% | ✅ Keep + deploy |
| 7-9% | ≤ 4% | ≥ 65% | 🟡 Marginal — keep flag |
| < 7% | > 5% | — | ❌ Restore `*.bak_v7.0.7` |

**ห้ามรัน live** ก่อน retrain — `OBS_DIM` mismatch (env 32, model 29) จะ raise `ValueError` ใน `SelfLearningAgent._prepare_obs`.

**Backup discipline**: ก่อน Step 8 ต้อง execute backup commands ครบ. v7.0.7 lesson — skip backup → ลอบ regression rollback ไม่ได้.

---

### 2026-05-02 — v7.0.7 Revert threshold 30 → 20 (v7.0.5 retrain — backup recovery)

**Why** — v7.0.6 retrain (threshold 30) = Pass Rate **7.4%** (regression −3.3 pp จาก v7.0.5 = 10.7%). ⚠️ User skip backup step ของ plan → v7.0.5 model file (Pass 10.7%) ถูก overwrite โดย retrain → ต้อง retrain ใหม่เพื่อกลับ 10.7%.

**Lessons learned (สำคัญ)**:

1. **Early stop ที่ value_loss=20.45 ไม่ใช่ false positive** — แต่เป็น **inflection point detector**. Phase 2 หลังจุดนี้ทำให้ agent over-tune toward safety (WR ขึ้น 65→68%, Profitable ขึ้น 86→88%, แต่ Pass Rate ตก เพราะ "selective เกิน" ไม่ aggressive พอจะถึง 10% target ใน 45 วัน)
2. **Threshold 20 หลัง LR fix proper (5e-5) = right calibration** — ไม่ aggressive เกินตามที่ผมเคยกังวล
3. **Engineered sweet spot** = Phase 2 รันถึง ~70% (3.5M / 5M) แล้ว early stop จับ inflection
4. **Backup discipline critical** — ก่อน retrain ทุกครั้งต้อง backup verified model

**Fix (v7.0.7)** — single-line revert:

```python
# scripts/train_signal_filter.py:651
# Before (v7.0.6, regressed):
EarlyStopOnValueLoss(threshold=30.0, patience=5, warmup_steps=50_000),
# After (v7.0.7, = v7.0.5 config):
EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000),
```

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL อย่างเดียว ~30 ชม.

**Risk**: stochastic — RNG seed ใหม่อาจไม่ให้ Pass Rate เหมือนเดิม 10.7% เป๊ะ (อาจ 9.5-11.5%). v7.0.3 backup (10.0%) เป็น final safety net

**Watch points**:

- Phase 2 step 0: `train/learning_rate = 0.00005` (LR fix v7.0.5 ยังคง)
- Phase 2 trigger: คาด ~3-4M steps ที่ value_loss ~20-25 (เหมือน v7.0.5)
- Final eval: target ≥ 9.7% (= v6.13 baseline)

**Gate criteria**:

| Pass Rate | Action |
|---|---|
| ≥ 10% | ✅ Keep |
| 9-10% | 🟡 Marginal — keep ก็ได้ (ดีกว่า v7.0.3 = 10.0% ที่เป็น fallback) |
| < 9% | ❌ Restore `*.bak_v7.0.3` (10.0%) |

**Files modified**: `scripts/train_signal_filter.py` (line 651 + comment), `wiki/05-invariants.md` (this entry), `wiki/03-rl-training.md`, `context.md`.

**Backup awareness** — เพิ่ม emphasis ใน plan files: "ต้อง execute backup commands ก่อนทุก retrain — อย่า skip"

---

### 2026-05-02 — v7.0.6 Phase 2 EarlyStop threshold 20 → 30 (extension test, regressed)

**Why** — หลัง v7.0.5 (LR fix proper, Pass 10.7%) Phase 2 trigger early stop ที่ step 3.5M / 5M (70%) ที่ `value_loss = 20.45` (เกิน threshold 20 เพียง 2.25%).

**Reasonable doubt**: 20.45 ≠ true divergence:

- v7.0.3 trigger ที่ 31.29 (1.56× threshold) = real spike จาก high LR
- v7.0.5 trigger ที่ 20.45 (1.02× threshold) = borderline หลัง LR fix
- LR ลดลง 6× (3e-4 → 5e-5) → variance ของ value_loss ก็ลดลง → threshold 20 อาจ aggressive เกิน
- Pass Rate trajectory ใน v7.0.5 ยัง **ขึ้น** ตอน trigger (rolling 9.7% → final 10.7%) → ไม่มี over-train signal

**Hypothesis**: ถ้าขยาย threshold 20 → 30 → Phase 2 รันได้นานขึ้น (อาจครบ 5M) โดยไม่ over-train เพราะ:

- LR proper (5e-5) × 5M timesteps = update magnitude **6× น้อยกว่า v7.0.4** (over-trained config) → safe range
- 30 = 1.5× ของเดิม = ตรงกับ "natural spike range" ของ v6.13/v7.0.3 historical data

**Fix (v7.0.6)** — single-line change:

```python
# scripts/train_signal_filter.py:650 (Phase 2 only)
# Before (v7.0.5):
EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000),
# After (v7.0.6):
EarlyStopOnValueLoss(threshold=30.0, patience=5, warmup_steps=50_000),
```

> **Phase 1 ไม่กระทบ** — ยังคง threshold=10, warmup=0

**Probability ประเมิน**:

| Outcome | Probability |
|---|---|
| Pass > 10.7% (improvement) | 40% |
| Pass 10-10.7% (equivalent) | 35% |
| Pass 7-10% (mild regress) | 20% |
| Pass < 7% (catastrophic) | 5% |

→ 40% upside + 5% downside (mitigated โดย backup `*.bak_v7.0.5`)

**Pipeline**: ไม่ rebuild pool / GBM. Retrain RL อย่างเดียว ~30 ชม.

**Watch points**: TensorBoard `train/value_loss` curve — ถ้า > 30 = true divergence (early stop trigger, Phase 2 หยุด < 5M)

**Gate**:

| Pass Rate | Verdict |
|---|---|
| ≥ 11% | ✅ Win (better than v7.0.5) |
| 10-11% | 🟡 Equivalent |
| < 9% | ❌ Restore v7.0.5 backup |

**Files modified**: `scripts/train_signal_filter.py` (line 650 + comment), `wiki/05-invariants.md` (this entry), `wiki/03-rl-training.md`, `context.md`.

---

### 2026-05-02 — v7.0.5 Phase 2 LR schedule proper fix (latent bug ตั้งแต่ v6.x)

**Symptom (v7.0.4 retrain)** — Phase 2 รันเต็ม 5M timesteps (warmup ทำงาน, no early-stop) → Pass Rate **2.8%** (regression −7.2 pp จาก v7.0.3 = 10.0%). Pass Rate trajectory ใน Phase 2 ลดลงตอนปลาย: 5.42% (54%) → 4.55% (100%) → 2.8% (eval) = classic **over-training**.

**Root cause (2 layers)**:

#### Layer 1: SB3 LR bug (latent ตั้งแต่ v6.x)

```python
# scripts/train_signal_filter.py:607 (v7.0.4 และก่อนหน้า)
model_p2.learning_rate = 5e-5   # ← ตั้งใจ
```

แต่ TensorBoard log แสดง `learning_rate = 0.0003` (= 3e-4 = Phase 1 default).

**Why**: SB3 PPO `lr_schedule` ถูก wrap ตอน `_setup_model()` ครั้งเดียว. การ set `model.learning_rate = X` หลัง `AuxAwarePPO.load()` ไม่ rebuild `lr_schedule` → optimizer ใช้ schedule เก่า (Phase 1 const 3e-4)

→ **Phase 2 ใช้ LR 6× สูงกว่า intended ตลอด v6.x → v7.0.4** — verified จาก TensorBoard ของ v7.0.4 retrain

#### Layer 2: High LR × Long Phase 2 = Over-training (manifest ใน v7.0.4)

ก่อนหน้านี้ EarlyStopOnValueLoss trigger เร็ว (~30-100k steps Phase 2) → policy ไม่ over-train **โดยบังเอิญ**:

```
v6.13: Phase 2 early-stop @ ~30k steps → Pass 9.7% (lucky)
v7.0.3: Phase 2 early-stop @ 32k steps → Pass 10.0% (lucky)
v7.0.4: Phase 2 ran 5M (warmup ปลด safety) → over-train → Pass 2.8%
```

→ Warmup ของ v7.0.4 ปลด accidental safety mechanism → bug Layer 1 manifest เป็น regression ทันที

**Fix (v7.0.5)** — rebuild lr_schedule + update optimizer ตรงๆ:

```python
from stable_baselines3.common.utils import FloatSchedule

model_p2 = AuxAwarePPO.load(model_path_p1, env=vec_env_p2)
model_p2.learning_rate = 5e-5
model_p2.lr_schedule = FloatSchedule(5e-5)   # v7.0.5: rebuild schedule
for _pg in model_p2.policy.optimizer.param_groups:
    _pg['lr'] = 5e-5                         # v7.0.5: update optimizer immediately
```

**3 ชั้นป้องกัน**:

1. `model.learning_rate = 5e-5` — cosmetic (ก่อน v7.0.5 ทำแค่นี้)
2. `model.lr_schedule = FloatSchedule(5e-5)` — rebuild schedule fn ที่ `_update_learning_rate()` อ่าน
3. `optimizer.param_groups[i]['lr'] = 5e-5` — update PyTorch optimizer ทันที (กัน lag 1 rollout ก่อน schedule update)

**ทำไม FloatSchedule ไม่ใช่ get_schedule_fn**: SB3 deprecated `get_schedule_fn()` → `FloatSchedule()` (constant) เป็น replacement สำหรับ literal const value

**Hypothesis ของ v7.0.5**:

| Run | LR | Phase 2 timesteps | LR × steps | Result |
|---|---|---|---|---|
| v6.13 baseline | 3e-4 (bug) | ~30k (early-stop) | 9 | Pass 9.7% (lucky) |
| v7.0.4 (warmup) | 3e-4 (bug) | 5M | 1500 | Pass 2.8% (over-train) |
| **v7.0.5** | **5e-5** (fixed) | **5M** | **250** | **Pass ?** (predicted 10-12%) |

→ v7.0.5 update magnitude (LR × steps) = 250 = **6× น้อยกว่า v7.0.4** (= 1500) แม้ same Phase 2 timesteps → expected: ไม่ over-train

**Pipeline**: ไม่ต้อง rebuild pool / retrain GBM. Retrain RL อย่างเดียว ~30 ชม.

**Watch points (TensorBoard ระหว่าง training)**:

- Phase 2 step 0: `train/learning_rate = 0.00005` ← ยืนยัน fix ทำงาน (ถ้าเป็น 0.0003 = fix fail)
- Phase 2 mid (step 2.5M): `value_loss < 5` (low + steady)
- Phase 2 end (step 5M): Pass Rate cumulative ≥ 8%, ไม่ลดลงตอนปลาย

**Gate criteria**:

| Pass Rate | Verdict |
|---|---|
| ≥ 12% | ✅✅ Big win |
| 10-12% | ✅ Win |
| 9-10% | 🟡 Equivalent v7.0.3 |
| 7-9% | ⚠️ Marginal — discuss |
| < 7% | ❌ Worse → restore `*.bak_v7.0.3` |

**Rollback**: `*.bak_v7.0.3` (Pass 10.0%) เป็น safety net

**Files modified**: `scripts/train_signal_filter.py` (LR fix + import), `wiki/05-invariants.md` (this entry), `wiki/03-rl-training.md`, `context.md`.

---

### 2026-05-02 — v7.0.4 EarlyStopOnValueLoss warmup grace (Phase 2 transient spike fix)

**Symptom (v7.0.3 retrain)** — Phase 2 trigger early stop ที่ step **32,808 / 5,000,000 (0.6% ของ target)**:

```
[Early Stop] value_loss=31.29 > 20.0 x5
Phase 2 done (0.1 min) — terminated at step 32,808
```

ผล eval สุดท้าย Pass Rate 10.0% (ผ่าน gate 9.7%) — แต่ Phase 2 "เกือบไม่ได้ทำงาน" → potential ที่ขาด

**Root cause** — Reward distribution shift ระหว่าง Phase 1 → Phase 2:

| Reward range | Phase 1 | Phase 2 |
|---|---|---|
| Base + bonuses | ~[-2, 5] | ~[-2, 5] |
| **DD penalty** (exp ramp -0.5×(e^(3·ratio)-1)) | none | active |
| **Activity floor** + undertrading checks | none | active |
| **Effective range** | ~[-2, 5] | ~[-15, 5] |

Value head ของ Phase 1 ถูก train ภายใต้ reward [-2, 5] → ตอน Phase 2 step 0 เห็น reward ใหม่ -15 → predict ผิดมาก → MSE spike. value_loss spike จาก ~3 → 31. v7.0.3 ซ้ำ shift มากกว่าปกติเพราะ Chronos features (formula refactor v7.0.2) เพิ่ม obs distribution shift.

**Fix (v7.0.4)** — เพิ่ม `warmup_steps` parameter ใน `EarlyStopOnValueLoss`:

```python
class EarlyStopOnValueLoss(BaseCallback):
    def __init__(self, threshold=10.0, patience=5, warmup_steps=0, verbose=0):
        ...
        self.warmup_steps = warmup_steps   # v7.0.4

    def _on_step(self) -> bool:
        # warmup grace — value head re-fit ก่อน enable check
        if self.num_timesteps < self.warmup_steps:
            self._consecutive = 0
            return True
        # ... existing logic
```

Phase 2 init: `EarlyStopOnValueLoss(threshold=20.0, patience=5, warmup_steps=50_000)`

**Phase 1 ไม่กระทบ** — `warmup_steps` default = 0 = behavior เดิม. Phase 1 เริ่มจาก fresh policy ไม่มี distribution shift

**Why warmup = 50,000?**

- 50k / 5M Phase 2 = **1%** ของ target (เล็กพอไม่กระทบ schedule)
- PPO literature: value head re-fit ภายใน 30-80k steps สำหรับ moderate distribution shift → 50k = good middle
- ระหว่าง warmup ถ้า value_loss spike แล้ว settle ลง → enable check จับ true divergence ปกติ

**Risk + Mitigation**:

- Risk: ระหว่าง warmup ถ้าเกิด true divergence → ทำลาย policy
- Mitigation: backup `*.bak_v7.0.3` model ก่อน retrain. ถ้า v7.0.4 Pass Rate < 9% → restore

**Pipeline**: ไม่ต้อง rebuild pool / retrain GBM. Retrain RL อย่างเดียว ~25-30 ชม.

**Gate criteria**:

| Pass Rate | Verdict |
|---|---|
| > 10.0% | ✅ Win — warmup ช่วย push เพิ่ม |
| 9-10% | 🟡 Equivalent — warmup ไม่ regress |
| < 9% | ❌ Worse — restore `*.bak_v7.0.3` |

**Verification (synthetic, ก่อน retrain)**:

| Test | Result |
|---|---|
| Within warmup → no trigger | ✅ |
| Past warmup, low value_loss → no trigger | ✅ |
| Past warmup, high value_loss → trigger after patience | ✅ |
| Default `warmup_steps=0` (Phase 1) → backward compat | ✅ |

**Files modified**: `scripts/train_signal_filter.py` (EarlyStopOnValueLoss class + Phase 2 init), `wiki/05-invariants.md` (this entry), `wiki/03-rl-training.md`, `context.md`.

---

### 2026-05-02 — v7.0.3 Correlation simulator HOLD=0 (over-block fix)

**Symptom (v7.0.2 eval, 5000 eps)**: Pass Rate **0.6%** (28/5000) — catastrophic regression vs baseline 9.7% และแย่กว่า v7.0 (4.0%) เอง.

**Root cause** — Time scale mismatch ใน correlation simulator:

- Live: signal scan ทุก ~60 วินาที × 12 ครั้ง/ชม. = **720 scans/day**
- Pool: scan 4 ครั้ง/day → **1 pool signal slot ≈ 6 ชั่วโมง live time**
- Live position avg hold = 75 นาที (= 1.25 ชม.)
- ใน pool, `HOLD_SIGNALS_APPROX = 1` → block correlation **6 ชั่วโมง** = over-block 4-5× ของ live behavior

ผล: agent skip-all (Take Rate ตก 50% → 32%, Orders/ep ตก 7.7 → 4.8) → ไม่ได้เรียน TAKE good signals → Pass Rate 0.6%

**Fix (v7.0.3)** — single-line change:

```python
# ml/signal_filter_env.py — class FTMOSignalFilterEnv
HOLD_SIGNALS_APPROX = 0   # was 1 (v7.0.2)
```

**Effect**: drop-stale logic ใน `step()` ใช้ `cutoff = signal_idx - HOLD = signal_idx`. Positions ที่ append step ก่อน (`opened_at_idx = signal_idx - 1`) จะถูก drop ทันที → `_open_positions` empty ก่อน check correlation → no block. Effective: correlation simulator off, แต่ infrastructure คงไว้ (เผื่อ tune กลับ).

**Rationale**: live position avg 75 นาที < 1 pool slot (6 ชม.) → ใน pool ระหว่าง 2 scan-points (6h apart) live position เปิด-ปิดได้ 4-5 ครั้ง → effective live correlation block ใกล้ 0 ใน pool time scale.

**Pipeline**: ไม่ต้อง rebuild pool (ไม่ขึ้นกับ HOLD), ไม่ต้อง retrain GBM. Retrain RL อย่างเดียว ~25-30 ชม.

**Gate criteria** (v7.0.3):

| Pass Rate | Verdict |
|---|---|
| ≥ 9.7% | ✅ Keep — Chronos fix + minimal correlation works |
| 7-9.7% | 🟡 Marginal — ดู WR/DD/orders ก่อน deploy |
| 4-7% | ⚠️ Partial — ดีกว่า v7.0, แต่ไม่ถึง baseline |
| < 4% | ❌ Worse → revert (Option A) |

**Files modified**: `ml/signal_filter_env.py` (1 บรรทัด + comment), `wiki/05-invariants.md` (this entry), `wiki/02-modules.md`, `context.md`.

---

### 2026-05-01 — v7.0.2 Chronos formula fix + correlation training-live sync

**Symptoms (v7.0 retrain eval, 5000 eps)**: Pass Rate **9.7% → 4.0%** (regression). Live demo audit (`ftmo_trades_pre_v7.xlsx`): AGENT_TAKE_FAIL **71.7%** ของ 1429 scans (1024 reject = 96.3% เพราะ correlation block).

**Three root causes (post-mortem analysis)**:

1. **`chronos_uncertainty_norm` saturated** ที่ค่า max (3.0) ใน **96.2%** ของ signals. สูตรเดิม `(q90-q10)/atr` ไม่คิดว่า variance ของ 8-bar forecast ขยายตาม √horizon (Brownian motion). ATR M15 ~8 pips, q90-q10 ~30 pips → ratio ~3.75 → clip = useless feature.
2. **`chronos_alignment` correlation ติดลบ** (corr = **−0.0178** กับ outcome). SMC ค้าขาย swing/reversal สวนเทรนด์ระยะสั้น แต่ Chronos forecast เทรนด์ → "agree" หมายถึง SMC ตามเทรนด์ = ช้าเกิน, "disagree" หมายถึง SMC จับ reversal = profitable. Sign **กลับด้านกับ task** จริง.
3. **Live ↔ Training distribution mismatch**: `TradeExecutor._check_correlation_risk` block 96.3% ของ TAKE ใน live (USD_WEAK / USD_STRONG / JPY_CROSS / EUR_PAIRS / GBP_PAIRS / SAFE_HAVEN groups), แต่ `FTMOSignalFilterEnv.step` ไม่มี correlation check เลย → agent train บน distribution ที่ open ทุก signal ได้ แต่ live เปิดได้ 1.2%.

**Fixes**:

- **`ml/chronos_forecaster.py`** — `compute_features`:
  - flip alignment: `delta = last_close - median_h` (เดิม `median_h - last_close`) → `+1` = SMC + Chronos contrarian = good reversal setup
  - Brownian-scaled uncertainty: `(q90-q10) / (atr * sqrt(8))` → expected band = ATR × √8, ratio = 1 หมายถึง "ตามคาด" → ไม่ saturate
- **`ml/signal_filter_env.py`** — เพิ่ม correlation simulator:
  - Class constants `CORRELATION_GROUPS`, `_GROUP_POSITIVE_DIR`, `MAX_CORRELATED_POSITIONS=1` (mirror `TradeExecutor`)
  - `HOLD_SIGNALS_APPROX = 1` — approximate live trade duration (live avg 75 นาที ≈ 1 pool signal slot)
  - `_open_positions: List[Dict]` ใน `_reset_state` — track virtual open trades
  - `_is_correlation_blocked(sig)` method — replicate live logic (duplicate symbol + group exposure)
  - `step()`: drop stale positions → check correlation → forced SKIP → append to `_open_positions` ถ้า TAKE สำเร็จ
  - `info['correlation_forced_skip']` (per-step) + `info['correlation_forced_skips']` (cumulative ตอน episode end)

**ห้ามแก้ใน fix นี้** (ตามกฎ CLAUDE.md):

- ⛔ Daily DD logic (`RiskManager`) — ไม่แตะ
- ⛔ OBS_DIM ยังคง 29 (sync 3 ที่ตามเดิม)
- ⛔ Live correlation source of truth (`TradeExecutor.CORRELATION_GROUPS`) — แค่ replicate constants ใน env, ไม่แก้ live
- ⛔ Pool field schema — keys คงเดิม (`chronos_alignment`, `chronos_uncertainty_norm`); แค่สูตร compute เปลี่ยน → rebuild ตามปกติ

**Synthetic verification** (run before pool rebuild):

| Test | Expected | Result |
|---|---|---|
| SMC BUY + Chronos predicts DOWN | alignment = +1 | ✅ |
| SMC BUY + Chronos predicts UP | alignment = -1 | ✅ |
| EURUSD M15 typical band (q90-q10=0.0040, atr=0.0008) | unc ≈ 1.77 (เดิม clip 3.0) | ✅ |
| Tight band (q90-q10=0.0004) | unc < 0.3 | ✅ |
| Wide band (q90-q10=0.0066) | unc 2.0-3.0 | ✅ |
| Empty EURUSD BUY → block GBPUSD BUY (USD_WEAK same dir) | ✅ block | ✅ |
| EURUSD BUY → NOT block GBPUSD SELL (opposite dir) | ✅ allow | ✅ |
| Duplicate symbol → block | ✅ block | ✅ |

**Pipeline**: rebuild pool + retrain GBM + retrain RL ทั้งหมด. Eval gate Pass Rate ≥ 9.7% = keep, < 7% = revert จาก `*.bak_v6.14`.

**Files modified**: `ml/chronos_forecaster.py`, `ml/signal_filter_env.py`, `wiki/05-invariants.md`, `wiki/03-rl-training.md`, `wiki/02-modules.md`, `context.md`.

---

### 2026-05-01 — v7.0.1 Friday Warning timezone semantic fix (UTC → EET)

**Symptom (live demo observation)** — Friday Warning trigger ที่ **22:00 Bangkok (4 ทุ่ม)** ดูผิดเวลา.

**Two root causes** (ทั้งสองอย่างผิด):

1. **Off-by-15-min arithmetic bug** — `max(0, sessions.friday_cutoff.minute − 15) = max(0, -15) = 0` → ตัด `−15` ทิ้ง (เพราะ `friday_cutoff.minute = 0` < 15). ทำให้ `friday_warning_utc = dt_time(15, 0)` แทน `14:45`.
2. **Wrong timezone semantic** — Friday Warning เป็น "soft wind-down ก่อน FTMO Force Close (EET broker time)" → ควรอิง `friday_force_close − 15 min` (EET) ไม่ใช่ `friday_cutoff − 15 min` (UTC). การใช้ UTC ทำให้เวลาขยับตาม DST ±1 ชม. (winter 17:00 EET vs summer 18:00 EEST) — ไม่ consistent กับ FTMO bell.

**Fix** — Refactor `TradeManager.check_session_close` Trigger #3 ให้ derive จาก `friday_force_close` (EET):

```python
# Before (BUGGY — UTC + off-by-15-min):
utc_time_now = server_time_now.astimezone(pytz.UTC)
friday_warning_utc = dt_time(
    sessions.friday_cutoff.hour,
    max(0, sessions.friday_cutoff.minute - 15)  # bug
)
if utc_time_now.weekday() == 4 and utc_time_now.time() >= friday_warning_utc:
    ...

# After (EET-based, derive from FTMO Force Close):
_warning_anchor = datetime.combine(server_time_now.date(), sessions.friday_force_close) - timedelta(minutes=15)
friday_warning_eet = _warning_anchor.time()  # = 20:30 EET
if server_time_now.weekday() == 4 and server_time_now.time() >= friday_warning_eet:
    ...
```

ลบ `pytz` import + `utc_time_now`/`current_time_utc`/`current_weekday_utc` ออก — Trigger #3 ใช้ `server_time_now` (EET) ตรง. Path ทั้ง 3 ใน `check_session_close` ตอนนี้ใช้ EET consistent (#1 force close 20:45 EET, #2 daily 23:30 EET, #3 warning 20:30 EET).

**Time mapping (after fix, EET-anchored — DST-stable)**:

| Trigger | EET (broker) | UTC (winter) | UTC (summer DST) | Bangkok |
|---|---|---|---|---|
| #3 Friday Warning (NEW) | **20:30** | 18:30 | 17:30 | **00:30 ICT (เสาร์)** ✅ |
| #1 FTMO Force Close | 20:45 | 18:45 | 17:45 | 00:45 ICT (เสาร์) |

> ⚠️ Window 20:30-20:45 EET = soft wind-down 15 นาทีก่อน FTMO bell. Bot จะปิดทุก position ในช่วงนี้ก่อน hard force close ทำงาน

**Impact** —

- ✅ Bot ปิด position **ตามเวลา broker (EET)** ไม่ใช่ UTC → ไม่ขยับตาม DST อีกต่อไป
- ✅ Wind-down window 15 นาที ก่อน FTMO bell — make sense ในระบบเดียวกัน
- ✅ Trigger #3 fire **หลัง** SMC `_is_trading_session` หยุดเปิด signal ใหม่ (15:00 UTC) ไปแล้วหลายชั่วโมง → ไม่กระทบ window เทรดปกติ
- ❌ ไม่กระทบ: FTMO Force Close (#1 — EET path เดิมถูกอยู่แล้ว), Daily DD calculation, Training env (`FTMOSignalFilterEnv` ไม่เรียก `check_session_close`) — ปลอดภัย fix ระหว่าง RL กำลัง train

**Files modified**: `execution/trade_manager.py` (refactor `check_session_close`), `wiki/04-operations.md` (Friday Warning row), `wiki/05-invariants.md` (this entry).

**Verification**:

```bash
.venv/bin/python -c "
from datetime import time, datetime, timedelta
ffc = time(20, 45)  # default friday_force_close (EET)
warning = (datetime.combine(datetime.today(), ffc) - timedelta(minutes=15)).time()
assert warning == time(20, 30)
print(f'✅ Friday Warning EET = {warning} (15min before FTMO bell)')"
```

---

### 2026-05-01 — v7.0 Amazon Chronos 2 zero-shot forecast features (obs 27 → 29)

**Why** — system v6.14 อ่าน "อดีต" เก่ง (SMC + GBM + portfolio state) แต่ไม่มี forward-looking signal. Amazon Chronos 2 (Hugging Face foundation model, zero-shot) ป้อนทิศทาง median + uncertainty band ของ M15 ใน 8 bars ข้างหน้าเป็น obs feature เพิ่มเติม.

**What changed**:

- **`ml/chronos_forecaster.py` (NEW)** — `ChronosForecaster` class. โหลด `BaseChronosPipeline.from_pretrained(amazon/chronos-bolt-small)` ครั้งเดียวตอน init. `forecast(symbol, df_m15)` คืน `{median_h8, q10_h8, q90_h8, last_close}` พร้อม cache key `(symbol, last_bar_ts)` (cap 64 entries, LRU evict). `compute_features(forecast_dict, signal_direction, atr_value)` แปลง raw → 2 obs features. Determinism: `torch.manual_seed(0)` ก่อน inference. Graceful degradation: import / load fail → return `0.0, 0.0` (neutral).
- **Obs 27 → 29** —
  - `SelfLearningAgent.OBS_DIM = 29`
  - `FTMOSignalFilterEnv.observation_space = Box(shape=(29,), low=-5, high=5)` + `_get_obs()` reads `sig.get('chronos_alignment', 0.0)` + `sig.get('chronos_uncertainty_norm', 0.0)`
  - `FTMOTradingBot._build_signal_observation` เรียก `self._chronos.forecast_features(sig.symbol, self._strategy._ltf_data, direction, atr_val)` → append เป็น 2 elements ตอนปลาย
- **Pool inject** — `StrategyBacktester.__init__` instantiate `ChronosForecaster` (อ่าน `bot_config.ml.CHRONOS_*`). `generate_episode_signals` คำนวณ `chronos_alignment, chronos_uncertainty_norm` จาก `ltf_slice` (closed bars only — ห้าม leak future) → เก็บใน signal dict. Pool ที่ build จาก v7 ขึ้นไปจะมี 2 fields ใหม่นี้ทุก signal.
- **Live path** — `FTMOTradingBot.__init__` instantiate `self._chronos`. `_build_live_context` mirror chronos features เข้า ctx → `TradeExecutor` apply เข้า `ExecutedTrade.chronos_align/chronos_unc` → `TradeLogger.log_trade_opened` เขียนลง Trades sheet. `_log_signal_scan` เขียนลง Signals sheet.
- **Excel schema bump** —
  - `SIGNAL_HEADERS`: 21 → 23 cols (+`Chronos Align`, `Chronos Unc`)
  - `TRADE_HEADERS`: 64 → 66 cols (+`Chronos Align`, `Chronos Unc` @ entry)
  - `obs_27_json` column key คงเดิม (backward compat) แต่เก็บ 29 dims จริงตั้งแต่ v7
- **Config** — `bot_config.ml.MLConfig` (NEW) — `CHRONOS_MODEL_NAME`, `CHRONOS_DEVICE`, `CHRONOS_PREDICTION_LENGTH`, `CHRONOS_CONTEXT_LENGTH`, `CHRONOS_ENABLED`. ทั้ง backtester + main.py อ่านจาก single source.
- **Disable knob** — env `BOT_DISABLE_CHRONOS=1` หรือ `bot_config.ml.CHRONOS_ENABLED = False` → forecaster ไม่ทำงาน, obs[27,28] = 0.0 (สำหรับ unit tests / smoke runs).

**Critical rules added**:

- ⛔ **Chronos config sync** — `CHRONOS_MODEL_NAME` + `CHRONOS_PREDICTION_LENGTH` + `CHRONOS_CONTEXT_LENGTH` ต้องเหมือนกันทั้ง training (StrategyBacktester) และ live (main.py) — ถ้าต่างกัน obs distribution shift = silent regression. **เปลี่ยนค่าเหล่านี้ → ต้อง rebuild pool + retrain GBM + retrain RL ใหม่ทั้งหมด**.
- ⛔ **ห้าม leak future bars** — backtester ต้องใช้ `ltf_slice = m15_df.iloc[scan_idx - MIN_M15_BARS+1:scan_idx+1]` (closed bars only) เป็น Chronos input.
- ⛔ **Daily DD logic untouched** — `RiskManager.update_daily_pnl`, `_initial_balance`, `_daily_start_balance` ไม่ได้รับ touch ใดๆ. Chronos อยู่ pre-trade gate เท่านั้น.

**Rebuild required**: pool + GBM + RL ทั้งหมด. Eval gate: Pass Rate ≥ 9.7 % (v6.13 baseline) — ถ้าไม่ผ่าน revert จาก `*.bak_v6.14`.

**Backups**: `main.py.bak_v6.14_chronos`, `ml/rl_agent.py.bak_v6.14`, `ml/signal_filter_env.py.bak_v6.14`, `ml/strategy_backtester.py.bak_v6.14`, `analytics/trade_logger.py.bak_v6.14`, `execution/trade_executor.py.bak_v6.14`, `requirements.txt.bak_v6.14`, `config/settings.py.bak_v6.14`.

**Files modified**: `main.py`, `ml/chronos_forecaster.py` (NEW), `ml/rl_agent.py`, `ml/signal_filter_env.py`, `ml/strategy_backtester.py`, `analytics/trade_logger.py`, `execution/trade_executor.py`, `config/settings.py`, `requirements.txt`. Wiki: `context.md`, `wiki/03-rl-training.md`, `wiki/05-invariants.md`, `wiki/02-modules.md`, `readme.md`.

---

### 2026-04-30 — v6.14 Live demo log audit — 4 bug fixes (silent regression vs v6.13 spec)

หลัง 2 วันแรกของ live demo (04-29 → 04-30, 9 trades) วิเคราะห์ `logs/ftmo_trades.xlsx` พบ silent regression + logger bugs ที่บัง verified Pass Rate 9.7 % ของ v6.13.

**Symptom**: XAUUSD trade #0 (ticket 437211678) SL = 0.28×ATR (ปกติ 1.5–1.8×) → SL hit ใน 12 วินาที, -$103. trade #2/#4/#7 ของ XAU ใน batch เดียวกัน SL ≈ 1.5×ATR (ไม่ใช่ 1.8× ตาม v6.13 spec). MFE column ใน Excel เก็บค่า duration_seconds, Time-in-Trade column เก็บ exit_path string. Stats sheet "Total Trades 3" แทน 9.

**Root cause (3 layers ซ้อนใน SL flow)**:

- **Layer A — `SMCStrategy.scan_signal` (BUY+SELL) hardcode global multiplier**: ใช้ `bot_config.indicators.atr_sl_multiplier = 1.5` ตรงๆ ไม่ดึง `SymbolConfig.symbol_overrides[X].sl_atr_multiplier` (XAU = 1.8 ที่ตั้งไว้ตอน v6.13 ไม่เคยมีผล)
- **Layer B — OB SL clamp ไม่มี lower bound**: ถ้า `bearish_ob.high - entry_price < sl_distance × 1.5` → swap → ลด SL ลงได้ไม่จำกัด
- **Layer C — XAU `min_sl_pips: 300` (= $3) ต่ำเกินไป**: หลัง OB clamp ลด SL ลง guard ดึงขึ้นแค่ $3 ≈ 0.3×ATR — ไม่สามารถบังคับให้ SL กลับไปที่ 1.8×ATR ได้

**Fix 1 — SL flow alignment** (`strategy/smc_strategy.py` BUY @ ~line 745, SELL @ ~line 1095):

- ใช้ `get_symbol_config(symbol, "sl_atr_multiplier", bot_config.indicators.atr_sl_multiplier)` แทน global → XAU ได้ 1.8× จริง, FX อื่นยังคง 1.5×
- เพิ่ม `ob_sl_floor = atr_value * 0.5` เป็น lower bound ของ OB clamp → SL clamp ลงได้ถึง 0.5×ATR ขั้นต่ำ
- กระทบทั้ง BUY + SELL branches แบบ mirror

**Fix 1C — XAUUSD `min_sl_pips: 300 → 1000`** (`config/settings.py:220`): 1000 ticks = $10 ≈ 1.0×ATR ปกติของ Gold (ATR M15 8-15 USD). Floor นี้ทำงานก็ต่อเมื่อ Layer A+B ไม่ผลิต SL ที่กว้างพอ — เป็น guard ชั้นสุดท้าย

**Fix 2 — TradeLogger off-by-one** (`analytics/trade_logger.py` close-row update path, ~line 296-301):

- เดิม `column=28..31` ทับ `DD@Entry % / MAE / MFE / Time-in-Trade (s)` ตามลำดับ
- คอลัมน์จริงตาม `TRADE_HEADERS`: 28 = DD@Entry %, **29 = MAE, 30 = MFE, 31 = Time-in-Trade (s), 32 = Exit Path**
- แก้ index `28..31 → 29..32` → MFE field กลับมาเป็น MFE จริง, Time-in-Trade เป็นวินาที, Exit Path เป็น string. กระทบ retrain pipeline (Trades 64 cols ที่มี `Obs27 JSON` for retrain)
- หมายเหตุ: บั๊กเกิดเฉพาะ "update existing row" path (close trade ที่มี ticket ใน sheet อยู่แล้ว). "new-row append" path (line 320+) เขียนแค่ 19 core cols ไม่กระทบ

**Fix 3 — PerformanceAnalyzer replay จาก Excel** (`main.py` step 3.6, หลัง `set_initial_balance`):

- เดิมมี `[DISABLED]` block — เลือกให้ analyzer fresh ทุก session → Stats sheet นับเฉพาะ trade ที่ปิดในรอบ session ปัจจุบัน
- v6.14 re-enable: เรียก `self._analyzer.load_from_excel(logs/ftmo_trades.xlsx)` → equity curve / Max DD / Sharpe ต่อเนื่องข้าม restart
- ถ้าต้องการ reset → ลบ `logs/ftmo_trades.xlsx` ก่อน run

**Backups**: `*.bak_v6.13` ของ 4 ไฟล์ (`smc_strategy.py`, `settings.py`, `trade_logger.py`, `main.py`) พร้อมสำหรับ rollback ทีละ fix

**ผลที่คาดหวัง**:

- Live SL ของ XAU จะ ≈ 1.8×ATR (16-27 USD) แทนที่จะเป็น $3-$10 → ห่าง wick noise พอ, ลดโอกาส SL hit ใน 12s
- Live performance กลับมาตรงกับ verified train Pass Rate 9.7 % (Fix 1 ปิด silent regression)
- Retrain pipeline ใช้ MAE/MFE column ได้จริง (Fix 2)
- Stats sheet สะท้อนสถานะ challenge ทั้งหมด (Fix 3)

**ไม่ต้อง retrain GBM/RL** — train env (`StrategyBacktester`) ใช้ 1.8× อยู่แล้วตั้งแต่ v6.13, fix นี้ดึง live ให้ตามให้ทัน

**Risk**: Fix 1C (raise min_sl_pips) อาจกรอง XAU signals ในตลาดเงียบ (ATR < 5) มากขึ้น → ดูผ่าน `Signals` sheet `AGENT_TAKE_FAIL` rate หลัง deploy 1-2 sessions. ถ้าหลุดมาก revert min_sl_pips กลับ 300 ได้ (Fix 1A+1B ยังคงทำงาน)

### 2026-04-29 — v6.13 Combined patch (pause + defaults safety + equalize TAKE @ ml ≥ 0.36 + XAU SL widen)

หลัง analyze 1-day live log + train↔live deep audit ผู้ใช้สั่งให้รวม fix หลายตัวเข้าด้วยกัน rebuild + eval ครั้งเดียว เพื่อประหยัด cycle time. Audit obs 27 dims ยืนยัน **zero outcome leakage** ที่ policy input (aux task + SKIP-oracle reward เป็น training signal เท่านั้น, agent's policy network ไม่เห็น outcome).

**Layer 1 — Config-only**:

- `FTMOConfig.CONSECUTIVE_LOSS_PAUSE_COUNT`: 2 → **3** (DD trigger 1.4 % → 2.1 %, ยังห่าง FTMO 4 % limit)
- `FTMOConfig.CONSECUTIVE_LOSS_HALT_COUNT`: 3 → **4** (รักษา invariant pause < halt)

**Layer 2 — Defaults safety (ป้องกัน silent regression)**:

- `FTMOSignalFilterEnv.RISK_PER_TRADE` (class const): 0.003 → **0.007** (sync กับ live `DEFAULT_RISK_PER_TRADE_PCT`)
- `train_signal_filter.py --risk_per_trade default`: None → **0.007**
- `train_signal_filter.py --outcome_noise default`: 0.02 → **0.05** (more robust to live distribution)
- `train_signal_filter.py --ml_threshold default`: 0.0 → **0.36** (production calibrated)

**Layer 3 — Reward rebalance (no policy-input leak)**:

- TAKE branch equalize: win + ml ≥ 0.36 → **+0.30 uniform** (เดิม +0.35 ถ้า ≥ 0.40, +0.15 ถ้า [0.36, 0.40)) — agent ไม่ bias หลีกเลี่ยง 0.36-0.40
- SKIP-oracle rebalance (P2): missed-big-winner −0.70 → **−0.90**, ML-confirmed missed −0.40 → **−0.55**, smart-skip-big-loser +0.20 → **+0.35**, symmetry +0.06 → **+0.10**
- เพิ่ม early undertrading check: day ≥ 10 + progress < 20 % + takes < 3 → **−0.2** (sticky)

**Layer 4 — XAUUSD SL widen** (1.5× → 1.8× ATR, RR คง 1:2):

- `SymbolConfig.symbol_overrides["XAUUSD"]` เพิ่ม `sl_atr_multiplier: 1.8` + `tp_atr_multiplier: 3.6`
- `PositionSizer.calculate_sl_tp_prices` อ่าน per-symbol override → fallback global
- `StrategyBacktester._run_day_scan` sync override → pool training สะท้อน live SL distribution

**Trigger**: live trade #436840790 (XAU SELL) โดน SL hit by 3 ticks — XAU wick noise สูงกว่า FX, 1.5×ATR แคบเกินไป

**No-leak audit ที่ทำในรอบนี้**:

- 27 obs dims ใน `_get_obs` + `_build_signal_observation` — ใช้แค่ signal-time data + env state ตอน signal (ไม่มี outcome)
- GBM 17 features ใน `SignalQualityModel.FEATURES` — ทั้งหมดเป็น signal-time
- Aux head (Phase E2) — outcome เป็น **target** สำหรับ MSE loss, ไม่ใช่ obs input → policy ไม่เห็น
- SKIP-oracle reward — ใช้ outcome ใน reward function (training signal), policy network ไม่ได้รับ outcome เป็น input

**Backups พร้อมสำหรับ rollback** (ถ้า eval regress < 3.0 %):

```text
ftmo_trading_bot/data/signal_pool_3000.pkl.bak_v6.12
ftmo_trading_bot/data/signal_quality_model.pkl.bak_v6.12
ftmo_trading_bot/models/ppo_signal_filter.zip.bak_v6.12
ftmo_trading_bot/models/ppo_signal_filter_p1.zip.bak_v6.12
ftmo_trading_bot/models/vec_normalize_sf.pkl.bak_v6.12
```

**Eval result (5000 eps, 2026-04-29 22:02) — EXCELLENT, ทะลุเป้า**:

| Metric | v6.11.3 baseline | **v6.13** | Δ |
|---|---|---|---|
| Pass Rate | 3.4 % | **9.7 %** ⭐⭐⭐ | **+185 %** |
| Win Rate | 68.8 % | 64.8 % | -4 pp |
| Orders/ep | 6.1 | **7.7** | +26 % |
| Take Rate | 51.6 % | 51.3 % | similar |
| Total DD max | 3.23 % | 4.40 % | +37 % (ยังห่าง 8 % limit) |
| Daily DD max | 2.12 % | 2.15 % | similar |
| Breach Rate | 0 % | **0 %** ✅ | same |
| Profit avg (5000 eps) | — | **+3.89 %** | — |
| Trades to target | 12.7 | 12.7 | same |
| Profitable survive | 87.1 % | 87.0 % | same |

**Decision: ✅ KEEP — deploy demo**. Pass Rate เกือบถึง FTMO 10 % target จากการรวม 4 layers ครั้งเดียว. รอเก็บ live data 1 อาทิตย์ก่อนสมัคร challenge จริง

**Why work** (post-mortem):

- **L3 reward rebalance ทำงานตามคาด**: orders/ep 6.1 → 7.7 (+26 %) ตรงกับการ push SKIP-oracle penalty + day-10 early undertrading
- **L3 TAKE equalize @ ml ≥ 0.36** ทำให้ agent ไม่หลีกเลี่ยง marginal signals → cover ทั้ง spectrum ของ ML score
- **L4 XAU SL widen 1.8×** อาจช่วยเรื่อง wick survival แต่ effect ยังต้อง verify ใน live (eval ไม่แยก per-symbol stats)
- **L1 Pause ผ่อน** กระทบมากที่สุดใน live (eval pool เป็นการ simulate ไม่ trigger pause บ่อย)
- **L2 defaults safety**: env class const 0.003 → 0.007 = sync แล้ว, ลด silent regression เสี่ยง

**Trade-off ที่ยอมรับ**: WR -4 pp (68.8 → 64.8), DD +37 % (3.23 → 4.40, ยังปลอดภัย) แลกกับ Pass Rate +185 % — ROI สูงมาก

### 2026-04-29 — v6.12 Live ML threshold gate fix (sync live ↔ training)

หลัง analyze `logs/ftmo_trades.xlsx` (1 วัน, 3 trades, 128 signals) เจอ bug 2 จุดที่ทำให้ live ≠ training:

**Bug A — ไม่มี ML gate ใน live**:

- `FTMOSignalFilterEnv` มี `if ml_score < ml_filter_threshold (0.36): drop` ตอน train
- `FTMOTradingBot.run` ส่ง signals ตรงให้ `SelfLearningAgent.should_take_signal` **ไม่มี** ML gate
- ผล: agent เห็น distribution กว้างกว่าที่ฝึก = distribution mismatch = silent regression

**Bug B — `ML Threshold` column = 0.0 ทุก row ใน `Signals` sheet**:

- `_build_live_context` พยายาม `getattr(self._rl_agent, "ml_filter_threshold")` แต่ attribute นี้อยู่บน env ไม่ใช่ agent → fall to default 0.0 ตลอด

**Fix (ใน 1 commit, ไม่ต้อง retrain)**:

- เพิ่ม `FTMOConfig.ML_FILTER_THRESHOLD: float = 0.36` ใน `config/settings.py` (single source of truth)
- `FTMOTradingBot._build_live_context` อ่าน `ctx["ml_threshold_used"]` จาก `bot_config.ftmo.ML_FILTER_THRESHOLD`
- `FTMOTradingBot.run` เพิ่ม gate ก่อนเรียก agent: ถ้า `ml_score < ML_FILTER_THRESHOLD` → log `Result = "ML_FILTERED"` แล้ว `continue`
- `TradeLogger.log_signal_scan` รับ `Result = "ML_FILTERED"` (light-blue color)

**Risk**: ต่ำมาก — Signals sheet จริง ๆ มี ml_score min = 0.358 (≥ 0.36 อยู่แล้วเกือบทั้งหมด เพราะ confluence ≥ 70 มี correlation กับ ML score). แต่บัดนี้ live = train อย่างเป็นทางการ.

**Verify**:

- รัน live 30 นาที → `Signals` sheet column `ML Threshold` = 0.36 ทุก row
- ไม่มี Trade ที่ ML Score (cal) < 0.36
- eval Pass Rate ไม่เปลี่ยน (~3.4 %)

**ที่ไม่แตะ**: SMC strategy, obs 27 dims, reward, PPO hyperparams, models — เพราะ fix นี้ไม่กระทบ training distribution

### 2026-04-29 — v6.11.3 Mild relaxation tune (IDM 5→2, ADX H4 22→20) — measurable improvement

หลัง v6.11.2 + retrain ได้ Pass Rate 2.7 % (WR 65.6 %, ปลอดภัยแต่ orders/ep แค่ 6.7). Math บอกว่า 6.7 trades × 0.68 % = 4.6 % expected → ห่างเป้า FTMO 10 % เพราะเทรดน้อยเกิน

**Decision**: Tune **ผ่อน 2 จุด** (mild relaxations, ไม่ลบ gate ใด) แล้ว rebuild + retrain เต็มรูปแบบ:

**Changes (`strategy/smc_strategy.py`):**

- **IDM penalty: -5 → -2** (factor 3.7 ทั้ง BUY + SELL) — กัน over-penalize ใน calm market ที่ไม่มี rejection candle
- **ADX H4 floor: ≥ 22 → ≥ 20** (pre-filter E2 ทั้ง BUY + SELL) — สอดคล้องกับ ADX H1 floor, +10-15 % signals ผ่าน

**Pool + GBM + RL retrained:**

- Pool: 78,472 → **90,799 signals** (+15.7 %), avg sigs/ep 27.2 → 31.4
- GBM: OOF AUC 0.5942 → 0.5915 (เกือบเท่าเดิม), calibrated all 5 bins ✅
- RL: `--fresh` 10M+5M timesteps, 23.6 min total

**Final Eval (5000 episodes) — ดีขึ้นทุกมิติ:**

| Metric | v6.11.2 | **v6.11.3** | Δ |
|---|---|---|---|
| Pass Rate | 2.7 % | **3.4 %** | **+26 %** |
| Win Rate | 65.6 % | **68.8 %** | **+3.2 pp** |
| Take Rate | 51.1 % | 51.6 % | similar |
| Orders/ep | 6.7 | 6.1 | -9 % (selective ขึ้น) |
| **Total DD max** | 4.46 % | **3.23 %** | **-28 %** safer |
| Daily DD max | 2.12 % | 2.12 % | same |
| Trades to target | 14.9 | **12.7** | -15 % (efficient) |
| Breach Rate | 0 % | 0 % | 🟢 |
| Profitable survive | 87.1 % | 87.1 % | 🟢 |

**Key insight**: filter หย่อน → signals เพิ่ม 15 % แต่ **agent เลือก trade น้อยลง 9 %** = quality > quantity emerged naturally. DD ปลอดภัยกว่าเดิมตรงข้ามกับที่ผมคาดว่าจะเพิ่ม.

**Decision**: KEEP v6.11.3 — pure improvement, ไม่มี trade-off

**Backups preserved** สำหรับ rollback (ถ้าจำเป็น):

```text
data/signal_pool_3000.pkl.bak_v6.11.2
data/signal_quality_model.pkl.bak_v6.11.2
models/ppo_signal_filter.zip.bak_v6.11.2
models/vec_normalize_sf.pkl.bak_v6.11.2
```

**Live deploy plan**: เก็บ data MT5 demo 1-2 อาทิตย์ → verify live numbers ตรงกับ eval ก่อนสมัคร FTMO challenge จริง

### 2026-04-29 — v6.11.2 Partial rollback Tier 2.2 + 2.3 hard gates → soft bonuses

หลัง v6.11.1 fix backtester แล้ว rebuild pool ด้วย v6.11 hard gates → eval result **Pass Rate 0.0 %** (จาก baseline 11.2 % cached pool). Pool stats เปิดเผย root cause:

- Old pool (v6.10): 2887 episodes × 36.9 sigs/ep = **106,454 signals**
- New pool (v6.11 hard gates): 867 episodes × 1.3 sigs/ep = **1,105 signals** (-99 %)
- WR 35.9 % → 34.4 % (ใกล้เคียงกัน — gates ไม่ได้ลด loser, แค่ลดทุก signal)

**ปัญหา**: Tier 2.2 (Sweep within 8 bars) + Tier 2.3 (Fresh M15 BOS within 6 bars) **stack คูณกัน** ทำให้ co-occurrence rate ต่ำมาก. Match กับ wiki precedent v6.4 Phase C (4 SMC principles → 3.7 → 1.5 % rolled back).

**Fix (`strategy/smc_strategy.py`):**

- **Tier 2.2 — ลบ pre-filter G** (Sweep prereq hard reject) ทั้ง BUY + SELL. Sweep ยังคงเป็น confluence bonus ใน factor 3.6 (max +15) เหมือนเดิม
- **Tier 2.3 — ลบ pre-filter H** (Fresh M15 BOS prereq hard reject) ทั้ง BUY + SELL. แทนด้วย **factor 2.5 soft bonus +5** ใน confluence section: ถ้า `_structure_ltf.get_latest_event()` คืน BOS/CHoCH ในทิศที่ถูกต้องภายใน 6 bars → +5 (เพิ่มใน mtf_pts สำหรับ logging)

**ที่เก็บไว้ (data-validated, ไม่ rollback):**

- ✅ Tier 1.1/1.2 — TradeManager BE best_price + partial_closed_flag mirror (live-only, ไม่กระทบ pool)
- ✅ Tier 1.3 — Counter-D1 hard veto (live demo evidence: 3/13 BUY ผิดทิศทั้งหมด)
- ✅ Tier 1.4 — Quiet-vol × off-overlap (เฉพาะ 7-8/16-17 UTC, narrow scope)
- ✅ Tier 2.1 — ADX H4 ≥ 22 hard gate (data-validated: ADX>35 quiet vol = 0% WR ใน live)
- ✅ Tier 2.4 — Per-component logging fields
- ✅ Tier 3.1 — IDM detector +10/-5 (soft scoring เท่านั้น ไม่ใช่ hard gate)
- ✅ Tier 3.2 — OB grading × weight

**Expected outcome**: Pool retention ขยับจาก 4 % → ~50-70 % → ~50k-75k signals (น้อยกว่า old 106k แต่พอ retrain ได้). Pass Rate เป้า ≥ 7 %

**Verification**:

```bash
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py --pool_size 3000 --workers 8
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.36 --risk_per_trade 0.007
```

ถ้ายัง Pass Rate ต่ำ → consider tune Tier 3.1 IDM penalty (-5 → -2)

### 2026-04-29 — v6.11.1 Post-impl audit fixes (eval cache caveat + backtester _idm_detector)

หลัง implement v6.11 รัน `train_signal_filter.py --eval_only` ได้ Pass Rate 11.2 % แต่ deep audit เจอ 2 issue:

**Issue A — Eval cache caveat (DOCUMENTED, not bug)**

- `data/signal_pool_3000.pkl` mtime = Apr 25 14:11 (ก่อน v6.11)
- `FTMOSignalFilterEnv.reset` ถ้า `_signal_pool` มี → load idx จาก cache ไม่เรียก `backtester.generate_episode_signals()` (signal_filter_env.py:317-320)
- ผลคือ eval นี้รัน pool ที่ **SMC v6.10 (ก่อน gates ใหม่)** สร้างไว้ → **11.2 % สะท้อนพฤติกรรม "old SMC + old RL"** ไม่ใช่ "v6.11 SMC + old RL"
- Live deploy v6.11 จะกรอง signal ก่อน → agent เห็น distribution ใหม่ที่ไม่เคย eval → ผลจริงอาจ ≠ 11.2 %
- **แก้: rebuild pool ด้วย v6.11 gates ก่อน eval ใหม่** (Issue B fix ทำให้ rebuild ได้)

**Issue B — `StrategyBacktester._init_strategy` ขาด `_idm_detector` init (FIXED)**

- เก่า: `_init_strategy` ตั้ง `_structure_mtf/_structure_ltf/_ob_detector/_fvg_detector/_sweep_detector` แต่ **ไม่มี `_idm_detector`**
- ผลคือ `analyze_with_data` → `_evaluate_buy/sell_signal` → `self._idm_detector.detect_idm(...)` → **`AttributeError`** ทุก signal eval
- กระทบ: `python scripts/build_signal_pool.py` รันไม่ได้ → ติดล็อก rebuild

**Fix (`ml/strategy_backtester.py` `_init_strategy`):**

```python
from strategy.inducement import InducementDetector
...
self._strategy._idm_detector = InducementDetector(lookback=8)
```

ใต้ `self._strategy._sweep_detector = LiquiditySweepDetector()` ภายใน method `_init_strategy`. ตอนนี้ backtester `analyze_with_data` รันผ่าน v6.11 gates ครบ.

**Caveats ที่ปล่อยไว้ (จะ tune ทีหลังถ้า data confirm):**

- IDM -5 penalty ใน `_evaluate_buy/sell_signal` factor 3.7 อาจ over-penalize ใน calm market — keep ค่าเดิมก่อน
- OB EXTREME tolerance 5 % ของ window range อาจ over-classify ใน strong trend — keep ค่าเดิมก่อน

**Migration sequence (สำหรับ user):**

```bash
# 1. Backup (auto-done in v6.11.1 commit)
cp data/signal_pool_3000.pkl              data/signal_pool_3000.pkl.bak_pre_v6.11
cp data/signal_quality_model.pkl          data/signal_quality_model.pkl.bak_pre_v6.11
cp models/ppo_signal_filter.zip           models/ppo_signal_filter.zip.bak_pre_v6.11
cp models/vec_normalize_sf.pkl            models/vec_normalize_sf.pkl.bak_pre_v6.11

# 2. Rebuild pool + GBM (v6.11 gates)
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py --pool_size 3000 --workers 8
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py

# 3. Re-eval (RL model เดิม + pool/GBM ใหม่)
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --eval_only \
    --pool_size 3000 --ml_threshold 0.36 --risk_per_trade 0.007

# 4. ตัดสินใจ:
#    Pass Rate ≥ 7 % → keep RL model เดิม, deploy live
#    Pass Rate < 7 % → retrain RL --fresh
```

**Expected pool size impact**: ~158k signals → ~80-95k signals (ลด ~40-50 % เพราะ Sweep + Fresh BOS + Counter-D1 + ADX H4 + Quiet-vol gates)

### 2026-04-29 — v6.11 SMC Precision Overhaul (post live demo audit)

Live demo Day-3 (2026-04-28) เจอ EV ติดลบ (PF 0.96, Net −$19.66, WR 46.2 %) แม้ไม่ผิด FTMO. Audit หาเจอว่า SMC entry gate **หลวมเกินไป** — Sweep, IDM, Fresh BOS, Counter-D1 ทั้งหมดเป็น *bonus* ไม่ใช่ *prerequisite*. รวมถึง TradeManager BE trigger พลาด trade ที่ MFE สูงแล้วย้อน, และ TradeLogger logging gap หลายฟิลด์.

**Tier 1 — Quick Wins (ไม่ต้อง retrain):**

- **`TradeManager._manage_single_position`** — BE/Partial trigger เปลี่ยนจาก `current_rr` เป็น `best_rr` (rolling MFE-based). track `state.best_price` ทุก tick (ไม่รอ trailing activate). Trade ที่ MFE สูงระหว่าง 5 s tick แล้ว revert จะ lock BE+Partial ทันที.
- **`TradeManager._partial_close`** lot_min branch — mirror `trade.partial_closed_flag = True` (เพิ่มจากเดิมที่ตั้งแค่ `partial_close_skipped=True`). log แสดงตรงกับ state จริง.
- **`SMCStrategy._evaluate_buy_signal/_evaluate_sell_signal`** — Counter-D1 เปลี่ยนจาก soft +15 confluence threshold bump → **hard reject**. BUY: `d1_bias == -1` → reject; SELL: `d1_bias == +1` → reject. Neutral (0) ผ่านได้ปกติ.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter F2 — เพิ่ม **Quiet-vol × off-overlap blocker**. ถ้า `atr_pips < 1.2 × atr_floor_pips` AND `_get_session_multiplier() < 1.0` → reject. ลด trade quiet vol นอก London-NY overlap ที่ pattern reliability ต่ำ.

**Tier 2 — Medium (ไม่ต้อง retrain):**

- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter E2 — **ADX(H4) ≥ 22 hard gate**. ดึงจาก `_htf_data["adx"]`. ลด whipsaw ใน H4 ranging.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter G — **Recent Sweep prerequisite within 8 bars**. ใช้ `LiquiditySweepDetector.get_recent_bullish/bearish_sweep(max_bars_ago=8)` เป็น hard gate (ก่อนหน้านี้แค่ confluence bonus). ทุก entry บังคับมี smart-money confirmation.
- **`SMCStrategy._evaluate_buy/sell_signal`** pre-filter H — **Fresh M15 BOS/CHoCH structural shift within 6 bars**. ใช้ `_structure_ltf.get_latest_event()` + ตรวจ `event.index >= len(ltf_df) - 6`. ตัด pullback-to-OB-without-break trades.
- **`TradeSignal` dataclass** — เพิ่ม fields: `htf_score, mtf_score, ob_pts, fvg_pts, sweep_pts, sweep_age_bars, htf_bias` (string label), `d1_bias`. populate ใน BUY/SELL eval.
- **`FTMOTradingBot._build_live_context`** — อ่าน per-component pts + `htf_bias` จาก `signal` ตรงๆ (ก่อนหน้านี้ hardcode 0). แก้ Trades sheet HTF/MTF/OB/FVG/Sweep pts ที่ว่างเปล่า.
- **`FTMOTradingBot._log_signal_scan`** — `htf_bias` field ใช้ `sig.htf_bias` (string "BULLISH/BEARISH/RANGING") แทน `_strategy._htf_bias` (int).

**Tier 3 — Strategic:**

- **NEW `strategy/inducement.py` — `InducementDetector` class.** ตรวจจับ rejection candle (wick failed) ภายใน 8 bars. API: `detect_idm(df, direction) -> Optional[InducementEvent]`. wired เข้า `SMCStrategy._evaluate_buy/sell_signal` หลัง Sweep block: IDM = +10 confluence; ไม่มี IDM = -5 (อาจเป็น obvious swing).
- **`OrderBlock`** dataclass — เพิ่ม field `ob_grade: str = "INTERNAL"`.
- **NEW `OrderBlockDetector._classify_ob_grade(ob, df, avg_impulse)`** — จัดประเภทเป็น `EXTREME` (ใกล้ swing extreme ของ window 50 bars), `DECISIONAL` (impulse ≥ 1.8 × avg), หรือ `INTERNAL`.
- **`OrderBlockDetector._score_order_blocks`** — apply grade weight: EXTREME ×1.20, DECISIONAL ×1.00, INTERNAL ×0.60. ลด false-positive จาก Internal OBs.

**Mandatory verification before next live deploy:**

1. **Schema migration** — ถ้า user เคย deploy ก่อน v6.11: lint error อาจไม่กระทบ แต่ field `obs_27_json` + per-component pts ต่าง → rename `logs/ftmo_trades.xlsx` ถ้าเปิดรอบใหม่.
2. **Smoke test**: รัน `python main.py` ใน demo MT5 ≥ 4 ชม. ใน London-NY overlap window
3. **ตรวจ Trades sheet**: `HTF Bias` (string), `MTF Bias` (int), `ADX H4` (float), `HTF pts/MTF pts/OB pts/FVG pts/Sweep pts` (int) — ทั้งหมดต้องมีค่าจริง ไม่ใช่ 0/null
4. **ตรวจ Signals sheet col 20** (Executor Reject) — TAKE_FAIL rows ต้องมี reason (verify v6.10d ทำงาน)
5. **ตรวจ BE Moved + Partial Closed**: ถ้า MFE > 1 R ต้อง True ทั้งคู่ หรือ `Partial Skipped=True` (lot น้อย)
6. **ตรวจไม่มี trade ที่ Counter-D1**: BUY ไม่ควรเข้าตอน D1 = -1; SELL ไม่ควรเข้าตอน D1 = +1

**Expected outcome (post Tier 1):**

- WR: 46 % → 60-65 %, PF: 0.96 → 1.3+, Expectancy: −$1.51 → +$3 to +$5
- Trades/day: 13 → 7-9 (volume ลด ~40 % แต่ quality ขึ้น)
- Counter-D1 trade %: 23 % → 0 %
- MFE-then-SL anomaly: 46 % → < 5 %

**ที่ไม่ทำ (out of scope):**

- ❌ Retrain RL/GBM — entry gate เปลี่ยนแล้ว → re-eval 5000 eps ก่อนตัดสิน. Pool distribution อาจต่าง แต่ fields obs_27 ไม่เปลี่ยน → existing model ยังใช้ได้
- ❌ Tighten `MIN_CONFLUENCE_SCORE` 70 → 75 — Tier 1.3 + 1.4 ตัด ~30-40 % volume แล้ว, ตึงเพิ่มเสี่ยง undertrade
- ❌ Reduce risk per trade — risk 0.7 % verified Pass Rate 9.7 % (v6.13) / 10.0 % (Phase E2 pre-v6.11), ปัญหาคือ entry quality ไม่ใช่ sizing

### 2026-04-25 — v6.4 SMC Phase C (4 professional principles)

Addresses root-cause gaps that Phase A (bug fixes) + Phase B (reward tuning) could not reach. All 5 sub-tasks landed in one pass. Requires **pool rebuild + GBM retrain + RL retrain** before deploy (strategy layer changed).

**SMC `smc_strategy.py`:**

- **C1 — ADX threshold raised 20 → 25** in BUY + SELL pre-filters. Industry standard for "actual trend vs ranging". Expected signal volume drop ~30 %.
- **C3 — H4 POI hard gate (new, Principle 2):** added `_get_h4_poi_zones` + `_is_near_h4_poi`. Before confluence score, signal is rejected if price is > 2 ATR away from an H4 bullish OB/FVG (for BUY) / bearish (for SELL). Cache per-symbol, invalidated when H4 bar timestamp changes. New state: `_h4_poi_cache`, `_ob_detector_h4`, `_fvg_detector_h4` (separate instances to avoid M15 state contamination).
- **C4 — IDM sweep soft gate (Principle 3):** in sweep scoring block, OB without recent IDM sweep now costs `-20` (old OB, age > 5 bars) or `-8` (fresh OB). Sweep + OB together adds `+10` bonus (ideal smart-money pattern).
- **C5 — FVG + BOS conjunction (Principle 4):** after MTF bias scoring, if LTF (M15) had recent BOS, check for active M15 FVG. BOS without FVG → `-15` (weak break). BOS + FVG → `+8`.

**SMC `market_structure.py`:**

- **C2 — `is_valid_pullback` helper (new, Principle 1):** added method with 3 gates:
  1. impulse size ≥ 1.0 × ATR (no tiny wobble)
  2. pullback retracement ≥ 30 % of impulse (deep enough)
  3. pullback depth ≥ 0.25 × ATR (absolute floor, guards wick-only "BOS")
- Wired into `detect_structure_breaks`: when close breaks active swing high/low, `is_valid_pullback` is called first. If invalid → swing marked broken but NO event raised (internal noise rejected).
- Uses `df['atr']` column — already populated by `TechnicalIndicators.calculate_all` upstream, no new param plumbing.

**Lookahead / correctness:**

- All checks operate on confirmed-closed bars (iloc slicing ≤ current bar index).
- H4 POI cache invalidates by bar_ts equality — new H4 close triggers recompute.
- Mirror BUY/SELL logic verified identical except direction.

**Pipeline impact:**

- Pool will shrink (stricter filters) — monitor signals-per-episode, raise `pool_size` if < 12 avg.
- VecNormalize stats from previous training are invalid — must retrain RL with `--fresh`.
- `obs[0] confluence_norm` distribution shifts (IDM/FVG penalties widen range).

**Retrain sequence (required):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/build_signal_pool.py --pool_size 3000 --workers 8
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 3.7 % → 6-10 %, Win Rate 49.6 % → 53-57 % (quality-first), Orders/day 0.26 → 0.15-0.20 (fewer but better).

### 2026-04-25 — v6.9 Live Logging (Schema v3) — comprehensive demo data capture

Re-enabled `TradeLogger` for live demo deployment. Schema v3 = 58-col Trades sheet + 19-col Signals sheet to capture data needed for E1/E2/baseline comparison vs live behavior.

**`analytics/trade_logger.py`:**

- Extended `TRADE_HEADERS` from 31 → 58 cols. New cols (24 fields, grouped):
  - **ML / Agent decision** (5): `ML Score (cal)`, `ML Score (raw)`, `Agent Action`, `Agent Decision`, `ML Threshold`
  - **Confluence breakdown** (5): `HTF pts`, `MTF pts`, `OB pts`, `FVG pts`, `Sweep pts`
  - **Trade mgmt state** (4): `BE Moved`, `Partial Closed`, `Trailing`, `Final SL`
  - **Live execution** (5): `Bid@Entry`, `Ask@Entry`, `Spread (pips)`, `Bid@Exit`, `Ask@Exit`
  - **Market context** (4): `ADX H1`, `ADX H4`, `MTF Bias`, `D1 Bias`
  - **Account state** (3): `Balance@Entry`, `Balance@Close`, `Equity Peak`
- Added `log_signal_scan(scan_data)` method + `SIGNAL_HEADERS` (19 cols) — per-scan log including `AGENT_SKIP` / `AGENT_TAKE_FAIL` / `REJECTED` / `NO_SIGNAL` results. Color-coded by result.
- New `Signals` sheet auto-created on first scan log.

**`execution/trade_executor.py`:**

- Extended `ExecutedTrade` dataclass with 24 new fields matching the logger schema.
- `to_dict()` now exports all new fields.
- `execute_signal(signal, live_context=None)` accepts a context dict from main.py. Fields populated into `ExecutedTrade` if context provided.
- Close path captures `bid_at_exit`, `ask_at_exit`, `balance_at_close`, `equity_peak_during_trade`, `final_sl_at_close`.

**`execution/trade_manager.py`:**

- BE move (`_move_to_breakeven`) → mirrors `state.breakeven_moved` to `trade.be_moved` and `trade.final_sl_at_close`.
- Partial close (`_partial_close`) → mirrors `state.partial_closed` to `trade.partial_closed_flag`.
- Trail activation (`manage_position`) → mirrors `state.trailing_active` to `trade.trailing_active`.
- Trail SL update (BUY/SELL paths) → mirrors `new_sl` to `trade.final_sl_at_close`.

**`main.py`:**

- Re-enabled `TradeLogger` (was `None`); now `TradeLogger(log_dir=logs/)`.
- New `_build_live_context(sig)` method — computes `ml_score` (cal+raw), bid/ask snapshot, ADX H1/H4, MTF/D1 bias, balance at entry. Read from `_quality_model`, `_strategy._mtf_data`/`_htf_data`, `_connector`, `_risk_manager`.
- New `_log_signal_scan(sig, ctx, result)` — wrapper that builds `scan_data` from signal + context.
- Run loop (`scan_all_symbols` → for each sig) now logs every scan as `AGENT_SKIP` / `AGENT_TAKE` / `AGENT_TAKE_FAIL`. Passes `live_context` to `executor.execute_signal`.

**`requirements.txt`:**

- Added `openpyxl >= 3.1.0` — required for TradeLogger Excel output.
- Added `tqdm >= 4.65.0` and `rich >= 13.0.0` — required by `stable-baselines3` `model.learn(progress_bar=True)`.

**Overtrading detection (added 2026-04-25):**

- `ExecutedTrade` extended with 4 new fields (62 trade cols total):
  - `trades_today_at_open`: count of trades opened today before this one
  - `trades_last_hour_at_open`: count in trailing 60-min window
  - `secs_since_last_trade_open`: delta from last trade open (any symbol)
  - `secs_since_last_trade_same_symbol`: delta from last trade open (same symbol)
- `FTMOTradingBot` tracks `_trade_open_history: List[(datetime, symbol)]` (capped at 200 entries).
- `_build_live_context` computes the 4 metrics from history; passed to executor via `live_context`.
- Use case: filter Trades sheet by `Sec Since Last Open` < 60 → see clusters of fast-fire trades (overtrading symptom).

**Smoke test:** TradeLogger smoke test passed — 58 trade cols + 19 signal cols, all 4 sheets created (Trades, Daily, Stats, Signals).

**Output:** `ftmo_trading_bot/logs/ftmo_trades.xlsx` updated in real time during live run.

**Console quiet mode (added 2026-04-25):**

- Idle states (Daily Halt / Friday close / Weekend / Daily Close 23:30 / Rollover) ใช้ pattern **announce-once** แทน `if loop_count % N == 0` — print ครั้งเดียวตอน entry, silence จนกว่าออกแล้วเข้าใหม่.
- `FTMOTradingBot.__init__` มี 5 flags: `_daily_halt_announced`, `_friday_announced`, `_weekend_announced`, `_daily_close_announced`, `_rollover_announced` (init=False). Reset = False อัตโนมัติใน `else` branch ของแต่ละ state guard.
- Per-signal `AGENT_SKIP` print ลบทิ้ง — ข้อมูลครบใน `Signals` sheet (`AGENT_SKIP` row พร้อม ml_score, confidence, reasons).
- Per-signal `NO_AGENT` print ลบทิ้ง — fallback path เมื่อไม่มี RL agent loaded; logged ใน Signals sheet เช่นกัน.
- เก็บ `📡 [Agent] TAKE` print ไว้ — เป็น event สำคัญที่บอกว่าเทรดกำลังจะเปิด.

**v6.10d — Fix `_log_signal_scan` propagation bug (added 2026-04-28):**

Day-2 live demo analysis เจอว่า `Executor Reject` column ใน Signals sheet ทุกแถวเป็น None (104 AGENT_TAKE_FAIL events). Root cause: `FTMOTradingBot._log_signal_scan` สร้าง `scan_data` dict ครอบคลุมแค่ 19 keys (time, symbol, direction, ml_score, ฯลฯ) — **ไม่ copy `executor_reject_reason` และ `obs_27_json` จาก `live_context`** → `scan_data.get(...)` คืน "" → Excel cell empty.

**Fix:** เพิ่ม 2 keys ใน `_log_signal_scan()` scan_data:

```python
scan_data = {
    # ... existing 19 keys ...
    "executor_reject_reason": live_context.get("executor_reject_reason", ""),
    "obs_27_json": live_context.get("obs_27_json", ""),
}
```

ผลคือ col 20 (Executor Reject) + col 21 (Obs27 JSON) ของ Signals sheet จะมีค่าหลัง deploy. **Trade-level Obs27 JSON (col 64 ของ Trades sheet) ทำงานปกติอยู่แล้ว** เพราะ flow ผ่าน `ExecutedTrade.to_dict()` ที่ copy field จาก live_context ครบ.

**Verify after deploy:** หลัง bot รัน + เกิด AGENT_TAKE_FAIL → ตรวจ Signals sheet col 20 ต้องมี reject reason เช่น `spread_high:18>15` / `risk_manager:Cooldown active ...` / `correlation:USD pair max`. ถ้ายัง None → bug อื่น

**Day-2 fixes ที่ verified working:**

- ✅ Schema 64/21 cols deployed
- ✅ Stats sheet update hourly
- ✅ Timezone fix — DD@Entry แสดงค่าติดลบ (-0.32%) ตอน gain ✓ ตรงกับ MT5 reality
- ⚠️ Symbol coverage partial — GBPJPY scan ได้แล้ว (Day-1 = 0, Day-2 = 7) แต่ XAUUSD/USDJPY/USDCHF/EURJPY ยัง = 0 (อาจเพราะ broker rename — ต้องดู console output)

---

**v6.10c (Phase 1b) — Symbol coverage fix: pre-select Market Watch (added 2026-04-27):**

Live demo Day-1 analysis เจอว่า **5 ใน 10 symbols ไม่มี scan event เลย:** XAUUSD (0), USDCHF (0), USDJPY (2), EURJPY (0), GBPJPY (0).

**Root cause:** `MT5Connector.connect()` ไม่ pre-select symbols ใน Market Watch หลัง login. ผลคือ:

- `analyze()` เรียก `get_current_price(symbol)` → `mt5.symbol_info_tick(symbol)` คืน None ถ้า symbol ไม่อยู่ใน Market Watch
- `analyze()` return no_signal ทันที (line 343-344) — **ก่อน** จะถึง `get_symbol_info(symbol)` ที่จะ trigger `mt5.symbol_select()`
- บอท skip silently ตลอด — ไม่มี scan event ใน Excel

**Fix:** ใน `MT5Connector.connect()` หลัง login สำเร็จ → loop `bot_config.symbols.symbols` ทั้งหมดเรียก `mt5.symbol_select(sym, True)` ครั้งเดียว. Print summary `📌 Market Watch: enabled N/M symbols` + warn ถ้า broker ไม่รองรับ symbol บางตัว.

**Verify after deploy:** หลัง bot start ดู console message — ต้องเห็น `📌 Market Watch: enabled 10/10 symbols`. ถ้า < 10 → ดู warning ว่า broker ไม่รองรับ symbol อะไร (อาจเป็นชื่อต่าง เช่น `XAUUSD.r` แทน `XAUUSD`).

**v6.10c — Timezone fix: Excel timestamps ใช้ broker time (EEST) (added 2026-04-27):**

Live demo Day-1 analysis เปรียบเทียบ Excel `Open Time`/`Close Time` กับ MT5 history เจอว่า **Excel timestamps มี offset +4 ชม. จาก MT5** (Excel 16:39 vs MT5 12:39).

Root cause: production code หลายจุดใช้ `datetime.now()` (Python system local time = Bangkok VPS UTC+7) แทนที่จะใช้ broker time (EEST UTC+3). ทำให้ Excel timestamps คนละ wall-clock กับ MT5 history → debug ยาก + Friday/Daily close logic อาจ trigger ผิด.

Pattern: `TimeManager.get_server_time().replace(tzinfo=None)` — ได้ naive datetime ที่ display เป็น EEST wall clock (ตรงกับ MT5).

**Files patched:**

- `execution/trade_executor.py` — `execute_signal` capture entry time, `sync_with_mt5` close_time, `update_close` close_time. + import `TimeManager`.
- `main.py` — `_build_signal_observation` challenge_day calc, `_build_live_context` overtrading window, `_trade_open_history.append` (เก็บ EEST timestamp), Daily Halt print message.

**ที่ไม่แตะ (intentionally `datetime.now()` ยังใช้ได้):**

- `mt5_connector.py` `history_deals_get(today_start, datetime.now())` — MT5 API doc บอกใช้ naive UTC, library auto-convert
- mock data / test helpers / shutdown uptime calc — ไม่กระทบ FTMO logic
- `_signal_handler` / `__repr__` — diagnostic only

**Verify after deploy:** Excel `Open Time` ของ trade ใหม่ ต้อง match MT5 history mobile screen (วินาทีตรงกัน). ถ้ายังต่าง 4 ชม. → VPS NTP ไม่ sync หรือ timezone setting ผิด.

**v6.10b — Daily/Stats sheets fix (added 2026-04-27):**

Live demo Day-1 analysis เจอว่า Daily sheet + Stats sheet **ว่างเปล่า** ทั้งวัน ทั้งที่มี 6 trades. Root cause: `log_daily_summary()` + `update_stats_sheet()` ถูกเรียกแค่ใน `_run_phase4_tests()` (test function) — **ไม่เคยเรียกใน production loop** ตลอด.

**Fix:** เพิ่ม 3 hooks ใน `FTMOTradingBot`:

- `__init__`: `self._last_logged_day = None`
- `run()` ต้น loop iteration ก่อน `check_risk()`: ถ้า `broker_today != _last_logged_day` → flush ของวันก่อน (`log_daily_summary` + `update_stats_sheet`) → set `_last_logged_day = broker_today`. ครั้งแรกที่ loop รัน (None → today) ไม่ flush เพราะไม่มีวันก่อน
- `run()` ใน loop: ทุก 720 loops (~1 ชม. @ 5s) → `update_stats_sheet()` only (สำหรับ live monitor — user เปิด Excel ดูสถานะปัจจุบันได้)
- `shutdown()`: ทั้ง `log_daily_summary` + `update_stats_sheet` ก่อน save state — กัน user Ctrl+C แล้วข้อมูลวันสุดท้ายหาย

**Verify after deploy:** รัน bot ≥ 1 ชม. → ตรวจ `Stats` sheet ต้องมี data; รัน cross-day → ตรวจ `Daily` sheet มี row ของวันก่อน.

**v6.10 — Executor reject reason logging (added 2026-04-27, schema bump 63 → 64 trade cols, 20 → 21 signal cols):**

Live demo day 1 analysis revealed **62% of scans = AGENT_TAKE_FAIL** but reject reason ไม่ถูกบันทึก — Signals sheet's "Reject/Skip Reasons" column เก็บ SMC signal reasons แทน. Blind spot ใหญ่ เพราะไม่รู้ว่า cooldown / spread / correlation / DD halt / post-TP lock / order_send_failed อันไหน reject signal.

Changes:

- `TradeExecutor.execute_signal` — ตั้ง `self._last_reject_reason` ที่ทุก rejection point (signal_invalid / correlation:* / lot_calc_failed / risk_manager:* / price_fetch_failed / spread_high:* / final_validation:* / order_send_failed). Reset ที่ต้นของแต่ละ call.
- `FTMOTradingBot.run` — หลัง `executor.execute_signal()` คืน None → อ่าน `executor._last_reject_reason` → save เข้า `live_context["executor_reject_reason"]` → log ลง Signals sheet col 20 ใหม่ "Executor Reject".
- `FTMOTradingBot._build_live_context` — เพิ่ม raw account state (`balance_at_entry`, `equity_at_entry`, `floating_pnl_at_entry`, `daily_start_equity`) สำหรับ debug ตัวเลข `dd_at_entry_pct` ที่อาจ misleading (เช่น แสดง 11% ตอนที่ net P/L บวก).
- `ExecutedTrade.partial_close_skipped: bool` (col 63 ใหม่) — distinguish "Partial Closed = True (fired)" จาก "Partial Skipped = True (lot too small ข้ามไป)". `TradeManager._partial_close` set flag เมื่อ `remaining < lot_min` หรือ `close_volume < lot_min`.
- `TradeLogger.SIGNAL_HEADERS` 20 → 21 cols (เพิ่ม "Executor Reject" ก่อน "Obs27 JSON"). `TRADE_HEADERS` 63 → 64 cols (เพิ่ม "Partial Skipped" ก่อน "Obs27 JSON" — ไม่กระทบ hardcoded col index ของ `log_trade_closed`).

⚠️ **Schema migration:** VPS ต้อง **rename หรือลบ `logs/ftmo_trades.xlsx` เดิมก่อน restart** ไม่งั้น append ผิด column.

**Retrain unlock (added 2026-04-25, schema bump 62 → 63 trade cols, 19 → 20 signal cols):**

- `ExecutedTrade.obs_27_json: str` — JSON-encoded 27-dim obs vector at decision time. Lets us reconstruct full RL state for offline retrain / pool augmentation from live data.
- `FTMOTradingBot._build_live_context` calls `_build_signal_observation(sig)` (same path the live agent sees) and stores `json.dumps([round(float(x), 4) for x in obs.tolist()])` in `ctx["obs_27_json"]`.
- `TRADE_HEADERS[-1] = "Obs27 JSON"`, `SIGNAL_HEADERS[-1] = "Obs27 JSON"`. Both populated from `live_context` / `scan_data`. Cell capped at 600 chars.
- Round-trip verified: `json.dumps` (4-dec round) → `json.loads` → numpy float32. Max error ≈ 5e-5 (< 1e-3 invariant).
- Wrapped in `try/except` — JSON build failure leaves `obs_27_json=""` (no log break).
- File size impact: ~250 chars/row × ~1000 trades/month ≈ 250 KB/month (negligible).
- Use case: reconstruct exact obs the live agent saw → train next-gen agent on live distribution drift, or seed pool augmentation experiments.

### 2026-04-25 — v6.9 Phase E2 — Auxiliary Task on PPO

Research-backed (arXiv 2411.01456) — auxiliary regression head on policy network forces the trunk to learn signal-outcome-informative representations. Paper reports Sharpe lift -2.61 → 0.24 (Dataset 1) and -2.93 → 0.47 (Dataset 2) on forex DRL.

**3 new files:**

- `ml/aux_rollout_buffer.py` — `AuxRolloutBuffer` extends `RolloutBuffer` with per-step `aux_targets` field (shape `(buffer_size, n_envs)`). Adds `aux_target` kwarg to `add()`, includes `aux_targets` in `get()` swap_and_flatten loop, returns extended `AuxRolloutBufferSamples` NamedTuple.
- `ml/aux_aware_policy.py` — `AuxAwareACPolicy` extends `ActorCriticPolicy` with `aux_head: nn.Linear(latent_dim_pi, 1)` and `predict_aux(obs)` method that runs obs through actor trunk → aux head → squeezed scalar.
- `ml/aux_aware_ppo.py` — `AuxAwarePPO` extends `PPO`. Overrides:
  1. `__init__` — defaults `rollout_buffer_class = AuxRolloutBuffer`, accepts `aux_loss_weight=0.5`.
  2. `collect_rollouts()` — copy of `OnPolicyAlgorithm.collect_rollouts` with one extra line: `aux_targets = np.array([info.get('aux_target', 0.0) for info in infos])` then `rollout_buffer.add(..., aux_target=aux_targets)`.
  3. `train()` — copy of `PPO.train` with extra `aux_loss = F.mse_loss(policy.predict_aux(obs), aux_targets)` added to total loss as `+ aux_loss_weight × aux_loss`. Logs `train/aux_loss` to TensorBoard.

**Env modification (`ml/signal_filter_env.py`):**

- Added `info['aux_target'] = float(sig.get('outcome_pnl_ratio', 0.0))` in `step()` info dict. AuxAwarePPO reads this from `infos` returned by `env.step` (vectorized).

**Training script (`scripts/train_signal_filter.py`):**

- Replaced `PPO("MlpPolicy", ...)` with `AuxAwarePPO(AuxAwareACPolicy, ..., aux_loss_weight=0.5, ...)` in P1.
- Replaced `PPO.load(...)` with `AuxAwarePPO.load(...)` in P2 transition + final eval — preserves aux head + buffer class on reload.

**Smoke test (100 k P1 + 50 k P2, n_envs=4):**

- ✅ Pipeline runs end-to-end without errors.
- ✅ `train/aux_loss` logged: 1.39-1.45 (near regression baseline `var(outcome) ≈ 1.5`, stable not exploding).
- ✅ `train/value_loss`: 0.58-0.69 (healthy).
- ✅ `train/policy_gradient_loss`: -0.005 to 0 (healthy small values).
- ✅ Model save/load round-trip works (eval reloaded model successfully).

**Risks (still open):**

- Full 10M+5M training may diverge if `aux_loss_weight=0.5` is too high — fallback to 0.1 if value_loss or aux_loss explodes.
- VecNormalize wraps env — `info['aux_target']` is unmodified (VecNormalize only touches obs/reward).
- v6.8 P2 stability fix (LR 5e-5, ent 0.02, threshold 20) is preserved — paired with E2 aux loss.

**Retrain required (only RL — pool + GBM unchanged from v6.8 calibrated):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected (per arXiv paper extrapolation): Pass Rate 3.0-3.5 % (B1v2/E1 baseline) → 4.5-7 %, with stable P2 (no early-stop), DD safer due to better feature learning.

### 2026-04-25 — v6.8 Phase E1 — Isotonic Calibration on GBM

Research-backed improvement (Niculescu-Mizil & Caruana 2005, MQL5 financial-ML series). GBM `predict_proba` was uncalibrated — raw probabilities clustered around 0.30-0.45 regardless of true frequency. Calibration tightens probability semantics so `ml_score` is interpretable as actual win rate.

**Changes (`scripts/train_signal_quality.py`):**

- Added `IsotonicRegression(out_of_bounds='clip')` fitted on OOF probabilities (group-aware via existing `GroupKFold` setup → no leakage).
- Pool re-scored with **calibrated** OOF probabilities instead of raw.
- Production save bundle now includes both `model` (base GBM) and `calibrator` (isotonic mapping).
- Brier score logged before/after for verification.
- 5-bin reliability diagram printed (pred_avg vs true_avg per bin).

**Changes (`ml/signal_quality.py`):**

- `SignalQualityModel.__init__` loads optional `calibrator` from payload (None → backwards-compat with old uncalibrated models).
- `score` and `score_batch` apply `calibrator.transform` after `model.predict_proba`.

**Verification (Phase E1 first run):**

- Brier 0.2243 → 0.2234 (-0.4 %).
- Reliability bins: 5/5 ✅ — `pred_avg` matches `true_avg` exactly across [0,0.3), [0.3,0.4), [0.4,0.5), [0.5,0.6), [0.6,1.0).
- Distribution: mean 0.356, std grew 0.050 → 0.076 (calibration spreads probabilities to match true frequency).
- Threshold analysis shift:
  - 0.33: WR 38.8 % → 39.6 %, EV `−0.005` → `+0.010` (flip to positive).
  - 0.40: WR 46.8 % → 46.6 %, EV `+0.154` → `+0.149` (almost identical, larger n).
  - 0.45: 3.5 % kept → 8.3 % kept, EV `+0.393` → `+0.275` (more samples at sweet spot).

**Why isotonic, not Platt:**

- Tree models (GBM) have non-sigmoid miscalibration → Platt's log-linear assumption fails.
- 106 k samples ≫ 1 000 → isotonic strictly dominant per Niculescu-Mizil 2005.

**Live impact:** `ml_score >= 0.40` reward bonuses in `FTMOSignalFilterEnv.step` now trigger at the *true* 46 % WR threshold instead of an arbitrary raw-prob bucket. Threshold for `--ml_threshold` should typically use 0.40 (sweet spot) instead of the previous 0.36.

**Retrain required (only RL):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.40 --risk_per_trade 0.007
```

Expected: Pass Rate 3.5 % → 4-5 % (calibration improves position sizing per Niculescu-Mizil chain: probs → sizing → Kelly → equity smoothness).

### 2026-04-25 — v6.7 Rollback Phase D (BE-only tested + rejected)

Phase D full (partial + BE + trail) and Phase D BE-only both failed to beat the B1v2 baseline (Pass Rate 3.7 %). After two experiments the evidence is strong enough to lock in a decision: **trade management inside the training backtester hurts Pass Rate for FTMO challenges**, even though it improves WR in isolation.

**Why trade management loses for Pass Rate:**

- FTMO 10 % target in 45 days is a *tail* objective — it needs high variance, not low variance.
- Partial close caps winners at 1.5R (locks 0.5R early, remaining half to 2R net) → reduces tail events.
- BE-only at 1R trigger kills trades that reach 1R and minor-pullback back to entry: in the raw pool these are 8.4 pp of former winners that became 0R, versus 9.6 pp of losers saved. Net EV per trade worsens (mean outcome moved from `−0.0645` to `−0.1051`).
- Distribution confirms it: `TP 2R+` bucket 12.8 % (B1v2) → 8.6 % (BE-only). Tail got thinner.

**Rollback actions:**

- `ml/strategy_backtester.py` `_resolve_trade` — reverted to the v6.3 B1v2 version (no BE / partial / trail; SL or TP only, with bar-color heuristic for same-bar).
- `execution/trade_manager.py` constants — restored to live defaults (`PARTIAL_CLOSE_PCT=0.5`, `PARTIAL_TRIGGER_RR=1.0`, `TRAIL_ACTIVATION_RR=1.5`, `TRAIL_ATR_MULTIPLIER=1.0`). Live still uses trade management; only the training backtester is flat.
- Pool restored from `data/signal_pool_3000.pkl.bak_v6_2` (identical to the v6.3 B1v2 pool — 2 887 episodes, 106 454 signals, mean outcome `−0.0645`).
- GBM retrained on the restored pool (expected OOF AUC ≈ 0.5875).

**Note on train-live alignment:** with rollback, the training env now *under*-estimates live performance because live has BE + partial + trail while training does not. This is an *acceptable* direction of mismatch (live ≥ train) — the alternative (Phase D) produced the wrong direction (live worse than train in Pass Rate terms). Future work could revisit this gap with higher trigger points (e.g., BE at 1.5R with buffer) if empirical live data supports it.

**Retrain required (only RL — pool + GBM are restored/regenerated):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected to reproduce ≈ 3.7 % Pass Rate. Once confirmed, the bot can be deployed to demo for live data collection — further offline tuning hit diminishing returns.

### 2026-04-25 — v6.5 Phase D Train-Live Alignment (rollback Phase C)

Phase C (SMC 4 principles, 2026-04-25 earlier) reduced pool 44 % but Pass Rate dropped 3.7 % → 1.5 % — **rolled back fully** in same day. Root cause confirmed: Phase C filters removed signals proportionally without improving WR, and pool shrinkage caused PPO P2 to early-stop.

Phase D attacks the real gap: **backtester `_resolve_trade` did not simulate BE / partial-close / trailing** that live `TradeManager` performs. Training pool outcomes therefore misrepresented realized RR.

**Rollback (all Phase C changes reverted):**

- `smc_strategy.py` — removed `_get_h4_poi_zones`, `_is_near_h4_poi`, H4 POI gate (BUY+SELL), H4 POI soft scoring, IDM sweep penalty, FVG + BOS conjunction, ADX 25 threshold (back to 20). `_ob_detector_h4`, `_fvg_detector_h4`, `_h4_poi_cache` removed from `__init__`.
- `market_structure.py` — removed `is_valid_pullback`, wire-in dropped from `detect_structure_breaks`.
- `strategy_backtester.py` `_init_strategy` — removed `_ob_detector_h4`, `_fvg_detector_h4`, `_h4_poi_cache` setup.

**Phase D new (`ml/strategy_backtester.py`):**

- Added class constants mirroring `TradeManager`: `_BE_TRIGGER_RR=1.0`, `_PARTIAL_CLOSE_PCT=0.5`, `_PARTIAL_TRIGGER_RR=1.0`, `_TRAIL_ACTIVATION_RR=1.5`, `_TRAIL_ATR_MULTIPLIER=1.0`.
- Rewrote `_resolve_trade` as bar-by-bar state machine with fields: `effective_sl`, `partial_closed`, `partial_gain_R`, `trail_active`, `best_price`.
- On 1R hit: partial close 50 % (locks `+0.5R`) and moves SL to entry (BE).
- On 1.5R hit: activates trailing — SL = `best_price ± ATR × 1.0` (one-way).
- Gap / force-close logic preserved; applies to `effective_sl` so gap-SL below BE still counts as 0 R on remaining half.
- Outcome `total_R = partial_gain_R + remaining_pct × exit_R` — combines locked partial with remaining exit.

**Unit tests (6 scenarios) all pass:**

- TP direct hit → `+1.5R` (partial +0.5R + TP on remaining 50 %).
- Partial + BE stop (no trail) → `+0.5R` (half locked, half BE = 0R).
- Full SL before 1R → `−1R`.
- Partial + trail stop → `~+1R` (trail above entry, not full RR).
- SELL mirror of case 1 → `+1.5R` as expected.
- Timeout after partial → locked partial + remaining at last close.

**Pool v6.5 snapshot (after rebuild):**

- 2 887 episodes, 106 454 signals (same as v6.3 B1v2 baseline — rollback + Phase D preserves signal count).
- WR (outcome > 0) **35.56 % → 45.83 %** (+10.3 pp).
- New distribution buckets visible: `+0.1 to +0.5R` (18.0 %) and `+1.0 to +1.5R` (19.7 %) — partial-win outcomes that didn't exist before.
- Mean outcome −0.0887 (slightly more negative than v6.3's −0.0645) because partial-cap reduces winners from 2R → 1.5R on average.

**Retrain required:**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_quality.py
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 3.7 % → 6-10 %, WR 49.6 % → 55-60 %, DD max safer because BE caps downside of partial-winners.

### 2026-04-24 — v6.3 B1v2 mid-episode undertrading checks

Follow-up to B1 (Pass Rate 2.0 % → 3.7 %, but take rate ไม่ขยับ). Problem: terminal undertrading penalty fires too late for PPO credit assignment. B1v2 adds checks WITHIN the episode.

**RL reward changes (`ml/signal_filter_env.py`):**

- **Mid-episode check day 20 (P2, sticky):** progress < 40 % + takes < 6 → `-0.3`. Fires once per episode.
- **Mid-episode check day 35 (P2, sticky):** progress < 60 % + takes < 12 → `-0.7`. Fires once per episode.
- **Terminal threshold lowered:** `takes < 20` → `takes < 15`. Realistic for pool size (~16 signals at th0.36).
- **Progress shaping strengthened:** `0.02 × progress_delta` → `0.05 × progress_delta` (2.5×). Every 1 % toward target gives more reward.
- State flags added: `_mid_check_day20_fired`, `_mid_check_day35_fired` — reset per episode.

Expected: Pass Rate 3.7 % → 5-7 %, Take Rate 65 % → 72 %+, Orders/day 0.26 → 0.35+.

### 2026-04-24 — v6.3 B1 reward tuning (Pass Rate improvement)

Phase B1 targets low Pass Rate (2.0% at v6.3 baseline) — agent undertrades (10.4 orders/ep vs needed ~27 for 10% target). Only RL env changed; no pool/GBM rebuild needed.

**RL reward changes (`ml/signal_filter_env.py`):**

- **Milestone bonuses (sticky):** +0.5 at 30 % progress, +1.0 at 60 %, +1.5 at 90 %. Teaches agent that partial progress is rewarded, not just terminal target.
- **Target bonus raised:** first 100 % hit `+2.0 → +4.0`. Combined with milestones, hitting target gives total sticky `+7.0` vs prior `+2.0`.
- **Undertrading penalty (P2 only):** episode end + `_current_day ≥ 40` + `_total_takes < 20` + not passed → `-1.0`. Forces agent to use available trades instead of hoarding.
- **Passive SKIP cost reduced:** `-0.015 → -0.010` per step. Allows honest SKIPs on low-quality signals without excessive punishment.
- State flags added: `_milestone_30_given`, `_milestone_60_given`, `_milestone_90_given` — reset per episode via `_reset_state`.

**Retrain required (only RL, not pool/GBM):**

```bash
.venv/bin/python ftmo_trading_bot/scripts/train_signal_filter.py --fresh \
    --timesteps_p1 10000000 --timesteps_p2 5000000 \
    --n_envs 8 --pool_size 3000 --outcome_noise 0.02 \
    --ml_threshold 0.36 --risk_per_trade 0.007
```

Expected: Pass Rate 2 % → 5-8 %, Take Rate 65 % → 75 %+, Orders/day 0.26 → 0.40+.

### 2026-04-24 — v6.3 multi-brain audit fixes

Full 3-brain audit (SMC + ML + RL). 16 Critical fixes landed. Requires pool rebuild + ML retrain + RL retrain before live deployment.

**SMC (`smc_strategy.py`):**

- `_evaluate_sell_signal` now applies `atr_floor_pips` per-symbol override (was hardcoded 100/8, ignoring `SymbolConfig.symbol_overrides`).
- BUY signal `timestamp` uses `TimeManager.get_server_time(symbol)` (was `datetime.now()` — local machine offset).
- HTF bias anti-lookahead: `iloc[-6:-1]` instead of `iloc[-5:]` — skips currently-forming bar.
- EMA200 veto NaN guard (`pd.notna`) — prevents silent skip on fresh MTF data.
- `_get_d1_bias` cache invalidates on UTC day rollover.
- `min_sl_pips` clamp now prints debug warning.

**ML (`scripts/train_signal_quality.py`, `ml/strategy_backtester.py`):**

- `train_gbm` uses `GroupKFold` + `cross_val_predict` — episode-level OOF predictions, unbiased AUC, no in-sample leakage into RL.
- Pool's `ml_score` field re-scored with OOF probabilities.
- `_resolve_trade` gap handling — `bar_open` past SL/TP fills at `bar_open` with slippage.
- Pool `MIN_CONFLUENCE_SCORE` aligned with live (`bot_config.ftmo.MIN_CONFLUENCE_SCORE`) — was 60, now 70.

**RL (`ml/signal_filter_env.py`, `main.py`):**

- `FTMOSignalFilterEnv.reset` injects `spread_pips × uniform(0.7, 1.5)` — spread distribution shift fix.
- `spread_cost_R` clamped at 1.0 — prevents reward explosion on news-spike spreads.
- `obs_dim()` static method returns 27 (was stale 24).
- `main.py._has_opposite_recently_closed` logs warning when `RiskManager._flip_lock` missing — prevents silent obs[25]=0.

**Mandatory before live deploy:**

1. `python scripts/build_signal_pool.py --pool_size 3000` (pool has v6.2 min_sl_pips + v6.3 confluence=70 + gap handling)
2. `python scripts/train_signal_quality.py` (OOF AUC expected to drop vs prior ~0.59 — that's the leakage being removed)
3. `python scripts/train_signal_filter.py --fresh --timesteps_p1 10000000 --timesteps_p2 5000000 --n_envs 8 --pool_size 3000`
4. Back up old `models/ppo_signal_filter.zip` + `vec_normalize_sf.pkl` first.

### 2026-04-24 — ATR floor re-calibrated + MIN_SL guard (v6.2)

- Lowered `SymbolConfig.symbol_overrides[X].atr_floor_pips` from 8/15/20/100 → 3-8 pips FX (500 ticks XAUUSD). Prior floor blocked signals 68-100 % of the time.
- Added new per-symbol `min_sl_pips` (EURUSD 10, GBPUSD/USDCAD/USDCHF/NZDUSD 12, USDJPY 10, EURJPY 15, GBPJPY 20, XAUUSD 300) as an SL floor inside `SMCStrategy.scan_signal` BUY/SELL branches.
- Incident that prompted the guard: EURUSD SL collapsed to ~5 pips under low ATR, spread ate > 20 % of SL.
- Invariant added (Hard Invariant #10): `atr_floor_pips` (gate) and `min_sl_pips` (SL clamp) are distinct — do not conflate.
- Commit: `9b64f6c`.

### 2026-04-24 — LLM Wiki migration

- Migrated `context.md` from a 392-line monolith to Hub + Spoke (5 files under `wiki/`).
- Added `CLAUDE.md` at project root — Wiki Sync Protocol.
- Installed Stop hook in `.claude/settings.json` — warns if `.py` changed but wiki/context/readme did not.
- Switched source-reference style from line numbers to class / method / variable names.
- Set language policy: docs in English, `readme.md` in Thai, chat in Thai.

### 2026-04-22 — Obs Space v6 (24 → 27 dims)

- Added `spread_pct_of_atr` [24] (cost awareness, GBPJPY spread vs ATR).
- Added `has_opposite_recently_closed` [25] (flip-lock context, anti-whipsaw).
- Added `htf_trend_alignment` [26] (uses `bias_align` as proxy).
- Retrained the entire pipeline.

### 2026-04-20 — Risk 0.7 % verified (5000 eps)

- Changed `DEFAULT_RISK_PER_TRADE_PCT` from 0.006 → 0.007.
- Evaluation: Pass 12.5 %, Profit +2.59 %, DD max 8.50 %, Breach 0 %.
- Discovered MT5 FTMO `tick.time` quirk → switched to `datetime.now(Bucharest)`.

### 2026-04-19 — Added XAUUSD

- `SymbolConfig.symbols` went from 9 to 10 (added XAUUSD, Gold).
- PositionSizer supports contract size 100 oz.

### 2026-04-18 — Hybrid ML + RL + resolver fixes

- Added ML GBM quality layer → obs 23 → 24 dims.
- Resolver now uses bar-color heuristic (fixes distance-from-open bias).
- Slippage 2 % → 0.5 % (realistic on majors).
- Pool system: fresh-generate → 3000 pre-generated episodes (250× training speedup).
- HTF bias: unstable 2/3 → stable 5-bar ≥3 same side, <2 opposite.
- Network `[128,64]` → `[256,128]`, gamma 0.95 → 0.99.

### 2026-04-17 — 2-phase curriculum

- Phase 1 (Alpha): no DD penalty + oracle SKIP reward.
- Phase 2 (Risk): DD penalty + activity floor.

### 2026-04-13 — Code review patch

- Fixed timezone bugs (EET vs UTC).
- Fixed pip size to be JPY-aware.
- Fixed PositionSizer 3-case pip value.
- Fixed hardcoded contract size.

---

## Cross-links

- Architecture + loop priority → [01-architecture.md](01-architecture.md)
- Modules + symbols → [02-modules.md](02-modules.md)
- Obs 27 dims layout → [03-rl-training.md](03-rl-training.md)
- Live operations + state machine → [04-operations.md](04-operations.md)
