"""
===============================================================================
FTMO Trading Bot — Signal Quality Model (ML filter)
===============================================================================
GBM classifier ที่ทำนาย probability ว่า signal จะ win
Trained บน pool data (outcomes pre-computed) — AUC ~0.58-0.60

Usage:
    model = SignalQualityModel("data/signal_quality_model.pkl")
    score = model.score(signal_dict)        # 0.0 - 1.0
    scores = model.score_batch(sig_list)    # ndarray

Architecture: Hybrid ML+RL
    Signal → SMC features → ML.score (probability of win)
                              ↓
                          RL obs feature
                              ↓
                        RL Agent → TAKE/SKIP
===============================================================================
"""
import math
import os
import pickle
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd


# v7.1 — Legacy 17-feature list (สำหรับ load model เก่าก่อน v7.1 retrain)
_FEATURES_V70 = [
    'confluence_score', 'rr_ratio', 'atr_pips', 'ob_score',
    'market_bias', 'bias_alignment', 'sl_distance_atr',
    'rsi_value', 'trend_strength', 'macd_histogram', 'ob_size_atr',
    'adx', 'stoch_k', 'bb_pctb', 'atr_change_ratio', 'price_roc',
    'direction',
]


def compute_temporal_features(
    timestamp: datetime,
    ltf_df: Optional[pd.DataFrame] = None,
    atr_floor_pips: float = 8.0,
    pip_size: float = 0.0001,
) -> Dict[str, float]:
    """
    v7.1 — คำนวณ 7 temporal/regime features สำหรับ GBM.

    เรียกได้ทั้งจาก build_signal_pool (pool-time, แทน ts ของ signal) และ
    `FTMOTradingBot._build_live_context` (live-time).

    Args:
        timestamp: signal timestamp (datetime, ควรเป็น UTC ถ้า aware)
        ltf_df: M15 OHLCV ที่มี column 'atr' (ใช้คำนวณ atr_zscore + regime)
        atr_floor_pips: per-symbol ATR floor (สำหรับ regime classifier)
        pip_size: 0.0001 (FX 4-digit) / 0.01 (JPY/XAU)

    Returns:
        Dict ของ 7 feature keys (พร้อม inject เข้า signal dict)
    """
    # Normalize timestamp → UTC-aware datetime
    if timestamp is None:
        return {
            'hour_of_day_sin': 0.0, 'hour_of_day_cos': 1.0,
            'day_of_week': 0.0, 'minutes_since_session_start': 0.0,
            'is_post_weekend_first_hour': 0.0,
            'volatility_regime_score': 1.0, 'atr_zscore_30bars': 0.0,
        }

    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.utcnow()

    # ใช้ UTC สำหรับ session calc (consistent กับ _is_session_warmup ใน SMC)
    if timestamp.tzinfo is not None:
        ts_utc = timestamp.utctimetuple()
        hour, minute, dow = ts_utc.tm_hour, ts_utc.tm_min, ts_utc.tm_wday
    else:
        hour, minute, dow = timestamp.hour, timestamp.minute, timestamp.weekday()

    # 1-2. Cyclical hour encoding (24h period) — model เรียนได้ว่า 23:00 vs 00:00 ใกล้กัน
    angle = 2 * math.pi * (hour + minute / 60.0) / 24.0
    hour_sin = math.sin(angle)
    hour_cos = math.cos(angle)

    # 3. Day of week (0=Mon, 4=Fri)
    day_of_week = float(min(dow, 4))  # cap at Fri (weekend ไม่ควร trade)

    # 4. Minutes since session start (London 08:00 UTC = 0; NY 13:00 UTC = 0; else linear)
    if 8 <= hour < 16:  # London/NY active window
        if hour < 13:
            mins_since = (hour - 8) * 60 + minute  # London-relative
        else:
            mins_since = (hour - 13) * 60 + minute  # NY-relative
    else:
        mins_since = 0.0
    mins_since_norm = float(min(mins_since, 480))  # cap at 8 hours

    # 5. Post-weekend first hour binary (Mon 08:00-09:00 UTC = London open)
    is_post_weekend = 1.0 if (dow == 0 and 8 <= hour < 12) else 0.0

    # 6-7. ATR-derived regime features
    atr_zscore = 0.0
    vol_regime_score = 1.0  # default 'normal'
    if ltf_df is not None and 'atr' in ltf_df.columns and len(ltf_df) >= 30:
        try:
            recent = ltf_df['atr'].tail(30)
            mean = float(recent.mean())
            std = float(recent.std())
            if std > 1e-10:
                atr_zscore = (float(ltf_df['atr'].iloc[-1]) - mean) / std

            atr_value = float(ltf_df['atr'].iloc[-1])
            atr_pips = atr_value / pip_size if pip_size > 0 else 0.0
            # Mirror ของ TechnicalIndicators.classify_volatility_regime
            if atr_pips < atr_floor_pips * 1.2:
                vol_regime_score = 0.0  # quiet
            elif atr_zscore > 2.0:
                vol_regime_score = 3.0  # explosive
            elif atr_zscore > 1.0 or atr_pips > atr_floor_pips * 3.0:
                vol_regime_score = 2.0  # high
            else:
                vol_regime_score = 1.0  # normal
        except Exception:
            pass

    return {
        'hour_of_day_sin': float(hour_sin),
        'hour_of_day_cos': float(hour_cos),
        'day_of_week': day_of_week,
        'minutes_since_session_start': mins_since_norm,
        'is_post_weekend_first_hour': is_post_weekend,
        'volatility_regime_score': vol_regime_score,
        'atr_zscore_30bars': float(atr_zscore),
    }


class SignalQualityModel:
    """ML wrapper สำหรับให้ probability ของ signal → win"""

    # Features ที่ model คาดหวัง (match ตอน train) — v7.1 = 24 features
    # Legacy 17 features สำหรับ backwards-compat: load model เก่า → ใช้ keys ใน payload
    FEATURES = _FEATURES_V70 + [
        # === v7.1 temporal/regime features (7 ใหม่) ===
        # ทั้ง 7 ตัวแก้ไม่มี temporal awareness ของ GBM — ปัญหา 2026-05-04 ที่ Monday warmup
        # signals มี ML cal score 0.65+ แต่ขาดทุนทั้งหมด
        'hour_of_day_sin', 'hour_of_day_cos',
        'day_of_week', 'minutes_since_session_start',
        'is_post_weekend_first_hour',
        'volatility_regime_score', 'atr_zscore_30bars',
    ]

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[SignalQuality] Model not found: {model_path}\n"
                f"  Run: python scripts/train_signal_quality.py"
            )
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
        self.model = payload['model']
        # Phase E1: optional isotonic calibrator (None → backwards-compat raw probs)
        self.calibrator = payload.get('calibrator', None)
        # v7.1 — backwards compat: ถ้า payload ไม่มี 'keys' = pre-v7.1 model (17 features)
        # ถ้ามี 'keys' = ใช้ตามที่ train เก็บไว้
        self.keys = payload.get('keys', _FEATURES_V70)
        # v7.1 — drift monitor: training distribution snapshot (optional)
        self._train_dist: Optional[Dict[str, np.ndarray]] = payload.get('train_dist', None)
        self.model_path = model_path

        # Drift state — เก็บ recent live features เพื่อใช้กับ KS test
        self._recent_features: List[Dict[str, float]] = []
        self._drift_window: int = 100  # check ทุก 100 signals

    @staticmethod
    def _extract(src: Any, key: str) -> float:
        """อ่านค่า feature จาก dict หรือ TradeSignal (getattr)"""
        if isinstance(src, dict):
            return float(src.get(key, 0.0))
        return float(getattr(src, key, 0.0))

    def _calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibrator if available, else return raw."""
        if self.calibrator is None:
            return raw_probs
        return self.calibrator.transform(raw_probs)

    def score(self, src: Union[Dict, Any]) -> float:
        """ทำนาย P(win) calibrated ของ signal 1 ตัว → [0.0, 1.0]"""
        features = np.array(
            [[self._extract(src, k) for k in self.keys]],
            dtype=np.float64,
        )
        raw = self.model.predict_proba(features)[:, 1]
        return float(self._calibrate(raw)[0])

    def score_batch(self, sources: List[Union[Dict, Any]]) -> np.ndarray:
        """Score signals หลายตัวพร้อมกัน (calibrated, เร็วกว่า loop)"""
        if not sources:
            return np.array([], dtype=np.float64)
        features = np.array(
            [[self._extract(s, k) for k in self.keys] for s in sources],
            dtype=np.float64,
        )
        raw = self.model.predict_proba(features)[:, 1]
        return self._calibrate(raw)

    # =========================================================================
    # v7.1 — Drift Monitor (KS test live vs training distribution)
    # =========================================================================

    def record_live_signal(self, features: Dict[str, float]) -> None:
        """เก็บ feature dict ของ live signal เพื่อใช้กับ drift detect (rolling window)"""
        self._recent_features.append(dict(features))
        # Cap window — เก็บ recent 2× window size เพื่อให้มี data เผื่อ
        max_keep = self._drift_window * 2
        if len(self._recent_features) > max_keep:
            self._recent_features = self._recent_features[-max_keep:]

    def detect_drift(self, ks_threshold: float = 0.15) -> Dict[str, float]:
        """
        v7.1 — KS test ระหว่าง recent live features กับ training distribution.

        ต้องมี `train_dist` ใน payload (เก็บ array ของแต่ละ feature ตอน train).
        ถ้า payload เก่าไม่มี → return {} (no-op).

        Args:
            ks_threshold: threshold สำหรับ KS statistic (default 0.15 = noticeable shift)

        Returns:
            dict ของ {feature_name: ks_statistic} เฉพาะ feature ที่ shift > threshold
        """
        if self._train_dist is None:
            return {}
        if len(self._recent_features) < self._drift_window:
            return {}
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            return {}

        drifts: Dict[str, float] = {}
        recent = self._recent_features[-self._drift_window:]
        for key in self.keys:
            train_vals = self._train_dist.get(key)
            if train_vals is None or len(train_vals) < 50:
                continue
            live_vals = np.array(
                [r.get(key, 0.0) for r in recent], dtype=np.float64
            )
            try:
                statistic, _ = ks_2samp(train_vals, live_vals)
                if float(statistic) > ks_threshold:
                    drifts[key] = float(statistic)
            except Exception:
                continue
        return drifts

    def __repr__(self) -> str:
        return f"SignalQualityModel(path={self.model_path})"
