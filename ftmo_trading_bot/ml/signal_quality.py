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
import os
import pickle
from typing import List, Union, Dict, Any

import numpy as np


class SignalQualityModel:
    """ML wrapper สำหรับให้ probability ของ signal → win"""

    # Features ที่ model คาดหวัง (match ตอน train)
    FEATURES = [
        'confluence_score', 'rr_ratio', 'atr_pips', 'ob_score',
        'market_bias', 'bias_alignment', 'sl_distance_atr',
        'rsi_value', 'trend_strength', 'macd_histogram', 'ob_size_atr',
        'adx', 'stoch_k', 'bb_pctb', 'atr_change_ratio', 'price_roc',
        'direction',
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
        self.keys = payload.get('keys', self.FEATURES)
        self.model_path = model_path

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

    def __repr__(self) -> str:
        return f"SignalQualityModel(path={self.model_path})"
