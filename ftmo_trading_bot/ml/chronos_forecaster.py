"""
===============================================================================
FTMO Trading Bot — Chronos 2 Zero-shot Forecaster (v7.0)
===============================================================================
Wrapper รอบ Amazon Chronos foundation model (Hugging Face) สำหรับ M15 forecast
ใช้เป็น obs feature แทนการเทรน fine-tune (zero-shot only).

Output features ที่ป้อนเข้า PPO obs:
  obs[27] chronos_alignment       — direction × sign(median_h+8 − close_now), clip ±1
  obs[28] chronos_uncertainty_norm — (q90 − q10) / atr_value, clip [0, 3]

⚠️ CRITICAL:
  - Model name ต้องเหมือนกันทั้ง training (StrategyBacktester) และ live (main.py)
    ใช้ `bot_config.ml.CHRONOS_MODEL_NAME` เป็น single source of truth.
  - Cache key: (symbol, last_bar_timestamp) — ห้าม cache ตาม object identity
    ของ DataFrame เพราะ DataFrame ใหม่ทุก scan
  - หาก import / load model ล้มเหลว → return neutral features (0, 0) แทนการ raise
    เพื่อ keep bot operational (degrades to v6.14 behavior with 2 zero dims)
  - torch.manual_seed(0) ก่อน inference batch — pool reproducibility
===============================================================================
"""

from __future__ import annotations

import os
import time
import logging
import warnings
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Suppress noisy transformers deprecation warning ที่ chronos lib ยังใช้ API เก่า:
#   "Passing a tuple of `past_key_values` is deprecated and will be removed
#    in Transformers v4.48.0..."
# warning นี้ emit ผ่าน HF transformers logger (ไม่ใช่ Python warnings module) →
# ต้อง suppress 2 ทาง: (a) Python warnings filter (สำหรับ DeprecationWarning ทั่วไป)
# (b) logging filter ของ transformers logger (สำหรับ warning_once spam)
warnings.filterwarnings("ignore", message=r".*past_key_values.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*past_key_values.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*past_key_values.*", category=UserWarning)


class _PastKeyValuesFilter(logging.Filter):
    """Filter HF transformers warning ที่ใช้ logging.warn(...) ตรง ๆ"""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "past_key_values" not in msg


# Apply once on import — กัน spam ตลอด pool gen + live
for _logger_name in ("transformers", "transformers.modeling_utils", "transformers.cache_utils"):
    logging.getLogger(_logger_name).addFilter(_PastKeyValuesFilter())

DEFAULT_MODEL_NAME = "amazon/chronos-bolt-small"
DEFAULT_PREDICTION_LENGTH = 8
DEFAULT_CONTEXT_LENGTH = 512
CACHE_MAX_ENTRIES = 64


class ChronosForecaster:
    """Zero-shot M15 forecaster using Amazon Chronos."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        prediction_length: int = DEFAULT_PREDICTION_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        verbose: int = 0,
    ):
        self.model_name = model_name
        self.device = device
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.verbose = int(verbose)

        self._pipeline: Optional[Any] = None
        self._available: bool = False
        self._cache: Dict[Tuple[str, Any], Dict[str, float]] = {}

        if os.environ.get("BOT_DISABLE_CHRONOS", "").strip() == "1":
            if self.verbose:
                print("⚠️ [Chronos] BOT_DISABLE_CHRONOS=1 → forecaster disabled (returns neutral)")
            return

        self._load_pipeline()

    def _load_pipeline(self):
        try:
            import torch
            try:
                torch.manual_seed(0)
            except Exception:
                pass

            # Suppress noisy "past_key_values is deprecated" via HF native API
            # (warning_once spam ที่ Python warnings filter ไม่จับ)
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except Exception:
                pass

            try:
                from chronos import BaseChronosPipeline
            except ImportError:
                from chronos import ChronosPipeline as BaseChronosPipeline

            t0 = time.time()
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            self._available = True
            if self.verbose:
                print(f"✅ [Chronos] โหลดโมเดล {self.model_name} สำเร็จ ({time.time()-t0:.1f}s)")
        except Exception as exc:
            self._pipeline = None
            self._available = False
            if self.verbose:
                print(f"⚠️ [Chronos] โหลดโมเดลไม่สำเร็จ ({exc}) → จะใช้ค่า neutral (0, 0)")

    @property
    def is_available(self) -> bool:
        return self._available

    def _cache_evict_if_full(self):
        if len(self._cache) > CACHE_MAX_ENTRIES:
            for k in list(self._cache.keys())[: len(self._cache) - CACHE_MAX_ENTRIES]:
                self._cache.pop(k, None)

    def forecast(self, symbol: str, df_m15: pd.DataFrame) -> Dict[str, float]:
        """
        Forecast ราคา close ใน prediction_length bars ข้างหน้า.

        Returns dict:
          - median_h8: float — median ราคา ณ bar +prediction_length
          - q10_h8:    float — 10th percentile
          - q90_h8:    float — 90th percentile
          - last_close: float — ราคา close ของ bar สุดท้ายใน input (สำหรับ feature compute)
          - cache_hit: 0/1
        """
        neutral = {
            "median_h8": float("nan"),
            "q10_h8": float("nan"),
            "q90_h8": float("nan"),
            "last_close": float("nan"),
            "cache_hit": 0,
        }

        if not self._available or df_m15 is None or len(df_m15) < 32:
            return neutral
        if "close" not in df_m15.columns:
            return neutral

        try:
            last_close = float(df_m15["close"].iloc[-1])
            ts_anchor = df_m15["time"].iloc[-1] if "time" in df_m15.columns else len(df_m15)
            cache_key = (symbol, ts_anchor)
            cached = self._cache.get(cache_key)
            if cached is not None:
                out = dict(cached)
                out["cache_hit"] = 1
                return out

            ctx = df_m15["close"].tail(self.context_length).to_numpy(dtype=np.float32)

            import torch
            torch.manual_seed(0)
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32)

            # Wrap inference ใน catch_warnings เพื่อ suppress noisy "past_key_values"
            # deprecation warning จาก transformers (chronos lib ยังใช้ legacy API).
            # warning ปลอดภัย — pin transformers==4.48.3 รองรับทั้ง legacy + new
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quantiles, _mean = self._pipeline.predict_quantiles(
                    context=ctx_tensor,
                    prediction_length=self.prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
            q = quantiles[0].cpu().numpy()
            q10_h = float(q[-1, 0])
            median_h = float(q[-1, 1])
            q90_h = float(q[-1, 2])

            out = {
                "median_h8": median_h,
                "q10_h8": q10_h,
                "q90_h8": q90_h,
                "last_close": last_close,
                "cache_hit": 0,
            }
            self._cache[cache_key] = {k: v for k, v in out.items() if k != "cache_hit"}
            self._cache_evict_if_full()
            return out
        except Exception as exc:
            if self.verbose:
                print(f"⚠️ [Chronos] forecast failed for {symbol}: {exc}")
            return neutral

    @staticmethod
    def compute_features(
        forecast_dict: Dict[str, float],
        signal_direction: float,
        atr_value: float,
    ) -> Tuple[float, float]:
        """
        แปลง raw forecast → 2 obs features

        v7.0.2 (2026-05-01): formula refactor หลังเจอ regression v7.0:
          A. flip sign ของ alignment — pool diagnostic v7.0 พบ corr(align, outcome) = -0.0178
             (anti-signal!). SMC ค้าขาย swing/reversal สวนเทรนด์ → Chronos forecast เทรนด์ →
             "Chronos ทำนายลง + SMC BUY" คือ reversal setup ที่ถูก ไม่ใช่ผิด.
             สูตรใหม่: forecast_dir = sign(last_close - median) → flip vs เดิม.
          B. Brownian-scale uncertainty — เดิม (q90-q10)/atr saturated ที่ 3.0 ใน 96.2% ของ
             signals (variance ของ 8-bar forecast ขยายตาม √8). หารเพิ่มด้วย √prediction_length
             → ไม่ saturate, distribution กระจาย.

        Args:
            forecast_dict: ผลจาก forecast() (อาจเป็น neutral dict ถ้า fail)
            signal_direction: +1 (BUY) / -1 (SELL)
            atr_value: ATR ราคา (ไม่ใช่ pips)

        Returns:
            (chronos_alignment, chronos_uncertainty_norm)
            ทั้งคู่ clip ตาม obs spec แล้ว
            alignment +1 = SMC + Chronos contrarian (good reversal setup)
            alignment -1 = SMC + Chronos agree on trend (SMC late, weak setup)
        """
        try:
            median_h = float(forecast_dict.get("median_h8", float("nan")))
            q10 = float(forecast_dict.get("q10_h8", float("nan")))
            q90 = float(forecast_dict.get("q90_h8", float("nan")))
            last_close = float(forecast_dict.get("last_close", float("nan")))

            if not np.isfinite(median_h) or not np.isfinite(last_close):
                return 0.0, 0.0

            # v7.0.2 Change A: flip sign — last_close > median = Chronos predicts down
            # → +1 ถ้า SMC BUY ขณะ Chronos ทำนายลง = reversal setup
            delta = last_close - median_h
            forecast_dir = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
            alignment = float(np.clip(signal_direction * forecast_dir, -1.0, 1.0))

            atr = float(atr_value) if atr_value and atr_value > 0 else 0.0
            if atr <= 0 or not np.isfinite(q90) or not np.isfinite(q10):
                uncertainty = 0.0
            else:
                # v7.1 Change: log1p-scale (smooth gradient even when band > ATR×√8)
                # ปัญหา v7.0.2: linear ratio (q90-q10)/(atr×√8) saturate ที่ 3.0 ในทุก signal
                # ของ live data 2026-05-04 (5/5 trades = 3.0 = no information).
                # log1p gives gradient ที่ทำงานทั้ง low + high uncertainty regimes:
                #   ratio=0  → log1p=0  → 0.0
                #   ratio=1  → log1p=0.69 → 0.35
                #   ratio=3  → log1p=1.39 → 0.69
                #   ratio=10 → log1p=2.40 → 1.20
                #   ratio=50 → log1p=3.93 → clip 3.0
                # → คนละ feature value ระหว่าง ratio 3 vs 10 vs 50 (เก่า: ทั้งหมด = 3)
                horizon_factor = 2.8284271247461903   # √8 (DEFAULT_PREDICTION_LENGTH)
                raw_ratio = (q90 - q10) / (atr * horizon_factor)
                uncertainty = float(np.clip(np.log1p(max(0.0, raw_ratio)) / 2.0, 0.0, 3.0))

            return alignment, uncertainty
        except Exception:
            return 0.0, 0.0

    def forecast_features(
        self,
        symbol: str,
        df_m15: pd.DataFrame,
        signal_direction: float,
        atr_value: float,
    ) -> Tuple[float, float]:
        """Convenience: forecast() + compute_features() in one call."""
        fc = self.forecast(symbol, df_m15)
        return self.compute_features(fc, signal_direction, atr_value)
