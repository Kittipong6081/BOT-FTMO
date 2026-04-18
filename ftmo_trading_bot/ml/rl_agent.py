"""
===============================================================================
FTMO Trading Bot — RL Agent (Signal Filter Mode)
===============================================================================
PPO Agent ที่ตัดสินใจ TAKE/SKIP signal แต่ละตัวจาก SMC Strategy

Observation Space (24 dims) — ต้องตรงกับ FTMOSignalFilterEnv.observation_space
                                และ main._build_signal_observation:
  Signal core      [0-11]:   confluence, rr, direction, atr, ob_score, bias_align,
                             sl_atr, rsi, macd, trend_strength, ob_size_atr, adx
  Market regime    [12-15]:  stoch_k, bb_pctb, atr_change_ratio, price_roc
  ML quality       [16]:     ml_score_norm (GBM P(win), mapped [-1,+1]) ⭐
  Portfolio state  [17-23]:  total_dd, daily_dd, progress, day, trades_today,
                             recent_wr, consecutive_losses

Action Space (1 dim, continuous [-1, 1]):
  > 0 → TAKE signal
  ≤ 0 → SKIP signal

⚠️ CRITICAL: ตอน Inference ต้อง normalize obs ด้วย VecNormalize stats ที่บันทึกตอน train
===============================================================================
"""

import os
import pickle
import numpy as np
from stable_baselines3 import PPO

class SelfLearningAgent:
    """AI Agent สำหรับกรอง signal จาก SMC Strategy — ตัดสินใจ TAKE/SKIP"""

    OBS_DIM: int = 24

    def __init__(
        self,
        model_dir: str = "models",
        verbose: int = 0,
    ):
        self.model_dir = model_dir
        self.verbose = int(verbose)
        self.model_path = os.path.join(self.model_dir, "ppo_signal_filter.zip")
        self.vec_normalize_path = os.path.join(self.model_dir, "vec_normalize_sf.pkl")

        self.model = None
        self._obs_mean = None
        self._obs_var = None
        self._clip_obs = 10.0

    def initialize_model(self, strict: bool = True):
        """
        โหลดโมเดล Signal Filter + VecNormalize stats

        Args:
            strict: True = raise error ถ้าโหลดไม่ได้ (live mode)
                    False = return quietly (dev/test)
        """
        self._load_normalize_stats()

        if os.path.exists(self.model_path):
            try:
                if self.verbose:
                    print("🧠 [Signal Filter] กำลังโหลด PPO Model...")
                self.model = PPO.load(self.model_path)
                if self.verbose:
                    print(f"   ✅ โหลดสำเร็จ จาก {self.model_path}")
                    if self._obs_mean is not None:
                        print(f"   ✅ VecNormalize stats loaded (obs scale ตรงกับ training)")
                    else:
                        print(f"   ⚠️ ไม่พบ VecNormalize stats")
                return
            except Exception as e:
                if strict:
                    raise RuntimeError(
                        f"[Signal Filter] โหลดโมเดลล้มเหลว: {e} — "
                        f"ต้องเทรนก่อน (python scripts/train_signal_filter.py)"
                    )
                if self.verbose:
                    print(f"⚠️ [Signal Filter] โหลดล้มเหลว ({e})")
        else:
            if strict:
                raise RuntimeError(
                    f"[Signal Filter] ไม่พบโมเดลที่ {self.model_path} — "
                    f"ต้องเทรนก่อน (python scripts/train_signal_filter.py)"
                )
            if self.verbose:
                print(f"⚠️ [Signal Filter] ไม่พบ model — agent จะไม่ทำงาน")

    def _load_normalize_stats(self):
        if not os.path.exists(self.vec_normalize_path):
            if self.verbose:
                print(f"⚠️ [Signal Filter] ไม่พบ {self.vec_normalize_path}")
            return

        try:
            with open(self.vec_normalize_path, 'rb') as f:
                vec_normalize = pickle.load(f)

            if hasattr(vec_normalize, 'obs_rms') and vec_normalize.obs_rms is not None:
                self._obs_mean = vec_normalize.obs_rms.mean.copy()
                self._obs_var = vec_normalize.obs_rms.var.copy()
                self._clip_obs = getattr(vec_normalize, 'clip_obs', 10.0)
            else:
                if self.verbose:
                    print(f"⚠️ [Signal Filter] VecNormalize ไม่มี obs_rms")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ [Signal Filter] โหลด VecNormalize ล้มเหลว ({e})")
            self._obs_mean = None
            self._obs_var = None

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is None or self._obs_var is None:
            return obs
        normalized = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
        return np.clip(normalized, -self._clip_obs, self._clip_obs).astype(np.float32)

    def _prepare_obs(self, signal_obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.initialize_model(strict=True)
        if self.model is None:
            raise RuntimeError(
                "[Signal Filter] Model ไม่พร้อมใช้งาน — โหลดโมเดลไม่สำเร็จ"
            )

        obs = np.asarray(signal_obs, dtype=np.float32).flatten()
        if obs.size != self.OBS_DIM:
            raise ValueError(
                f"[Signal Filter] Observation มี {obs.size} dims แต่ต้องการ {self.OBS_DIM} "
                f"(สัญลักษณ์ว่าฝั่ง env/agent ไม่ตรงกัน — ตรวจ FTMOSignalFilterEnv.observation_space)"
            )
        return self._normalize_obs(obs)

    def should_take_signal(self, signal_obs: np.ndarray) -> bool:
        """
        ตัดสินใจว่าควร TAKE signal นี้หรือไม่

        Args:
            signal_obs: observation OBS_DIM dims (signal features + portfolio state)

        Returns:
            True = TAKE, False = SKIP
        """
        normalized_obs = self._prepare_obs(signal_obs)
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        return float(action[0]) > 0.0

    def get_action_confidence(self, signal_obs: np.ndarray) -> float:
        """
        คืนค่า action raw value สำหรับ logging/debugging
        ค่าบวกมาก = มั่นใจ TAKE, ค่าลบมาก = มั่นใจ SKIP
        """
        normalized_obs = self._prepare_obs(signal_obs)
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        return float(action[0])
