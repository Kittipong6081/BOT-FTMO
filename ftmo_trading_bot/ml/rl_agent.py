"""
===============================================================================
FTMO Trading Bot — RL Agent (สมองส่วนการเรียนรู้ด้วยตนเอง PPO)
===============================================================================
ควบคุมและบริหาร AI โมเดลด้วยอัลกอริทึม Proximal Policy Optimization (PPO)
จากคลัง Stable-Baselines3 โดยจะทำการฝึกสอน (Train) จากประวัติ + Backtest
แล้วนำค่าน้ำหนักไปปรับ Parameter จริงในระบบเทรด

Observation Space (16 dims — V3):
  [total_dd, daily_dd, progress, sortino, last_win_loss, volatility, day%, recent_wr,
   regime_trend, atr_zscore, day_of_week, regime_consistency, cum_pnl_norm,
   avg_rr_achieved, dd_velocity, profit_factor]
Action Space (5 dims, continuous [-1, 1]):
  [risk, confluence, atr_sl_mult, rr_ratio, max_daily_trades]

⚠️ CRITICAL: ตอน Inference ต้อง normalize obs ด้วย VecNormalize stats ที่บันทึกตอน train
   ไม่งั้น Agent จะเห็นโลกที่ scale ต่างจากที่มันเรียนรู้มา → output เป็นขยะ
===============================================================================
"""

import os
import pickle
import numpy as np
from stable_baselines3 import PPO

from ml.rl_environment import FTMOOptimizationEnv


class SelfLearningAgent:
    """ปัญญาประดิษฐ์ (AI Agent) สำหรับรับรู้ปัญหาการเทรดและปรับตัว"""

    # ขนาด Observation Space ต้องตรงกับที่กำหนดใน FTMOOptimizationEnv (V3: 13 → 16)
    OBS_DIM: int = 16

    def __init__(
        self,
        excel_path: str = "logs/ftmo_trades.xlsx",
        model_dir: str = "models",
        data_dir: str = None,
        verbose: int = 0,
    ):
        self.excel_path = excel_path
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.verbose = int(verbose)
        self.model_path = os.path.join(self.model_dir, "ppo_ftmo_agent.zip")
        self.vec_normalize_path = os.path.join(self.model_dir, "vec_normalize.pkl")

        # เตรียมตัวแปรโมเดล
        self.model = None

        # VecNormalize stats สำหรับ normalize obs ตอน inference
        # ⚠️ CRITICAL: ต้องโหลดค่านี้มาใช้ ไม่งั้น obs scale จะไม่ตรงกับตอน train
        self._obs_mean = None   # Running mean ของ observation
        self._obs_var = None    # Running variance ของ observation
        self._clip_obs = 10.0   # Clip range (ตรงกับ train_ppo.py)

    def initialize_model(self, strict: bool = True):
        """
        โหลดโมเดลที่เทรนแล้ว + VecNormalize stats สำหรับ live trading

        Args:
            strict: ถ้า True (ค่าเริ่มต้น) จะ raise RuntimeError เมื่อโหลดโมเดลไม่ได้
                    → ป้องกันการใช้ random policy เทรดเงินจริงเด็ดขาด
                    ถ้า False จะ fallback เป็น untrained model (สำหรับ dev/test เท่านั้น)
        """
        # ขั้นตอน 1: โหลด VecNormalize stats (ต้องทำก่อนโหลด model)
        self._load_normalize_stats()

        # ขั้นตอน 2: โหลด PPO model
        if os.path.exists(self.model_path):
            try:
                if self.verbose:
                    print("🧠 [RL Agent] พบสมองเดิม... กำลังโหลด PPO Model")
                self.model = PPO.load(self.model_path)
                if self.verbose:
                    print(f"   ✅ โหลดสำเร็จ จาก {self.model_path}")
                    if self._obs_mean is not None:
                        print(f"   ✅ VecNormalize stats โหลดเรียบร้อย (obs scale ตรงกับ training)")
                    else:
                        print(f"   ⚠️ ไม่พบ VecNormalize stats — obs จะไม่ถูก normalize!")
                return
            except Exception as e:
                if strict:
                    raise RuntimeError(
                        f"[RL Agent] โหลดโมเดลที่เทรนแล้วล้มเหลว: {e} — "
                        f"ปฏิเสธการใช้ random policy ในโหมด live"
                    )
                if self.verbose:
                    print(f"⚠️ [RL Agent] โหลดโมเดลเก่าล้มเหลว ({e}) — สร้างใหม่ (untrained)")
                self._create_new_model()
        else:
            if strict:
                raise RuntimeError(
                    f"[RL Agent] ไม่พบโมเดลที่ {self.model_path} — "
                    f"ต้องเทรน PPO ก่อน (python scripts/train_ppo.py)"
                )
            if self.verbose:
                print("🧠 [RL Agent] ไม่มีสมองเดิม... สร้าง PPO Model ใหม่ทั้งหมด")
            self._create_new_model()

    def _load_normalize_stats(self):
        """
        โหลดค่า running mean / variance จาก VecNormalize ที่บันทึกตอน train
        เพื่อนำไป normalize observation ก่อนส่งเข้า agent ตอน live inference

        ⚠️ ถ้าไม่โหลดค่านี้: obs ตอน live จะมี scale ต่างจากตอน train
           เช่น ตอน train obs ผ่าน normalize แล้ว (mean≈0, std≈1)
           แต่ตอน live obs เป็น raw (mean≈0.5, std≈0.3) → agent เห็นค่าแปลกๆ
        """
        if not os.path.exists(self.vec_normalize_path):
            if self.verbose:
                print(f"⚠️ [RL Agent] ไม่พบ {self.vec_normalize_path} — obs จะไม่ถูก normalize")
            return

        try:
            with open(self.vec_normalize_path, 'rb') as f:
                vec_normalize = pickle.load(f)

            # VecNormalize เก็บ obs_rms เป็น RunningMeanStd object
            # ที่มี .mean (np.ndarray) และ .var (np.ndarray)
            if hasattr(vec_normalize, 'obs_rms') and vec_normalize.obs_rms is not None:
                self._obs_mean = vec_normalize.obs_rms.mean.copy()
                self._obs_var = vec_normalize.obs_rms.var.copy()
                self._clip_obs = getattr(vec_normalize, 'clip_obs', 10.0)

                if self.verbose:
                    print(f"   📊 VecNormalize obs stats loaded: mean shape={self._obs_mean.shape}")
            else:
                if self.verbose:
                    print(f"⚠️ [RL Agent] VecNormalize ไม่มี obs_rms — ข้ามการ normalize")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ [RL Agent] โหลด VecNormalize ล้มเหลว ({e}) — ข้ามการ normalize")
            self._obs_mean = None
            self._obs_var = None

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation ให้ตรงกับที่ PPO เห็นตอน training

        สูตร (เหมือนกับ SB3 VecNormalize.normalize_obs):
            normalized = (obs - running_mean) / sqrt(running_var + epsilon)
            clipped = clip(normalized, -clip_obs, clip_obs)

        Args:
            obs: raw observation array shape (13,)

        Returns:
            normalized observation array shape (13,)
        """
        if self._obs_mean is None or self._obs_var is None:
            return obs

        normalized = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
        return np.clip(normalized, -self._clip_obs, self._clip_obs).astype(np.float32)

    def _create_new_model(self):
        """
        สร้าง PPO Model ใหม่ — ใช้ architecture เดียวกับ train_ppo.py เป๊ะ!

        ⚠️ WARNING: ถ้าค่าต่างจาก train_ppo.py → model ที่เทรนมาจะใช้ไม่ได้
        ถ้าจะเทรนจริง ให้ใช้ scripts/train_ppo.py เท่านั้น
        """
        from stable_baselines3.common.env_util import make_vec_env

        env_fn = lambda: FTMOOptimizationEnv(
            excel_log_path=self.excel_path,
            data_dir=self.data_dir,
        )
        vec_env = make_vec_env(env_fn, n_envs=1)

        # ⚠️ MUST match train_ppo.py architecture exactly! (V4)
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            ent_coef=0.005,
            gamma=0.97,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            clip_range=0.15,
            clip_range_vf=0.2,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 64], vf=[128, 64]),
            ),
        )

    def train_on_historical(self, timesteps: int = 4096):
        """
        ฝึกสอน AI จาก Mini-Backtest Environment
        Agent จะเรียนรู้จากการ simulate การเทรดแบบ day-by-day ในข้อมูลจริง
        """
        if self.model is None:
            self.initialize_model(strict=False)

        if self.verbose:
            print(f"📊 [RL Agent] กำลังเรียนรู้จาก Mini-Backtest Engine จำนวน {timesteps} timesteps...")
        try:
            self.model.learn(total_timesteps=timesteps, progress_bar=(self.verbose > 0))
            self.model.save(self.model_path)
            if self.verbose:
                print(f"✅ [RL Agent] เรียนรู้เสร็จสิ้นและบันทึกสมองไว้ที่ {self.model_path}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [RL Agent] เกิดข้อผิดพลาดขณะเรียนรู้: {e}")

    def get_optimized_parameters(self, current_observation: np.ndarray = None) -> dict:
        """
        ให้ AI ทำนายค่าพารามิเตอร์ที่ดีที่สุด (Inference) สำหรับตั้งค่า Bot ในวันถัดไป

        Pipeline:
        1. รับ raw observation (13 dims) จาก main.py
        2. Normalize ด้วย VecNormalize stats (จากตอน train)
        3. ส่งเข้า PPO.predict() → ได้ action (4 dims)
        4. แปลง action → Trading Parameters ผ่าน map_actions_to_parameters()

        Args:
            current_observation: ค่าสถานะปัจจุบัน (16 dims) — ถ้ายกเว้นจะใช้ Blank state

        Returns:
            Dict ของ 4 parameters: risk, confluence, atr_mult, rr_ratio
        """
        if self.model is None:
            self.initialize_model()

        if current_observation is None:
            # Blank state = วันแรกของ Challenge (ทุกอย่างเป็น 0)
            current_observation = np.zeros(self.OBS_DIM, dtype=np.float32)
        else:
            current_observation = np.asarray(current_observation, dtype=np.float32).flatten()
            # Defensive: ถ้าขนาดไม่ตรง → pad หรือ truncate
            if current_observation.size != self.OBS_DIM:
                padded = np.zeros(self.OBS_DIM, dtype=np.float32)
                n = min(current_observation.size, self.OBS_DIM)
                padded[:n] = current_observation[:n]
                current_observation = padded

        # ⚠️ CRITICAL: Normalize obs ก่อนส่งเข้า PPO
        # ตอน training obs จะผ่าน VecNormalize ก่อนเข้า network
        # ตอน live ต้องทำเอง — ไม่งั้น agent เห็นค่าผิด scale ทั้งหมด
        normalized_obs = self._normalize_obs(current_observation)

        # AI ทายว่า Action ใดดีที่สุด (Deterministic = True เพื่อความปลอดภัยในการเทรดจริง)
        action, _states = self.model.predict(normalized_obs, deterministic=True)

        # ใช้ Static Method → ไม่ต้องสร้าง Env instance ใหม่
        optimized_params = FTMOOptimizationEnv.map_actions_to_parameters(action)

        return optimized_params
