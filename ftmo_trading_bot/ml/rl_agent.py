"""
===============================================================================
FTMO Trading Bot — RL Agent (สมองส่วนการเรียนรู้ด้วยตนเอง PPO)
===============================================================================
ควบคุมและบริหาร AI โมเดลด้วยอัลกอริทึม Proximal Policy Optimization (PPO)
จากคลัง Stable-Baselines3 โดยจะทำการฝึกสอน (Train) จากประวัติ + Backtest
แล้วนำค่าน้ำหนักไปปรับ Parameter จริงในระบบเทรด

Observation Space (13 dims — V2):
  [total_dd, daily_dd, progress, sortino, last_win_loss, volatility, day%, recent_wr,
   regime_trend, atr_zscore, day_of_week, regime_consistency, cum_pnl_norm]
Action Space (4 dims, continuous [-1, 1]):
  [risk, confluence, atr_sl_mult, rr_ratio]
===============================================================================
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ml.rl_environment import FTMOOptimizationEnv


class SelfLearningAgent:
    """ปัญญาประดิษฐ์ (AI Agent) สำหรับรับรู้ปัญหาการเทรดและปรับตัว"""

    # ขนาด Observation Space ต้องตรงกับที่กำหนดใน FTMOOptimizationEnv (V2: 8 → 13)
    OBS_DIM: int = 13

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

        # Environment factory (vectorized สำหรับ PPO)
        self.env_fn = lambda: FTMOOptimizationEnv(
            excel_log_path=self.excel_path,
            data_dir=self.data_dir,
        )
        self.vec_env = make_vec_env(self.env_fn, n_envs=1)

        # เตรียมตัวแปรโมเดล
        self.model = None

    def initialize_model(self, strict: bool = False):
        """
        โหลดโมเดลเก่ามาเรียนรู้ต่อ (Transfer Learning / Continual Learning) หากมี, ไม่งั้นสร้างใหม่

        Args:
            strict: ถ้า True จะ raise RuntimeError เมื่อโหลดโมเดลที่เทรนแล้วไม่ได้
                    (ใช้ตอน live trading — ไม่ยอมรับ untrained random policy)
        """
        if os.path.exists(self.model_path):
            try:
                if self.verbose:
                    print("🧠 [RL Agent] พบสมองเดิม... กำลังโหลด PPO Model")
                self.model = PPO.load(self.model_path, env=self.vec_env)
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

    def _create_new_model(self):
        """สร้าง PPO Model ใหม่ — Policy อนุรักษ์นิยม (FTMO-friendly)"""
        # ent_coef = 0.01 กระตุ้นการสำรวจเล็กน้อย
        # gamma = 0.999 ให้น้ำหนักอนาคตสูง (FTMO = 30 วัน, จำเป็นต้องมองยาว)
        # learning_rate เริ่มต้น 3e-4 ตาม SB3 default เหมาะสำหรับ FTMO
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=0,
            ent_coef=0.01,
            gamma=0.999,
            learning_rate=3e-4,
            n_steps=512,
        )

    def train_on_historical(self, timesteps: int = 4096):
        """
        ฝึกสอน AI จาก Mini-Backtest Environment
        Agent จะเรียนรู้จากการ simulate การเทรดแบบ day-by-day ในข้อมูลจริง
        """
        if self.model is None:
            self.initialize_model()

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

        Args:
            current_observation: ค่าสถานะปัจจุบัน (13 dims) — ถ้ายกเว้นจะใช้ Blank state

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

        # AI ทายว่า Action ใดดีที่สุด (Deterministic = True เพื่อความปลอดภัยในการเทรดจริง)
        action, _states = self.model.predict(current_observation, deterministic=True)

        # ใช้ Static Method → ไม่ต้องสร้าง Env instance ใหม่
        optimized_params = FTMOOptimizationEnv.map_actions_to_parameters(action)

        return optimized_params
