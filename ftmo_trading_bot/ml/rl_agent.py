"""
===============================================================================
FTMO Trading Bot — RL Agent (สมองส่วนการเรียนรู้ด้วยตนเอง PPO)
===============================================================================
ควบคุมและบริหาร AI โมเดลด้วยอัลกอริทึม Proximal Policy Optimization (PPO) 
จากคลัง Stable-Baselines3 โดยจะทำการฝึกสอน (Train) จากประวัติ และนำค่าน้ำหนัก
ไปปรับ Parameter จริงในระบบเทรด
===============================================================================
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ml.rl_environment import FTMOOptimizationEnv

class SelfLearningAgent:
    """ปัญญาประดิษฐ์ (AI Agent) สำหรับรับรู้ปัญหาการเทรดและปรับตัว"""
    
    def __init__(self, excel_path: str = "logs/trading_log.xlsx", model_dir: str = "models", verbose: int = 0):
        self.excel_path = excel_path
        self.model_dir = model_dir
        self.verbose = int(verbose)
        self.model_path = os.path.join(self.model_dir, "ppo_ftmo_agent.zip")
        
        # สร้าง Environment ของเรา
        self.env_fn = lambda: FTMOOptimizationEnv(excel_log_path=self.excel_path)
        self.vec_env = make_vec_env(self.env_fn, n_envs=1)
        
        # เตรียมตัวแปรโมเดล
        self.model = None

    def initialize_model(self):
        """โหลดโมเดลเก่ามาเรียนรู้ต่อ (Transfer Learning / Continual Learning) หากมี, ไม่งั้นสร้างใหม่"""
        if os.path.exists(self.model_path):
            if self.verbose:
                print("🧠 [RL Agent] พบกล้ามเนื้อเนื้อสมองเดิม... กำลังโหลด PPO Model เพื่อเรียนรู้ต่อ")
            self.model = PPO.load(self.model_path, env=self.vec_env)
        else:
            if self.verbose:
                print("🧠 [RL Agent] ไม่มีสมองเดิม... สร้าง PPO Model ใหม่ทั้งหมด")
            # เน้น Policy ให้อนุรักษ์นิยมด้วยค่า Ent_coef กระตุ้นการสำรวจเล็กน้อย และ gamma สูง (เน้นอนาคตระยะยาว FTMO)
            self.model = PPO(
                "MlpPolicy", 
                self.vec_env, 
                verbose=0, 
                ent_coef=0.01, 
                gamma=0.999 
            )

    def train_on_historical(self, timesteps: int = 2048):
        """
        สั่งให้ AI เสพประวัติศาสตร์การเทรดที่ผ่านมาใน Excel ทบทวนข้อผิดพลาด
        และอัพเดทค่าน้ำหนักสู่พารามิเตอร์ของระบบที่ดีที่สุด
        """
        if self.model is None:
            self.initialize_model()

        if self.verbose:
            print(f"📊 [RL Agent] กำลังเรียนรู้จากประวัติเทรดล่าสุดจำนวน {timesteps} timesteps...")
        try:
            self.model.learn(total_timesteps=timesteps, progress_bar=(self.verbose>0))
            self.model.save(self.model_path)
            if self.verbose:
                print(f"✅ [RL Agent] เรียนรู้เสร็จสิ้นและบันทึกสมองไว้ที่ {self.model_path}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [RL Agent] เกิดข้อผิดพลาดขณะเรียนรู้: {e}")

    def get_optimized_parameters(self, current_observation: np.ndarray = None) -> dict:
        """
        ให้ AI ทำนาย/วางแผนค่าน้ำหนักล่าสุด (Inference) สำหรับใช้ตั้งค่าบอทในวันถัดไป
        
        Args:
            current_observation: ค่าสถิติปัจจุบัน [total_dd, daily_dd, progress, sortino, last_win_loss]
                                ถ้ายกเว้นไว้ จะใช้การประเมินจาก env ล่าสุด
        """
        if self.model is None:
            self.initialize_model()

        if current_observation is None:
            # ใช้ Blank state หรือสภาวะทรงตัวเป็นฐานถ้ายังไม่ได้เริ่มเทรดของวัน
            current_observation = np.zeros(5, dtype=np.float32)

        # AI ทายว่า Action ใดดีที่สุด (Deterministic = True เพื่อความปลอดภัยของการเทรดจริง ไม่สุ่มมั่ว)
        action, _states = self.model.predict(current_observation, deterministic=True)
        
        # ถอด Action เป็น Parameter
        temp_env = FTMOOptimizationEnv()
        optimized_params = temp_env.map_actions_to_parameters(action)
        
        return optimized_params
