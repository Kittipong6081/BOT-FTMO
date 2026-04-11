"""
===============================================================================
FTMO Trading Bot — RL Environment (สภาพแวดล้อมฝึกซ้อมฉบับ Gymnasium)
===============================================================================
ทำหน้าที่เป็นสะพานเชื่อมระหว่าง PPO Agent (สมอง) และ Trading System (ระบบ)
ใช้สำหรับการ Training แบบ Offline จากข้อมูลประวัติเทรดใน Excel
เพื่อหาพารามิเตอร์ระบบ (Action) ที่ดีที่สุดเทียบกับข้อมูลสภาพตลาดล่าสุด (State)
===============================================================================
"""

import os
import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ml.reward_function import FTMORewardCalculator

class FTMOOptimizationEnv(gym.Env):
    """
    คลาส Environment สำหรับการ Optimize การตั้งค่า Bot เพื่อผ่านการทดสอบแบบปลอดภัยที่สุด
    """
    
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, excel_log_path: str = "logs/trading_log.xlsx", max_steps: int = 100):
        super(FTMOOptimizationEnv, self).__init__()
        
        self.excel_path = excel_log_path
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward function custom FTMO
        self.reward_calculator = FTMORewardCalculator()
        
        # ประวัติความก้าวหน้า
        self.trades_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        self._load_historical_data()

        # ==========================================
        # Observation Space (State ที่มองเห็น)
        # ==========================================
        # ให้ Agent มองเห็นสถานะตัวเองปัจจุบัน จำนวน 5 ค่า (Normalize ล่วงหน้าเข้าช่วง [-1.0, 1.0]):
        # 1. Total Drawdown % (0 ถึง -0.10)
        # 2. Daily Drawdown % (0 ถึง -0.05)
        # 3. Target Progress % (0 ถึง 1.0)
        # 4. Previous Sortino Ratio (1D)
        # 5. Last Trade Win/Loss (1=Win, -1=Loss, 0=None)
        
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(5,), dtype=np.float32
        )

        # ==========================================
        # Action Space (การกระทำที่กดสวิตช์ได้)
        # ==========================================
        # การเปลี่ยนพารามิเตอร์ระบบ 4 ค่า ให้ออกมาเป็น Continuous ระหว่าง [-1.0, 1.0]
        # จากนั้นตรรกะใน Agent จะเอาไปแปลงสเกล (Map to bounds) เช่น:
        # [0] Risk Per Trade           : [-1.0, 1.0] -> [0.5% ถึง 1.0%]
        # [1] Min Confluence Score     : [-1.0, 1.0] -> [50 ถึง 80]
        # [2] ATR SL Multiplier        : [-1.0, 1.0] -> [1.0 ถึง 2.5]
        # [3] Minimum Risk:Reward Ratio: [-1.0, 1.0] -> [1.5 ถึง 3.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.current_stats = {
            'daily_dd_pct': 0.0,
            'total_dd_pct': 0.0,
            'sortino_ratio': 0.0,
            'target_progress_pct': 0.0,
            'balance': 100000.0,
            'last_trade_win': 0
        }
        
        self.previous_stats = self.current_stats.copy()

    def _load_historical_data(self):
        """รับข้อมูลย้อนหลังจาก Trade Logger"""
        if os.path.exists(self.excel_path):
            try:
                self.trades_df = pd.read_excel(self.excel_path, sheet_name='Trades')
                self.stats_df = pd.read_excel(self.excel_path, sheet_name='Stats')
            except Exception as e:
                print(f"⚠️ [RL Env] ไม่สามารถอ่านไหล์ประวัติได้: {e}")
                self.trades_df = pd.DataFrame()
                self.stats_df = pd.DataFrame()

    def _get_obs(self) -> np.ndarray:
        """แปลง current_stats ออกมาเป็น Array สำหรับ Observation Space"""
        obs = np.array([
            min(0.0, max(-1.0, self.current_stats['total_dd_pct'] / 0.10)),  # แปลงเทียบ 10%
            min(0.0, max(-1.0, self.current_stats['daily_dd_pct'] / 0.05)),  # แปลงเทียบ 5%
            min(1.0, max(-1.0, self.current_stats['target_progress_pct'] / 100.0)),
            min(5.0, max(-5.0, self.current_stats['sortino_ratio'])),        # Clip ที่ +-5
            float(self.current_stats['last_trade_win'])
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """เริ่ม Episode ใหม่"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # รีเซ็ตอ้างอิงสถานะไปตริเริ่มพอร์ต 1 แสนเหรียญ
        self.current_stats = {
            'daily_dd_pct': 0.0,
            'total_dd_pct': 0.0,
            'sortino_ratio': 0.0,
            'target_progress_pct': 0.0,
            'balance': 100000.0,
            'last_trade_win': 0
        }
        self.previous_stats = self.current_stats.copy()
        
        info = {}
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        """
        Agent สั่ง Action (เปลี่ยนค่าพารามิเตอร์) -> Environment ตอบสนอง (Reward)
        และขยับไปยัง Step ถัดไป หรือ State ใหม่
        """
        self.current_step += 1
        self.previous_stats = self.current_stats.copy()

        # ==========================================
        # ในโลกปกติ: Action -> เอาไปเทรดกับ Backtest หรือ Simulator
        # เนื่องจากเราเป็นแบบ Continual Adaptation:
        # สมมติผลกระทบ (Mock Impact) ว่าระบบตั้งค่าแบบนี้ จะเกิดอะไรขึ้นในวันต่อไป
        # แต่เพื่อความสมจริง หากมีการเชื่อมต่อระบบจำลอง Backtest ให้ Call ที่นี่
        # สำหรับโครงสร้างนี้ ผมจะเลียนแบบผลลัพธ์จาก Historical Data เพื่อคำนวณ Reward คร่าวๆ
        # ==========================================
        
        # สมมติเราดึงข้อมูลแถวถัดไปจาก Excel History (ถ้ามี) มาแทนค่า 
        # เพื่อป้อนกลับเป็น State ให้ Agent ฝึกเข้าใจสถานการณ์เก่า
        if not self.stats_df.empty and self.current_step < len(self.stats_df):
            row = self.stats_df.iloc[self.current_step]
        else:
            # ใช้การกะประมาณ (Stochastic Simulation) หากข้อมูลหมด หรือไม่มี
            # PPO จะเรียนรู้การป้องกันความเสี่ยง แม้อาจเทรดจำลอง
            # ในกรณีเทรดจริง สภาพจะมาจาการอัพเดทจริงหลังจากเทรด 1 วัน หรือ 1 สัปดาห์
            pass 

        # ตัวอย่างจำลอง (Simulate Impact) - Agent ที่เล่นปลอดภัย จะได้ Sortino ดีขึ้น
        risk_action = action[0]       # ลบ = เสี่ยงต่ำ (0.5%), บวก = เสี่ยงสูง (1.0%)
        conf_action = action[1]       # ลบ = คอนเฟิร์มน้อย (60), บวก = คอนเฟิร์มเยอะ (80)
        
        # ปรับจำลองผลกระทบต่อสถิติปัจจุบัน
        if risk_action > 0.5 and conf_action < 0.0:
            # ความเสี่ยงสูง มั่วสูง -> มีสิทธิขาดทุนติดลบ (Daily DD ลบ)
            self.current_stats['daily_dd_pct'] += np.random.uniform(0.01, 0.03)
            self.current_stats['last_trade_win'] = -1
        elif risk_action <= 0.0 and conf_action >= 0.0:
            # ปลอดภัยสูง -> ได้กำไรสม่ำเสมอ หรือขาดทุนนิดหน่อย
            self.current_stats['target_progress_pct'] += np.random.uniform(0.1, 1.0)
            self.current_stats['last_trade_win'] = 1
            self.current_stats['sortino_ratio'] = min(3.0, self.current_stats['sortino_ratio'] + 0.1)

        # สะสม Total
        self.current_stats['total_dd_pct'] = max(self.current_stats['total_dd_pct'], self.current_stats['daily_dd_pct'])
        
        # คำนวณ Reward
        reward = self.reward_calculator.calculate_reward(self.current_stats, self.previous_stats)

        # จบ Episode เมื่อไร? (บรรลุเป้า , ตาย , หรือหมดรอบ)
        terminated = False
        truncated = False
        
        if self.current_stats['daily_dd_pct'] >= self.reward_calculator.daily_limit or \
           self.current_stats['total_dd_pct'] >= self.reward_calculator.total_limit:
            terminated = True
            
        elif self.current_stats['target_progress_pct'] >= 100.0:
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True

        info = {'current_step': self.current_step, 'action': action}
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.current_step} | Total DD: {self.current_stats['total_dd_pct']:.2%} | Target: {self.current_stats['target_progress_pct']:.2%}")

    def map_actions_to_parameters(self, action: np.ndarray) -> dict:
        """แปลง Action สเปซกลับเป็นพารามิเตอร์จริงสำหรับไปทับค่า `bot_config` ใน `main.py`"""
        # [0] risk
        val_risk = 0.0075 + (action[0] * 0.0025)  # -1->0.5%, 1->1.0%
        # [1] confluence
        val_conf = 65.0 + (action[1] * 15.0)      # -1->50, 1->80
        # [2] atr
        val_atr = 1.75 + (action[2] * 0.75)       # -1->1.0, 1->2.5
        # [3] rr
        val_rr = 2.25 + (action[3] * 0.75)        # -1->1.5, 1->3.0

        return {
            'risk_per_trade_pct': round(val_risk, 4),
            'min_confluence_score': int(val_conf),
            'atr_sl_multiplier': round(val_atr, 1),
            'preferred_risk_reward_ratio': round(val_rr, 1)
        }
