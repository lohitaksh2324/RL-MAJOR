"""
Standalone training runner — run from project root:
    .venv\Scripts\python.exe run_training.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.rl.train_ppo import train_ppo
train_ppo(total_timesteps=150000)
