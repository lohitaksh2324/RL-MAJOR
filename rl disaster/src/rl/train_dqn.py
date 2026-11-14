"""
DQN Training Pipeline for RL-Based Disaster Evacuation System.
Provides comparative deep RL algorithm against primary PPO.
"""

import os
import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from ..env.evacuation_env import DisasterEvacuationEnv
from .train_ppo import EvacuationMetricsCallback

def train_dqn(
    total_timesteps: int = 40000,
    save_path: str = "models/dqn_evac.zip",
    metrics_path: str = "results/dqn_training_metrics.json"
) -> DQN:
    """Trains DQN evacuation agent for benchmark comparison."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    raw_env = DisasterEvacuationEnv()
    env = Monitor(raw_env)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=30000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0
    )

    callback = EvacuationMetricsCallback(check_freq=3000, verbose=1)

    print(f"Starting DQN training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(save_path)
    print(f"DQN model saved to {save_path}")

    with open(metrics_path, "w") as f:
        json.dump(callback.history, f, indent=2)
    print(f"DQN training metrics saved to {metrics_path}")

    return model
