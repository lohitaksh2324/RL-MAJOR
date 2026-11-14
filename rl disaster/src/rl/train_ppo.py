"""
PPO Training Pipeline for RL-Based Disaster Evacuation System.
Improvements v2:
  - 150 000 total timesteps (5x more)
  - Larger 256-unit, 3-layer MLP policy
  - Dynamic hazard curriculum: after 40k steps, random barricades/car-blasts
    are injected mid-episode so PPO natively learns to detour
  - Anti-reversal reward already baked into the environment
  - Lower entropy coefficient (0.005) so policy converges to decisive routing
"""

import os
import json
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ..env.evacuation_env import DisasterEvacuationEnv


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DynamicHazardWrapper(gym.Wrapper if False else object):
    """
    Lightweight wrapper that adds random mid-episode hazards after `warmup_steps`
    training steps have passed. This forces the agent to encounter blocked-path
    situations during training itself, not just at demo time.
    """
    pass  # implemented inline in CurriculumCallback


# ──────────────────────────────────────────────────────────────────────────────
# Custom callbacks
# ──────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Two responsibilities:
    1. Injects a random hazard every `hazard_interval` steps once training
       passes `warmup_steps` (dynamic curriculum).
    2. Logs evacuation-specific metrics every `check_freq` steps.
    """

    def __init__(
        self,
        check_freq: int = 3000,
        warmup_steps: int = 40000,
        hazard_interval: int = 800,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.warmup_steps = warmup_steps
        self.hazard_interval = hazard_interval
        self.history = {
            "timesteps": [],
            "mean_reward": [],
            "success_rate": [],
            "mean_steps": [],
            "mean_exposure": [],
        }
        self._last_hazard_step = 0

    # ------------------------------------------------------------------
    def _inject_curriculum_hazard(self):
        """Drop 2-3 random hazards simultaneously on road cells not adjacent to exits."""
        env = self.training_env.envs[0].unwrapped  # unwrap Monitor
        road_cells = list(env.grid_map.road_cells)
        safe_exits = set(env.grid_map.exits)

        candidates = [
            c for c in road_cells
            if c not in safe_exits and not env.disaster.is_blocked(*c)
            and min(abs(c[0] - e[0]) + abs(c[1] - e[1]) for e in safe_exits) > 2
        ]
        if not candidates:
            return

        # Inject 1, 2, or 3 hazards — ramps up with training progress
        if self.num_timesteps < 80000:
            n_hazards = 1
        elif self.num_timesteps < 150000:
            n_hazards = 2
        else:
            n_hazards = 3
        n_hazards = min(n_hazards, len(candidates))
        chosen = random.sample(candidates, n_hazards)

        hazard_types = ["BARRICADE", "CAR_BLAST", "BUILDING_COLLAPSE"]
        for i, target in enumerate(chosen):
            h_type = hazard_types[i % len(hazard_types)]
            if h_type == "BARRICADE":
                env.disaster.place_barricade(target[0], target[1], reason="TRAINING CURRICULUM")
            elif h_type == "CAR_BLAST":
                env.disaster.trigger_car_blast(target)
            else:
                env.disaster.trigger_building_collapse(road_pos=target)

        # Update distance baseline so reward shaping doesn't spike
        env.prev_distance_to_nearest_exit = env._get_distance_to_nearest_safe_exit(env.agent_pos)

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # Curriculum: inject hazard
        if (self.num_timesteps >= self.warmup_steps and
                self.num_timesteps - self._last_hazard_step >= self.hazard_interval):
            try:
                self._inject_curriculum_hazard()
                self._last_hazard_step = self.num_timesteps
            except Exception:
                pass  # never crash training

        # Metrics logging
        if self.n_calls % self.check_freq == 0:
            eval_env = DisasterEvacuationEnv()
            successes = 0
            steps_list, exposure_list, rewards_list = [], [], []

            for _ in range(20):
                obs, _ = eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, step_info = eval_env.step(action)
                    ep_reward += reward
                    done = term or trunc

                rewards_list.append(ep_reward)
                if step_info.get("evacuated", False):
                    successes += 1
                steps_list.append(step_info.get("steps", 0))
                exposure_list.append(step_info.get("hazard_exposure", 0.0))

            mean_r = float(np.mean(rewards_list))
            succ_rate = float(successes / 20.0)
            mean_s = float(np.mean(steps_list))
            mean_exp = float(np.mean(exposure_list))

            self.history["timesteps"].append(self.num_timesteps)
            self.history["mean_reward"].append(mean_r)
            self.history["success_rate"].append(succ_rate)
            self.history["mean_steps"].append(mean_s)
            self.history["mean_exposure"].append(mean_exp)

            if self.verbose > 0:
                curriculum_on = "[ON]" if self.num_timesteps >= self.warmup_steps else "WARMING UP"
                print(
                    f"[PPO {self.num_timesteps:7d}] "
                    f"Reward: {mean_r:7.2f} | "
                    f"Success: {succ_rate*100:5.1f}% | "
                    f"Avg Steps: {mean_s:5.1f} | "
                    f"Hazard Exp: {mean_exp:4.2f} | "
                    f"Curriculum: {curriculum_on}"
                )

        return True


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps: int = 150000,
    save_path: str = "models/ppo_evac.zip",
    metrics_path: str = "results/ppo_training_metrics.json"
) -> PPO:
    """Trains the PPO evacuation agent with single distant exit and dynamic curriculum."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    raw_env = DisasterEvacuationEnv()
    env = Monitor(raw_env)

    # Larger 3-layer MLP (256 units) for better generalisation across detour situations
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,     # slightly lower for stability
        n_steps=2048,             # more rollout data per update
        batch_size=128,
        n_epochs=10,
        gamma=0.995,              # slightly longer horizon (rewards reaching exit)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,           # low entropy -> decisive policy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    callback = CurriculumCallback(
        check_freq=3000,
        warmup_steps=30000,       # first 30k steps: learn distant exit routing
        hazard_interval=600,      # then drop hazards every 600 steps
        verbose=1
    )

    print("=" * 65)
    print("PPO TRAINING: Single Distant Exit (150k steps + Curriculum)")
    print("=" * 65)
    print(f"  Network : 256 -> 256 -> 128  (pi & vf)")
    print(f"  Steps   : {total_timesteps:,}")
    print(f"  Obs dim : {raw_env.observation_space.shape[0]}  (inc. topological direction vector)")
    print(f"  Curriculum warmup : 30,000 steps then hazards every 600 steps")
    print("=" * 65)

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    model.save(save_path)
    print(f"\n[DONE] PPO model saved -> {save_path}")

    with open(metrics_path, "w") as f:
        json.dump(callback.history, f, indent=2)
    print(f"[DONE] Training metrics saved -> {metrics_path}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    model: PPO,
    num_episodes: int = 50,
    scenario=None
) -> dict:
    """Evaluates a trained policy across multiple test episodes."""
    env = DisasterEvacuationEnv(scenario=scenario)
    successes = 0
    total_steps, total_exposure, total_congestion, total_rewards = [], [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            done = term or trunc

        total_rewards.append(ep_reward)
        if info.get("evacuated", False):
            successes += 1
        total_steps.append(info.get("steps", 0))
        total_exposure.append(info.get("hazard_exposure", 0.0))
        total_congestion.append(info.get("congestion_encountered", 0.0))

    return {
        "success_rate": successes / num_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_steps": float(np.mean(total_steps)),
        "mean_hazard_exposure": float(np.mean(total_exposure)),
        "mean_congestion": float(np.mean(total_congestion)),
    }
