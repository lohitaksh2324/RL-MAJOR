"""
Metrics tracking module for disaster evacuation evaluation.
Stores and aggregates results across episodes and algorithms.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

@dataclass
class EpisodeMetrics:
    success: bool
    steps: int
    hazard_exposure: float
    congestion: float
    total_reward: float
    exit_reached: Any = None

@dataclass
class BenchmarkResult:
    algorithm_name: str
    scenario_name: str
    num_episodes: int
    success_rate: float
    mean_steps: float
    std_steps: float
    mean_hazard_exposure: float
    std_hazard_exposure: float
    mean_congestion: float
    mean_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def aggregate_metrics(
    algorithm_name: str,
    scenario_name: str,
    episodes: List[EpisodeMetrics]
) -> BenchmarkResult:
    """Aggregate a list of episode results into benchmark summary statistics."""
    num_episodes = len(episodes)
    if num_episodes == 0:
        return BenchmarkResult(
            algorithm_name=algorithm_name,
            scenario_name=scenario_name,
            num_episodes=0,
            success_rate=0.0,
            mean_steps=0.0,
            std_steps=0.0,
            mean_hazard_exposure=0.0,
            std_hazard_exposure=0.0,
            mean_congestion=0.0,
            mean_reward=0.0
        )

    success_rate = float(sum(1 for e in episodes if e.success) / num_episodes)
    steps = [e.steps for e in episodes]
    exposures = [e.hazard_exposure for e in episodes]
    congestions = [e.congestion for e in episodes]
    rewards = [e.total_reward for e in episodes]

    return BenchmarkResult(
        algorithm_name=algorithm_name,
        scenario_name=scenario_name,
        num_episodes=num_episodes,
        success_rate=success_rate,
        mean_steps=float(np.mean(steps)),
        std_steps=float(np.std(steps)),
        mean_hazard_exposure=float(np.mean(exposures)),
        std_hazard_exposure=float(np.std(exposures)),
        mean_congestion=float(np.mean(congestions)),
        mean_reward=float(np.mean(rewards))
    )
