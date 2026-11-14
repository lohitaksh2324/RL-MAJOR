"""
Benchmarking framework comparing PPO against Baselines (A*, Dijkstra).
Evaluates algorithms across 3x3 city grid scenarios:
- Static Damaged Environment (known hazards)
- Dynamic Car Blasts (sudden vehicular explosions)
- Dynamic Building Crashes (structural collapse dumping rubble)
- Compound Catastrophe (simultaneous blasts, building crashes, and spreading fire)
"""

from typing import List, Dict, Any, Optional
import os
import json
from stable_baselines3 import PPO

from ..config import ScenarioConfig, CarSpec, BuildingCrashSpec
from ..env.evacuation_env import DisasterEvacuationEnv
from ..baselines.astar import AStarPlanner
from ..baselines.dijkstra import DijkstraPlanner
from .metrics import EpisodeMetrics, BenchmarkResult, aggregate_metrics

def create_benchmark_scenarios() -> List[ScenarioConfig]:
    """Generates standard scenarios for the 3x3 city grid."""
    return [
        ScenarioConfig(
            name="1_Static_Damaged_City",
            description="Static city grid with pre-damaged road debris. Hazards are known upfront.",
            is_dynamic=False,
            fire_start_positions=[(22, 23)],
            fire_spread_prob=0.08,
            cars=[
                CarSpec(row=8, col=4, color="yellow"),
                CarSpec(row=15, col=12, color="blue"),
                CarSpec(row=22, col=8, color="silver"),
            ],
            building_crashes=[],
            static_hazards=[(15, 8), (15, 9)],  # Damaged crossroad
            exit_congestions={0: 0.1, 1: 0.1, 2: 0.1}
        ),
        ScenarioConfig(
            name="2_Dynamic_Car_Blast",
            description="Vehicle on central avenue explodes at step 6, blocking the primary evacuation route with fire & wreckage.",
            is_dynamic=True,
            fire_start_positions=[(22, 23)],
            fire_spread_prob=0.15,
            cars=[
                CarSpec(row=8, col=4, color="yellow", explodes_at_step=-1),
                CarSpec(row=15, col=16, color="red", explodes_at_step=6),   # Car blast cuts off corridor!
                CarSpec(row=8, col=12, color="blue", explodes_at_step=-1),
                CarSpec(row=22, col=12, color="silver", explodes_at_step=-1),
            ],
            building_crashes=[],
            static_hazards=[],
            exit_congestions={0: 0.15, 1: 0.1, 2: 0.1}
        ),
        ScenarioConfig(
            name="3_Dynamic_Building_Crash",
            description="Building B5 suffers structural collapse at step 8, dumping concrete rubble onto the avenue.",
            is_dynamic=True,
            fire_start_positions=[(22, 23)],
            fire_spread_prob=0.12,
            cars=[
                CarSpec(row=8, col=4, color="yellow"),
                CarSpec(row=22, col=8, color="silver"),
            ],
            building_crashes=[
                BuildingCrashSpec(
                    building_id="B5",
                    rubble_cells=[(9, 11), (9, 12), (10, 8), (10, 9)],
                    crashes_at_step=8
                )
            ],
            static_hazards=[],
            exit_congestions={0: 0.1, 1: 0.1, 2: 0.1}
        ),
        ScenarioConfig(
            name="4_Compound_Catastrophe",
            description="Extreme disaster: Building B8 collapses while vehicle explodes on Mid-Avenue, with spreading fire.",
            is_dynamic=True,
            fire_start_positions=[(22, 23), (23, 22)],
            fire_spread_prob=0.22,
            cars=[
                CarSpec(row=15, col=16, color="red", explodes_at_step=5),
                CarSpec(row=8, col=12, color="blue", explodes_at_step=9),
            ],
            building_crashes=[
                BuildingCrashSpec(
                    building_id="B8",
                    rubble_cells=[(16, 11), (16, 12), (17, 8), (17, 9)],
                    crashes_at_step=6
                )
            ],
            static_hazards=[(15, 8)],
            exit_congestions={0: 0.25, 1: 0.2, 2: 0.1}
        )
    ]

def evaluate_baseline_agent(
    planner_cls,
    scenario: ScenarioConfig,
    num_episodes: int = 25,
    replan_dynamically: bool = False
) -> List[EpisodeMetrics]:
    """Runs evaluation for A* or Dijkstra baseline."""
    results = []
    for ep in range(num_episodes):
        env = DisasterEvacuationEnv(scenario=scenario)
        obs, _ = env.reset(seed=ep * 101)
        planner = planner_cls(replan_dynamically=replan_dynamically)
        planner.plan_initial_path(env)

        done = False
        ep_reward = 0.0
        while not done:
            action = planner.select_action(env)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            done = term or trunc

        results.append(EpisodeMetrics(
            success=info.get("evacuated", False),
            steps=info.get("steps", 0),
            hazard_exposure=info.get("hazard_exposure", 0.0),
            congestion=info.get("congestion_encountered", 0.0),
            total_reward=ep_reward,
            exit_reached=info.get("exit_reached", None)
        ))
    return results

def evaluate_rl_agent(
    model,
    scenario: ScenarioConfig,
    num_episodes: int = 25
) -> List[EpisodeMetrics]:
    """Runs evaluation for a trained PPO model."""
    results = []
    for ep in range(num_episodes):
        env = DisasterEvacuationEnv(scenario=scenario)
        obs, _ = env.reset(seed=ep * 101)

        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            done = term or trunc

        results.append(EpisodeMetrics(
            success=info.get("evacuated", False),
            steps=info.get("steps", 0),
            hazard_exposure=info.get("hazard_exposure", 0.0),
            congestion=info.get("congestion_encountered", 0.0),
            total_reward=ep_reward,
            exit_reached=info.get("exit_reached", None)
        ))
    return results

def run_benchmark(
    ppo_model_path: Optional[str] = "models/ppo_evac.zip",
    num_episodes_per_scenario: int = 25,
    output_path: str = "results/benchmark_results.json"
) -> List[BenchmarkResult]:
    """Executes comparative benchmark across all algorithms and scenarios."""
    scenarios = create_benchmark_scenarios()
    all_summaries: List[BenchmarkResult] = []

    ppo_model = None
    if ppo_model_path and os.path.exists(ppo_model_path):
        ppo_model = PPO.load(ppo_model_path)
        print("Loaded trained PPO model.")

    for scenario in scenarios:
        print(f"\n--- Benchmarking Scenario: {scenario.name} ---")

        # 1. Static A*
        astar_static = evaluate_baseline_agent(
            AStarPlanner, scenario, num_episodes=num_episodes_per_scenario, replan_dynamically=False
        )
        res_astar_s = aggregate_metrics("A* (Static)", scenario.name, astar_static)
        all_summaries.append(res_astar_s)
        print(f"A* (Static): Succ={res_astar_s.success_rate*100:.1f}%, Steps={res_astar_s.mean_steps:.1f}, Exp={res_astar_s.mean_hazard_exposure:.2f}")

        # 2. Dynamic Replanning A*
        astar_dyn = evaluate_baseline_agent(
            AStarPlanner, scenario, num_episodes=num_episodes_per_scenario, replan_dynamically=True
        )
        res_astar_d = aggregate_metrics("A* (Replanning)", scenario.name, astar_dyn)
        all_summaries.append(res_astar_d)
        print(f"A* (Replanning): Succ={res_astar_d.success_rate*100:.1f}%, Steps={res_astar_d.mean_steps:.1f}, Exp={res_astar_d.mean_hazard_exposure:.2f}")

        # 3. Static Dijkstra
        dijk_static = evaluate_baseline_agent(
            DijkstraPlanner, scenario, num_episodes=num_episodes_per_scenario, replan_dynamically=False
        )
        res_dijk_s = aggregate_metrics("Dijkstra (Static)", scenario.name, dijk_static)
        all_summaries.append(res_dijk_s)
        print(f"Dijkstra (Static): Succ={res_dijk_s.success_rate*100:.1f}%, Steps={res_dijk_s.mean_steps:.1f}, Exp={res_dijk_s.mean_hazard_exposure:.2f}")

        # 4. PPO Agent
        if ppo_model is not None:
            ppo_res = evaluate_rl_agent(ppo_model, scenario, num_episodes=num_episodes_per_scenario)
            res_ppo = aggregate_metrics("PPO (Adaptive RL)", scenario.name, ppo_res)
            all_summaries.append(res_ppo)
            print(f"PPO (Adaptive RL): Succ={res_ppo.success_rate*100:.1f}%, Steps={res_ppo.mean_steps:.1f}, Exp={res_ppo.mean_hazard_exposure:.2f}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([s.to_dict() for s in all_summaries], f, indent=2)
    print(f"\nAll benchmark results exported to {output_path}")

    return all_summaries
