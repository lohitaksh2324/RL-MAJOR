"""
Plotting utilities for disaster evacuation benchmarks and training curves.
Generates publication-quality charts using Matplotlib.
"""

import json
import os
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(
    metrics_path: str = "results/ppo_training_metrics.json",
    save_path: str = "results/ppo_training_curves.png"
):
    """Plot PPO training learning curves (Reward, Success Rate, Steps)."""
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} does not exist.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    timesteps = data.get("timesteps", [])
    rewards = data.get("mean_reward", [])
    successes = data.get("success_rate", [])
    steps = data.get("mean_steps", [])

    if not timesteps:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PPO Evacuation Policy Training Curves", fontsize=15, fontweight="bold")

    # 1. Reward
    axes[0].plot(timesteps, rewards, color="#1f77b4", linewidth=2.2, marker="o")
    axes[0].set_title("Mean Episode Reward", fontsize=12)
    axes[0].set_xlabel("Environment Timesteps")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Success Rate
    axes[1].plot(timesteps, [s * 100 for s in successes], color="#2ca02c", linewidth=2.2, marker="s")
    axes[1].set_title("Evacuation Success Rate (%)", fontsize=12)
    axes[1].set_xlabel("Environment Timesteps")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Evacuation Steps
    axes[2].plot(timesteps, steps, color="#ff7f0e", linewidth=2.2, marker="^")
    axes[2].set_title("Average Evacuation Steps", fontsize=12)
    axes[2].set_xlabel("Environment Timesteps")
    axes[2].set_ylabel("Steps to Exit")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_benchmark_results(
    results_path: str = "results/benchmark_results.json",
    save_path: str = "results/benchmark_comparison.png"
):
    """Generates comparative bar charts for benchmark metrics across algorithms."""
    if not os.path.exists(results_path):
        print(f"Benchmark results file {results_path} does not exist.")
        return

    with open(results_path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        return

    scenarios = sorted(list(set(d["scenario_name"] for d in data)))
    algorithms = sorted(list(set(d["algorithm_name"] for d in data)))

    # Structure data: {metric: {algo: [val_scen1, val_scen2, ...]}}
    metrics = ["success_rate", "mean_steps", "mean_hazard_exposure", "mean_congestion"]
    metric_titles = [
        "Evacuation Success Rate (%)",
        "Evacuation Time (Steps)",
        "Hazard Exposure (Lower is better)",
        "Congestion Encountered"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    axes = axes.flatten()

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    x = np.arange(len(scenarios))
    bar_width = 0.8 / len(algorithms)

    for m_idx, (m_key, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[m_idx]
        for a_idx, algo in enumerate(algorithms):
            values = []
            for sc in scenarios:
                match = [d for d in data if d["scenario_name"] == sc and d["algorithm_name"] == algo]
                if match:
                    val = match[0][m_key]
                    if m_key == "success_rate":
                        val *= 100.0
                    values.append(val)
                else:
                    values.append(0.0)

            offset = (a_idx - (len(algorithms) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=algo if m_idx == 0 else "",
                color=colors[a_idx % len(colors)]
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        # Format scenario names nicely
        short_scenarios = [s.replace("_", " ") for s in scenarios]
        ax.set_xticklabels(short_scenarios, rotation=15, ha="right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add shared legend at top
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(algorithms), fontsize=11)
    plt.subplots_adjust(top=0.90, hspace=0.35)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Benchmark comparison charts saved to {save_path}")
