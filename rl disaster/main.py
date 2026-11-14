"""
Main CLI entry point for RL-Based Disaster Evacuation System.
Provides commands for training PPO, benchmarking against baselines, and running visualization demos.
"""

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Based Disaster Evacuation System")
    parser.add_argument("--train", action="store_true", help="Train the primary PPO evacuation agent")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps for PPO training (default: 50000)")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmark comparing PPO vs A* vs Dijkstra")
    parser.add_argument("--episodes", type=int, default=25, help="Number of benchmark evaluation episodes per scenario")
    parser.add_argument("--demo-pygame", action="store_true", help="Launch interactive 2D Pygame simulator")
    parser.add_argument("--demo-web", action="store_true", help="Launch interactive Streamlit web dashboard")
    parser.add_argument("--test", action="store_true", help="Run rapid diagnostic test suite")
    return parser.parse_args()

def run_tests():
    print("=== Running Diagnostic System Tests ===")
    from src.env.evacuation_env import DisasterEvacuationEnv
    from src.baselines.astar import AStarPlanner
    from src.baselines.dijkstra import DijkstraPlanner
    from src.evaluation.benchmark import create_benchmark_scenarios

    env = DisasterEvacuationEnv()
    obs, info = env.reset(seed=123)
    assert obs.shape == (118,), f"Unexpected obs shape: {obs.shape}"
    print("[PASS] Gymnasium environment initialized.")

    for action in range(5):
        obs, r, term, trunc, info = env.step(action)
    print("[PASS] Environment actions step successfully.")

    astar = AStarPlanner()
    astar.plan_initial_path(env)
    assert len(astar.planned_path) > 0, "A* failed to compute path"
    print(f"[PASS] A* baseline computed path (length: {len(astar.planned_path)}).")

    dijk = DijkstraPlanner()
    dijk.plan_initial_path(env)
    assert len(dijk.planned_path) > 0, "Dijkstra failed to compute path"
    print(f"[PASS] Dijkstra baseline computed path (length: {len(dijk.planned_path)}).")

    scenarios = create_benchmark_scenarios()
    assert len(scenarios) == 4, "Expected 4 benchmark scenarios"
    print(f"[PASS] {len(scenarios)} benchmark scenarios initialized.")

    print("=== All Diagnostic Tests Passed! ===")

def main():
    args = parse_args()

    # Default action if no flag is provided
    if len(sys.argv) == 1:
        print("\nRL-Based Disaster Evacuation System")
        print("Usage:")
        print("  python main.py --test            : Run quick component diagnostics")
        print("  python main.py --train           : Train primary PPO model")
        print("  python main.py --benchmark       : Run PPO vs A* vs Dijkstra benchmark")
        print("  python main.py --demo-pygame     : Launch interactive 2D Pygame GUI")
        print("  python main.py --demo-web        : Launch Streamlit web dashboard")
        print("\nRunning quick component test now...\n")
        run_tests()
        return

    if args.test:
        run_tests()

    if args.train:
        print(f"\n[1/2] Initiating PPO Training ({args.timesteps} timesteps)...")
        from src.rl.train_ppo import train_ppo
        from src.evaluation.plotting import plot_training_curves

        train_ppo(total_timesteps=args.timesteps)
        print("\n[2/2] Generating PPO Training Learning Curves...")
        plot_training_curves()
        print("PPO Training & Curve Generation Complete!")

    if args.benchmark:
        print(f"\nInitiating Comparative Benchmark across 4 disaster scenarios ({args.episodes} episodes each)...")
        from src.evaluation.benchmark import run_benchmark
        from src.evaluation.plotting import plot_benchmark_results

        run_benchmark(num_episodes_per_scenario=args.episodes)
        print("\nGenerating Comparative Visualizations...")
        plot_benchmark_results()
        print("Benchmark Complete! Check `results/benchmark_comparison.png`")

    if args.demo_pygame:
        print("\nLaunching Pygame Simulator...")
        from demo_pygame import run_interactive_demo
        run_interactive_demo()

    if args.demo_web:
        print("\nLaunching Streamlit Dashboard...")
        os.system("streamlit run src/visualization/streamlit_app.py")

if __name__ == "__main__":
    main()
