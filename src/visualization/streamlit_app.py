"""
Interactive Streamlit Dashboard for 3x3 Superblock City Evacuation.
Features:
- Visual rendering of 9 building blocks (B1 to B9) with clean labels
- Barricades with 'BLOCKED' warning labels (no lava)
- Mode toggle: [Static Environment] vs [Dynamic Environment (Sudden Roadblock)]
- Algorithm choices: [Dijkstra], [A* Search], [PPO (Adaptive RL)]
- Dynamic adaptive rerouting visualization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from stable_baselines3 import PPO

from src.env.evacuation_env import DisasterEvacuationEnv
from src.config import ScenarioConfig
from src.baselines.astar import AStarPlanner
from src.baselines.dijkstra import DijkstraPlanner
from src.config import CELL_BUILDING, CELL_EXIT

st.set_page_config(
    page_title="City Evacuation: Static vs Dynamic Rerouting",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 3x3 City Block Evacuation Simulator")
st.markdown(
    """
    **Interactive Autonomous Routing & Dynamic Obstacle Avoidance**
    - **Static Environment**: Route is pre-planned to avoid known stationary roadworks.
    - **Dynamic Environment**: While moving toward the exit, a **barricade marked 'BLOCKED'** suddenly cuts off the active path. The agent detects the blockage in real-time, recalculates, and redirects through open alternative avenues!
    """
)

tab_sim, tab_benchmark, tab_about = st.tabs(["🎮 Interactive Simulation", "📊 Benchmark & Analytics", "📖 System Overview"])

with tab_sim:
    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        st.subheader("1. Environment Mode")
        env_mode = st.radio(
            "Select Environment Mode:",
            ["Dynamic Environment (Sudden Roadblock)", "Static Environment (Fixed Route)"],
            index=0
        )
        is_dynamic = ("Dynamic" in env_mode)

        st.subheader("2. Routing Algorithm")
        planner_choice = st.radio(
            "Select Routing Algorithm:",
            ["Dijkstra", "A* Search", "PPO (Adaptive RL)"],
            index=0
        )

        st.subheader("3. Simulation Settings")
        max_steps = st.slider("Step Cap:", min_value=20, max_value=120, value=70, step=10)
        auto_block_step = st.slider("Dynamic Blockage Trigger (Step #):", min_value=2, max_value=20, value=6) if is_dynamic else -1
        random_seed = st.number_input("Random Seed:", min_value=0, max_value=9999, value=42)

        run_btn = st.button("🚀 Run Evacuation Simulation", type="primary")

    with col_view:
        st.subheader("City Grid Map & Live Evacuation Route")
        canvas_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if run_btn:
        scen = ScenarioConfig(
            name="Dynamic_Env" if is_dynamic else "Static_Env",
            description="Dynamic roadblock" if is_dynamic else "Static environment",
            is_dynamic=is_dynamic,
            fire_start_positions=[],
            static_hazards=[(15, 8), (15, 9)] if not is_dynamic else [],
            exit_congestions={0: 0.1, 1: 0.1, 2: 0.1}
        )

        env = DisasterEvacuationEnv(scenario=scen, max_steps=max_steps)
        obs, _ = env.reset(seed=int(random_seed))

        # Setup Planner
        planner = None
        ppo_model = None
        if planner_choice == "PPO (Adaptive RL)":
            model_path = "models/ppo_evac.zip"
            if os.path.exists(model_path):
                ppo_model = PPO.load(model_path)
            else:
                st.warning(f"Trained PPO model not found at `{model_path}`! Using fallback.")
        elif planner_choice == "Dijkstra":
            planner = DijkstraPlanner(replan_dynamically=is_dynamic)
            planner.plan_initial_path(env)
        elif planner_choice == "A* Search":
            planner = AStarPlanner(replan_dynamically=is_dynamic)
            planner.plan_initial_path(env)

        agent_trajectory = [env.agent_pos]
        done = False
        step = 0
        total_reward = 0.0
        block_happened = False
        reroute_happened = False

        with st.spinner("Simulating city routing dynamics..."):
            while not done and step < max_steps:
                # Dynamic blockage trigger
                if is_dynamic and not block_happened and step == auto_block_step:
                    target_path = planner.planned_path[planner.path_index:] if (planner and planner.planned_path) else []
                    blk = env.disaster.block_path_ahead(env.agent_pos, target_path)
                    if blk:
                        block_happened = True
                        if planner:
                            new_path = planner.find_path_to_exits(env, env.agent_pos, env.grid_map.exits, avoid_fire=True)
                            if new_path:
                                planner.planned_path = new_path
                                planner.path_index = 0
                                reroute_happened = True

                if ppo_model is not None:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                elif planner is not None:
                    action = planner.select_action(env)
                else:
                    action = env.action_space.sample()

                obs, reward, term, trunc, info = env.step(int(action))
                total_reward += reward
                agent_trajectory.append(env.agent_pos)
                step += 1
                done = term or trunc

        # Render Final Simulation Canvas
        fig, ax = plt.subplots(figsize=(9, 9))
        grid_map = env.grid_map
        disaster = env.disaster

        ax.set_xlim(0, grid_map.width)
        ax.set_ylim(grid_map.height, 0)
        ax.set_facecolor("#1e222b")

        # 1. Draw Asphalt Roads
        for r in range(grid_map.height):
            for c in range(grid_map.width):
                if grid_map.grid[r, c] != CELL_BUILDING:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor="#2b303a", edgecolor="#3a4250", linewidth=0.5))

        # 2. Draw 9 Building Blocks (B1 to B9) - Clean labels
        for b_name, (r1, r2, c1, c2) in grid_map.building_blocks.items():
            w = c2 - c1 + 1
            h = r2 - r1 + 1
            b_color = "#3a4b5c" if b_name == "B5" else "#34495e"
            ax.add_patch(patches.Rectangle((c1, r1), w, h, facecolor=b_color, edgecolor="#202b35", linewidth=2))
            ax.text(c1 + w / 2, r1 + h / 2, b_name, color="#ecf0f1", fontsize=11, fontweight="bold",
                    ha="center", va="center")

        # 3. Draw Exits
        for er, ec in grid_map.exits:
            ax.add_patch(patches.Rectangle((ec, er), 1, 1, facecolor="#2ecc71", edgecolor="#27ae60", linewidth=2))
            ax.text(ec + 0.5, er + 0.5, "EXIT", color="#0e3d1e", fontsize=7, fontweight="bold", ha="center", va="center")

        # 4. Draw Barricades with 'BLOCKED' Text (NO lava)
        all_barricades = disaster.barricade_cells.union(disaster.rubble_cells).union(disaster.car_wrecks).union(disaster.static_hazards)
        for r, c in all_barricades:
            # Orange hazard barricade board
            ax.add_patch(patches.Rectangle((c + 0.05, r + 0.2), 0.9, 0.6, facecolor="#f39c12", edgecolor="#d35400", linewidth=1.5))
            # Red center badge
            ax.add_patch(patches.Rectangle((c + 0.12, r + 0.32), 0.76, 0.36, facecolor="#c0392b", edgecolor="#ffffff", linewidth=1))
            ax.text(c + 0.5, r + 0.5, "BLOCKED", color="#ffffff", fontsize=5.5, fontweight="bold", ha="center", va="center")

        # 5. Draw Civilian Cars
        for car in disaster.cars:
            if not car.is_exploded and (car.row, car.col) not in all_barricades:
                ax.add_patch(patches.Rectangle((car.col + 0.15, car.row + 0.25), 0.7, 0.5, facecolor="#f1c40f", edgecolor="#2c3e50"))

        # 6. Draw Evacuation Trajectory
        if len(agent_trajectory) > 1:
            traj_c = [p[1] + 0.5 for p in agent_trajectory]
            traj_r = [p[0] + 0.5 for p in agent_trajectory]
            ax.plot(traj_c, traj_r, color="#00d2d3", linewidth=3.5, marker="o", markersize=3, label="Evacuation Route")

        # Mark Start and Final
        sr, sc = agent_trajectory[0]
        er, ec = agent_trajectory[-1]
        ax.plot(sc + 0.5, sr + 0.5, marker="*", markersize=14, color="#f1c40f", label="Start Point")
        final_color = "#2ecc71" if info.get("evacuated") else "#e74c3c"
        ax.plot(ec + 0.5, er + 0.5, marker="X", markersize=12, color=final_color, label="Final Position")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=9)
        canvas_placeholder.pyplot(fig)
        plt.close(fig)

        # Status & Metrics
        evac_status = "✅ Successfully Evacuated to Safe Haven" if info.get("evacuated") else ("🚫 Stopped / Blocked by Barricade" if term else "⏳ Timed Out")
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outcome", evac_status)
            m2.metric("Routing Algorithm", planner_choice)
            m3.metric("Steps Taken", f"{step} steps")
            m4.metric("Dynamic Reroute", "Yes (Adaptive)" if reroute_happened else ("N/A" if not is_dynamic else "Blocked"))

with tab_benchmark:
    st.subheader("Algorithmic Benchmark Comparison")
    p_path = "results/benchmark_comparison.png"
    if os.path.exists(p_path):
        st.image(p_path, caption="Benchmark Comparison across Scenarios", use_container_width=True)
    b_path = "results/benchmark_results.json"
    if os.path.exists(b_path):
        with open(b_path, "r") as f:
            b_data = json.load(f)
        st.dataframe(b_data, use_container_width=True)

with tab_about:
    st.subheader("System Architecture")
    st.markdown(
        """
        - **3x3 Superblock Grid (B1 to B9)**: Clean modern city layout with interconnecting avenues and streets.
        - **Real-Time Barricades with 'BLOCKED' Sign**: Realistic roadwork and emergency hazard barricades replace spreading lava.
        - **Multi-Algorithm Routing**: Switch between **Dijkstra**, **A***, and **PPO (Adaptive RL)** with one click.
        - **Dynamic Sudden Blockage & Rerouting**: When a barricade drops right ahead of the vehicle, the system immediately recalculates a new bypass route to the exit!
        """
    )
