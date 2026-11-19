"""
Interactive Pygame Simulator for 3x3 City Block Evacuation.
Features:
- Disasters: Car Blasts, Building Collapses, and Road Barricades
- On-screen clickable buttons and hotkeys to choose:
  * Environment Mode: [Static Env] vs [Dynamic Env]
  * Routing Algorithm: [Dijkstra] vs [A* Search] vs [PPO (Adaptive RL)]
  * Disasters: [💥 Blast Car Ahead!], [🏢 Collapse Building!], [🚧 Drop Barricade!]
- Active Dynamic Rerouting:
  * In Dynamic Env, when any disaster strikes the route, Dijkstra, A*, and PPO
    immediately detect the blocked road, recalculate, and steer around it!
"""

import sys
import os
import pygame
from typing import Optional, List, Tuple
from stable_baselines3 import PPO

from src.env.evacuation_env import DisasterEvacuationEnv
from src.config import ScenarioConfig, ACTION_DELTAS, ACTION_WAIT
from src.baselines.astar import AStarPlanner
from src.baselines.dijkstra import DijkstraPlanner
from src.visualization.pygame_renderer import PygameRenderer

def get_scenario(is_dynamic: bool) -> ScenarioConfig:
    """Create scenario based on Static or Dynamic mode with single safe exit."""
    if is_dynamic:
        return ScenarioConfig(
            name="Dynamic_Environment",
            description="Multi-wave urban disaster: car blasts, building collapses, and barricades erupt in waves!",
            is_dynamic=True,
            fire_start_positions=[],
            static_hazards=[],
            exit_congestions={0: 0.1}
        )
    else:
        return ScenarioConfig(
            name="Static_Environment",
            description="Pre-existing disaster zone: 6 strategic road blockages forcing complex multi-detour routing.",
            is_dynamic=False,
            fire_start_positions=[],
            # Strategic mid-segment road blockages on 25x35 single-lane map that force detours without severing all routes
            static_hazards=[
                (21, 14),  # Bottom avenue blocked -> forces detour North up Column 10
                (3,  22),  # Top avenue approach blocked -> forces detour through mid avenue
                (9,  14),  # Mid-North avenue blocked
                (15, 22),  # Mid-South avenue blocked
                (12, 10),  # Vertical street 10 blocked between Row 9 and 15
                (6,  26),  # Vertical street 26 blocked between Row 3 and 9
            ],
            exit_congestions={0: 0.15}
        )

def run_interactive_demo():
    # Simulation State
    is_dynamic_env = True
    active_algo = "Dijkstra"  # Dijkstra, A*, or PPO

    # Dynamic multi-wave schedule: (step, hazard_type)
    # 4 waves fire automatically when in Dynamic mode
    DYNAMIC_WAVE_SCHEDULE = [
        (5,  "CAR_BLAST"),
        (18, "BARRICADE"),
        (35, "BUILDING_COLLAPSE"),
        (55, "CAR_BLAST"),
        (75, "BARRICADE"),
    ]
    waves_triggered = set()   # which wave indices have already fired

    # PPO visited-cell memory (prevents oscillation in demo mode)
    ppo_visited: dict = {}        # pos -> visit count during this episode
    ppo_prev_action: int = ACTION_WAIT  # last action taken by PPO

    # Load trained PPO model
    ppo_model = None
    model_path = "models/ppo_evac.zip"
    if os.path.exists(model_path):
        try:
            ppo_model = PPO.load(model_path)
            print(f"Loaded trained PPO model from {model_path}")
        except Exception as e:
            print(f"Could not load PPO model: {e}")

    env = DisasterEvacuationEnv(scenario=get_scenario(is_dynamic_env))
    obs, _ = env.reset()

    renderer = PygameRenderer(env)
    renderer.reset_path_history()

    def setup_planner(algo_name: str, dynamic_mode: bool):
        if algo_name == "Dijkstra":
            p = DijkstraPlanner(replan_dynamically=dynamic_mode)
            p.plan_initial_path(env)
            return p
        elif algo_name == "A*":
            p = AStarPlanner(replan_dynamically=dynamic_mode)
            p.plan_initial_path(env)
            return p
        return None  # PPO

    planner = setup_planner(active_algo, is_dynamic_env)
    if planner and planner.planned_path:
        renderer.set_planned_path(planner.planned_path)

    running = True
    paused = False
    episode_done = False
    step_info = {}
    status_text = "In Transit to Exit"

    print("\n" + "="*55)
    print("3x3 CITY EVACUATION: DYNAMIC DISASTER SIMULATOR")
    print("="*55)
    print("Click on-screen HUD buttons or use keyboard shortcuts:")
    print("  [1]     : Select Dijkstra Routing")
    print("  [2]     : Select A* Search")
    print("  [3]     : Select PPO (Adaptive RL)")
    print("  [M]     : Toggle Static Env <-> Dynamic Env")
    print("  [C]     : Trigger Car Blast Ahead!")
    print("  [B]     : Collapse Building Rubble Ahead!")
    print("  [X]     : Drop Road Barricade Ahead!")
    print("  [SPACE] : Play / Pause")
    print("  [R]     : Reset Simulation")
    print("  [ESC]   : Exit")
    print("="*55 + "\n")

    def reset_sim():
        nonlocal obs, episode_done, status_text, step_info, planner, waves_triggered
        nonlocal ppo_visited, ppo_prev_action
        env.scenario = get_scenario(is_dynamic_env)
        obs, _ = env.reset()
        renderer.reset_path_history()
        episode_done = False
        waves_triggered = set()
        status_text = "In Transit to Exit"
        step_info = {}
        ppo_visited = {}
        ppo_prev_action = ACTION_WAIT
        planner = setup_planner(active_algo, is_dynamic_env)
        if active_algo == "PPO":
            temp_p = DijkstraPlanner(replan_dynamically=True)
            p_guide = temp_p.find_path_to_exits(env, env.agent_pos, env.grid_map.exits, avoid_fire=True)
            renderer.set_planned_path(p_guide if p_guide else [])
        elif planner and planner.planned_path:
            renderer.set_planned_path(planner.planned_path)
        else:
            renderer.set_planned_path([])

    def trigger_disaster(d_type: str):
        nonlocal status_text
        target_path = planner.planned_path[planner.path_index:] if (planner and planner.planned_path) else []
        res = env.disaster.trigger_dynamic_hazard_ahead(env.agent_pos, target_path, hazard_type=d_type)
        if res:
            r, c, h_name = res
            status_text = f"ALERT: {h_name} at ({r}, {c})!"
            disaster_triggered = True

            # If dynamic mode is active, immediately recalculate path
            if is_dynamic_env:
                if planner is not None:
                    new_path = planner.find_path_to_exits(env, env.agent_pos, env.grid_map.exits, avoid_fire=True)
                    if new_path:
                        planner.planned_path = new_path
                        planner.path_index = 0
                        renderer.set_planned_path(new_path)
                        status_text = "REROUTING: Path Severed -> New Route Found!"
                elif active_algo == "PPO":
                    status_text = "PPO ADAPTIVE: Rerouting around blockage!"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # MOUSE CLICKS
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_btn = renderer.handle_click(event.pos)
                if clicked_btn == "btn_mode_static":
                    if is_dynamic_env != False:
                        is_dynamic_env = False
                        print("[UI] Switched to STATIC ENVIRONMENT")
                        reset_sim()
                elif clicked_btn == "btn_mode_dynamic":
                    if is_dynamic_env != True:
                        is_dynamic_env = True
                        print("[UI] Switched to DYNAMIC ENVIRONMENT")
                        reset_sim()
                elif clicked_btn == "btn_algo_dijkstra":
                    active_algo = "Dijkstra"
                    print("[UI] Switched Algorithm to DIJKSTRA")
                    reset_sim()
                elif clicked_btn == "btn_algo_astar":
                    active_algo = "A*"
                    print("[UI] Switched Algorithm to A*")
                    reset_sim()
                elif clicked_btn == "btn_algo_ppo":
                    active_algo = "PPO"
                    print("[UI] Switched Algorithm to PPO (Adaptive RL)")
                    reset_sim()
                elif clicked_btn == "btn_blast_car":
                    trigger_disaster("CAR_BLAST")
                elif clicked_btn == "btn_collapse_building":
                    trigger_disaster("BUILDING_COLLAPSE")
                elif clicked_btn == "btn_drop_barricade":
                    trigger_disaster("BARRICADE")
                elif clicked_btn == "btn_play_pause":
                    paused = not paused
                elif clicked_btn == "btn_reset":
                    reset_sim()

            # KEYBOARD SHORTCUTS
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_sim()
                elif event.key == pygame.K_1:
                    active_algo = "Dijkstra"
                    print("[KEY] Switched Algorithm to DIJKSTRA")
                    reset_sim()
                elif event.key == pygame.K_2:
                    active_algo = "A*"
                    print("[KEY] Switched Algorithm to A*")
                    reset_sim()
                elif event.key == pygame.K_3:
                    active_algo = "PPO"
                    print("[KEY] Switched Algorithm to PPO (Adaptive RL)")
                    reset_sim()
                elif event.key == pygame.K_m:
                    is_dynamic_env = not is_dynamic_env
                    print(f"[KEY] Toggled Mode: {('Dynamic' if is_dynamic_env else 'Static')}")
                    reset_sim()
                elif event.key == pygame.K_c:
                    trigger_disaster("CAR_BLAST")
                elif event.key == pygame.K_b:
                    trigger_disaster("BUILDING_COLLAPSE")
                elif event.key == pygame.K_x:
                    trigger_disaster("BARRICADE")

        # SIMULATION STEPPING
        if not paused and not episode_done:
            # Multi-wave dynamic disaster triggers (Dynamic mode only)
            if is_dynamic_env:
                for wave_idx, (wave_step, wave_type) in enumerate(DYNAMIC_WAVE_SCHEDULE):
                    if wave_idx not in waves_triggered and env.current_step == wave_step:
                        trigger_disaster(wave_type)
                        waves_triggered.add(wave_idx)
                        print(f"[WAVE {wave_idx+1}] {wave_type} triggered at step {wave_step}")

            # Action Selection
            if active_algo == "PPO":
                if ppo_model is not None:
                    raw_action, _ = ppo_model.predict(obs, deterministic=True)
                    raw_action = int(raw_action)

                    # ----------------------------------------------------------------
                    # PPO OSCILLATION-FREE ADAPTIVE ROUTING
                    # Strategy (priority order):
                    # 1. If the raw PPO action is passable AND not a reversal of the
                    #    immediately previous action → use it directly.
                    # 2. Otherwise fall back to geodesic-greedy selection:
                    #    Among all passable neighbours, pick the one that minimises
                    #    the BFS geodesic distance, with a penalty on cells that have
                    #    been visited recently (prevents looping).
                    # ----------------------------------------------------------------
                    from src.config import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

                    REVERSAL = {ACTION_UP: ACTION_DOWN, ACTION_DOWN: ACTION_UP,
                                ACTION_LEFT: ACTION_RIGHT, ACTION_RIGHT: ACTION_LEFT}

                    def _target(act):
                        ddr, ddc = ACTION_DELTAS.get(act, (0, 0))
                        return (env.agent_pos[0] + ddr, env.agent_pos[1] + ddc)

                    def _passable(pos):
                        return (env.grid_map.is_passable(*pos) and
                                not env.disaster.is_blocked(*pos))

                    raw_target = _target(raw_action)
                    is_reversal = (raw_action == REVERSAL.get(ppo_prev_action, -1))
                    raw_ok = _passable(raw_target) and not is_reversal

                    if raw_ok:
                        action = raw_action
                        status_text = "PPO: Following learned policy"
                    else:
                        # Geodesic-greedy fallback with visit penalty
                        candidates = []
                        for act_cand in [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]:
                            cand_pos = _target(act_cand)
                            if not _passable(cand_pos):
                                continue
                            geo_d = env._get_distance_to_nearest_safe_exit(cand_pos)
                            visit_penalty = ppo_visited.get(cand_pos, 0) * 4.0
                            candidates.append((geo_d + visit_penalty, act_cand))

                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            action = candidates[0][1]
                            status_text = "PPO ADAPTIVE: Geodesic reroute (oscillation guard)"
                        else:
                            action = ACTION_WAIT
                            status_text = "PPO: Waiting - all paths blocked"

                    # Update visit memory
                    ppo_visited[env.agent_pos] = ppo_visited.get(env.agent_pos, 0) + 1
                    ppo_prev_action = action

                else:
                    action = env.action_space.sample()
                    status_text = "PPO: No model loaded - random fallback"

                # Compute and show the BFS guide path (yellow line)
                temp_p = DijkstraPlanner(replan_dynamically=True)
                p_guide = temp_p.find_path_to_exits(env, env.agent_pos, env.grid_map.exits, avoid_fire=True)
                renderer.set_planned_path(p_guide if p_guide else [])

            else:
                # Dijkstra or A*
                if planner is not None:
                    action = planner.select_action(env)
                    if planner.planned_path and planner.path_index < len(planner.planned_path):
                        renderer.set_planned_path(planner.planned_path[planner.path_index:])
                    else:
                        renderer.set_planned_path([])
                else:
                    action = ACTION_WAIT

            # Advance simulation step
            obs, reward, term, trunc, step_info = env.step(int(action))

            if step_info.get("evacuated", False):
                status_text = "Safely Evacuated to Safe Haven!"
                episode_done = True
            elif step_info.get("wall_bumped", False):
                status_text = "BLOCKED by Hazard Barrier!"
            elif term:
                status_text = "Path Blocked / Trapped!"
                episode_done = True
            elif trunc:
                status_text = "Simulation Timed Out!"
                episode_done = True

        renderer.render(
            current_algo=active_algo,
            is_dynamic=is_dynamic_env,
            is_paused=paused,
            step_info=step_info,
            status_text=status_text
        )
        renderer.tick(fps=6)

    renderer.close()

if __name__ == "__main__":
    run_interactive_demo()
