"""
Custom Gymnasium Environment for 3x3 Superblock City Evacuation.
Features:
- Topological Geodesic BFS Distance Potential for reward shaping
- Topological direction vector in observation (explicit detour compass - key fix for oscillation)
- Anti-oscillation reward penalty for immediately reversing direction
- Diverse urban hazards: Car Blasts, Building Collapses, and Road Barricades
- 122-dimensional state space
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional, List

from ..config import (
    GRID_WIDTH, GRID_HEIGHT,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT,
    ACTION_DELTAS,
    STEP_PENALTY, GOAL_REWARD, FIRE_PENALTY, SMOKE_PENALTY,
    RUBBLE_PENALTY, WALL_COLLISION_PENALTY,
    PROGRESS_REWARD_WEIGHT, CONGESTION_PENALTY_WEIGHT,
    MAX_STEPS_PER_EPISODE,
    CELL_BUILDING, CELL_EXIT, ScenarioConfig
)
from .grid_map import GridMap
from .disaster_model import DisasterModel
from .crowd_model import CrowdModel

# Anti-oscillation penalty weight
ANTI_OSCILLATION_PENALTY = -3.0
# Reward for moving closer (on top of geodesic progress)
DETOUR_BONUS = 1.5


class DisasterEvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_width: int = GRID_WIDTH,
        grid_height: int = GRID_HEIGHT,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        scenario: Optional[ScenarioConfig] = None,
        render_mode: Optional[str] = None,
        local_window_radius: int = 2  # 5x5 local window (radius 2)
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.scenario = scenario
        self.render_mode = render_mode
        self.local_radius = local_window_radius
        self.local_size = 2 * local_window_radius + 1  # 5

        # Initialize sub-components
        self.grid_map = GridMap(width=grid_width, height=grid_height)
        self.disaster = DisasterModel(grid_map=self.grid_map)
        self.crowd = CrowdModel(grid_map=self.grid_map)

        # Agent state
        self.agent_pos: Tuple[int, int] = (22, 22)
        self.prev_distance_to_nearest_exit = 0.0
        self.current_step = 0
        self.cumulative_hazard_exposure = 0.0
        self.cumulative_congestion = 0.0
        self.car_blast_count = 0
        self.building_crash_count = 0

        # Anti-oscillation: keep last N positions to detect back-and-forth
        self._pos_history: deque = deque(maxlen=8)
        self._prev_pos: Optional[Tuple[int, int]] = None

        # Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: WAIT
        self.action_space = spaces.Discrete(5)

        # Observation space formulation (dim = 122):
        # 1. Agent pos normalized: 2
        # 2. Exits info (3 exits * 5 = 15)
        # 3. Topological direction vector (topo_dr, topo_dc, geo_dist_norm, stuck): 4
        # 4. Local 5x5 window (4 channels: wall, car blast/wreck, rubble/barricade, crowd) = 100
        # 5. Normalized step count: 1
        # Total: 2 + 15 + 4 + 100 + 1 = 122
        obs_dim = 2 + (len(self.grid_map.exits) * 5) + 4 + (self.local_size * self.local_size * 4) + 1
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # BFS helpers
    # ------------------------------------------------------------------

    def _get_topological_direction(self, pos: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Returns the unit direction (dr, dc) of the FIRST STEP on the unblocked
        BFS shortest path to the nearest open exit.
        This gives the network an explicit detour compass — the key fix for oscillation.

        Returns: (topo_dr, topo_dc, geo_dist_norm, stuck_flag)
          topo_dr, topo_dc ∈ {-1, 0, 1}  — direction of first step on optimal path
          geo_dist_norm ∈ [0, 1]          — normalised geodesic distance to exit
          stuck_flag ∈ {0, 1}             — 1 if completely trapped (no open path)
        """
        r, c = pos
        targets = set(e for e in self.grid_map.exits if not self.disaster.is_blocked(*e))
        if not targets:
            targets = set(self.grid_map.exits)

        # BFS that stores parent pointers
        visited: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {pos: None}
        queue: deque = deque([pos])
        first_step_cell: Optional[Tuple[int, int]] = None

        while queue:
            curr = queue.popleft()
            if curr in targets and curr != pos:
                # Backtrack to find the first step taken from `pos`
                node = curr
                while visited[node] != pos and visited[node] is not None:
                    node = visited[node]
                first_step_cell = node
                break
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + ddr, curr[1] + ddc
                neighbor = (nr, nc)
                if neighbor not in visited:
                    if self.grid_map.is_passable(nr, nc) and not self.disaster.is_blocked(nr, nc):
                        visited[neighbor] = curr
                        queue.append(neighbor)

        if first_step_cell is not None:
            topo_dr = float(first_step_cell[0] - r)  # ∈ {-1, 0, 1}
            topo_dc = float(first_step_cell[1] - c)  # ∈ {-1, 0, 1}
            stuck = 0.0
        else:
            topo_dr = 0.0
            topo_dc = 0.0
            stuck = 1.0

        geo_dist = self._get_distance_to_nearest_safe_exit(pos)
        geo_dist_norm = min(geo_dist / float(self.grid_height + self.grid_width), 1.0)

        return topo_dr, topo_dc, geo_dist_norm, stuck

    def _get_distance_to_nearest_safe_exit(self, pos: Tuple[int, int]) -> float:
        """
        True topological geodesic distance (unblocked BFS shortest path)
        to the nearest reachable exit. Used for potential-based reward shaping.
        """
        r, c = pos
        if pos in self.grid_map.exits or pos in self.grid_map.safe_zones:
            return 0.0

        targets = set(e for e in self.grid_map.exits if not self.disaster.is_blocked(*e))
        if not targets:
            targets = set(self.grid_map.exits)

        visited: Dict[Tuple[int, int], int] = {pos: 0}
        queue: deque = deque([pos])

        while queue:
            curr = queue.popleft()
            curr_d = visited[curr]

            if curr in targets:
                return float(curr_d)

            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + ddr, curr[1] + ddc
                neighbor = (nr, nc)
                if neighbor not in visited:
                    if self.grid_map.is_passable(nr, nc) and not self.disaster.is_blocked(nr, nc):
                        visited[neighbor] = curr_d + 1
                        queue.append(neighbor)

        # Fallback: no open path exists
        min_manhattan = min(abs(r - er) + abs(c - ec) for er, ec in targets)
        return float(min_manhattan + 50.0)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = []
        ar, ac = self.agent_pos

        # 1. Agent position normalised [0, 1]
        obs.append(ar / float(self.grid_height))
        obs.append(ac / float(self.grid_width))

        # 2. Information about each exit (relative offset, dist, blocked flag, crowd)
        for exit_idx, (er, ec) in enumerate(self.grid_map.exits):
            rel_dr = (er - ar) / float(self.grid_height)
            rel_dc = (ec - ac) / float(self.grid_width)
            dist = (abs(er - ar) + abs(ec - ac)) / float(self.grid_height + self.grid_width)
            is_blocked = 1.0 if self.disaster.is_blocked(er, ec) else 0.0
            congestion = self.crowd.get_exit_congestion(exit_idx)
            obs.extend([rel_dr, rel_dc, dist, is_blocked, congestion])

        # 3. Topological direction vector (explicit detour compass)
        topo_dr, topo_dc, geo_dist_norm, stuck = self._get_topological_direction(self.agent_pos)
        obs.extend([topo_dr, topo_dc, geo_dist_norm, stuck])

        # 4. Local sensing window (5x5 around agent)
        # Channels:
        # Ch0: Solid building walls and perimeter barriers
        # Ch1: Car blasts & burning wrecks
        # Ch2: Collapsed building rubble & barricades
        # Ch3: Traffic/Crowd congestion
        for dr in range(-self.local_radius, self.local_radius + 1):
            for dc in range(-self.local_radius, self.local_radius + 1):
                r = ar + dr
                c = ac + dc
                if not self.grid_map.is_valid_cell(r, c):
                    obs.extend([1.0, 0.0, 0.0, 0.0])  # Boundary barrier
                else:
                    is_building = 1.0 if self.grid_map.grid[r, c] == CELL_BUILDING else 0.0
                    is_car_blast = 1.0 if (
                        (r, c) in self.disaster.car_wrecks or
                        self.disaster.hazard_types.get((r, c)) == "CAR_BLAST"
                    ) else 0.0
                    is_rubble_or_barricade = 1.0 if (
                        (r, c) in self.disaster.rubble_cells or
                        (r, c) in self.disaster.barricade_cells
                    ) else 0.0
                    crowd_val = self.crowd.get_cell_congestion(r, c)
                    obs.extend([is_building, is_car_blast, is_rubble_or_barricade, crowd_val])

        # 5. Normalised time step
        obs.append(self.current_step / float(self.max_steps))

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.disaster.rng = np.random.default_rng(seed)
            self.crowd.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.cumulative_hazard_exposure = 0.0
        self.cumulative_congestion = 0.0
        self.car_blast_count = 0
        self.building_crash_count = 0
        self._pos_history.clear()
        self._prev_pos = None

        # Configure disaster based on scenario
        initial_fire = self.scenario.fire_start_positions if self.scenario else None
        static_hazards = self.scenario.static_hazards if self.scenario else None
        cars = self.scenario.cars if self.scenario else None
        exit_congestions = self.scenario.exit_congestions if self.scenario else {0: 0.1, 1: 0.1, 2: 0.1}

        self.disaster.reset(initial_fire=initial_fire, static_hazards=static_hazards, cars=cars)
        self.crowd.reset(exit_congestions=exit_congestions)

        # Agent spawn: spawn at STARTING SECTOR (bottom-left) or far points across city
        exit_r, exit_c = self.grid_map.exits[0] if self.grid_map.exits else (3, 32)
        if self.scenario is not None:
            # Starting sector at bottom-left, plus distant points on southern and western arteries
            scenario_spawns = [
                (21, 3),   # Primary START SECTOR origin (bottom-left)
                (21, 2),   # Staging bay 1
                (22, 2),   # Staging bay 2
                (21, 10),  # South-West intersection
                (21, 18),  # South-Central avenue
                (15, 3),   # Mid-West street
                (15, 10),  # Mid-West crossroad
            ]
            valid_spawns = [
                p for p in scenario_spawns
                if self.grid_map.is_passable(*p) and not self.disaster.is_blocked(*p)
                and (abs(p[0] - exit_r) + abs(p[1] - exit_c) >= 15)
            ]
            if valid_spawns:
                self.agent_pos = valid_spawns[int(self.np_random.integers(0, len(valid_spawns)))]
            else:
                self.agent_pos = (21, 3)
        else:
            # General training: spawn across road cells with distance >= 15 from exit
            open_roads = [
                p for p in self.grid_map.road_cells
                if not self.disaster.is_blocked(*p)
                and p not in self.grid_map.exits
                and (abs(p[0] - exit_r) + abs(p[1] - exit_c) >= 15)
            ]
            if open_roads:
                self.agent_pos = open_roads[int(self.np_random.integers(0, len(open_roads)))]
            else:
                self.agent_pos = (21, 3)

        self._pos_history.append(self.agent_pos)
        self._prev_pos = self.agent_pos
        self.prev_distance_to_nearest_exit = self._get_distance_to_nearest_safe_exit(self.agent_pos)

        info = {
            "initial_pos": self.agent_pos,
            "exits": list(self.grid_map.exits)
        }
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action = int(action)
        dr, dc = ACTION_DELTAS.get(action, (0, 0))
        target_r = self.agent_pos[0] + dr
        target_c = self.agent_pos[1] + dc

        reward = STEP_PENALTY
        terminated = False
        truncated = False
        wall_bumped = False
        prev_pos = self.agent_pos

        # 1. Attempt movement
        if not self.grid_map.is_passable(target_r, target_c):
            wall_bumped = True
            reward += WALL_COLLISION_PENALTY
        elif self.disaster.is_blocked(target_r, target_c):
            wall_bumped = True
            reward += RUBBLE_PENALTY
        else:
            self.agent_pos = (target_r, target_c)

        # 2. Anti-oscillation penalty:
        #    Heavy penalty if agent returns to the cell it was at 2 steps ago (pure back-and-forth)
        if len(self._pos_history) >= 2:
            two_steps_ago = list(self._pos_history)[-2]
            if self.agent_pos == two_steps_ago and self.agent_pos != prev_pos:
                reward += ANTI_OSCILLATION_PENALTY

        # Lighter penalty for revisiting any cell in the recent 6-step window
        if self.agent_pos != prev_pos and self.agent_pos in self._pos_history:
            reward += ANTI_OSCILLATION_PENALTY * 0.5

        self._pos_history.append(self.agent_pos)
        self._prev_pos = prev_pos

        # 3. Check dynamic building collapse triggers in scenario
        if self.scenario and self.scenario.is_dynamic:
            for crash_spec in self.scenario.building_crashes:
                if crash_spec.crashes_at_step == self.current_step:
                    self.disaster.trigger_building_collapse(
                        crash_spec.building_id,
                        crash_spec.rubble_cells[0] if crash_spec.rubble_cells else None
                    )
                    self.building_crash_count += 1
                    self.prev_distance_to_nearest_exit = self._get_distance_to_nearest_safe_exit(self.agent_pos)

        # 4. Advance disaster and crowd simulation
        old_blasted_count = len(self.disaster.car_wrecks)
        self.disaster.step(self.current_step)
        new_blasted_count = len(self.disaster.car_wrecks)
        if new_blasted_count > old_blasted_count:
            self.car_blast_count += (new_blasted_count - old_blasted_count)

        self.crowd.step()

        # 5. Check hazard exposure at new agent position
        curr_pos = self.agent_pos
        if self.disaster.is_blocked(*curr_pos):
            reward += FIRE_PENALTY
            self.cumulative_hazard_exposure += 1.0
            terminated = True
        elif curr_pos in self.disaster.smoke_cells:
            reward += SMOKE_PENALTY
            self.cumulative_hazard_exposure += 0.25

        # 6. Check crowd/traffic congestion
        cell_congestion = self.crowd.get_cell_congestion(*curr_pos)
        self.cumulative_congestion += cell_congestion
        reward += CONGESTION_PENALTY_WEIGHT * cell_congestion

        # 7. Check arrival at SAFE zone / exit
        evacuated = False
        exit_reached = None
        if not terminated and (curr_pos in self.grid_map.exits or curr_pos in self.grid_map.safe_zones):
            if curr_pos in self.grid_map.exits:
                exit_idx = self.grid_map.exits.index(curr_pos)
            else:
                exit_idx = 0
            exit_reached = exit_idx

            if self.disaster.is_blocked(*curr_pos):
                reward += FIRE_PENALTY
                terminated = True
            else:
                cong = self.crowd.get_exit_congestion(exit_idx)
                reward += GOAL_REWARD - (cong * 15.0)
                evacuated = True
                terminated = True

        # 8. Potential-based geodesic progress shaping (with bonus for actual forward progress)
        if not terminated:
            curr_dist = self._get_distance_to_nearest_safe_exit(curr_pos)
            raw_progress = self.prev_distance_to_nearest_exit - curr_dist
            progress = float(np.clip(raw_progress, -2.0, 2.0))
            reward += PROGRESS_REWARD_WEIGHT * progress
            # Extra bonus for strictly getting closer (motivates taking the detour)
            if raw_progress > 0:
                reward += DETOUR_BONUS * min(raw_progress, 1.0)
            self.prev_distance_to_nearest_exit = curr_dist

        # 9. Episode truncation
        if not terminated and self.current_step >= self.max_steps:
            truncated = True

        info = {
            "evacuated": evacuated,
            "exit_reached": exit_reached,
            "wall_bumped": wall_bumped,
            "steps": self.current_step,
            "hazard_exposure": self.cumulative_hazard_exposure,
            "congestion_encountered": self.cumulative_congestion,
            "car_blasts_during_ep": self.car_blast_count,
            "building_crashes_during_ep": self.building_crash_count,
        }
        return self._get_obs(), reward, terminated, truncated, info
