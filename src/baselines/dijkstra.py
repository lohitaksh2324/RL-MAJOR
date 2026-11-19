"""
Dijkstra's Algorithm Baseline for 3x3 Superblock City Evacuation.
Computes uniform-cost shortest path across city roads to exits.
"""

import heapq
from typing import List, Tuple, Optional
from ..config import (
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT,
    ACTION_DELTAS
)
from ..env.evacuation_env import DisasterEvacuationEnv

class DijkstraPlanner:
    def __init__(self, replan_dynamically: bool = False, avoid_hazards: bool = True):
        self.replan_dynamically = replan_dynamically
        self.avoid_hazards = avoid_hazards
        self.planned_path: List[Tuple[int, int]] = []
        self.path_index: int = 0

    def find_path_to_exits(
        self,
        env: DisasterEvacuationEnv,
        start: Tuple[int, int],
        exits: List[Tuple[int, int]],
        avoid_fire: bool = True
    ) -> Optional[List[Tuple[int, int]]]:
        """Uniform-cost search from start to the nearest reachable exit."""
        frontier = []
        heapq.heappush(frontier, (0.0, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        reached_exit = None

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            # Check if an exit is reached
            if current in exits:
                if not avoid_fire or not env.disaster.is_blocked(*current):
                    reached_exit = current
                    break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not env.grid_map.is_passable(nr, nc):
                    continue

                if env.disaster.is_blocked(nr, nc):
                    continue

                step_cost = 1.0
                if self.avoid_hazards and neighbor in env.disaster.smoke_cells:
                    step_cost += 2.0
                if self.avoid_hazards:
                    step_cost += env.crowd.get_cell_congestion(nr, nc) * 3.0

                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))
                    came_from[neighbor] = current

        if not reached_exit or reached_exit not in came_from:
            return None

        # Reconstruct path
        path = []
        curr = reached_exit
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    def plan_initial_path(self, env: DisasterEvacuationEnv):
        self.planned_path = self.find_path_to_exits(
            env, env.agent_pos, env.grid_map.exits, avoid_fire=self.avoid_hazards
        ) or []
        self.path_index = 0

    def select_action(self, env: DisasterEvacuationEnv) -> int:
        if self.replan_dynamically:
            new_path = self.find_path_to_exits(
                env, env.agent_pos, env.grid_map.exits, avoid_fire=self.avoid_hazards
            )
            if new_path:
                self.planned_path = new_path
                self.path_index = 0
        elif not self.planned_path:
            self.plan_initial_path(env)

        if not self.planned_path or self.path_index >= len(self.planned_path):
            return ACTION_WAIT

        next_pos = self.planned_path[self.path_index]
        self.path_index += 1

        dr = next_pos[0] - env.agent_pos[0]
        dc = next_pos[1] - env.agent_pos[1]

        for act, (adr, adc) in ACTION_DELTAS.items():
            if (dr, dc) == (adr, adc):
                return act

        return ACTION_WAIT
