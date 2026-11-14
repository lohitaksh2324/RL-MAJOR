"""
A* Pathfinding Baseline for 3x3 Superblock City Evacuation.
Supports static single-shot planning as well as dynamic replanning when buildings collapse or cars explode.
"""

import heapq
from typing import List, Tuple, Optional, Set
import numpy as np
from ..config import (
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT,
    ACTION_DELTAS
)
from ..env.evacuation_env import DisasterEvacuationEnv

class AStarPlanner:
    def __init__(self, replan_dynamically: bool = False, avoid_hazards: bool = True):
        self.replan_dynamically = replan_dynamically
        self.avoid_hazards = avoid_hazards
        self.planned_path: List[Tuple[int, int]] = []
        self.path_index: int = 0

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path_to_exits(
        self,
        env: DisasterEvacuationEnv,
        start: Tuple[int, int],
        exits: List[Tuple[int, int]],
        avoid_fire: bool = True
    ) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path from start to the closest reachable exit using A*."""
        best_path = None
        min_cost = float("inf")

        for exit_pos in exits:
            # Skip exit if it is blocked
            if avoid_fire and env.disaster.is_blocked(*exit_pos):
                continue

            path = self._astar_search(env, start, exit_pos, avoid_fire)
            if path and len(path) < min_cost:
                min_cost = len(path)
                best_path = path

        return best_path

    def _astar_search(
        self,
        env: DisasterEvacuationEnv,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        avoid_fire: bool
    ) -> Optional[List[Tuple[int, int]]]:
        frontier = []
        heapq.heappush(frontier, (0.0, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                # Check passability
                if not env.grid_map.is_passable(nr, nc):
                    continue

                # Blocked by any hazard (car blast, rubble, barricade)
                if env.disaster.is_blocked(nr, nc):
                    continue

                # Step cost (higher for smoke/congestion)
                step_cost = 1.0
                if self.avoid_hazards and neighbor in env.disaster.smoke_cells:
                    step_cost += 2.0
                if self.avoid_hazards:
                    step_cost += env.crowd.get_cell_congestion(nr, nc) * 3.0

                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return None

        # Reconstruct path
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    def plan_initial_path(self, env: DisasterEvacuationEnv):
        """Compute initial path at the beginning of the episode."""
        self.planned_path = self.find_path_to_exits(
            env, env.agent_pos, env.grid_map.exits, avoid_fire=self.avoid_hazards
        ) or []
        self.path_index = 0

    def select_action(self, env: DisasterEvacuationEnv) -> int:
        """
        Determine the next action to take.
        If replanning dynamically, re-computes route if current path is blocked.
        """
        if self.replan_dynamically:
            # Recompute path considering current dynamic hazards
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
