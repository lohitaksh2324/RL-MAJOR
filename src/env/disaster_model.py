"""
Disaster model simulating diverse urban hazards:
- Car Blasts: civilian vehicles exploding into flames, smoke, and charred wreckage
- Building Collapses: building facades cracking and dumping jagged concrete rubble across streets
- Road Barricades: traffic barricades stamped with 'BLOCKED' warning signs
- Dynamic path blocking: triggers sudden catastrophes right on the agent's route to force active rerouting
"""

import numpy as np
from typing import Set, Tuple, List, Optional, Dict
from collections import deque
from ..config import CarSpec, BuildingCrashSpec
from .grid_map import GridMap

class DisasterModel:
    def __init__(
        self,
        grid_map: GridMap,
        fire_spread_prob: float = 0.0,
        smoke_radius: int = 1,
        rng: Optional[np.random.Generator] = None
    ):
        self.grid_map = grid_map
        self.rng = rng or np.random.default_rng()

        # Hazardous cell sets
        self.barricade_cells: Set[Tuple[int, int]] = set()     # Standard traffic barricades
        self.car_wrecks: Set[Tuple[int, int]] = set()          # Blasted / exploded car wrecks
        self.rubble_cells: Set[Tuple[int, int]] = set()        # Collapsed building rubble
        self.static_hazards: Set[Tuple[int, int]] = set()      # Static road construction
        self.smoke_cells: Set[Tuple[int, int]] = set()         # Billowing smoke / dust haze
        self.hazard_types: Dict[Tuple[int, int], str] = {}     # (r, c) -> "CAR_BLAST" | "BUILDING_COLLAPSE" | "BARRICADE"

        self.cars: List[CarSpec] = []                          # Active vehicles on roads
        self.crashed_buildings: Set[str] = set()               # Labeled buildings that crashed
        self.recent_events: List[str] = []                     # Event logs for UI HUD

    @property
    def fire_cells(self) -> Set[Tuple[int, int]]:
        """Alias for backward compatibility with RL observation tensors."""
        return set(self.hazard_types.keys())

    def reset(
        self,
        initial_fire: Optional[List[Tuple[int, int]]] = None,
        static_hazards: Optional[List[Tuple[int, int]]] = None,
        cars: Optional[List[CarSpec]] = None
    ):
        """Reset disaster state."""
        self.barricade_cells.clear()
        self.car_wrecks.clear()
        self.rubble_cells.clear()
        self.static_hazards.clear()
        self.smoke_cells.clear()
        self.hazard_types.clear()
        self.crashed_buildings.clear()
        self.recent_events.clear()

        # Place initial static barricades if provided
        if static_hazards is not None:
            for cell in static_hazards:
                if self.grid_map.is_passable(*cell):
                    self.place_barricade(cell[0], cell[1], reason="STATIC ROADWORK")
        elif initial_fire is not None:
            for cell in initial_fire:
                if self.grid_map.is_passable(*cell):
                    self.place_barricade(cell[0], cell[1], reason="INITIAL BLOCKAGE")

        # Place vehicles along roads across the 25x35 game city
        if cars is not None:
            self.cars = [CarSpec(c.row, c.col, c.color, c.is_exploded, c.explodes_at_step) for c in cars]
        else:
            self.cars = [
                CarSpec(row=9, col=6, color="yellow", explodes_at_step=-1),
                CarSpec(row=9, col=22, color="blue", explodes_at_step=-1),
                CarSpec(row=15, col=14, color="red", explodes_at_step=-1),
                CarSpec(row=15, col=29, color="silver", explodes_at_step=-1),
                CarSpec(row=21, col=14, color="white", explodes_at_step=-1),
                CarSpec(row=21, col=22, color="blue", explodes_at_step=-1),
                CarSpec(row=3, col=14, color="red", explodes_at_step=-1),
                CarSpec(row=3, col=22, color="yellow", explodes_at_step=-1),
            ]

    def place_barricade(self, r: int, c: int, reason: str = "ROAD CLOSED") -> bool:
        """Place standard road barricade with 'BLOCKED' sign."""
        if self.grid_map.is_passable(r, c) and (r, c) not in self.grid_map.exits:
            self.barricade_cells.add((r, c))
            self.hazard_types[(r, c)] = "BARRICADE"
            msg = f"ROAD BARRICADE placed at ({r}, {c}) [{reason}]"
            self.recent_events.append(msg)
            return True
        return False

    def trigger_car_blast(self, pos: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Cause vehicle explosion into flames, smoke, and charred wreckage."""
        target_pos = None

        if pos is not None and self.grid_map.is_passable(*pos):
            target_pos = pos
        else:
            # Find an unexploded car on the road
            for car in self.cars:
                if not car.is_exploded:
                    car.is_exploded = True
                    target_pos = (car.row, car.col)
                    break

        if target_pos:
            r, c = target_pos
            self.car_wrecks.add(target_pos)
            self.smoke_cells.add(target_pos)
            self.hazard_types[target_pos] = "CAR_BLAST"

            # Smoke diffuses to neighbors
            for nr, nc in self.grid_map.get_cardinal_neighbors(r, c):
                if self.grid_map.is_passable(nr, nc):
                    self.smoke_cells.add((nr, nc))

            msg = f"CAR EXPLOSION! Vehicle blasted at ({r}, {c}) - Lane Blocked!"
            self.recent_events.append(msg)
            return target_pos

        return None

    def trigger_building_collapse(self, building_id: Optional[str] = None, road_pos: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Cause building structural failure, scattering concrete rubble slabs across the road."""
        if building_id is None or building_id not in self.grid_map.building_blocks:
            # Pick a default building
            building_id = list(self.grid_map.building_blocks.keys())[0]

        if building_id in self.grid_map.building_blocks:
            self.crashed_buildings.add(building_id)

            if road_pos is not None and self.grid_map.is_passable(*road_pos):
                rubble_cells = [road_pos]
            else:
                all_adj = self.grid_map.get_building_adjacent_road_cells(building_id)
                rubble_cells = all_adj[:3] if len(all_adj) >= 3 else all_adj

            for cell in rubble_cells:
                if self.grid_map.is_valid_cell(*cell) and cell not in self.grid_map.exits:
                    self.rubble_cells.add(cell)
                    self.smoke_cells.add(cell)
                    self.hazard_types[cell] = "BUILDING_COLLAPSE"

            msg = f"BUILDING COLLAPSE! {building_id} crumbled concrete rubble across avenue!"
            self.recent_events.append(msg)
            return rubble_cells[0] if rubble_cells else None

        return None

    def trigger_dynamic_hazard_ahead(
        self,
        agent_pos: Tuple[int, int],
        planned_path: List[Tuple[int, int]],
        hazard_type: str = "auto"
    ) -> Optional[Tuple[int, int, str]]:
        """
        Finds the road cell 2 to 4 steps ahead of the agent along its path
        and triggers a sudden catastrophe (Car Blast, Building Collapse, or Barricade),
        forcing an active reroute!
        """
        target_cell = None

        if planned_path and len(planned_path) > 1:
            for cell in planned_path[2:6]:
                if self.grid_map.is_passable(*cell) and cell not in self.grid_map.exits and cell != agent_pos:
                    target_cell = cell
                    break

            if target_cell is None:
                for cell in planned_path[1:]:
                    if self.grid_map.is_passable(*cell) and cell not in self.grid_map.exits and cell != agent_pos:
                        target_cell = cell
                        break

        if target_cell is None:
            r, c = agent_pos
            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self.grid_map.is_passable(nr, nc) and (nr, nc) not in self.grid_map.exits:
                    target_cell = (nr, nc)
                    break

        if target_cell is None:
            return None

        # Determine disaster type
        if hazard_type == "auto":
            # Cycle or pick based on count
            count = len(self.hazard_types)
            choice = count % 3
            if choice == 0:
                h_type = "CAR_BLAST"
            elif choice == 1:
                h_type = "BUILDING_COLLAPSE"
            else:
                h_type = "BARRICADE"
        else:
            h_type = hazard_type

        # Trigger chosen disaster
        if h_type == "CAR_BLAST":
            self.trigger_car_blast(target_cell)
        elif h_type == "BUILDING_COLLAPSE":
            # Identify nearest building to target_cell
            nearest_b = "B5"
            min_bd = float("inf")
            for b_name, (br1, br2, bc1, bc2) in self.grid_map.building_blocks.items():
                center_r = (br1 + br2) / 2.0
                center_c = (bc1 + bc2) / 2.0
                d = abs(target_cell[0] - center_r) + abs(target_cell[1] - center_c)
                if d < min_bd:
                    min_bd = d
                    nearest_b = b_name
            self.trigger_building_collapse(nearest_b, target_cell)
        else:
            self.place_barricade(target_cell[0], target_cell[1], reason="SUDDEN ROADWORK")

        return (target_cell[0], target_cell[1], h_type)

    def block_path_ahead(self, agent_pos: Tuple[int, int], planned_path: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Legacy helper for backward-compatibility."""
        res = self.trigger_dynamic_hazard_ahead(agent_pos, planned_path, hazard_type="auto")
        return (res[0], res[1]) if res else None

    def step(self, current_step: int = 0):
        """Advance time-based car blockages without lava spreading."""
        for idx, car in enumerate(self.cars):
            if not car.is_exploded and car.explodes_at_step == current_step:
                self.trigger_car_blast((car.row, car.col))

    def is_blocked(self, r: int, c: int) -> bool:
        """Check if cell is impassable due to barricade, rubble, or wreck."""
        return (
            (r, c) in self.hazard_types or
            (r, c) in self.barricade_cells or
            (r, c) in self.rubble_cells or
            (r, c) in self.car_wrecks or
            (r, c) in self.static_hazards
        )

    def get_hazard_level(self, r: int, c: int) -> float:
        """Return 1.0 if blocked, 0.0 otherwise."""
        return 1.0 if self.is_blocked(r, c) else 0.0
