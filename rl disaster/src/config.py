"""
Configuration module for the RL-Based Disaster Evacuation System.
Defines simulation grid parameters, cell types, actions, reward weights, and disaster dynamics.
Configured for a 3x3 Superblock City Grid (9 building blocks B1 to B9 with interconnecting avenues and streets).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

# Grid Dimensions (25x35 clean 2D game map: 1-lane roads, parks, buildings, start, exit)
GRID_WIDTH = 35
GRID_HEIGHT = 25

# Cell Identifiers
CELL_ROAD = 0           # Clear asphalt road
CELL_EMPTY = 0          # Alias for backward compatibility
CELL_BUILDING = 1       # Solid building block / wall
CELL_WALL = 1           # Alias for backward compatibility
CELL_PARK = 2           # Green park / lawn / trees (impassable to vehicles)
CELL_EXIT = 3           # SAFE zone / emergency evacuation exit
CELL_SAFE_ZONE = 3      # Safe approach zone
CELL_START = 4          # Starting evacuation sector
CELL_RUBBLE = 5         # Collapsed building debris on road
CELL_STATIC_HAZARD = 5  # Alias for backward compatibility
CELL_FIRE = 6           # Active fire (from car explosions or disaster spread)
CELL_SMOKE = 7          # Smoke diffusion cloud
CELL_CAR = 8            # Vehicle on road (passable or park obstruction)
CELL_CAR_WRECK = 9      # Blasted / exploded car wreckage (impassable road blockage)

# Discrete Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT = 4

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_WAIT: "WAIT",
}

# (row_offset, col_offset)
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_WAIT: (0, 0),
}

# Reward & Penalty Weights
STEP_PENALTY = -0.5               # Discourages unnecessary delays
GOAL_REWARD = 150.0               # Large terminal reward for reaching SAFE zone
FIRE_PENALTY = -75.0              # Fatal/severe penalty for entering fire / explosion
SMOKE_PENALTY = -6.0              # Penalty for breathing smoke
RUBBLE_PENALTY = -25.0            # Penalty for navigating collapsed building rubble
STATIC_HAZARD_PENALTY = -25.0     # Alias
WALL_COLLISION_PENALTY = -2.0     # Penalty for bumping into buildings / barriers
PROGRESS_REWARD_WEIGHT = 2.5      # Potential-based reward shaping towards nearest safe exit
CONGESTION_PENALTY_WEIGHT = -3.0  # Penalty for crowd/traffic congestion

# Simulation Limits
MAX_STEPS_PER_EPISODE = 200

# Disaster Dynamics Parameters
FIRE_SPREAD_PROBABILITY = 0.20    # Probability of adjacent cell igniting per step
SMOKE_DIFFUSION_RADIUS = 2       # Radius around fire cells that get filled with smoke

@dataclass
class CarSpec:
    """Specification of a car in the city grid."""
    row: int
    col: int
    color: str = "blue"          # "blue", "red", "yellow", "silver", "white"
    is_exploded: bool = False
    explodes_at_step: int = -1    # Step at which dynamic explosion triggers (-1 for never)

@dataclass
class BuildingCrashSpec:
    """Specification of building structural failure spilling rubble into streets."""
    building_id: str             # e.g., "B5", "B8"
    rubble_cells: List[Tuple[int, int]]
    crashes_at_step: int = -1    # Step at which collapse triggers (-1 for never)

@dataclass
class ScenarioConfig:
    """Disaster scenario configuration for 3x3 city grid."""
    name: str
    description: str
    is_dynamic: bool = False
    fire_start_positions: List[Tuple[int, int]] = field(default_factory=list)
    fire_spread_prob: float = 0.15
    cars: List[CarSpec] = field(default_factory=list)
    building_crashes: List[BuildingCrashSpec] = field(default_factory=list)
    static_hazards: List[Tuple[int, int]] = field(default_factory=list)
    exit_congestions: Dict[int, float] = field(default_factory=dict)
    block_nearest_exit_at_step: int = -1
