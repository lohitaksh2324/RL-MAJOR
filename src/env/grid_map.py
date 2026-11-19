"""
Grid Map representation for realistic urban city layout with greenery, buildings, and roads.
Matches the architectural plan image (50x28 grid) with:
- Connected asphalt road corridors, curved suburban loops, and main avenues
- Green parks, lawns, and tree zones
- Varied building blocks with courtyard footprints
- STARTING SECTOR at bottom-left
- SAFE EXIT zone at top-right
"""

import os
import json
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from ..config import (
    GRID_WIDTH, GRID_HEIGHT,
    CELL_ROAD, CELL_BUILDING, CELL_PARK, CELL_EXIT, CELL_SAFE_ZONE, CELL_START
)

MAP_DATA_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")

class GridMap:
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.exits: List[Tuple[int, int]] = []
        self.safe_zones: Set[Tuple[int, int]] = set()
        self.walls: Set[Tuple[int, int]] = set()
        self.parks: Set[Tuple[int, int]] = set()
        self.start_zones: Set[Tuple[int, int]] = set()
        self.building_blocks: Dict[str, Tuple[int, int, int, int]] = {}
        self.road_cells: Set[Tuple[int, int]] = set()
        self.build_city_layout()

    def build_city_layout(self):
        """Loads the realistic 50x28 urban layout with interconnected roads, buildings, and greenery."""
        if os.path.exists(MAP_DATA_PATH):
            with open(MAP_DATA_PATH, "r") as f:
                data = json.load(f)
            self.grid = np.array(data["grid"], dtype=np.int32)
            self.height, self.width = self.grid.shape
        else:
            self.grid = np.zeros((self.height, self.width), dtype=np.int32)
            # Default fallback layout
            self.grid.fill(CELL_ROAD)

        self.walls.clear()
        self.parks.clear()
        self.exits.clear()
        self.safe_zones.clear()
        self.start_zones.clear()
        self.road_cells.clear()

        # Classify cells
        exit_cells = []
        for r in range(self.height):
            for c in range(self.width):
                val = self.grid[r, c]
                if val == CELL_ROAD:
                    self.road_cells.add((r, c))
                elif val == CELL_BUILDING:
                    self.walls.add((r, c))
                elif val == CELL_PARK:
                    self.parks.add((r, c))
                elif val == CELL_EXIT or val == CELL_SAFE_ZONE:
                    exit_cells.append((r, c))
                    self.safe_zones.add((r, c))
                elif val == CELL_START:
                    self.start_zones.add((r, c))

        # Designate primary exit anchor at top-right
        primary_exit = (3, 32)
        self.exits = [primary_exit]
        self.grid[primary_exit] = CELL_EXIT
        self.safe_zones.add(primary_exit)
        self.safe_zones.add((3, 33))
        self.safe_zones.add((2, 33))

        # Set up named building clusters for collision/disaster effects
        self.building_blocks = {
            "B1_WestApartments": (4, 8, 5, 9),
            "B2_NorthMall": (4, 7, 12, 17),
            "B3_TechPlaza": (5, 8, 20, 25),
            "B4_CityHospital": (4, 8, 28, 31),
            "B5_MidMarket": (11, 14, 5, 9),
            "B6_CentralOffice": (11, 14, 12, 15),
            "B7_Library": (11, 14, 20, 24),
            "B8_CivicCenter": (11, 14, 28, 31),
            "B9_SouthResidences": (17, 20, 5, 9),
            "B10_Hotel": (17, 20, 12, 17),
            "B11_School": (17, 20, 20, 25),
            "B12_TransitHub": (17, 20, 28, 31),
        }

    def is_valid_cell(self, r: int, c: int) -> bool:
        """Check if coordinates lie inside the grid boundaries."""
        return 0 <= r < self.height and 0 <= c < self.width

    def is_passable(self, r: int, c: int) -> bool:
        """Check if a cell is passable (only roads, starts, and exits can be driven on)."""
        if not self.is_valid_cell(r, c):
            return False
        val = self.grid[r, c]
        return val in (CELL_ROAD, CELL_EXIT, CELL_SAFE_ZONE, CELL_START)

    def get_cardinal_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return valid 4-connected neighbor cells."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_valid_cell(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def get_free_road_cells(self) -> List[Tuple[int, int]]:
        """Return all open asphalt road cells."""
        return sorted(list(self.road_cells))

    def get_building_adjacent_road_cells(self, building_id: str) -> List[Tuple[int, int]]:
        """Get road cells immediately touching a specific building block (for crash debris)."""
        if building_id not in self.building_blocks:
            return []
        r1, r2, c1, c2 = self.building_blocks[building_id]
        adj_cells = set()
        for r in range(max(0, r1 - 1), min(self.height, r2 + 2)):
            for c in range(max(0, c1 - 1), min(self.width, c2 + 2)):
                if (r, c) in self.road_cells:
                    adj_cells.add((r, c))
        return sorted(list(adj_cells))
