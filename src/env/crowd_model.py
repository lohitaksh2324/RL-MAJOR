"""
Crowd and congestion model for the Disaster Evacuation System.
Simulates crowd bottlenecking, exit queues, and density-induced movement delays.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from .grid_map import GridMap

class CrowdModel:
    def __init__(self, grid_map: GridMap, rng: Optional[np.random.Generator] = None):
        self.grid_map = grid_map
        self.rng = rng or np.random.default_rng()
        # Density per cell (0.0: empty, 1.0: jammed)
        self.cell_density = np.zeros((grid_map.height, grid_map.width), dtype=np.float32)
        # Exit congestion level: exit_index -> density float [0.0, 1.0]
        self.exit_congestion: Dict[int, float] = {}

    def reset(self, exit_congestions: Optional[Dict[int, float]] = None):
        """Reset crowd levels across the map and exits."""
        self.cell_density.fill(0.0)
        self.exit_congestion.clear()

        num_exits = len(self.grid_map.exits)
        for i in range(num_exits):
            if exit_congestions and i in exit_congestions:
                self.exit_congestion[i] = float(exit_congestions[i])
            else:
                # Default exit congestion: Exit 0 has rush, Exit 1 & 2 are clearer
                self.exit_congestion[i] = 0.8 if i == 0 else 0.1

        # Project congestion around exit vicinities
        for exit_idx, (er, ec) in enumerate(self.grid_map.exits):
            c_val = self.exit_congestion[exit_idx]
            for r in range(max(0, er - 2), min(self.grid_map.height, er + 3)):
                for c in range(max(0, ec - 2), min(self.grid_map.width, ec + 3)):
                    dist = abs(r - er) + abs(c - ec)
                    if dist <= 2 and self.grid_map.is_passable(r, c):
                        self.cell_density[r, c] = max(self.cell_density[r, c], c_val / (dist + 1))

    def step(self):
        """
        Dynamic crowd fluctuation:
        Crowd around exits gradually evacuates or fluctuates slightly.
        """
        for exit_idx in self.exit_congestion:
            # Slight natural decrease as evacuees filter through doors
            self.exit_congestion[exit_idx] = max(0.05, self.exit_congestion[exit_idx] * 0.98)

    def get_cell_congestion(self, r: int, c: int) -> float:
        """Return crowd density at cell (r, c) between 0.0 and 1.0."""
        if 0 <= r < self.grid_map.height and 0 <= c < self.grid_map.width:
            return float(self.cell_density[r, c])
        return 0.0

    def get_exit_congestion(self, exit_idx: int) -> float:
        """Return congestion factor for a specific exit index."""
        return self.exit_congestion.get(exit_idx, 0.0)
