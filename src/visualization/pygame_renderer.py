"""
High-Fidelity 2D Game Renderer for Disaster Evacuation Simulator.
Game style:
- Clean, crisp vector/tile graphics (NO AI image distortion or misalignment!)
- Single-lane dark charcoal asphalt roads with crisp sidewalk curbs and dashed dividers
- Lush green parks with trees, flowers, and park trails
- Varied building architecture (residential, commercial, civic, industrial) with 3D drop-shadows and lit windows
- STARTING SECTOR at bottom-left with amber hazard stripes and glowing beacon
- SAFE EXIT at top-right with pulsing emerald green evacuation zone
- 100% pixel-perfect alignment: agents and barricades align strictly to road tiles with ZERO clipping through buildings
"""

import sys
import os
import math
import pygame
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..config import (
    GRID_WIDTH, GRID_HEIGHT,
    CELL_ROAD, CELL_BUILDING, CELL_PARK, CELL_EXIT, CELL_SAFE_ZONE, CELL_START
)
from ..env.evacuation_env import DisasterEvacuationEnv

# Game Color Palette
COLOR_BG = (14, 18, 24)
COLOR_ASPHALT = (42, 46, 54)
COLOR_ASPHALT_CURB = (60, 66, 78)
COLOR_ASPHALT_LINE = (241, 196, 15)
COLOR_CROSSWALK = (220, 225, 235)

COLOR_GRASS_BASE = (58, 125, 68)
COLOR_GRASS_DARK = (46, 105, 55)
COLOR_TREE_FOLIAGE = (34, 85, 42)
COLOR_TREE_TRUNK = (85, 55, 35)

COLOR_ROOF_DEFAULT = (65, 75, 95)
COLOR_ROOF_CIVIC = (75, 85, 110)
COLOR_ROOF_BRICK = (120, 60, 55)
COLOR_ROOF_MALL = (55, 85, 95)
COLOR_BUILDING_WALL = (40, 46, 58)
COLOR_WINDOW_LIT = (255, 235, 140)
COLOR_WINDOW_DARK = (30, 36, 46)

COLOR_SAFE_GREEN = (46, 204, 113)
COLOR_SAFE_BORDER = (39, 174, 96)
COLOR_START_AMBER = (243, 156, 18)
COLOR_START_BORDER = (211, 84, 0)
COLOR_DANGER_RED = (231, 76, 60)

# HUD Colors
COLOR_HUD_BG = (18, 22, 30)
COLOR_HUD_PANEL = (25, 32, 44)
COLOR_HUD_BORDER = (45, 56, 75)
COLOR_BTN_DEFAULT = (35, 44, 60)
COLOR_BTN_HOVER = (50, 64, 88)
COLOR_BTN_ACTIVE = (41, 128, 185)
COLOR_BTN_ACTIVE_BORDER = (52, 152, 219)
COLOR_TEXT_MAIN = (240, 243, 246)
COLOR_TEXT_MUTED = (145, 155, 170)

CAR_COLORS = {
    "yellow": (241, 196, 15),
    "blue": (52, 152, 219),
    "red": (231, 76, 60),
    "silver": (189, 195, 199),
    "white": (245, 245, 250)
}

class PygameRenderer:
    def __init__(self, env: DisasterEvacuationEnv, cell_size: int = 26, hud_width: int = 340):
        self.env = env
        self.cell_size = cell_size
        self.grid_width_px = env.grid_width * cell_size    # 35 * 26 = 910 px
        self.grid_height_px = env.grid_height * cell_size  # 25 * 26 = 650 px
        self.hud_width = hud_width

        self.window_width = self.grid_width_px + hud_width # 1250 px
        self.window_height = max(self.grid_height_px, 660)

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("City Disaster Evacuation Simulator: Pure 2D Game World")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        self.font_header = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.font_main = pygame.font.SysFont("Segoe UI", 11)
        self.font_bold = pygame.font.SysFont("Segoe UI", 11, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 10)
        self.font_btn = pygame.font.SysFont("Segoe UI", 11, bold=True)
        self.font_blocked = pygame.font.SysFont("Arial Black", 8, bold=True)

        self.agent_path_history: List[Tuple[int, int]] = []
        self.planned_path_visual: List[Tuple[int, int]] = []
        self.anim_tick = 0
        self.buttons: Dict[str, pygame.Rect] = {}

    def reset_path_history(self):
        self.agent_path_history.clear()
        if hasattr(self.env, "agent_pos"):
            self.agent_path_history.append(self.env.agent_pos)

    def set_planned_path(self, path: List[Tuple[int, int]]):
        self.planned_path_visual = list(path) if path else []

    def draw_world(self):
        """Draws the crisp tile-based game world."""
        grid_map = self.env.grid_map
        cs = self.cell_size

        for r in range(grid_map.height):
            for c in range(grid_map.width):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                val = grid_map.grid[r, c]

                # 1. PARKS & GREENERY
                if val == CELL_PARK:
                    # Alternating subtle grass texture
                    col = COLOR_GRASS_BASE if (r + c) % 2 == 0 else COLOR_GRASS_DARK
                    pygame.draw.rect(self.screen, col, rect)

                    # Decorative scattered trees on specific park tiles
                    if (r * 7 + c * 13) % 5 == 0:
                        center_x = rect.centerx
                        center_y = rect.centery
                        # Trunk
                        pygame.draw.rect(self.screen, COLOR_TREE_TRUNK, (center_x - 2, center_y + 2, 4, 6))
                        # Foliage
                        pygame.draw.circle(self.screen, (20, 65, 30), (center_x, center_y - 2), 7)
                        pygame.draw.circle(self.screen, COLOR_TREE_FOLIAGE, (center_x, center_y - 4), 6)
                        pygame.draw.circle(self.screen, (45, 115, 55), (center_x - 2, center_y - 5), 3)

                # 2. ASPHALT ROADS (Single-lane)
                elif val == CELL_ROAD:
                    pygame.draw.rect(self.screen, COLOR_ASPHALT, rect)
                    pygame.draw.rect(self.screen, COLOR_ASPHALT_CURB, rect, 1)

                    # Road markings (center dashed lines along avenues)
                    if r in (3, 9, 15, 21) and c % 2 == 0:
                        pygame.draw.line(self.screen, COLOR_ASPHALT_LINE, (rect.left, rect.centery), (rect.right, rect.centery), 1)
                    if c in (3, 10, 18, 26, 32) and r % 2 == 0:
                        pygame.draw.line(self.screen, COLOR_ASPHALT_LINE, (rect.centerx, rect.top), (rect.centerx, rect.bottom), 1)

                # 3. BUILDINGS
                elif val == CELL_BUILDING:
                    # Drop shadow
                    pygame.draw.rect(self.screen, (10, 14, 20), (rect.x + 2, rect.y + 2, cs, cs), border_radius=2)

                    # Choose roof color by location
                    roof_col = COLOR_ROOF_BRICK if r > 16 else (COLOR_ROOF_MALL if c > 20 else COLOR_ROOF_DEFAULT)
                    pygame.draw.rect(self.screen, roof_col, rect, border_radius=2)
                    pygame.draw.rect(self.screen, COLOR_BUILDING_WALL, rect, 1, border_radius=2)

                    # Rooftop windows / HVAC vents
                    wx = rect.x + 5
                    wy = rect.y + 5
                    w_col = COLOR_WINDOW_LIT if (r + c + self.anim_tick // 15) % 4 == 0 else COLOR_WINDOW_DARK
                    pygame.draw.rect(self.screen, w_col, (wx, wy, 5, 5))
                    pygame.draw.rect(self.screen, w_col, (rect.right - 10, wy, 5, 5))
                    pygame.draw.rect(self.screen, w_col, (wx, rect.bottom - 10, 5, 5))
                    pygame.draw.rect(self.screen, w_col, (rect.right - 10, rect.bottom - 10, 5, 5))

                # 4. START SECTOR (Bottom-Left)
                elif val == CELL_START:
                    pygame.draw.rect(self.screen, (55, 40, 15), rect)
                    # Hazard diagonal stripes
                    pygame.draw.line(self.screen, COLOR_START_AMBER, rect.topleft, rect.bottomright, 2)
                    pygame.draw.rect(self.screen, COLOR_START_BORDER, rect, 1)

                # 5. SAFE EXIT (Top-Right)
                elif val == CELL_EXIT or val == CELL_SAFE_ZONE:
                    pulse = int(140 + 70 * math.sin(self.anim_tick * 0.15))
                    s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                    s.fill((46, 204, 113, pulse))
                    self.screen.blit(s, rect.topleft)
                    pygame.draw.rect(self.screen, COLOR_SAFE_BORDER, rect, 2)

        # Draw Safe Exit Banner Badge
        e_rect = pygame.Rect(32 * cs, 1 * cs + 4, 3 * cs, 14)
        pygame.draw.rect(self.screen, (15, 20, 28, 220), e_rect, border_radius=3)
        pygame.draw.rect(self.screen, COLOR_SAFE_GREEN, e_rect, 1, border_radius=3)
        lbl_exit = self.font_small.render("SAFE EXIT", True, COLOR_SAFE_GREEN)
        self.screen.blit(lbl_exit, (e_rect.centerx - lbl_exit.get_width() // 2, e_rect.y + 1))

    def draw_cars(self):
        """Draw civilian cars cleanly on roads."""
        disaster = self.env.disaster
        cs = self.cell_size

        for car in disaster.cars:
            if car.is_exploded or disaster.is_blocked(car.row, car.col):
                continue

            cx = car.col * cs
            cy = car.row * cs
            color = CAR_COLORS.get(car.color, (52, 152, 219))

            w = int(cs * 0.75)
            h = int(cs * 0.50)
            rect = pygame.Rect(cx + (cs - w) // 2, cy + (cs - h) // 2, w, h)

            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, (20, 24, 30), rect, 1, border_radius=3)
            # Windshield
            win = pygame.Rect(rect.x + 3, rect.y + 2, w - 6, h - 4)
            pygame.draw.rect(self.screen, (30, 42, 55), win, border_radius=1)

    def draw_hazards_and_disasters(self):
        """Draw barricades, explosions, and rubble strictly on grid cells."""
        disaster = self.env.disaster
        cs = self.cell_size

        # 1. Smoke diffusion
        for r, c in disaster.smoke_cells:
            if self.env.grid_map.is_valid_cell(r, c):
                x = c * cs
                y = r * cs
                smoke_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
                alpha = int(90 + 35 * math.sin(self.anim_tick * 0.2 + r + c))
                smoke_surf.fill((65, 70, 78, alpha))
                self.screen.blit(smoke_surf, (x, y))

        # 2. Car explosions
        for r, c in disaster.car_wrecks:
            self.draw_car_blast(r, c)

        # 3. Building rubble
        for r, c in disaster.rubble_cells:
            self.draw_building_rubble(r, c)

        # 4. Barricades
        for r, c in disaster.barricade_cells:
            self.draw_barricade(r, c)

    def draw_car_blast(self, r: int, c: int):
        cs = self.cell_size
        x = c * cs
        y = r * cs

        # Burnt chassis
        chassis = pygame.Rect(x + 3, y + 6, cs - 6, cs - 12)
        pygame.draw.rect(self.screen, (25, 25, 28), chassis, border_radius=2)

        # Animated flames
        flame_h = int(7 + 4 * math.sin(self.anim_tick * 0.4 + x))
        pygame.draw.polygon(self.screen, (243, 156, 18), [(x + cs // 2, y + 2 - flame_h), (x + 4, y + cs - 4), (x + cs - 4, y + cs - 4)])
        pygame.draw.polygon(self.screen, (231, 76, 60), [(x + cs // 2, y + 5 - flame_h // 2), (x + 6, y + cs - 4), (x + cs - 6, y + cs - 4)])

        self.draw_blocked_badge(x, y)

    def draw_building_rubble(self, r: int, c: int):
        cs = self.cell_size
        x = c * cs
        y = r * cs

        pygame.draw.polygon(self.screen, (95, 90, 85), [(x + 3, y + 15), (x + 10, y + 4), (x + cs - 4, y + 10), (x + cs - 7, y + cs - 5)])
        pygame.draw.polygon(self.screen, (60, 55, 50), [(x + 6, y + cs - 6), (x + 14, y + 7), (x + cs - 5, y + cs - 5)])
        pygame.draw.line(self.screen, (180, 70, 50), (x + 4, y + 12), (x + cs - 5, y + 8), 2)

        self.draw_blocked_badge(x, y)

    def draw_barricade(self, r: int, c: int):
        cs = self.cell_size
        x = c * cs
        y = r * cs

        # Crisp A-frame barrier across the 1-lane street
        bar_w = cs - 2
        bar_h = 10
        bar_rect = pygame.Rect(x + 1, y + 6, bar_w, bar_h)

        pygame.draw.rect(self.screen, (245, 245, 245), bar_rect, border_radius=2)
        pygame.draw.rect(self.screen, (30, 30, 35), bar_rect, 1)

        # Orange reflective stripes
        for sx in range(bar_rect.x, bar_rect.right - 3, 5):
            pygame.draw.polygon(self.screen, (230, 100, 20), [
                (sx, bar_rect.bottom),
                (sx + 3, bar_rect.top),
                (sx + 5, bar_rect.top),
                (sx + 2, bar_rect.bottom)
            ])

        # Legs
        pygame.draw.line(self.screen, (40, 45, 50), (bar_rect.x + 2, bar_rect.bottom), (bar_rect.x, y + cs - 2), 2)
        pygame.draw.line(self.screen, (40, 45, 50), (bar_rect.right - 2, bar_rect.bottom), (bar_rect.right, y + cs - 2), 2)

        self.draw_blocked_badge(x, y)

    def draw_blocked_badge(self, x: int, y: int):
        cs = self.cell_size
        badge_w = cs + 2
        badge_h = 10
        rect = pygame.Rect(x - 1, y + cs - 10, badge_w, badge_h)

        pygame.draw.rect(self.screen, (192, 57, 43), rect, border_radius=2)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=2)
        txt = self.font_blocked.render("BLOCKED", True, (255, 255, 255))
        self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def draw_paths_and_agent(self):
        """Draw historical trail, planned guide path, and agent."""
        cs = self.cell_size

        # Historical trail
        if len(self.agent_path_history) > 1:
            pts = [(c * cs + cs // 2, r * cs + cs // 2) for (r, c) in self.agent_path_history]
            pygame.draw.lines(self.screen, (52, 152, 219), False, pts, 3)

        # Golden planned path
        if len(self.planned_path_visual) > 1:
            guide_pts = [(c * cs + cs // 2, r * cs + cs // 2) for (r, c) in self.planned_path_visual]
            pygame.draw.lines(self.screen, (241, 196, 15), False, guide_pts, 3)
            for pt in guide_pts:
                pygame.draw.circle(self.screen, (241, 196, 15), pt, 3)

        # Evacuation Agent
        ar, ac = self.env.agent_pos
        ax = ac * cs + cs // 2
        ay = ar * cs + cs // 2

        # Pulsing beacon
        rad = int(9 + 3 * math.sin(self.anim_tick * 0.3))
        s_beacon = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s_beacon, (52, 152, 219, 130), (rad + 1, rad + 1), rad, 2)
        self.screen.blit(s_beacon, (ax - rad - 1, ay - rad - 1))

        # Core circle
        pygame.draw.circle(self.screen, (52, 152, 219), (ax, ay), 7)
        pygame.draw.circle(self.screen, (231, 76, 60), (ax, ay), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), 1)

    def draw_hud_and_buttons(
        self,
        current_algo: str,
        is_dynamic: bool,
        is_paused: bool,
        step_info: dict,
        status_text: str
    ):
        hud_x = self.grid_width_px
        hud_w = self.hud_width
        hud_h = self.window_height

        pygame.draw.rect(self.screen, COLOR_HUD_BG, (hud_x, 0, hud_w, hud_h))
        pygame.draw.line(self.screen, COLOR_HUD_BORDER, (hud_x, 0), (hud_x, hud_h), 2)

        x = hud_x + 14
        y = 12
        bw = hud_w - 28

        title_surf = self.font_header.render("CITY EVACUATION SYSTEM", True, COLOR_TEXT_MAIN)
        self.screen.blit(title_surf, (x, y))
        y += 20
        sub_surf = self.font_small.render("1-Lane Streets • Strict Detour Rerouting", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (x, y))
        y += 24

        # 1. Environment Mode
        sec1_title = self.font_title.render("1. ENVIRONMENT MODE", True, (241, 196, 15))
        self.screen.blit(sec1_title, (x, y))
        y += 18

        btn_w_half = (bw - 8) // 2
        rect_static = pygame.Rect(x, y, btn_w_half, 28)
        rect_dynamic = pygame.Rect(x + btn_w_half + 8, y, btn_w_half, 28)
        self.buttons["btn_mode_static"] = rect_static
        self.buttons["btn_mode_dynamic"] = rect_dynamic

        col_s = COLOR_BTN_ACTIVE if not is_dynamic else COLOR_BTN_DEFAULT
        bor_s = COLOR_BTN_ACTIVE_BORDER if not is_dynamic else COLOR_HUD_BORDER
        pygame.draw.rect(self.screen, col_s, rect_static, border_radius=4)
        pygame.draw.rect(self.screen, bor_s, rect_static, 2 if not is_dynamic else 1, border_radius=4)
        txt_s = self.font_btn.render("Static Env", True, COLOR_TEXT_MAIN)
        self.screen.blit(txt_s, (rect_static.centerx - txt_s.get_width() // 2, rect_static.centery - txt_s.get_height() // 2))

        col_d = COLOR_BTN_ACTIVE if is_dynamic else COLOR_BTN_DEFAULT
        bor_d = COLOR_BTN_ACTIVE_BORDER if is_dynamic else COLOR_HUD_BORDER
        pygame.draw.rect(self.screen, col_d, rect_dynamic, border_radius=4)
        pygame.draw.rect(self.screen, bor_d, rect_dynamic, 2 if is_dynamic else 1, border_radius=4)
        txt_d = self.font_btn.render("Dynamic Env", True, COLOR_TEXT_MAIN)
        self.screen.blit(txt_d, (rect_dynamic.centerx - txt_d.get_width() // 2, rect_dynamic.centery - txt_d.get_height() // 2))
        y += 36

        # 2. Routing Algorithm
        sec2_title = self.font_title.render("2. ROUTING ALGORITHM", True, (241, 196, 15))
        self.screen.blit(sec2_title, (x, y))
        y += 18

        btn_w_third = (bw - 12) // 3
        rect_dijk = pygame.Rect(x, y, btn_w_third, 28)
        rect_astar = pygame.Rect(x + btn_w_third + 6, y, btn_w_third, 28)
        rect_ppo = pygame.Rect(x + (btn_w_third + 6) * 2, y, btn_w_third, 28)
        self.buttons["btn_algo_dijkstra"] = rect_dijk
        self.buttons["btn_algo_astar"] = rect_astar
        self.buttons["btn_algo_ppo"] = rect_ppo

        for r_btn, name, label in [(rect_dijk, "Dijkstra", "Dijkstra"), (rect_astar, "A*", "A* Search"), (rect_ppo, "PPO", "PPO (RL)")]:
            is_active = (current_algo == name)
            col = COLOR_BTN_ACTIVE if is_active else COLOR_BTN_DEFAULT
            bor = COLOR_BTN_ACTIVE_BORDER if is_active else COLOR_HUD_BORDER
            pygame.draw.rect(self.screen, col, r_btn, border_radius=4)
            pygame.draw.rect(self.screen, bor, r_btn, 2 if is_active else 1, border_radius=4)
            txt = self.font_btn.render(label, True, COLOR_TEXT_MAIN)
            self.screen.blit(txt, (r_btn.centerx - txt.get_width() // 2, r_btn.centery - txt.get_height() // 2))
        y += 36

        # 3. Dynamic Disaster Triggers
        sec3_title = self.font_title.render("3. TRIGGER DYNAMIC DISASTERS", True, (241, 196, 15))
        self.screen.blit(sec3_title, (x, y))
        y += 18

        rect_blast = pygame.Rect(x, y, bw, 28)
        self.buttons["btn_blast_car"] = rect_blast
        pygame.draw.rect(self.screen, (169, 50, 38), rect_blast, border_radius=4)
        pygame.draw.rect(self.screen, (231, 76, 60), rect_blast, 1, border_radius=4)
        txt_b = self.font_btn.render("💥 Trigger Car Blast Ahead!", True, (255, 255, 255))
        self.screen.blit(txt_b, (rect_blast.centerx - txt_b.get_width() // 2, rect_blast.centery - txt_b.get_height() // 2))
        y += 32

        rect_collapse = pygame.Rect(x, y, bw, 28)
        self.buttons["btn_collapse_building"] = rect_collapse
        pygame.draw.rect(self.screen, (142, 68, 173), rect_collapse, border_radius=4)
        pygame.draw.rect(self.screen, (175, 122, 197), rect_collapse, 1, border_radius=4)
        txt_c = self.font_btn.render("🏢 Collapse Building Rubble!", True, (255, 255, 255))
        self.screen.blit(txt_c, (rect_collapse.centerx - txt_c.get_width() // 2, rect_collapse.centery - txt_c.get_height() // 2))
        y += 32

        rect_barricade = pygame.Rect(x, y, bw, 28)
        self.buttons["btn_drop_barricade"] = rect_barricade
        pygame.draw.rect(self.screen, (211, 84, 0), rect_barricade, border_radius=4)
        pygame.draw.rect(self.screen, (230, 126, 34), rect_barricade, 1, border_radius=4)
        txt_bar = self.font_btn.render("🚧 Drop Road Barricade!", True, (255, 255, 255))
        self.screen.blit(txt_bar, (rect_barricade.centerx - txt_bar.get_width() // 2, rect_barricade.centery - txt_bar.get_height() // 2))
        y += 34

        # Play/Pause and Reset
        rect_pause = pygame.Rect(x, y, btn_w_half, 28)
        rect_reset = pygame.Rect(x + btn_w_half + 8, y, btn_w_half, 28)
        self.buttons["btn_play_pause"] = rect_pause
        self.buttons["btn_reset"] = rect_reset

        pygame.draw.rect(self.screen, COLOR_BTN_DEFAULT, rect_pause, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, rect_pause, 1, border_radius=4)
        pause_label = "▶ Resume" if is_paused else "⏸ Pause"
        txt_p = self.font_btn.render(pause_label, True, COLOR_TEXT_MAIN)
        self.screen.blit(txt_p, (rect_pause.centerx - txt_p.get_width() // 2, rect_pause.centery - txt_p.get_height() // 2))

        pygame.draw.rect(self.screen, COLOR_BTN_DEFAULT, rect_reset, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, rect_reset, 1, border_radius=4)
        txt_r = self.font_btn.render("⟳ Reset", True, COLOR_TEXT_MAIN)
        self.screen.blit(txt_r, (rect_reset.centerx - txt_r.get_width() // 2, rect_reset.centery - txt_r.get_height() // 2))
        y += 36

        # Telemetry Panel
        status_panel = pygame.Rect(x, y, bw, 42)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL, status_panel, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_HUD_BORDER, status_panel, 1, border_radius=4)

        stat_col = COLOR_SAFE_GREEN if "Evacuated" in status_text else (COLOR_DANGER_RED if "Blocked" in status_text or "Trapped" in status_text else (241, 196, 15))
        stat_surf = self.font_bold.render(f"Status: {status_text}", True, stat_col)
        self.screen.blit(stat_surf, (status_panel.x + 8, status_panel.y + 4))

        mode_str = f"Mode: {('Dynamic' if is_dynamic else 'Static')} | Grid: 25x35"
        mode_surf = self.font_small.render(mode_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(mode_surf, (status_panel.x + 8, status_panel.y + 22))
        y += 48

        steps_val = step_info.get("steps", self.env.current_step)
        car_blasts_val = sum(1 for v in self.env.disaster.hazard_types.values() if v == "CAR_BLAST")
        collapses_val = sum(1 for v in self.env.disaster.hazard_types.values() if v == "BUILDING_COLLAPSE")
        barricades_val = sum(1 for v in self.env.disaster.hazard_types.values() if v == "BARRICADE")

        metrics = [
            ("Steps Taken", f"{steps_val} / {self.env.max_steps}"),
            ("Agent Position", f"{self.env.agent_pos}"),
            ("Car Blasts (Active)", f"{car_blasts_val}"),
            ("Building Collapses", f"{collapses_val}"),
            ("Barricades (Active)", f"{barricades_val}"),
        ]

        for lbl, val_str in metrics:
            lbl_surf = self.font_main.render(lbl, True, COLOR_TEXT_MUTED)
            val_surf = self.font_bold.render(val_str, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl_surf, (x, y))
            self.screen.blit(val_surf, (x + bw - val_surf.get_width(), y))
            y += 18

        y += 6
        self.screen.blit(self.font_bold.render("DISASTER EVENT FEED:", True, (241, 196, 15)), (x, y))
        y += 16
        events = self.env.disaster.recent_events[-3:] if self.env.disaster.recent_events else ["[Normal] Clear city traffic conditions"]
        for ev in events:
            ev_surf = self.font_small.render(f"• {ev[:40]}", True, (240, 160, 160) if "EXPLOSION" in ev or "COLLAPSE" in ev or "BARRICADE" in ev else COLOR_TEXT_MUTED)
            self.screen.blit(ev_surf, (x, y))
            y += 14

    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        for btn_name, rect in self.buttons.items():
            if rect.collidepoint(mouse_pos):
                return btn_name
        return None

    def render(
        self,
        current_algo: str = "Dijkstra",
        is_dynamic: bool = True,
        is_paused: bool = False,
        step_info: Optional[dict] = None,
        status_text: str = "In Transit"
    ):
        self.anim_tick += 1
        if step_info is None:
            step_info = {}

        if self.env.agent_pos not in self.agent_path_history:
            self.agent_path_history.append(self.env.agent_pos)

        self.screen.fill(COLOR_BG)
        self.draw_world()
        self.draw_cars()
        self.draw_hazards_and_disasters()
        self.draw_paths_and_agent()
        self.draw_hud_and_buttons(current_algo, is_dynamic, is_paused, step_info, status_text)
        pygame.display.flip()

    def tick(self, fps: int = 8):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
