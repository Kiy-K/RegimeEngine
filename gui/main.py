#!/usr/bin/env python3
"""
GRAVITAS Engine — Air Strip One Strategic Map GUI

A real-time strategic map viewer for the 1984 war simulation.
Shows 35 sectors across British Isles + France/Benelux/Netherlands,
6 sea zones, fleet positions, land garrisons, BLF resistance,
and the war correspondent's dispatches.

Usage:
    .venv/bin/python gui/main.py [--seed 42] [--turns 100] [--speed 1.0]

Controls:
    SPACE     — Pause/Resume auto-play
    N         — Next turn (when paused)
    +/-       — Speed up/slow down
    S         — Toggle sector names
    F         — Toggle fleet display
    ESC/Q     — Quit
"""

import sys
import os
import time
import argparse

# Ensure project root on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pygame

from gravitas.llm_game import (
    create_game, step_game, summarize_turn, GameState,
    generate_visible_events, format_commentary_prompt,
)

# ═══════════════════════════════════════════════════════════════════════════ #
# Colors                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREY = (30, 30, 35)
MED_GREY = (60, 60, 65)
LIGHT_GREY = (120, 120, 130)
TEXT_DIM = (160, 160, 170)

# Faction colors
OCEANIA_BLUE = (40, 80, 160)
OCEANIA_LIGHT = (70, 120, 200)
EURASIA_RED = (160, 40, 40)
EURASIA_LIGHT = (200, 70, 70)
BLF_GREEN = (40, 160, 60)
BLF_LIGHT = (70, 200, 90)
CONTESTED_YELLOW = (200, 180, 40)

# Sea zones
SEA_DARK = (20, 40, 70)
SEA_MID = (30, 60, 100)
SEA_LIGHT = (40, 80, 130)

# UI
PANEL_BG = (25, 25, 30)
PANEL_BORDER = (50, 50, 60)
BUTTON_BG = (50, 60, 80)
BUTTON_HOVER = (70, 80, 110)
BUTTON_TEXT = (200, 200, 210)
ACCENT_GOLD = (220, 180, 60)
ACCENT_RED = (220, 60, 60)
ACCENT_GREEN = (60, 200, 80)

# ═══════════════════════════════════════════════════════════════════════════ #
# Sector Positions — geographic layout of British Isles + France/Benelux     #
# Coordinates are in screen pixels (designed for 1400×900 window)            #
# ═══════════════════════════════════════════════════════════════════════════ #

SECTOR_POS = {
    # Oceania — South England
    0:  (340, 310),   # London
    1:  (410, 340),   # Dover
    2:  (320, 370),   # Portsmouth
    3:  (280, 360),   # Southampton
    4:  (400, 310),   # Canterbury
    5:  (370, 360),   # Brighton
    # Oceania — Southwest + Wales
    6:  (220, 340),   # Bristol
    7:  (170, 390),   # Plymouth
    8:  (190, 310),   # Cardiff
    # Oceania — Midlands + North
    9:  (300, 260),   # Birmingham
    10: (280, 210),   # Manchester
    11: (240, 200),   # Liverpool
    12: (320, 210),   # Leeds
    # Oceania — East Anglia
    13: (400, 260),   # Norwich
    # Oceania — Scotland
    14: (300, 130),   # Edinburgh
    15: (250, 130),   # Glasgow
    # Oceania — Ireland
    16: (140, 210),   # Dublin
    17: (170, 160),   # Belfast
    # Eurasia — Channel Front
    18: (470, 370),   # Calais
    19: (450, 400),   # Dunkirk
    20: (420, 430),   # Le Havre
    21: (370, 450),   # Cherbourg
    # Eurasia — Northern France
    22: (470, 430),   # Amiens
    23: (430, 460),   # Rouen
    24: (490, 400),   # Lille
    # Eurasia — Benelux (expanded)
    25: (510, 370),   # Brussels
    26: (520, 340),   # Antwerp
    27: (550, 310),   # Rotterdam
    28: (550, 280),   # Amsterdam
    29: (540, 400),   # Luxembourg
    # Eurasia — Central France
    30: (470, 490),   # Paris
    31: (460, 540),   # Orleans
    32: (510, 570),   # Lyon
    # Eurasia — Atlantic France
    33: (310, 490),   # Brest
    34: (380, 560),   # Bordeaux
}

# Sea zone approximate centers (for overlay labels)
SEA_ZONE_POS = {
    0: (440, 355),   # Dover Strait
    1: (370, 400),   # Western Channel
    2: (430, 250),   # North Sea
    3: (180, 180),   # Irish Sea
    4: (250, 470),   # Bay of Biscay
    5: (100, 100),   # North Atlantic
}

SEA_ZONE_NAMES = [
    "Dover Strait", "W. Channel", "North Sea",
    "Irish Sea", "Bay of Biscay", "N. Atlantic",
]

# Adjacency for drawing connections between sectors
SECTOR_CONNECTIONS = [
    (0, 1), (0, 2), (0, 4), (0, 5), (0, 9),  # London hub
    (1, 4), (1, 5), (2, 3), (2, 5), (3, 6),
    (6, 7), (6, 8), (6, 9), (8, 9),
    (9, 10), (9, 12), (10, 11), (10, 12), (11, 12),
    (12, 13), (13, 14), (14, 15), (11, 15),
    (11, 16), (15, 17), (16, 17),
    # Eurasia internal
    (18, 19), (18, 24), (18, 25), (19, 20), (19, 24),
    (20, 21), (20, 22), (20, 23), (22, 23), (22, 24), (22, 30),
    (23, 30), (24, 25), (25, 26), (25, 29),
    (26, 27), (27, 28), (29, 30),
    (30, 31), (31, 32), (31, 34),
    (21, 33), (33, 34),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# GUI Application                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class AirStripOneGUI:
    def __init__(self, seed=42, max_turns=100, speed=1.0):
        pygame.init()
        self.W, self.H = 1400, 900
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("GRAVITAS Engine — Air Strip One 1984")

        # Fonts
        self.font_title = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 13)
        self.font_small = pygame.font.SysFont("monospace", 11)
        self.font_tiny = pygame.font.SysFont("monospace", 9)

        # Game
        self.game = create_game(seed=seed, max_turns=max_turns)
        self.rng = np.random.default_rng(seed)
        self.feedback = {}
        self.turn_log = []
        self.dispatch = ""
        self.visible_events = ""

        # State
        self.paused = True
        self.speed = speed
        self.last_step_time = 0
        self.show_names = True
        self.show_fleets = True
        self.selected_sector = None
        self.running = True

        # Map offset for scrolling
        self.map_offset_x = 0
        self.map_offset_y = 0

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            dt = clock.tick(30) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n and self.paused:
                    self._step_turn()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.speed = min(10.0, self.speed * 1.5)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(0.1, self.speed / 1.5)
                elif event.key == pygame.K_s:
                    self.show_names = not self.show_names
                elif event.key == pygame.K_f:
                    self.show_fleets = not self.show_fleets
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Check sector clicks
                for sid, (sx, sy) in SECTOR_POS.items():
                    if (mx - sx) ** 2 + (my - sy) ** 2 < 225:  # 15px radius
                        self.selected_sector = sid
                        break

    def _update(self, dt):
        if not self.paused and not self.game.game_over:
            self.last_step_time += dt * self.speed
            if self.last_step_time >= 1.0:
                self.last_step_time = 0
                self._step_turn()

    def _step_turn(self):
        if self.game.game_over:
            return
        self.feedback = step_game(self.game, self.rng)
        self.visible_events = generate_visible_events(self.game, self.feedback)
        # Simple dispatch (no LLM in GUI mode)
        self.dispatch = self.visible_events
        self.turn_log.append({
            "turn": self.game.turn,
            "scores": dict(self.game.faction_scores),
            "events": self.visible_events,
        })

    def _draw(self):
        self.screen.fill(DARK_GREY)

        # Map area (left 700px)
        self._draw_map()

        # Right panel (700-1400)
        self._draw_right_panel()

        # Bottom dispatch bar
        self._draw_bottom_bar()

        # Top bar
        self._draw_top_bar()

    # ── Map Drawing ────────────────────────────────────────────────────── #

    def _draw_map(self):
        # Sea background
        sea_rect = pygame.Rect(0, 60, 700, 680)
        pygame.draw.rect(self.screen, SEA_DARK, sea_rect)

        # Draw connections
        for a, b in SECTOR_CONNECTIONS:
            if a in SECTOR_POS and b in SECTOR_POS:
                pa = SECTOR_POS[a]
                pb = SECTOR_POS[b]
                color = MED_GREY
                pygame.draw.line(self.screen, color, pa, pb, 1)

        # Draw sea zone labels
        for zid, (zx, zy) in SEA_ZONE_POS.items():
            label = self.font_tiny.render(SEA_ZONE_NAMES[zid], True, SEA_LIGHT)
            self.screen.blit(label, (zx - label.get_width() // 2, zy))

        # Draw sectors
        for sid, (sx, sy) in SECTOR_POS.items():
            owner = self.game.cluster_owners.get(sid, -1)

            # Color by owner
            if owner == 0:
                color = OCEANIA_BLUE
                border = OCEANIA_LIGHT
            elif owner == 1:
                color = EURASIA_RED
                border = EURASIA_LIGHT
            elif owner == 2:
                color = BLF_GREEN
                border = BLF_LIGHT
            else:
                color = MED_GREY
                border = LIGHT_GREY

            # Check if contested (multiple factions have land units)
            if self.game.land is not None:
                alive = self.game.land.alive_garrison(sid)
                factions = set(u.faction_id for u in alive)
                if len(factions) > 1:
                    color = CONTESTED_YELLOW
                    border = WHITE

            radius = 12
            # Larger for capitals
            if sid == 0 or sid == 30:  # London or Paris
                radius = 16

            # Selected highlight
            if sid == self.selected_sector:
                pygame.draw.circle(self.screen, ACCENT_GOLD, (sx, sy), radius + 4, 2)

            pygame.draw.circle(self.screen, color, (sx, sy), radius)
            pygame.draw.circle(self.screen, border, (sx, sy), radius, 2)

            # Sector name
            if self.show_names and sid < len(self.game.cluster_names):
                name = self.game.cluster_names[sid]
                label = self.font_tiny.render(name, True, WHITE)
                self.screen.blit(label, (sx - label.get_width() // 2, sy + radius + 2))

            # Land unit count badge
            if self.game.land is not None:
                alive = self.game.land.alive_garrison(sid)
                if len(alive) > 0:
                    badge = self.font_tiny.render(str(len(alive)), True, WHITE)
                    bx = sx + radius - 2
                    by = sy - radius - 2
                    pygame.draw.circle(self.screen, BLACK, (bx, by), 7)
                    self.screen.blit(badge, (bx - badge.get_width() // 2, by - badge.get_height() // 2))

        # Fleet indicators in sea zones
        if self.show_fleets:
            for zid in range(min(6, len(self.game.naval.sea_zones))):
                zone = self.game.naval.sea_zones[zid]
                zx, zy = SEA_ZONE_POS.get(zid, (0, 0))
                oc = sum(1 for f in zone.fleets if f.faction_id == 0 for _ in f.operational_ships)
                eu = sum(1 for f in zone.fleets if f.faction_id == 1 for _ in f.operational_ships)
                if oc > 0 or eu > 0:
                    fleet_text = f"{oc}O/{eu}E"
                    ft = self.font_tiny.render(fleet_text, True, ACCENT_GOLD)
                    self.screen.blit(ft, (zx - ft.get_width() // 2, zy + 12))

    # ── Right Panel ────────────────────────────────────────────────────── #

    def _draw_right_panel(self):
        panel = pygame.Rect(700, 60, 700, self.H - 60)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.line(self.screen, PANEL_BORDER, (700, 60), (700, self.H), 2)

        x0 = 715
        y = 75

        # ── Faction Scores ──
        y = self._draw_section_header("FACTION SCORES", x0, y)
        for fid, fname in self.game.faction_names.items():
            score = self.game.faction_scores.get(fid, 0)
            colors = {0: OCEANIA_LIGHT, 1: EURASIA_LIGHT, 2: BLF_LIGHT}
            c = colors.get(fid, WHITE)
            text = f"  {fname}: {score:.0f}"
            self._text(text, x0, y, c, self.font_medium)
            y += 16
        y += 8

        # ── Military Overview ──
        y = self._draw_section_header("MILITARY FORCES", x0, y)
        oc_ships = len(self.game.naval.faction_ships(0))
        eu_ships = len(self.game.naval.faction_ships(1))
        oc_sqs = len(self.game.air.faction_squadrons(0))
        eu_sqs = len(self.game.air.faction_squadrons(1))
        oc_land = 0
        eu_land = 0
        if self.game.land:
            for units in self.game.land.garrisons.values():
                for u in units:
                    if u.is_alive:
                        if u.faction_id == 0: oc_land += 1
                        elif u.faction_id == 1: eu_land += 1

        self._text(f"  Naval:  Oceania {oc_ships} | Eurasia {eu_ships}", x0, y, TEXT_DIM); y += 14
        self._text(f"  Air:    Oceania {oc_sqs} | Eurasia {eu_sqs}", x0, y, TEXT_DIM); y += 14
        self._text(f"  Land:   Oceania {oc_land} | Eurasia {eu_land}", x0, y, TEXT_DIM); y += 14
        y += 8

        # ── BLF Status ──
        if self.game.resistance:
            blf = self.game.resistance
            y = self._draw_section_header("BLF RESISTANCE", x0, y)
            esc_names = {0:"Dormant",1:"Whispers",2:"Sabotage",3:"Organized",4:"Open Revolt",5:"REVOLUTION",6:"Betrayed"}
            esc = esc_names.get(blf.escalation.value, "?")
            c = ACCENT_RED if blf.escalation.value >= 4 else BLF_LIGHT if blf.escalation.value >= 2 else TEXT_DIM
            self._text(f"  Escalation: {esc} (Level {blf.escalation.value})", x0, y, c); y += 14
            self._text(f"  Members: {blf.total_members} | Cells: {len(blf.active_cells)} | Arms: {blf.arms_caches}", x0, y, TEXT_DIM); y += 14
            w_status = "ALIVE" if blf.winston.is_alive and not blf.winston.is_captured else "CAPTURED" if blf.winston.is_captured else "DEAD"
            self._text(f"  Winston: {w_status} | Heat: {blf.winston.detection_heat:.0%} | Legend: {blf.winston.legend_level:.0%}", x0, y, TEXT_DIM); y += 14
            y += 8

        # ── Invasions ──
        active_inv = [inv for inv in self.game.invasions if inv.is_active]
        if active_inv:
            y = self._draw_section_header(f"ACTIVE INVASIONS ({len(active_inv)})", x0, y)
            for inv in active_inv[:5]:
                origin = self.game.cluster_names[inv.origin_cluster] if 0 <= inv.origin_cluster < len(self.game.cluster_names) else "air"
                target = self.game.cluster_names[inv.target_cluster] if inv.target_cluster < len(self.game.cluster_names) else "?"
                side = "O" if inv.faction_id == 0 else "E"
                c = OCEANIA_LIGHT if inv.faction_id == 0 else EURASIA_LIGHT
                self._text(f"  [{side}] {origin}->{target} [{inv.phase.name}]", x0, y, c); y += 14
            y += 8

        # ── Research ──
        if self.game.research:
            y = self._draw_section_header("RESEARCH", x0, y)
            for fid in [0, 1]:
                fr = self.game.research.factions.get(fid)
                if fr:
                    fname = "O" if fid == 0 else "E"
                    active = [p for p in fr.active_projects if not p.is_complete]
                    completed = sum(fr.levels.values())
                    c = OCEANIA_LIGHT if fid == 0 else EURASIA_LIGHT
                    if active:
                        names = ", ".join(p.branch.name[:4] for p in active)
                        self._text(f"  [{fname}] {completed} techs | Researching: {names}", x0, y, c)
                    else:
                        self._text(f"  [{fname}] {completed} techs | IDLE", x0, y, c)
                    y += 14
            y += 8

        # ── Selected Sector Detail ──
        if self.selected_sector is not None and self.selected_sector < len(self.game.cluster_names):
            sid = self.selected_sector
            y = self._draw_section_header(f"SECTOR: {self.game.cluster_names[sid]} (#{sid})", x0, y)
            owner = self.game.cluster_owners.get(sid, -1)
            owner_name = self.game.faction_names.get(owner, "None")
            self._text(f"  Owner: {owner_name}", x0, y, TEXT_DIM); y += 14

            if sid < len(self.game.cluster_data):
                d = self.game.cluster_data[sid]
                self._text(f"  Stability: {d[0]:.0%} | Hazard: {d[1]:.0%} | Resources: {d[2]:.0%}", x0, y, TEXT_DIM); y += 14
                self._text(f"  Military: {d[3]:.0%} | Trust: {d[4]:.0%} | Polarization: {d[5]:.0%}", x0, y, TEXT_DIM); y += 14

            if self.game.land:
                alive = self.game.land.alive_garrison(sid)
                if alive:
                    by_faction = {}
                    for u in alive:
                        by_faction.setdefault(u.faction_id, []).append(u)
                    for fid, units in by_faction.items():
                        fname = self.game.faction_names.get(fid, f"F{fid}")
                        self._text(f"  {fname}: {len(units)} units ({sum(u.hp for u in units):.0f} HP)", x0, y, TEXT_DIM); y += 14

    # ── Top Bar ────────────────────────────────────────────────────────── #

    def _draw_top_bar(self):
        bar = pygame.Rect(0, 0, self.W, 55)
        pygame.draw.rect(self.screen, BLACK, bar)
        pygame.draw.line(self.screen, PANEL_BORDER, (0, 55), (self.W, 55), 2)

        # Title
        title = self.font_title.render("AIR STRIP ONE — 1984", True, ACCENT_GOLD)
        self.screen.blit(title, (15, 8))

        # Turn info
        week = self.game.turn
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month = month_names[min((week // 4) % 12, 11)]
        year = 1984 + week // 52

        turn_text = f"WEEK {week}/{self.game.max_turns} | {month} {year}"
        tt = self.font_large.render(turn_text, True, WHITE)
        self.screen.blit(tt, (300, 12))

        # Scores
        o_score = self.game.faction_scores.get(0, 0)
        e_score = self.game.faction_scores.get(1, 0)
        score_text = f"Oceania: {o_score:.0f}  |  Eurasia: {e_score:.0f}"
        st = self.font_large.render(score_text, True, WHITE)
        self.screen.blit(st, (550, 12))

        # Controls
        status = "PAUSED" if self.paused else f"RUNNING x{self.speed:.1f}"
        sc = ACCENT_RED if self.paused else ACCENT_GREEN
        status_text = self.font_medium.render(status, True, sc)
        self.screen.blit(status_text, (15, 35))

        controls = "SPACE=Play/Pause  N=Next  +/-=Speed  S=Names  F=Fleets  ESC=Quit"
        ct = self.font_tiny.render(controls, True, TEXT_DIM)
        self.screen.blit(ct, (200, 38))

        if self.game.game_over:
            winner = self.game.faction_names.get(self.game.winner, "?")
            go = self.font_title.render(f"GAME OVER — {winner} WINS", True, ACCENT_GOLD)
            self.screen.blit(go, (self.W // 2 - go.get_width() // 2, 12))

    # ── Bottom Bar (Dispatch) ──────────────────────────────────────────── #

    def _draw_bottom_bar(self):
        bar_h = 160
        bar = pygame.Rect(0, self.H - bar_h, 700, bar_h)
        pygame.draw.rect(self.screen, BLACK, bar)
        pygame.draw.line(self.screen, PANEL_BORDER, (0, self.H - bar_h), (700, self.H - bar_h), 2)

        x0 = 10
        y = self.H - bar_h + 5
        header = self.font_medium.render("WAR CORRESPONDENT DISPATCH", True, ACCENT_GOLD)
        self.screen.blit(header, (x0, y))
        y += 18

        # Word-wrap the dispatch
        if self.dispatch:
            words = self.dispatch.split()
            line = ""
            max_w = 680
            for word in words:
                test = line + " " + word if line else word
                tw = self.font_small.size(test)[0]
                if tw > max_w:
                    self._text(line, x0, y, TEXT_DIM, self.font_small)
                    y += 13
                    line = word
                    if y > self.H - 10:
                        break
                else:
                    line = test
            if line and y < self.H - 10:
                self._text(line, x0, y, TEXT_DIM, self.font_small)

    # ── Helpers ────────────────────────────────────────────────────────── #

    def _text(self, text, x, y, color=WHITE, font=None):
        if font is None:
            font = self.font_medium
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_section_header(self, title, x, y):
        self._text(title, x, y, ACCENT_GOLD, self.font_large)
        y += 20
        pygame.draw.line(self.screen, PANEL_BORDER, (x, y - 4), (x + 300, y - 4), 1)
        return y


# ═══════════════════════════════════════════════════════════════════════════ #
# Main                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(description="Air Strip One — Strategic Map GUI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    gui = AirStripOneGUI(seed=args.seed, max_turns=args.turns, speed=args.speed)
    gui.run()


if __name__ == "__main__":
    main()
