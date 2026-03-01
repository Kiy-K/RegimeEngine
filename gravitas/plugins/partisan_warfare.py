"""
partisan_warfare.py — Partisan Warfare Plugin for GRAVITAS Engine.

Models asymmetric partisan warfare as an automated third force operating
from contested and Soviet-controlled sectors:

1. **Recruitment**: Partisans recruit from sectors with low Axis stability
   and high trust (population sympathetic). Costs Soviet resources.
   Recruitment rate = base × trust × (1 − σ_axis) — nonlinear in both.

2. **Sabotage**: Targeted attacks on Axis logistics links and rail hubs.
   Success probability depends on link vulnerability and partisan strength.
   Disrupts supply flow for multiple turns. Cascading downstream effects.

3. **Ambush**: Hit-and-run attacks on Axis military in adjacent sectors.
   Damage = partisan_strength × surprise_factor × terrain_bonus.
   Forest terrain gives 30% bonus. Ambush degrades Axis morale (trust).

4. **Stealth & Detection**: Partisans have a detection probability each turn.
   detection_prob = base × axis_military × (1 − forest_bonus).
   If detected: partisan strength reduced, hazard spike in sector.

5. **Movement**: Partisans relocate after actions (hit-and-run doctrine).
   Prefer forest/contested sectors. Movement through open terrain is risky.

6. **Propaganda**: Partisans conduct propaganda in contested zones,
   increasing Soviet trust and polarization against Axis.

Partisan units are NOT controlled by either RL agent — they are an
autonomous stochastic force that creates uncertainty both sides must
adapt to. This prevents either agent from perfectly predicting outcomes.

Author: GRAVITAS Engine
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gravitas.plugins import GravitasPlugin

logger = logging.getLogger("gravitas.plugins.partisan_warfare")


# ─────────────────────────────────────────────────────────────────────────── #
# Partisan unit                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class PartisanUnit:
    """A partisan unit operating behind enemy lines."""
    uid: int
    sector: int
    strength: float = 0.10       # [0, 1] — combat effectiveness
    experience: float = 0.0      # [0, 1] — accumulated experience
    stealth: float = 0.85        # [0, 1] — chance to avoid detection
    morale: float = 0.70         # [0, 1] — willingness to fight
    cooldown: int = 0            # turns until next action

    @property
    def effective_strength(self) -> float:
        """Combat effectiveness factoring in experience and morale."""
        return self.strength * (0.5 + 0.5 * self.experience) * self.morale

    def tick_cooldown(self):
        self.cooldown = max(0, self.cooldown - 1)


# ─────────────────────────────────────────────────────────────────────────── #
# Partisan decision engine                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _select_sabotage_target(
    unit: PartisanUnit,
    axis_sectors: Set[int],
    adjacency: Dict[int, List[int]],
    c_arr: np.ndarray,
    N: int,
) -> Optional[int]:
    """Select best adjacent Axis sector to sabotage (highest resources)."""
    neighbors = adjacency.get(unit.sector, [])
    candidates = [(j, c_arr[j, 2]) for j in neighbors if j in axis_sectors and j < N]
    if not candidates:
        return None
    # Probabilistic: weight by resource level (target high-value)
    targets, weights = zip(*candidates)
    weights = np.array(weights) + 0.01
    weights /= weights.sum()
    return int(np.random.choice(targets, p=weights))


def _select_ambush_target(
    unit: PartisanUnit,
    axis_sectors: Set[int],
    adjacency: Dict[int, List[int]],
    c_arr: np.ndarray,
    N: int,
) -> Optional[int]:
    """Select best adjacent Axis sector to ambush (highest military, lowest σ)."""
    neighbors = adjacency.get(unit.sector, [])
    candidates = [(j, c_arr[j, 3] * (1.0 - c_arr[j, 0])) for j in neighbors if j in axis_sectors and j < N]
    if not candidates:
        return None
    targets, scores = zip(*candidates)
    scores = np.array(scores) + 0.01
    scores /= scores.sum()
    return int(np.random.choice(targets, p=scores))


def _select_movement_target(
    unit: PartisanUnit,
    safe_sectors: Set[int],
    adjacency: Dict[int, List[int]],
    terrain_map: Dict[int, str],
    N: int,
) -> Optional[int]:
    """Select movement destination (prefer forest/contested sectors)."""
    neighbors = adjacency.get(unit.sector, [])
    candidates = []
    for j in neighbors:
        if j >= N:
            continue
        if j in safe_sectors:
            terrain = terrain_map.get(j, "open")
            score = 1.0
            if terrain == "forest":
                score = 3.0
            elif terrain == "urban":
                score = 1.5
            candidates.append((j, score))
    if not candidates:
        return None
    targets, scores = zip(*candidates)
    scores = np.array(scores)
    scores /= scores.sum()
    return int(np.random.choice(targets, p=scores))


# ─────────────────────────────────────────────────────────────────────────── #
# Plugin                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class PartisanWarfarePlugin(GravitasPlugin):
    """
    Partisan warfare plugin — autonomous asymmetric force.

    Manages a pool of partisan units that recruit, sabotage, ambush,
    and move independently of both RL agents. Creates irreducible
    uncertainty in the simulation that neither agent can fully control.
    """

    name = "partisan_warfare"
    version = "1.0.0"
    description = (
        "Autonomous partisan warfare: recruitment, sabotage, ambush, "
        "stealth/detection, movement, and propaganda."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}

        # ── Tunable parameters ────────────────────────────────────── #
        self.recruitment_interval = cfg.get("recruitment_interval", 25)
        self.action_interval = cfg.get("action_interval", 8)
        self.max_partisans = cfg.get("max_partisans", 6)
        self.recruitment_resource_cost = cfg.get("recruitment_resource_cost", 0.08)
        self.recruitment_base_strength = cfg.get("recruitment_base_strength", 0.12)
        self.sabotage_resource_damage = cfg.get("sabotage_resource_damage", 0.12)
        self.sabotage_hazard_boost = cfg.get("sabotage_hazard_boost", 0.10)
        self.ambush_military_damage = cfg.get("ambush_military_damage", 0.06)
        self.ambush_trust_damage = cfg.get("ambush_trust_damage", 0.04)
        self.detection_base_prob = cfg.get("detection_base_prob", 0.12)
        self.detection_strength_loss = cfg.get("detection_strength_loss", 0.30)
        self.propaganda_trust_boost = cfg.get("propaganda_trust_boost", 0.03)
        self.propaganda_polar_boost = cfg.get("propaganda_polar_boost", 0.02)
        self.experience_gain_per_action = cfg.get("experience_gain_per_action", 0.05)
        self.morale_decay_rate = cfg.get("morale_decay_rate", 0.01)
        self.action_cooldown = cfg.get("action_cooldown", 5)

        # ── Scenario data ─────────────────────────────────────────── #
        self._axis_clusters: List[int] = []
        self._soviet_clusters: List[int] = []
        self._contested_clusters: List[int] = []
        self._adjacency: Dict[int, List[int]] = {}
        self._terrain_map: Dict[int, str] = {}
        self._recruitment_sectors: List[int] = []  # sectors where partisans can spawn

        # ── Partisan units ────────────────────────────────────────── #
        self._partisans: List[PartisanUnit] = []
        self._next_uid: int = 0

        # ── Random state ──────────────────────────────────────────── #
        self._rng = np.random.default_rng(42)

    def on_reset(self, world, **kwargs):
        """Initialize partisan force."""
        N = world.n_clusters
        self._partisans = []
        self._next_uid = 0
        self._adjacency = {}
        self._terrain_map = {}

        engine = kwargs.get("engine", None)
        if engine and hasattr(engine, "_scenario_meta"):
            meta = engine._scenario_meta
            self._axis_clusters = meta.get("axis_clusters", [])
            self._soviet_clusters = meta.get("soviet_clusters", [])
            self._contested_clusters = meta.get("contested_clusters", [])

            for sector in meta.get("sectors", []):
                sid = sector.get("id", -1)
                self._terrain_map[sid] = sector.get("terrain", "open")

            for link in meta.get("logistics_links", []):
                a, b = link.get("from", -1), link.get("to", -1)
                self._adjacency.setdefault(a, []).append(b)
                self._adjacency.setdefault(b, []).append(a)

        # Fallback
        if not self._axis_clusters:
            env = kwargs.get("env", None)
            if env and hasattr(env, "axis_clusters"):
                self._axis_clusters = list(env.axis_clusters)
                self._soviet_clusters = list(env.soviet_clusters)
                self._contested_clusters = list(getattr(env, "contested_clusters", []))

        if not self._adjacency:
            for i in range(N):
                neighbors = []
                if i > 0:
                    neighbors.append(i - 1)
                if i < N - 1:
                    neighbors.append(i + 1)
                self._adjacency[i] = neighbors

        # Recruitment sectors: contested + forest sectors adjacent to Soviet territory
        self._recruitment_sectors = list(self._contested_clusters)
        soviet_set = set(self._soviet_clusters)
        for i in range(N):
            if self._terrain_map.get(i) == "forest":
                # Check if adjacent to Soviet sector
                if any(j in soviet_set for j in self._adjacency.get(i, [])):
                    if i not in self._recruitment_sectors:
                        self._recruitment_sectors.append(i)

        # Spawn initial partisan unit in contested zone
        if self._contested_clusters:
            self._spawn_partisan(self._contested_clusters[0], strength=0.10)

        return world

    def on_step(self, world, turn: int, **kwargs) -> object:
        """Execute partisan operations."""
        N = world.n_clusters
        c_arr = world.cluster_array()

        axis_set = set(self._axis_clusters)
        soviet_set = set(self._soviet_clusters)
        contested_set = set(self._contested_clusters)
        safe_sectors = soviet_set | contested_set

        # ── Tick cooldowns ───────────────────────────────────────── #
        for p in self._partisans:
            p.tick_cooldown()

        # ── Recruitment ──────────────────────────────────────────── #
        if turn % self.recruitment_interval == 0 and len(self._partisans) < self.max_partisans:
            c_arr = self._do_recruitment(c_arr, N, soviet_set, turn)

        # ── Partisan actions ─────────────────────────────────────── #
        if turn % self.action_interval == 0:
            for p in list(self._partisans):
                if p.cooldown > 0:
                    continue

                # Decide action based on situation
                action = self._decide_action(p, c_arr, N, axis_set, soviet_set)

                if action == "sabotage":
                    c_arr = self._do_sabotage(p, c_arr, N, axis_set, turn)
                elif action == "ambush":
                    c_arr = self._do_ambush(p, c_arr, N, axis_set, turn)
                elif action == "propaganda":
                    c_arr = self._do_propaganda(p, c_arr, N, contested_set, turn)
                elif action == "move":
                    self._do_movement(p, safe_sectors, N)

                # Gain experience
                if action in ("sabotage", "ambush"):
                    p.experience = min(1.0, p.experience + self.experience_gain_per_action)

                p.cooldown = self.action_cooldown

        # ── Detection checks ─────────────────────────────────────── #
        if turn % self.action_interval == 0:
            c_arr = self._do_detection(c_arr, N, axis_set, turn)

        # ── Morale decay ─────────────────────────────────────────── #
        for p in self._partisans:
            p.morale = max(0.1, p.morale - self.morale_decay_rate)
            # Morale recovery from Soviet resources
            if p.sector in soviet_set and p.sector < N:
                p.morale = min(1.0, p.morale + 0.005)

        # ── Remove destroyed partisans ───────────────────────────── #
        self._partisans = [p for p in self._partisans if p.strength > 0.02]

        # ── Write back ───────────────────────────────────────────── #
        world = self._apply_cluster_array(world, c_arr)

        # ── Log status ───────────────────────────────────────────── #
        if turn % 50 == 0:
            self.log_event(
                turn=turn,
                message="partisan_status",
                data={
                    "active_units": len(self._partisans),
                    "total_strength": round(sum(p.strength for p in self._partisans), 3),
                    "sectors": [p.sector for p in self._partisans],
                    "mean_experience": round(
                        np.mean([p.experience for p in self._partisans]) if self._partisans else 0.0, 3
                    ),
                },
            )

        return world

    # ── Partisan operations ───────────────────────────────────────── #

    def _decide_action(
        self,
        unit: PartisanUnit,
        c_arr: np.ndarray,
        N: int,
        axis_set: Set[int],
        soviet_set: Set[int],
    ) -> str:
        """Simple stochastic decision tree for partisan action."""
        neighbors = self._adjacency.get(unit.sector, [])
        has_axis_neighbor = any(j in axis_set and j < N for j in neighbors)

        if not has_axis_neighbor:
            # No Axis nearby — move or propagandize
            return self._rng.choice(["move", "propaganda"], p=[0.6, 0.4])

        # Near Axis: choose based on strength and situation
        if unit.effective_strength > 0.15:
            # Strong enough for direct action
            return self._rng.choice(
                ["sabotage", "ambush", "propaganda", "move"],
                p=[0.40, 0.30, 0.15, 0.15],
            )
        else:
            # Weak — prefer sabotage (lower risk) or movement
            return self._rng.choice(
                ["sabotage", "propaganda", "move"],
                p=[0.45, 0.25, 0.30],
            )

    def _do_recruitment(
        self,
        c_arr: np.ndarray,
        N: int,
        soviet_set: Set[int],
        turn: int,
    ) -> np.ndarray:
        """Recruit a new partisan unit."""
        if not self._recruitment_sectors:
            return c_arr

        # Find best recruitment sector (high trust, low enemy presence)
        best_sector = -1
        best_score = -1.0
        for s in self._recruitment_sectors:
            if s >= N:
                continue
            trust = c_arr[s, 4]
            anti_axis = 1.0 - c_arr[s, 0]  # low σ = discontent
            score = trust * 0.6 + anti_axis * 0.4
            if score > best_score:
                best_score = score
                best_sector = s

        if best_sector < 0:
            return c_arr

        # Pay resource cost from nearest Soviet sector
        cost_sector = None
        for s in self._soviet_clusters:
            if s < N and c_arr[s, 2] > self.recruitment_resource_cost * 2:
                cost_sector = s
                break

        if cost_sector is None:
            return c_arr

        c_arr[cost_sector, 2] -= self.recruitment_resource_cost
        c_arr[cost_sector, 2] = max(0.0, c_arr[cost_sector, 2])

        # Strength scales with local trust (sympathetic population)
        strength = self.recruitment_base_strength * (0.5 + 0.5 * c_arr[best_sector, 4])
        self._spawn_partisan(best_sector, strength)

        self.log_event(
            turn=turn,
            message="partisan_recruited",
            data={
                "sector": best_sector,
                "strength": round(strength, 3),
                "cost_sector": cost_sector,
            },
        )
        return c_arr

    def _do_sabotage(
        self,
        unit: PartisanUnit,
        c_arr: np.ndarray,
        N: int,
        axis_set: Set[int],
        turn: int,
    ) -> np.ndarray:
        """Sabotage an adjacent Axis sector's resources."""
        target = _select_sabotage_target(unit, axis_set, self._adjacency, c_arr, N)
        if target is None:
            return c_arr

        terrain = self._terrain_map.get(unit.sector, "open")
        terrain_bonus = 1.3 if terrain == "forest" else 1.0

        # Sabotage damage scales with partisan effectiveness
        damage = self.sabotage_resource_damage * unit.effective_strength * terrain_bonus
        hazard = self.sabotage_hazard_boost * unit.effective_strength

        # Stochastic success (not guaranteed)
        success_prob = 0.6 + 0.3 * unit.experience
        if self._rng.random() > success_prob:
            # Failed sabotage — minor exposure
            unit.stealth = max(0.0, unit.stealth - 0.05)
            return c_arr

        c_arr[target, 2] = np.clip(c_arr[target, 2] - damage, 0.0, 1.0)
        c_arr[target, 1] = np.clip(c_arr[target, 1] + hazard, 0.0, 2.0)

        self.log_event(
            turn=turn,
            message="partisan_sabotage",
            data={
                "partisan_uid": unit.uid,
                "target_sector": target,
                "damage": round(damage, 3),
                "from_sector": unit.sector,
            },
        )
        return c_arr

    def _do_ambush(
        self,
        unit: PartisanUnit,
        c_arr: np.ndarray,
        N: int,
        axis_set: Set[int],
        turn: int,
    ) -> np.ndarray:
        """Ambush Axis military in adjacent sector."""
        target = _select_ambush_target(unit, axis_set, self._adjacency, c_arr, N)
        if target is None:
            return c_arr

        terrain = self._terrain_map.get(unit.sector, "open")
        terrain_bonus = 1.3 if terrain == "forest" else 1.0

        # Ambush damage
        mil_damage = self.ambush_military_damage * unit.effective_strength * terrain_bonus
        trust_damage = self.ambush_trust_damage * unit.effective_strength

        # Stochastic
        success_prob = 0.5 + 0.35 * unit.experience
        if self._rng.random() > success_prob:
            # Failed ambush — partisan takes damage
            unit.strength = max(0.0, unit.strength - 0.03)
            unit.stealth = max(0.0, unit.stealth - 0.10)
            return c_arr

        c_arr[target, 3] = np.clip(c_arr[target, 3] - mil_damage, 0.0, 1.0)
        c_arr[target, 4] = np.clip(c_arr[target, 4] - trust_damage, 0.0, 1.0)
        c_arr[target, 1] = np.clip(c_arr[target, 1] + 0.03, 0.0, 2.0)  # combat hazard

        # Partisan takes some damage too
        counter_damage = 0.02 * c_arr[target, 3]  # stronger garrison = more risk
        unit.strength = max(0.0, unit.strength - counter_damage)

        self.log_event(
            turn=turn,
            message="partisan_ambush",
            data={
                "partisan_uid": unit.uid,
                "target_sector": target,
                "mil_damage": round(mil_damage, 3),
                "from_sector": unit.sector,
            },
        )
        return c_arr

    def _do_propaganda(
        self,
        unit: PartisanUnit,
        c_arr: np.ndarray,
        N: int,
        contested_set: Set[int],
        turn: int,
    ) -> np.ndarray:
        """Conduct propaganda in contested/local sectors."""
        # Propaganda in own sector and adjacent contested sectors
        targets = [unit.sector]
        for j in self._adjacency.get(unit.sector, []):
            if j in contested_set and j < N:
                targets.append(j)

        for t in targets:
            if t >= N:
                continue
            trust_boost = self.propaganda_trust_boost * unit.effective_strength
            polar_boost = self.propaganda_polar_boost * unit.effective_strength
            c_arr[t, 4] = np.clip(c_arr[t, 4] + trust_boost, 0.0, 1.0)
            c_arr[t, 5] = np.clip(c_arr[t, 5] + polar_boost, 0.0, 1.0)

        return c_arr

    def _do_movement(self, unit: PartisanUnit, safe_sectors: Set[int], N: int):
        """Move partisan to a safer/better sector."""
        target = _select_movement_target(unit, safe_sectors, self._adjacency, self._terrain_map, N)
        if target is not None:
            unit.sector = target

    def _do_detection(
        self,
        c_arr: np.ndarray,
        N: int,
        axis_set: Set[int],
        turn: int,
    ) -> np.ndarray:
        """Check if partisans are detected by Axis forces."""
        for p in list(self._partisans):
            if p.sector >= N:
                continue

            # Detection probability depends on Axis military nearby and terrain
            nearby_axis_mil = 0.0
            for j in self._adjacency.get(p.sector, []):
                if j in axis_set and j < N:
                    nearby_axis_mil = max(nearby_axis_mil, c_arr[j, 3])

            # Also check if partisan is IN an Axis sector
            if p.sector in axis_set:
                nearby_axis_mil = max(nearby_axis_mil, c_arr[p.sector, 3])

            terrain = self._terrain_map.get(p.sector, "open")
            forest_bonus = 0.3 if terrain == "forest" else 0.0
            urban_bonus = 0.15 if terrain == "urban" else 0.0

            detection_prob = self.detection_base_prob * nearby_axis_mil * (1.0 - forest_bonus - urban_bonus)
            detection_prob *= (1.0 - p.stealth * 0.5)  # stealth reduces detection

            if self._rng.random() < detection_prob:
                # Detected! Partisan takes damage
                p.strength *= (1.0 - self.detection_strength_loss)
                p.morale = max(0.1, p.morale - 0.15)

                # Hazard spike in sector
                c_arr[p.sector, 1] = np.clip(c_arr[p.sector, 1] + 0.08, 0.0, 2.0)

                self.log_event(
                    turn=turn,
                    message="partisan_detected",
                    data={
                        "partisan_uid": p.uid,
                        "sector": p.sector,
                        "remaining_strength": round(p.strength, 3),
                    },
                )

                # Try to flee
                safe = set(self._soviet_clusters) | set(self._contested_clusters)
                self._do_movement(p, safe, N)

        return c_arr

    # ── Helpers ───────────────────────────────────────────────────── #

    def _spawn_partisan(self, sector: int, strength: float = 0.10):
        """Create a new partisan unit."""
        unit = PartisanUnit(
            uid=self._next_uid,
            sector=sector,
            strength=strength,
            stealth=0.80 + self._rng.random() * 0.15,
            morale=0.65 + self._rng.random() * 0.20,
        )
        self._partisans.append(unit)
        self._next_uid += 1

    @staticmethod
    def _apply_cluster_array(world, c_arr):
        """Write modified cluster array back to world via copy_with_clusters."""
        from dataclasses import replace as dc_replace
        clusters = [
            dc_replace(
                cluster,
                sigma=float(c_arr[i, 0]),
                hazard=float(c_arr[i, 1]),
                resource=float(c_arr[i, 2]),
                military=float(c_arr[i, 3]),
                trust=float(c_arr[i, 4]),
                polar=float(c_arr[i, 5]),
            )
            for i, cluster in enumerate(world.clusters)
            if i < c_arr.shape[0]
        ]
        return world.copy_with_clusters(clusters)


# Module-level alias required by plugin loader convention
Plugin = PartisanWarfarePlugin
