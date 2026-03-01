"""
nonlinear_combat.py — Nonlinear Combat Dynamics Plugin for GRAVITAS Engine.

Replaces linear combat resolution with historically-grounded nonlinear mechanics
that are harder for RL agents to exploit:

1. **Lanchester Square Law**: Attrition scales with force ratio² (S-curve),
   making concentration of force powerful but with diminishing returns.

2. **Diminishing Returns**: Military effectiveness = m^α where α < 1.
   Stacking 2× military does NOT yield 2× combat power.

3. **Breakthrough Sigmoid**: Meaningful territorial gains require force
   concentration above a threshold. Below it, attacker bleeds but defender
   σ doesn't drop. Prevents "death by a thousand cuts" exploitation.

4. **Combat Fatigue**: Sigmoid-based effectiveness decay tracking sustained
   combat turns per sector. Fatigue is low initially, accelerates, plateaus.

5. **Terrain Multipliers**: Nonlinear defender advantage based on terrain type
   (forest, urban, fortified, rail, open). Read from scenario YAML.

6. **Fog of War Variance**: When attacking sectors with low intelligence,
   damage variance increases (stochastic penalty for blind attacks).

Mathematical foundations:
  - Lanchester: fr = A/(A+D+ε), lanchester = fr²/(fr²+(1-fr)²)
  - Diminishing returns: eff = m^0.7
  - Breakthrough: σ_dmg = base × sigmoid((fr - threshold) × steepness)
  - Fatigue: fat = sigmoid((cum_turns - midpoint) × rate)
  - All functions are smooth and differentiable for stable training.

Author: GRAVITAS Engine
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gravitas.plugins import GravitasPlugin

logger = logging.getLogger("gravitas.plugins.nonlinear_combat")

# ─────────────────────────────────────────────────────────────────────────── #
# Terrain defaults (overridden by scenario YAML)                               #
# ─────────────────────────────────────────────────────────────────────────── #

DEFAULT_TERRAIN = {
    "urban":     {"defender_multiplier": 2.0, "hazard_multiplier": 1.3, "movement_cost": 1.5},
    "fortified": {"defender_multiplier": 2.5, "hazard_multiplier": 1.0, "movement_cost": 2.0},
    "forest":    {"defender_multiplier": 1.5, "hazard_multiplier": 0.8, "movement_cost": 1.8, "partisan_bonus": 0.3},
    "rail":      {"defender_multiplier": 0.8, "hazard_multiplier": 1.0, "movement_cost": 0.5, "sabotage_vulnerability": 0.2},
    "open":      {"defender_multiplier": 1.0, "hazard_multiplier": 1.0, "movement_cost": 1.0},
}


# ─────────────────────────────────────────────────────────────────────────── #
# Nonlinear math primitives                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def _lanchester_factor(attacker_mil: float, defender_mil: float, eps: float = 1e-6) -> float:
    """
    Lanchester Square Law S-curve.

    Returns value in [0, 1] representing attacker's combat advantage.
    0.5 = equal forces, >0.5 = attacker advantage.

    The square law means concentrating force yields superlinear advantage,
    but the S-curve saturates, preventing infinite exploitation.
    """
    fr = attacker_mil / (attacker_mil + defender_mil + eps)
    fr_sq = fr * fr
    inv_sq = (1.0 - fr) * (1.0 - fr)
    return fr_sq / (fr_sq + inv_sq + eps)


def _diminishing_returns(military: float, alpha: float = 0.7) -> float:
    """
    Sublinear military effectiveness.

    eff = m^α where α < 1 means doubling military gives < 2× effect.
    Prevents agents from "stacking" all military in one sector.
    """
    return np.clip(military, 0.0, 1.0) ** alpha


def _breakthrough_probability(
    force_ratio: float,
    threshold: float = 0.6,
    steepness: float = 8.0,
) -> float:
    """
    Sigmoid breakthrough function.

    Returns probability [0, 1] of meaningful territorial gain.
    Below threshold: attacker bleeds but defender σ doesn't drop much.
    Above threshold: breakthrough — large σ damage to defender.

    The sigmoid transition prevents sharp discontinuities that
    agents could exploit by hovering at the threshold.
    """
    return _sigmoid((force_ratio - threshold) * steepness)


def _combat_fatigue(
    cumulative_combat_turns: int,
    midpoint: float = 120.0,
    rate: float = 0.025,
) -> float:
    """
    Sigmoid fatigue curve.

    Low fatigue early → accelerating degradation → plateau near 1.0.
    Prevents agents from sustaining indefinite combat in one sector.
    """
    return _sigmoid((cumulative_combat_turns - midpoint) * rate)


# ─────────────────────────────────────────────────────────────────────────── #
# Plugin                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class NonlinearCombatPlugin(GravitasPlugin):
    """
    Nonlinear combat dynamics plugin.

    Modifies cluster_array after each ODE step to apply:
    - Lanchester attrition between adjacent opposing sectors
    - Diminishing returns on military concentration
    - Breakthrough mechanics for territorial gains
    - Combat fatigue accumulation
    - Terrain-dependent combat modifiers
    """

    name = "nonlinear_combat"
    version = "1.0.0"
    description = (
        "Nonlinear combat dynamics: Lanchester square law, diminishing returns, "
        "breakthrough sigmoid, combat fatigue, and terrain multipliers."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}

        # ── Tunable parameters ────────────────────────────────────── #
        self.alpha = cfg.get("diminishing_returns_alpha", 0.7)
        self.breakthrough_threshold = cfg.get("breakthrough_threshold", 0.58)
        self.breakthrough_steepness = cfg.get("breakthrough_steepness", 8.0)
        self.breakthrough_sigma_damage = cfg.get("breakthrough_sigma_damage", 0.06)
        self.attrition_base = cfg.get("attrition_base", 0.03)
        self.hazard_combat_rate = cfg.get("hazard_combat_rate", 0.04)
        self.fatigue_midpoint = cfg.get("fatigue_midpoint", 120.0)
        self.fatigue_rate = cfg.get("fatigue_rate", 0.025)
        self.fatigue_attrition_mult = cfg.get("fatigue_attrition_mult", 0.5)
        self.fog_variance_scale = cfg.get("fog_variance_scale", 0.15)
        self.trigger_turn_interval = cfg.get("trigger_turn_interval", 1)

        # ── Scenario data (set during on_reset) ──────────────────── #
        self._terrain_map: Dict[int, str] = {}
        self._terrain_defs: Dict[str, Dict] = dict(DEFAULT_TERRAIN)
        self._axis_clusters: List[int] = []
        self._soviet_clusters: List[int] = []
        self._contested_clusters: List[int] = []
        self._adjacency: Dict[int, List[int]] = {}

        # ── Per-sector state ──────────────────────────────────────── #
        self._combat_turns: Dict[int, int] = {}

        # ── Random state ──────────────────────────────────────────── #
        self._rng = np.random.default_rng(42)

    def on_reset(self, world, **kwargs):
        """Initialize terrain map, adjacency, and combat turn counters."""
        N = world.n_clusters
        self._combat_turns = {i: 0 for i in range(N)}

        # Try to read scenario metadata from engine
        engine = kwargs.get("engine", None)
        if engine and hasattr(engine, "_scenario_meta"):
            meta = engine._scenario_meta
            self._axis_clusters = meta.get("axis_clusters", [])
            self._soviet_clusters = meta.get("soviet_clusters", [])
            self._contested_clusters = meta.get("contested_clusters", [])

            # Build terrain map from sectors
            for sector in meta.get("sectors", []):
                sid = sector.get("id", -1)
                self._terrain_map[sid] = sector.get("terrain", "open")

            # Load terrain definitions
            terrain_defs = meta.get("terrain", {})
            if terrain_defs:
                self._terrain_defs.update(terrain_defs)

            # Build adjacency from logistics_links
            for link in meta.get("logistics_links", []):
                a, b = link.get("from", -1), link.get("to", -1)
                self._adjacency.setdefault(a, []).append(b)
                self._adjacency.setdefault(b, []).append(a)

        # Fallback: if no metadata, infer from world
        if not self._axis_clusters:
            env = kwargs.get("env", None)
            if env and hasattr(env, "axis_clusters"):
                self._axis_clusters = list(env.axis_clusters)
                self._soviet_clusters = list(env.soviet_clusters)
                self._contested_clusters = list(getattr(env, "contested_clusters", []))

        # Fallback adjacency: sequential neighbors
        if not self._adjacency:
            for i in range(N):
                neighbors = []
                if i > 0:
                    neighbors.append(i - 1)
                if i < N - 1:
                    neighbors.append(i + 1)
                self._adjacency[i] = neighbors

        return world

    def on_step(self, world, turn: int, **kwargs) -> object:
        """Apply nonlinear combat dynamics each step."""
        if turn % self.trigger_turn_interval != 0:
            return world

        N = world.n_clusters
        c_arr = world.cluster_array()  # (N, 6): σ, h, r, m, τ, φ

        axis_set = set(self._axis_clusters)
        soviet_set = set(self._soviet_clusters)
        contested_set = set(self._contested_clusters)

        # ── Identify combat zones ────────────────────────────────── #
        # A sector is in combat if it has opposing forces nearby
        for i in range(N):
            neighbors = self._adjacency.get(i, [])
            i_side = self._get_side(i, axis_set, soviet_set, contested_set)
            terrain = self._terrain_map.get(i, "open")
            terrain_def = self._terrain_defs.get(terrain, DEFAULT_TERRAIN["open"])

            # Check for adjacent opposing sectors
            opposing_mil = 0.0
            opposing_count = 0
            for j in neighbors:
                if j >= N:
                    continue
                j_side = self._get_side(j, axis_set, soviet_set, contested_set)
                if self._are_opposing(i_side, j_side):
                    opposing_mil += _diminishing_returns(c_arr[j, 3], self.alpha)
                    opposing_count += 1

            if opposing_count == 0:
                # No combat — reduce fatigue slowly
                self._combat_turns[i] = max(0, self._combat_turns.get(i, 0) - 2)
                continue

            # ── This sector is in a combat zone ──────────────────── #
            self._combat_turns[i] = self._combat_turns.get(i, 0) + 1

            own_mil = _diminishing_returns(c_arr[i, 3], self.alpha)
            defender_mult = terrain_def.get("defender_multiplier", 1.0)
            hazard_mult = terrain_def.get("hazard_multiplier", 1.0)

            # Effective military with defender terrain bonus
            effective_own = own_mil * (defender_mult if i_side != "contested" else 1.0)
            effective_opp = opposing_mil / max(opposing_count, 1)

            # ── Lanchester factor ────────────────────────────────── #
            lanchester = _lanchester_factor(effective_opp, effective_own)

            # ── Combat fatigue ───────────────────────────────────── #
            fatigue = _combat_fatigue(
                self._combat_turns[i],
                midpoint=self.fatigue_midpoint,
                rate=self.fatigue_rate,
            )

            # ── Fog of war variance ──────────────────────────────── #
            fog_noise = self._rng.normal(0, self.fog_variance_scale * (1.0 - c_arr[i, 0]))

            # ── Attrition: military loss ─────────────────────────── #
            base_attrition = self.attrition_base * lanchester
            fatigue_attrition = self.fatigue_attrition_mult * fatigue
            total_attrition = (base_attrition + fatigue_attrition) * (1.0 + fog_noise)
            total_attrition = np.clip(total_attrition, 0.0, 0.15)

            c_arr[i, 3] = np.clip(c_arr[i, 3] - total_attrition, 0.0, 1.0)

            # ── Hazard increase from combat ──────────────────────── #
            hazard_increase = self.hazard_combat_rate * lanchester * hazard_mult
            hazard_increase *= (1.0 + 0.3 * fatigue)  # fatigue makes combat messier
            c_arr[i, 1] = np.clip(c_arr[i, 1] + hazard_increase, 0.0, 2.0)

            # ── Breakthrough: σ damage ───────────────────────────── #
            breakthrough = _breakthrough_probability(
                lanchester,
                threshold=self.breakthrough_threshold,
                steepness=self.breakthrough_steepness,
            )
            sigma_damage = self.breakthrough_sigma_damage * breakthrough
            sigma_damage *= (1.0 + fog_noise * 0.5)  # uncertainty in breakthrough effect
            sigma_damage = np.clip(sigma_damage, 0.0, 0.12)
            c_arr[i, 0] = np.clip(c_arr[i, 0] - sigma_damage, 0.0, 1.0)

            # ── Resource drain from combat ───────────────────────── #
            resource_drain = 0.01 * lanchester * (1.0 + fatigue * 0.5)
            c_arr[i, 2] = np.clip(c_arr[i, 2] - resource_drain, 0.0, 1.0)

            # ── Trust erosion in combat zones ────────────────────── #
            trust_erosion = 0.005 * lanchester
            c_arr[i, 4] = np.clip(c_arr[i, 4] - trust_erosion, 0.0, 1.0)

        # ── Write back to world ──────────────────────────────────── #
        world = self._apply_cluster_array(world, c_arr)

        # ── Log events for significant combat ────────────────────── #
        for i in range(N):
            if self._combat_turns.get(i, 0) > 0 and turn % 25 == 0:
                fatigue = _combat_fatigue(
                    self._combat_turns[i],
                    midpoint=self.fatigue_midpoint,
                    rate=self.fatigue_rate,
                )
                if fatigue > 0.3:
                    self.log_event(
                        turn=turn,
                        message="combat_fatigue",
                        data={
                            "sector": i,
                            "combat_turns": self._combat_turns[i],
                            "fatigue": round(fatigue, 3),
                            "military": round(float(c_arr[i, 3]), 3),
                        },
                    )

        return world

    # ── Helpers ───────────────────────────────────────────────────── #

    @staticmethod
    def _get_side(
        sector_id: int,
        axis_set: set,
        soviet_set: set,
        contested_set: set,
    ) -> str:
        if sector_id in axis_set:
            return "axis"
        elif sector_id in soviet_set:
            return "soviet"
        elif sector_id in contested_set:
            return "contested"
        return "unknown"

    @staticmethod
    def _are_opposing(side_a: str, side_b: str) -> bool:
        """Check if two sides are opposing forces."""
        if side_a == "contested" or side_b == "contested":
            return True  # contested zones always have friction
        opposing_pairs = {
            ("axis", "soviet"), ("soviet", "axis"),
        }
        return (side_a, side_b) in opposing_pairs

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
Plugin = NonlinearCombatPlugin
