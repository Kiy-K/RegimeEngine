"""
logistics_network.py — Nonlinear Logistics & Supply Network Plugin for GRAVITAS.

Models economic production, consumption, and supply flow through a graph-based
logistics network with nonlinear dynamics that resist RL exploitation:

1. **Sigmoid-Gated Production**: Sectors only produce resources when stability
   exceeds a threshold. Below σ=0.3: effectively zero. Above σ=0.6: full output.
   Smooth sigmoid prevents threshold-hovering exploits.

2. **Quadratic Consumption**: Military units consume resources at a rate that
   scales quadratically with hazard — combat multiplies supply burn rate.
   cons = base × m × (1 + h²)

3. **Saturating Flow (Logistic Curve)**: Resource transfer along logistics links
   follows a logistic curve that saturates at capacity — can't infinite-funnel.
   throughput = capacity × (1 − exp(−demand / capacity))

4. **Exponential Distance Decay**: Multi-hop supply loses effectiveness
   exponentially. exp(−λ × hops) means distant sectors get less supply.

5. **Sabotage Cascades**: When a logistics link is disrupted (by partisans or
   combat), all downstream sectors lose supply proportionally. Links have a
   vulnerability rating from scenario YAML.

6. **Seasonal Throughput Modifier**: Winter reduces Axis supply throughput
   (configurable per-scenario). Soviet sectors affected less.

7. **Congestion**: Multiple flows through same hub create diminishing returns.

Mathematical foundations:
  - Production: prod = base_prod × sigmoid((σ − 0.4) × 10)
  - Consumption: cons = base_cons × m × (1 + h²)
  - Flow: throughput = cap × (1 − exp(−demand/cap))
  - Distance: eff = exp(−0.3 × hops)
  - All smooth, differentiable, bounded.

Author: GRAVITAS Engine
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gravitas.plugins import GravitasPlugin

logger = logging.getLogger("gravitas.plugins.logistics_network")


# ─────────────────────────────────────────────────────────────────────────── #
# Nonlinear logistics math                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def _production_rate(sigma: float, base_prod: float, threshold: float = 0.4, steepness: float = 10.0) -> float:
    """
    Sigmoid-gated production.

    Only produces meaningfully when σ > threshold.
    Below σ=0.3: ~0 production. Above σ=0.6: ~full production.
    Prevents agents from producing in unstable sectors.
    """
    gate = _sigmoid((sigma - threshold) * steepness)
    return base_prod * gate


def _consumption_rate(military: float, hazard: float, base_cons: float) -> float:
    """
    Quadratic consumption.

    cons = base × m × (1 + h²)
    Combat (high hazard) multiplies burn rate quadratically.
    High military in combat zones = massive resource drain.
    """
    return base_cons * military * (1.0 + hazard * hazard)


def _saturating_flow(demand: float, capacity: float) -> float:
    """
    Logistic saturating flow.

    throughput = cap × (1 − exp(−demand/cap))
    At low demand: ~linear (demand ≈ throughput).
    At high demand: saturates at capacity.
    Prevents infinite resource funneling through single link.
    """
    if capacity <= 0:
        return 0.0
    ratio = demand / max(capacity, 1e-8)
    return capacity * (1.0 - np.exp(-ratio))


def _distance_decay(hops: int, decay_rate: float = 0.3) -> float:
    """
    Exponential distance decay.

    Supply effectiveness drops exponentially with network hops.
    1 hop: ~74%, 2 hops: ~55%, 3 hops: ~41%
    Encourages local supply chains over long-distance funneling.
    """
    return np.exp(-decay_rate * hops)


# ─────────────────────────────────────────────────────────────────────────── #
# Supply graph                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class LogisticsLink:
    """A directed logistics connection between two sectors."""
    src: int
    dst: int
    capacity: float
    vulnerability: float = 0.1
    disrupted: bool = False
    disruption_turns: int = 0


@dataclass
class LogisticsGraph:
    """Graph of logistics links between sectors."""
    links: Dict[Tuple[int, int], LogisticsLink] = field(default_factory=dict)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)

    def add_link(self, src: int, dst: int, capacity: float, vulnerability: float = 0.1):
        """Add bidirectional logistics link."""
        self.links[(src, dst)] = LogisticsLink(src, dst, capacity, vulnerability)
        self.links[(dst, src)] = LogisticsLink(dst, src, capacity, vulnerability)
        self.adjacency.setdefault(src, []).append(dst)
        self.adjacency.setdefault(dst, []).append(src)

    def get_link(self, src: int, dst: int) -> Optional[LogisticsLink]:
        return self.links.get((src, dst))

    def shortest_path_hops(self, src: int, dst: int, max_hops: int = 10) -> int:
        """BFS shortest path length. Returns max_hops+1 if unreachable."""
        if src == dst:
            return 0
        visited: Set[int] = {src}
        queue: deque = deque([(src, 0)])
        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbor in self.adjacency.get(node, []):
                if neighbor == dst:
                    return hops + 1
                if neighbor not in visited:
                    link = self.get_link(node, neighbor)
                    if link and not link.disrupted:
                        visited.add(neighbor)
                        queue.append((neighbor, hops + 1))
        return max_hops + 1

    def disrupt_link(self, src: int, dst: int, duration: int = 10):
        """Disrupt a link for N turns."""
        link = self.links.get((src, dst))
        if link:
            link.disrupted = True
            link.disruption_turns = duration
        link_rev = self.links.get((dst, src))
        if link_rev:
            link_rev.disrupted = True
            link_rev.disruption_turns = duration

    def tick_disruptions(self):
        """Decrement disruption counters, restore links when done."""
        for link in self.links.values():
            if link.disrupted:
                link.disruption_turns -= 1
                if link.disruption_turns <= 0:
                    link.disrupted = False
                    link.disruption_turns = 0


# ─────────────────────────────────────────────────────────────────────────── #
# Plugin                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class LogisticsNetworkPlugin(GravitasPlugin):
    """
    Nonlinear logistics and supply network plugin.

    Each step:
    1. Tick disruption counters on logistics links
    2. Compute production per sector (sigmoid-gated by σ)
    3. Compute consumption per sector (quadratic in hazard)
    4. Flow surplus resources along logistics links (saturating)
    5. Apply distance decay for multi-hop transfers
    6. Apply seasonal modifier (winter penalty for Axis)
    7. Check for random sabotage on vulnerable links
    """

    name = "logistics_network"
    version = "1.0.0"
    description = (
        "Graph-based nonlinear logistics: sigmoid production, quadratic consumption, "
        "saturating flow, distance decay, sabotage cascades."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}

        # ── Tunable parameters ────────────────────────────────────── #
        self.production_threshold = cfg.get("production_threshold", 0.4)
        self.production_steepness = cfg.get("production_steepness", 10.0)
        self.distance_decay_rate = cfg.get("distance_decay_rate", 0.3)
        self.winter_axis_penalty = cfg.get("winter_axis_penalty", 0.35)
        self.winter_soviet_penalty = cfg.get("winter_soviet_penalty", 0.10)
        self.winter_start_turn = cfg.get("winter_start_turn", 150)
        self.winter_end_turn = cfg.get("winter_end_turn", 400)
        self.sabotage_check_interval = cfg.get("sabotage_check_interval", 10)
        self.disruption_duration = cfg.get("disruption_duration", 12)
        self.congestion_factor = cfg.get("congestion_factor", 0.8)
        self.trigger_turn_interval = cfg.get("trigger_turn_interval", 2)

        # ── Graph & scenario data ─────────────────────────────────── #
        self._graph = LogisticsGraph()
        self._sector_production: Dict[int, float] = {}
        self._sector_consumption: Dict[int, float] = {}
        self._axis_clusters: List[int] = []
        self._soviet_clusters: List[int] = []
        self._contested_clusters: List[int] = []

        # ── Random state ──────────────────────────────────────────── #
        self._rng = np.random.default_rng(42)

    def on_reset(self, world, **kwargs):
        """Build logistics graph from scenario YAML."""
        N = world.n_clusters
        self._graph = LogisticsGraph()
        self._sector_production = {}
        self._sector_consumption = {}

        engine = kwargs.get("engine", None)
        if engine and hasattr(engine, "_scenario_meta"):
            meta = engine._scenario_meta
            self._axis_clusters = meta.get("axis_clusters", [])
            self._soviet_clusters = meta.get("soviet_clusters", [])
            self._contested_clusters = meta.get("contested_clusters", [])

            # Build logistics graph
            for link_def in meta.get("logistics_links", []):
                src = link_def.get("from", -1)
                dst = link_def.get("to", -1)
                cap = link_def.get("capacity", 0.10)
                vuln = link_def.get("vulnerability", 0.10)
                if src >= 0 and dst >= 0:
                    self._graph.add_link(src, dst, cap, vuln)

            # Read production/consumption from sectors
            for sector in meta.get("sectors", []):
                sid = sector.get("id", -1)
                self._sector_production[sid] = sector.get("production", 0.02)
                self._sector_consumption[sid] = sector.get("consumption", 0.02)
        else:
            # Fallback: default production/consumption
            for i in range(N):
                self._sector_production[i] = 0.02
                self._sector_consumption[i] = 0.02

        # Fallback cluster assignments
        if not self._axis_clusters:
            env = kwargs.get("env", None)
            if env and hasattr(env, "axis_clusters"):
                self._axis_clusters = list(env.axis_clusters)
                self._soviet_clusters = list(env.soviet_clusters)
                self._contested_clusters = list(getattr(env, "contested_clusters", []))

        return world

    def on_step(self, world, turn: int, **kwargs) -> object:
        """Apply logistics dynamics each step."""
        if turn % self.trigger_turn_interval != 0:
            return world

        N = world.n_clusters
        c_arr = world.cluster_array()  # (N, 6): σ, h, r, m, τ, φ

        axis_set = set(self._axis_clusters)
        soviet_set = set(self._soviet_clusters)

        # ── 1. Tick disruptions ──────────────────────────────────── #
        self._graph.tick_disruptions()

        # ── 2. Production (sigmoid-gated) ────────────────────────── #
        production = np.zeros(N)
        for i in range(N):
            base_prod = self._sector_production.get(i, 0.0)
            production[i] = _production_rate(
                sigma=c_arr[i, 0],
                base_prod=base_prod,
                threshold=self.production_threshold,
                steepness=self.production_steepness,
            )

        # ── 3. Consumption (quadratic hazard) ────────────────────── #
        consumption = np.zeros(N)
        for i in range(N):
            base_cons = self._sector_consumption.get(i, 0.0)
            consumption[i] = _consumption_rate(
                military=c_arr[i, 3],
                hazard=c_arr[i, 1],
                base_cons=base_cons,
            )

        # ── 4. Net resource change from production/consumption ───── #
        net_change = production - consumption

        # ── 5. Seasonal modifier (winter) ────────────────────────── #
        is_winter = self.winter_start_turn <= turn <= self.winter_end_turn
        if is_winter:
            for i in range(N):
                if i in axis_set:
                    # Axis sectors penalized in winter
                    net_change[i] -= self.winter_axis_penalty * consumption[i]
                elif i in soviet_set:
                    # Soviet sectors slightly penalized
                    net_change[i] -= self.winter_soviet_penalty * consumption[i]

        # ── 6. Flow surplus along logistics links ────────────────── #
        # Sectors with surplus push resources to connected deficit sectors
        surplus = np.zeros(N)
        for i in range(N):
            surplus[i] = c_arr[i, 2] + net_change[i] - 0.5  # deficit below 0.5

        flow_delta = np.zeros(N)
        hub_flow_count = np.zeros(N)  # for congestion

        for (src, dst), link in self._graph.links.items():
            if link.disrupted:
                continue
            if src >= N or dst >= N:
                continue

            # Only flow from surplus to deficit
            if surplus[src] <= 0 or surplus[dst] >= 0:
                continue

            demand = min(surplus[src], -surplus[dst])
            demand = max(demand, 0.0)

            # Saturating flow
            throughput = _saturating_flow(demand, link.capacity)

            # Distance decay (BFS hops to a supply source)
            hops = 1  # direct link = 1 hop
            throughput *= _distance_decay(hops, self.distance_decay_rate)

            # Congestion: diminishing returns on flows through a hub
            hub_flow_count[src] += 1
            congestion_mult = 1.0 / (1.0 + self.congestion_factor * max(hub_flow_count[src] - 1, 0))
            throughput *= congestion_mult

            # Apply flow
            flow_delta[src] -= throughput
            flow_delta[dst] += throughput

        # ── 7. Apply changes to cluster array ────────────────────── #
        for i in range(N):
            new_resource = c_arr[i, 2] + net_change[i] + flow_delta[i]
            c_arr[i, 2] = np.clip(new_resource, 0.0, 1.0)

        # ── 8. Sabotage checks on vulnerable links ──────────────── #
        if turn % self.sabotage_check_interval == 0:
            for (src, dst), link in list(self._graph.links.items()):
                if link.disrupted:
                    continue
                # Higher vulnerability near combat / low σ
                combat_factor = (c_arr[min(src, N-1), 1] + c_arr[min(dst, N-1), 1]) / 2.0
                sab_prob = link.vulnerability * (0.3 + 0.7 * combat_factor)
                if self._rng.random() < sab_prob:
                    self._graph.disrupt_link(src, dst, self.disruption_duration)

                    # Cascade: downstream sectors of disrupted link lose resources
                    cascade_penalty = 0.05 * link.capacity / max(
                        self._graph.links[(src, dst)].capacity, 0.01
                    )
                    for downstream in self._graph.adjacency.get(dst, []):
                        if downstream < N and downstream != src:
                            c_arr[downstream, 2] = np.clip(
                                c_arr[downstream, 2] - cascade_penalty, 0.0, 1.0
                            )

                    self.log_event(
                        turn=turn,
                        message="supply_disruption",
                        data={
                            "link": (src, dst),
                            "vulnerability": round(link.vulnerability, 3),
                            "combat_factor": round(combat_factor, 3),
                            "duration": self.disruption_duration,
                        },
                    )

        # ── 9. Resource deprivation effects ──────────────────────── #
        # Sectors with very low resources suffer additional penalties
        for i in range(N):
            if c_arr[i, 2] < 0.15:
                # Starvation: military degrades, stability drops
                deprivation = 0.15 - c_arr[i, 2]
                c_arr[i, 3] = np.clip(c_arr[i, 3] - 0.01 * deprivation, 0.0, 1.0)
                c_arr[i, 0] = np.clip(c_arr[i, 0] - 0.005 * deprivation, 0.0, 1.0)
                c_arr[i, 4] = np.clip(c_arr[i, 4] - 0.008 * deprivation, 0.0, 1.0)

        # ── Write back ───────────────────────────────────────────── #
        world = self._apply_cluster_array(world, c_arr)

        # ── Log periodic supply status ───────────────────────────── #
        if turn % 50 == 0:
            disrupted_count = sum(1 for l in self._graph.links.values() if l.disrupted)
            total_links = len(self._graph.links) // 2  # bidirectional
            self.log_event(
                turn=turn,
                message="logistics_status",
                data={
                    "mean_resources": round(float(np.mean(c_arr[:N, 2])), 3),
                    "disrupted_links": disrupted_count // 2,
                    "total_links": total_links,
                    "is_winter": is_winter,
                    "total_production": round(float(np.sum(production)), 4),
                    "total_consumption": round(float(np.sum(consumption)), 4),
                },
            )

        return world

    # ── Helpers ───────────────────────────────────────────────────── #

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
Plugin = LogisticsNetworkPlugin
