"""
gravitas.plugins.axis_airlift — Axis Stalingrad Airlift (Luftwaffe Supply).

Historical basis:
    After the Soviet encirclement of the German 6th Army in November 1942,
    Göring promised to resupply the pocket by air. The Luftwaffe attempted
    daily airlift operations from airfields at Tatsinskaya and Morozovsk,
    but delivered only a fraction of the required 300 tons/day. As Soviet
    fighters and AA strengthened, losses mounted and deliveries dwindled.

Mechanic:
    Every `trigger_turn_interval` steps (default 40), if the Axis Supply
    Corridor (cluster 5) is not completely cut off (σ > sigma_threshold)
    and the hazard is not extreme, a small resource boost is delivered to
    the encircled Axis sectors (clusters 0, 1, 2). The boost decays over
    time to simulate the historical degradation of the airlift capacity.

Config (in custom.yaml):
    plugin_configs:
      axis_airlift:
        trigger_turn_interval: 40
        supply_corridor_cluster: 5
        target_clusters: [0, 1, 2]
        sigma_threshold: 0.3
        hazard_threshold: 0.8
        base_resource_boost: 0.04
        decay_rate: 0.005
        resource_cap: 1.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from gravitas.plugins import GravitasPlugin

if TYPE_CHECKING:
    from gravitas_engine.core.gravitas_state import GravitasWorld


class Plugin(GravitasPlugin):
    """Axis airlift operations to resupply encircled Stalingrad pocket."""

    name = "axis_airlift"
    description = (
        "Simulates Luftwaffe airlift operations delivering diminishing "
        "supplies to encircled Axis sectors — historically inadequate "
        "but politically mandated by Göring's guarantee."
    )
    version = "1.0.0"

    DEFAULTS = {
        "trigger_turn_interval": 40,
        "supply_corridor_cluster": 5,
        "target_clusters": [0, 1, 2],
        "sigma_threshold": 0.3,
        "hazard_threshold": 0.8,
        "base_resource_boost": 0.04,
        "decay_rate": 0.005,
        "resource_cap": 1.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._cfg = {**self.DEFAULTS, **(config or {})}
        self._delivery_count = 0
        self._current_boost = self._cfg["base_resource_boost"]

    def on_reset(self, world: "GravitasWorld", **kwargs) -> "GravitasWorld":
        """Reset airlift state on episode start."""
        world = super().on_reset(world, **kwargs)
        self._delivery_count = 0
        self._current_boost = self._cfg["base_resource_boost"]
        return world

    def on_step(self, world: "GravitasWorld", turn: int, **kwargs) -> "GravitasWorld":
        """Attempt airlift delivery if supply corridor is not cut off."""
        from gravitas_engine.core.gravitas_state import ClusterState

        cfg = self._cfg
        interval = cfg["trigger_turn_interval"]

        if turn <= 0 or turn % interval != 0:
            return world

        N = world.n_clusters
        corridor_idx = cfg["supply_corridor_cluster"]
        if corridor_idx >= N:
            return world

        c_arr = world.cluster_array()
        corridor_sigma = c_arr[corridor_idx, 0]
        corridor_hazard = c_arr[corridor_idx, 1]

        # Check if supply corridor is cut off
        if corridor_sigma <= cfg["sigma_threshold"]:
            self.log_event(
                turn,
                f"Axis supply corridor collapsed (σ={corridor_sigma:.3f}), "
                f"airlift impossible — 6th Army starving.",
            )
            return world

        if corridor_hazard >= cfg["hazard_threshold"]:
            self.log_event(
                turn,
                f"Axis supply corridor under extreme fire (h={corridor_hazard:.3f}), "
                f"Luftwaffe transports shot down.",
            )
            return world

        # Apply diminishing airlift to target clusters
        c_arr_mod = c_arr.copy()
        targets_supplied = []

        for t_idx in cfg["target_clusters"]:
            if t_idx >= N:
                continue
            old_res = c_arr_mod[t_idx, 2]
            c_arr_mod[t_idx, 2] = min(
                cfg["resource_cap"],
                c_arr_mod[t_idx, 2] + self._current_boost,
            )
            targets_supplied.append((t_idx, old_res, c_arr_mod[t_idx, 2]))

        if targets_supplied:
            new_clusters = [
                ClusterState.from_array(c_arr_mod[i]) for i in range(N)
            ]
            world = world.copy_with_clusters(new_clusters)
            self._delivery_count += 1

            # Decay boost for next delivery (historical degradation)
            self._current_boost = max(
                0.005,
                self._current_boost - cfg["decay_rate"],
            )

            details = ", ".join(
                f"Cluster {i} res {old:.2f}→{new:.2f}"
                for i, old, new in targets_supplied
            )
            self.log_event(
                turn,
                f"Luftwaffe airlift delivery #{self._delivery_count}: "
                f"{details}. Next capacity: {self._current_boost:.3f}/drop.",
                data={
                    "delivery_number": self._delivery_count,
                    "boost_applied": float(self._current_boost + cfg["decay_rate"]),
                    "next_boost": float(self._current_boost),
                    "targets": [
                        {"cluster": i, "before": float(o), "after": float(n)}
                        for i, o, n in targets_supplied
                    ],
                },
            )

        return world

    def on_episode_end(self, world: "GravitasWorld", turn: int, **kwargs) -> None:
        """Log airlift summary."""
        self.log_event(
            turn,
            f"Episode ended. Total airlift deliveries: {self._delivery_count}. "
            f"Final capacity: {self._current_boost:.3f}/drop.",
        )

    @property
    def delivery_count(self) -> int:
        """Number of successful airlift deliveries this episode."""
        return self._delivery_count
