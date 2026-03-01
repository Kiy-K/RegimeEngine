"""
gravitas.plugins.soviet_reinforcements — Conditional Soviet Volga Reinforcements.

Historical basis:
    Throughout the Battle of Stalingrad (Aug 1942 – Feb 1943), Soviet forces
    received nightly reinforcements via barge crossings over the Volga River.
    These resupply operations were only possible when the Volga Crossing points
    were securely held and not under concentrated German artillery/air attack.

Mechanic:
    Every `trigger_turn_interval` steps (default 50), if the Volga Crossing
    sector (cluster 3) is stable (σ > sigma_threshold) and not under heavy
    fire (hazard < hazard_threshold), the Soviet Strategic Reserve (cluster 6)
    receives a military and resource boost.

Config (in custom.yaml):
    plugin_configs:
      soviet_reinforcements:
        trigger_turn_interval: 50
        volga_cluster: 3
        reserve_cluster: 6
        sigma_threshold: 0.5
        hazard_threshold: 0.7
        military_boost: 0.10
        resource_boost: 0.05
        military_cap: 0.95
        resource_cap: 1.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from gravitas.plugins import GravitasPlugin

if TYPE_CHECKING:
    from gravitas_engine.core.gravitas_state import GravitasWorld


class Plugin(GravitasPlugin):
    """Conditional Soviet reinforcements via Volga barge crossings."""

    name = "soviet_reinforcements"
    description = (
        "Adds military and resource boosts to Soviet Strategic Reserve "
        "when Volga Crossing is securely held — simulating historical "
        "nightly barge reinforcements."
    )
    version = "1.0.0"

    # Default configuration
    DEFAULTS = {
        "trigger_turn_interval": 50,
        "volga_cluster": 3,
        "reserve_cluster": 6,
        "sigma_threshold": 0.5,
        "hazard_threshold": 0.7,
        "military_boost": 0.10,
        "resource_boost": 0.05,
        "military_cap": 0.95,
        "resource_cap": 1.0,
        "soviet_clusters": [3, 6],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Merge defaults with user config
        self._cfg = {**self.DEFAULTS, **(config or {})}
        self._reinforcement_count = 0

    def on_reset(self, world: "GravitasWorld", **kwargs) -> "GravitasWorld":
        """Reset reinforcement counter on episode start."""
        world = super().on_reset(world, **kwargs)
        self._reinforcement_count = 0
        return world

    def on_step(self, world: "GravitasWorld", turn: int, **kwargs) -> "GravitasWorld":
        """Check Volga conditions and apply reinforcements if met."""
        from gravitas_engine.core.gravitas_state import ClusterState

        cfg = self._cfg
        interval = cfg["trigger_turn_interval"]
        volga_idx = cfg["volga_cluster"]
        reserve_idx = cfg["reserve_cluster"]
        soviet_clusters = cfg["soviet_clusters"]

        if turn <= 0 or turn % interval != 0:
            return world

        N = world.n_clusters
        if volga_idx >= N or reserve_idx >= N:
            return world

        if volga_idx not in soviet_clusters:
            return world

        c_arr = world.cluster_array()
        volga_sigma = c_arr[volga_idx, 0]
        volga_hazard = c_arr[volga_idx, 1]

        # Check conditions: stable crossing, not under heavy fire
        if volga_sigma <= cfg["sigma_threshold"]:
            self.log_event(turn, f"Volga Crossing unstable (σ={volga_sigma:.3f}), "
                          f"no reinforcements dispatched.")
            return world

        if volga_hazard >= cfg["hazard_threshold"]:
            self.log_event(turn, f"Volga Crossing under heavy fire (h={volga_hazard:.3f}), "
                          f"barges cannot cross.")
            return world

        # Apply reinforcements
        c_arr_mod = c_arr.copy()
        old_mil = c_arr_mod[reserve_idx, 3]
        old_res = c_arr_mod[reserve_idx, 2]

        c_arr_mod[reserve_idx, 3] = min(
            cfg["military_cap"],
            c_arr_mod[reserve_idx, 3] + cfg["military_boost"],
        )
        c_arr_mod[reserve_idx, 2] = min(
            cfg["resource_cap"],
            c_arr_mod[reserve_idx, 2] + cfg["resource_boost"],
        )

        new_mil = c_arr_mod[reserve_idx, 3]
        new_res = c_arr_mod[reserve_idx, 2]

        new_clusters = [
            ClusterState.from_array(c_arr_mod[i]) for i in range(N)
        ]
        world = world.copy_with_clusters(new_clusters)
        self._reinforcement_count += 1

        self.log_event(
            turn,
            f"Volga reinforcements arrive! Barges cross under fire. "
            f"Reserve military {old_mil:.2f}→{new_mil:.2f}, "
            f"resources {old_res:.2f}→{new_res:.2f} "
            f"(delivery #{self._reinforcement_count})",
            data={
                "volga_sigma": float(volga_sigma),
                "volga_hazard": float(volga_hazard),
                "military_before": float(old_mil),
                "military_after": float(new_mil),
                "resource_before": float(old_res),
                "resource_after": float(new_res),
                "delivery_number": self._reinforcement_count,
            },
        )

        return world

    def on_episode_end(self, world: "GravitasWorld", turn: int, **kwargs) -> None:
        """Log summary at episode end."""
        self.log_event(
            turn,
            f"Episode ended. Total Volga reinforcements delivered: "
            f"{self._reinforcement_count}",
        )

    @property
    def reinforcement_count(self) -> int:
        """Number of successful reinforcement deliveries this episode."""
        return self._reinforcement_count
