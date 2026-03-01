"""
PopShock — Custom regime-specific shocks for the population subsystem.

Design:
  - Shocks are defined in regime YAML under `custom_shocks`.
  - Each shock has a name, probability, and effect (string or callable).
  - Effects modify `GravitasWorld` and/or `WorldPopState` (e.g., resource drains,
    morale hits, trust collapse).
  - Called during `PopWrapper.step()` after population dynamics.

Example (Waterloo):
  - BritishNavalBlockade: Reduces AngloAllied food/industrial resources.
  - FrenchDeserters: Converts soldiers → urban laborers.
  - GrouchyDelay: Lowers Prussian trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .pop_state import WorldPopState, PopVector
from gravitas_engine.core.gravitas_state import GravitasWorld, ClusterState, GlobalState


# ─────────────────────────────────────────────────────────────────────────── #
# Shock definitions                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CustomShock:
    """Registry entry for a custom shock."""
    name: str
    probability: float
    effect: str  # Parsed into a callable or left as metadata


SHOCK_REGISTRY: Dict[str, CustomShock] = {}


def register_custom_shocks(config: List[Dict[str, Any]]) -> None:
    """Parse YAML shocks into the global registry."""
    for shock_cfg in config:
        SHOCK_REGISTRY[shock_cfg["name"]] = CustomShock(
            name=shock_cfg["name"],
            probability=shock_cfg["probability"],
            effect=shock_cfg.get("effect", ""),
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Shock effects                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _apply_british_naval_blockade(
    world: GravitasWorld,
    pop_state: Optional[WorldPopState],
    rng: np.random.Generator,
    target_cluster: int = 0,
    nation_name: str = "AngloAllied",
) -> Tuple[GravitasWorld, Optional[WorldPopState]]:
    """Reduce target nation's resources by 10%/5%."""
    clusters = list(world.clusters)
    if target_cluster < len(clusters):
        cluster = clusters[target_cluster]
        clusters[target_cluster] = cluster.copy_with(
            resource=float(np.clip(cluster.resource * 0.90, 0.0, 1.0))
        )
    return world.copy_with_clusters(clusters), pop_state


def _apply_french_deserters(
    world: GravitasWorld,
    pop_state: Optional[WorldPopState],
    rng: np.random.Generator,
) -> Tuple[GravitasWorld, Optional[WorldPopState]]:
    """Convert 10% of French soldiers to urban laborers."""
    if pop_state is None:
        return world, None
    
    pops = list(pop_state.pops)
    for i, pop in enumerate(pops):
        # TODO: Replace with nation tag check
        if i == 1:  # Placeholder: assume cluster 1 is France
            soldier_idx = 8
            laborer_idx = 2
            if soldier_idx < pop.n_archetypes:
                deserters = pop.sizes[soldier_idx] * 0.10
                new_sizes = pop.sizes.copy()
                new_sizes[soldier_idx] -= deserters
                new_sizes[laborer_idx] += deserters
                # Create a new PopVector with updated sizes
                from .pop_state import PopVector
                new_pop = PopVector(
                    sizes=new_sizes,
                    satisfaction=pop.satisfaction,
                    radicalization=pop.radicalization,
                    income=pop.income,
                    ethnic_shares=pop.ethnic_shares,
                    ethnic_tension=pop.ethnic_tension,
                )
                pops[i] = new_pop
    return world, pop_state.copy_with_pops(tuple(pops))


def _apply_grouchy_delay(
    world: GravitasWorld,
    pop_state: Optional[WorldPopState],
    rng: np.random.Generator,
    target_cluster: int = 2,
    nation_name: str = "Prussia",
) -> Tuple[GravitasWorld, Optional[WorldPopState]]:
    """Reduce target nation's trust by 20%."""
    clusters = list(world.clusters)
    if target_cluster < len(clusters):
        cluster = clusters[target_cluster]
        clusters[target_cluster] = cluster.copy_with(
            trust=float(np.clip(cluster.trust * 0.80, 0.0, 1.0))
        )
    return world.copy_with_clusters(clusters), pop_state


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_custom_shocks(
    world: GravitasWorld,
    pop_state: Optional[WorldPopState],
    shock_configs: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[GravitasWorld, Optional[WorldPopState]]:
    """Apply multiple shocks in parallel. Returns modified (world, pop_state)."""
    from joblib import Parallel, delayed
    
    def _apply_single(shock_cfg):
        shock_name = shock_cfg["name"]
        target_cluster = shock_cfg.get("target_cluster", 0)
        nation_name = shock_cfg.get("target_nation", "")
        return apply_custom_shock(
            world, pop_state, shock_name, rng, target_cluster, nation_name
        )
    
    # Parallelize if >3 shocks
    if len(shock_configs) > 3:
        results = Parallel(n_jobs=-1)(
            delayed(_apply_single)(cfg) for cfg in shock_configs
        )
        # Merge results (last shock wins for overlapping clusters)
        for w, p in results:
            world = w
            if p is not None:
                pop_state = p
    else:
        for cfg in shock_configs:
            world, pop_state = _apply_single(cfg)
    
    return world, pop_state


def apply_custom_shock(
    world: GravitasWorld,
    pop_state: Optional[WorldPopState],
    shock_name: str,
    rng: np.random.Generator,
    target_cluster: int = 0,
    nation_name: str = "",
) -> Tuple[GravitasWorld, Optional[WorldPopState]]:
    """Apply a single shock by name. Returns modified (world, pop_state)."""
    if shock_name not in SHOCK_REGISTRY:
        return world, pop_state

    if shock_name == "BritishNavalBlockade":
        return _apply_british_naval_blockade(world, pop_state, rng, target_cluster, nation_name)
    elif shock_name == "FrenchDeserters":
        return _apply_french_deserters(world, pop_state, rng)
    elif shock_name == "GrouchyDelay":
        return _apply_grouchy_delay(world, pop_state, rng, target_cluster, nation_name)
    else:
        return world, pop_state