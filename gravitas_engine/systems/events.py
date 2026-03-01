"""
Poisson-process event system for exogenous shocks.

Simulates discrete, probabilistic events (e.g., Economic Crises, Scandals)
that instantly perturb the regime state outside the continuous ODE integration.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

from ..core.factions import recompute_system_state
from ..core.parameters import SystemParameters
from ..core.state import FactionState, RegimeState


def check_and_apply_events(
    state: RegimeState, params: SystemParameters
) -> RegimeState:
    """Evaluate probabilistic shocks and apply them to the state.

    Args:
        state:  Current regime state.
        params: System parameters.

    Returns:
        New RegimeState if a shock occurred, otherwise the original state.
    """
    # Base probabilities (could be moved to params)
    p_economic_crisis = 0.005  # 0.5% chance per step
    p_scandal = 0.01          # 1.0% chance per step
    p_windfall = 0.005        # 0.5% chance per step

    shock_applied = False
    
    # We will build new arrays if a shock happens
    powers = state.get_faction_powers()
    rads = state.get_faction_radicalizations()
    cohs = state.get_faction_cohesions()
    mems = state.get_faction_memories()
    wealths = state.get_faction_wealths()
    
    gdp = state.system.state_gdp
    pillars = np.array(state.system.pillars, dtype=np.float64)

    # 1. Economic Crisis
    if random.random() < p_economic_crisis:
        shock_applied = True
        gdp = max(0.0, gdp - 0.3)
        # All factions lose wealth proportionately, memory spikes
        wealths *= 0.7
        mems = np.clip(mems + 0.2, 0.0, 1.0)
        
    # 2. Scandal (hits the most powerful faction's cohesion)
    if random.random() < p_scandal:
        shock_applied = True
        dominant_idx = int(np.argmax(powers))
        cohs[dominant_idx] = max(0.0, cohs[dominant_idx] - 0.3)
        # Pillar control takes a hit
        pillars = np.clip(pillars - 0.1, 0.0, 1.0)
        
    # 3. Windfall
    if random.random() < p_windfall:
        shock_applied = True
        gdp = min(1.0, gdp + 0.2)
        # Rejuvenates all factions' wealth
        wealths = np.clip(wealths + 0.1, 0.0, 1.0)

    if not shock_applied:
        return state

    # Reconstruct state
    factions: List[FactionState] = []
    for i in range(state.n_factions):
        factions.append(
            state.factions[i].copy_with(
                power=float(powers[i]),
                radicalization=float(rads[i]),
                cohesion=float(cohs[i]),
                memory=float(mems[i]),
                wealth=float(wealths[i]),
            )
        )
        
    new_system = state.system
    new_system = new_system.__class__(
        legitimacy=new_system.legitimacy,
        cohesion=new_system.cohesion,
        fragmentation=new_system.fragmentation,
        instability=new_system.instability,
        mobilization=new_system.mobilization,
        repression=new_system.repression,
        elite_alignment=new_system.elite_alignment,
        volatility=new_system.volatility,
        exhaustion=new_system.exhaustion,
        state_gdp=gdp,
        pillars=tuple(float(x) for x in pillars),
    )
    
    shocked_state = RegimeState(
        factions=factions,
        system=new_system,
        affinity_matrix=state.affinity_matrix,
        step=state.step,
        hierarchical=getattr(state, "hierarchical", None),
    )
    return recompute_system_state(shocked_state, params)
