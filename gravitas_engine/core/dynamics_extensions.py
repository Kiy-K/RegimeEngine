"""
Extended dynamics for Phase 2: Economics and Topology.

These functions compute the deterministic derivatives for the newly added
variables: Wealth, GDP, Pillars, and Affinity Matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import SystemParameters
from .state import RegimeState


def compute_economic_derivatives(
    state: RegimeState, params: SystemParameters
) -> tuple[NDArray[np.float64], float]:
    """Compute d(Wealth_i)/dt and d(GDP)/dt.

    Returns:
        d_wealth: Array of shape (n_factions,).
        d_gdp: Scalar derivative for GDP.
    """
    gdp = state.system.state_gdp
    inst = state.system.instability
    vol = state.system.volatility
    exh = state.system.exhaustion

    # GDP grows when stable, shrinks when unstable/volatile
    # Bounded by exhaustion.
    growth = params.alpha_gdp * (1.0 - inst) * (1.0 - vol) * (1.0 - gdp)
    drag = params.beta_gdp * (inst + vol) * gdp
    d_gdp = (growth - drag) * (1.0 - exh)

    # Factions extract wealth proportional to their power and the total GDP
    powers = state.get_faction_powers()
    wealths = state.get_faction_wealths()
    
    # Wealth extraction diminishes as factions get richer (logistic bounded)
    extraction = params.wealth_extraction * powers * gdp * (1.0 - wealths)
    
    # Baseline wealth decay if they don't have power
    decay = 0.05 * wealths * (1.0 - powers)
    
    d_wealth = extraction - decay

    return d_wealth, float(d_gdp)


def compute_topological_derivatives(
    state: RegimeState, params: SystemParameters
) -> tuple[NDArray[np.float64], tuple[tuple[float, ...], ...]]:
    """Compute d(Pillars)/dt and the drift for the Affinity Matrix.

    Returns:
        d_pillars: Array of shape (n_pillars,).
        affinity_drift: Tuple of tuples representing dA/dt.
    """
    # For now, Pillars naturally drift towards Legitimacy, but are eroded by Instability.
    pillars = np.array(state.system.pillars, dtype=np.float64)
    if len(pillars) == 0:
        pillars = np.zeros(params.n_pillars)
        
    leg = state.system.legitimacy
    inst = state.system.instability
    
    # Pillars revert to Legitimacy at a slow rate, eroded heavily by Instability.
    d_pillars = 0.1 * (leg - pillars) - 0.2 * inst * pillars
    
    # Affinity Matrix drift:
    # Factions with similar radicalization gain affinity; 
    # Factions with dissimilar radicalization lose affinity.
    n = state.n_factions
    rads = state.get_faction_radicalizations()
    
    if not state.affinity_matrix:
        current_aff = np.eye(n)
    else:
        current_aff = np.array(state.affinity_matrix)
        
    rad_diffs = np.abs(rads[:, None] - rads[None, :])
    # if diff < 0.2, they align. If > 0.5, they polarize.
    alignment_force = 0.5 - rad_diffs
    
    # Decay towards neutral (0) if there's no strong force
    decay = -0.05 * current_aff
    
    d_affinity = 0.1 * alignment_force + decay
    # Diagonal stays 1
    np.fill_diagonal(d_affinity, 0.0)
    
    # Convert to tuple
    aff_tuple = tuple(tuple(float(x) for x in row) for row in d_affinity)
    return d_pillars, aff_tuple
