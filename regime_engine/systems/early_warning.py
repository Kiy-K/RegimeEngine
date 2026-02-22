"""
Early warning index and survival metrics for multi-agent RL.

Compressed spatial signals: variance, clustering, exhaustion trend, volatility.
Agents act before EWI crosses threshold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.state import RegimeState


def early_warning_index(
    state: RegimeState,
    unrest_variance: float = 0.0,
    clustering_index: float = 0.0,
    exhaustion_growth_rate: float = 0.0,
    w_variance: float = 0.3,
    w_clustering: float = 0.35,
    w_exh_rate: float = 0.2,
    w_volatility: float = 0.15,
) -> float:
    """Single scalar early warning index ∈ [0, 1]. Act before it crosses threshold.

    EWI = w_v * unrest_variance + w_c * clustering + w_e * exhaustion_growth + w_v * volatility
    (all terms normalized to ~[0,1])
    """
    vol = state.system.volatility
    exh = state.system.exhaustion
    # Normalize exhaustion growth (can be negative); assume typical range [-0.05, 0.05]
    exh_norm = float(np.clip((exhaustion_growth_rate + 0.05) / 0.1, 0.0, 1.0))
    ewi = (
        w_variance * float(np.clip(unrest_variance, 0.0, 1.0))
        + w_clustering * float(np.clip(clustering_index, 0.0, 1.0))
        + w_exh_rate * exh_norm
        + w_volatility * vol
    )
    return float(np.clip(ewi, 0.0, 1.0))


def exhaustion_growth_rate(
    current_exhaustion: float,
    previous_exhaustion: float,
    dt: float = 0.01,
) -> float:
    """Approximate d(Exh)/dt from last step (for observation)."""
    if dt <= 0:
        return 0.0
    return (current_exhaustion - previous_exhaustion) / dt


def volatility_spike_indicator(
    current_volatility: float,
    previous_volatility: float,
    threshold: float = 0.1,
) -> float:
    """1.0 if volatility jumped by more than threshold, else 0.0. For spike rate."""
    return 1.0 if (current_volatility - previous_volatility) >= threshold else 0.0
