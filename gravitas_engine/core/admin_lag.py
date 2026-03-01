"""
Administrative lag: policy pipeline buffers.

Policies enter the district pipeline and apply gradually with time constant
τ_delay,i ∝ 1 / admin_capacity_i. So high admin_capacity → fast application.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import SystemParameters


def pipeline_derivative(
    pipeline_buffers: NDArray[np.float64],
    regime_policy: NDArray[np.float64],
    admin_capacity: NDArray[np.float64],
    params: SystemParameters,
) -> NDArray[np.float64]:
    """Compute d(buffer)/dt for each district.

    db_i/dt = (regime_policy - b_i) / τ_i,  τ_i = tau_delay_base / max(ε, a_i).

    regime_policy: (n_policy_dims,) or (1,) broadcast to (N_D, n_policy_dims).
    admin_capacity: (N_D,).
    pipeline_buffers: (N_D, n_policy_dims).

    Returns:
        (N_D, n_policy_dims) derivative.
    """
    n_d, n_pol = pipeline_buffers.shape
    a = np.maximum(admin_capacity, params.tau_delay_eps)
    tau = params.tau_delay_base / a
    # regime_policy: if 1D of length n_pol, broadcast to (n_d, n_pol)
    if regime_policy.ndim == 1:
        target = np.broadcast_to(regime_policy, (n_d, n_pol)).copy()
    else:
        target = np.asarray(regime_policy, dtype=np.float64)
        if target.shape != (n_d, n_pol):
            target = np.broadcast_to(target, (n_d, n_pol)).copy()
    rate = 1.0 / np.maximum(tau[:, np.newaxis], 1e-6)
    return rate * (target - pipeline_buffers)


def effective_policy_at_districts(
    pipeline_buffers: NDArray[np.float64],
    implementation_efficiency: NDArray[np.float64],
    admin_capacity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Policy effect per district: buffer * implementation_efficiency * admin_capacity.

    Returns:
        (N_D, n_policy_dims) — applied policy (for use in dynamics).
    """
    eff = (pipeline_buffers * implementation_efficiency[:, np.newaxis]
           * admin_capacity[:, np.newaxis])
    return np.clip(eff, 0.0, 1.0)
