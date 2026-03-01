"""
System parameters for the Adaptive Memory-Driven Regime Engine.

All parameters are immutable, named constants chosen to satisfy the stability
constraints proven in the design specification:

  - beta_mem > 1             (memory decay dominates; Mem* = I(1-Coh)/beta_mem < 1)
  - beta_rad + gamma_rad > 1 (radicalization self-limiting at Rad=1)
  - 0 < dt <= 0.05           (RK4 accuracy and Lyapunov stability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SystemParameters:
    """Immutable, fully validated system parameters.

    Every parameter is documented with its role in the ODE system and the
    constraint required for the corresponding stability proof.
    """

    # ------------------------------------------------------------------ #
    # Faction count                                                        #
    # ------------------------------------------------------------------ #
    n_factions: int = 3
    """Number of political factions (2–6)."""

    # ------------------------------------------------------------------ #
    # Power dynamics  —  Eq. P                                            #
    # dP_i/dt = alpha_P*(1-Exh)*[E*Coh_i - beta_P*Rad_i*(1-Coh_i)       #
    #                              - gamma_P*F*P_i]                        #
    # ------------------------------------------------------------------ #
    alpha_power: float = 0.10
    """Power growth time-scale (> 0)."""

    beta_power: float = 0.30
    """Radicalization drag on power (> 0)."""

    gamma_power: float = 0.20
    """Fragmentation damping on power (> 0)."""

    # ------------------------------------------------------------------ #
    # Radicalization dynamics  —  Eq. R                                   #
    # dRad_i/dt = alpha_rad*(1-Exh)*[Mem_i*(1-Coh_i)*(1-R_sys)           #
    #              - beta_rad*Rad_i*Coh_i - gamma_rad*Rad_i^2]            #
    # STABILITY: beta_rad + gamma_rad > 1  (chosen: 0.40 + 0.65 = 1.05)  #
    # ------------------------------------------------------------------ #
    alpha_rad: float = 0.15
    """Radicalization growth time-scale (> 0)."""

    beta_rad: float = 0.40
    """Cohesion damping on radicalization (>  0; beta_rad+gamma_rad > 1)."""

    gamma_rad: float = 0.65
    """Quadratic self-damping on radicalization (> 0; beta_rad+gamma_rad > 1)."""

    # ------------------------------------------------------------------ #
    # Cohesion dynamics  —  Eq. C_i                                       #
    # dCoh_i/dt = alpha_coh*(1-Exh)*[(1-F)*(1-Rad_i)                     #
    #              - beta_coh*|P_i - P_bar| - Coh_i*Rad_i^2]             #
    # ------------------------------------------------------------------ #
    alpha_coh: float = 0.12
    """Cohesion recovery time-scale (> 0)."""

    beta_coh: float = 0.35
    """Power-imbalance drag on cohesion (> 0)."""

    # ------------------------------------------------------------------ #
    # Memory dynamics  —  Eq. Mem                                         #
    # dMem_i/dt = alpha_mem*[I*(1-Coh_i) - beta_mem*Mem_i]               #
    # STABILITY: beta_mem > 1  (Mem* = I(1-Coh)/beta_mem < 1)            #
    # ------------------------------------------------------------------ #
    alpha_mem: float = 0.08
    """Memory accumulation time-scale (> 0)."""

    beta_mem: float = 1.20
    """Memory decay rate (> 1 required for boundedness proof)."""

    # ------------------------------------------------------------------ #
    # Fragmentation  —  F = 1 - exp(-lambda_frag * Gini(P))              #
    # ------------------------------------------------------------------ #
    lambda_frag: float = 3.0
    """Fragmentation sensitivity to power Gini (> 0)."""

    # ------------------------------------------------------------------ #
    # Volatility  —  V = tanh(kappa_v * M * (1-Exh) * (1+I))            #
    # ------------------------------------------------------------------ #
    kappa_v: float = 2.5
    """Volatility amplification factor (> 0)."""

    # ------------------------------------------------------------------ #
    # Exhaustion dynamics  —  Eq. Exh                                     #
    # dExh/dt = alpha_exh*[V*I*(1-Exh) - beta_exh*(1-V)*(1-I)*Exh]      #
    # ------------------------------------------------------------------ #
    alpha_exh: float = 0.05
    """Exhaustion accumulation time-scale (> 0)."""

    beta_exh: float = 0.30
    """Exhaustion recovery rate (> 0)."""

    # ------------------------------------------------------------------ #
    # Integration parameters                                              #
    # ------------------------------------------------------------------ #
    dt: float = 0.01
    """RK4 time step (0 < dt <= 0.05 for numerical stability)."""

    max_steps: int = 10_000
    """Maximum simulation steps (> 0)."""

    # ------------------------------------------------------------------ #
    # Random seed for reproducibility                                      #
    # ------------------------------------------------------------------ #
    seed: int = 0
    """PRNG seed for deterministic runs (any non-negative integer)."""

    # ------------------------------------------------------------------ #
    # Phase 2 Expansion: Stochastic, Economic, Topological                 #
    # ------------------------------------------------------------------ #
    sigma_noise: float = 0.05
    """Wiener process noise intensity (>= 0)."""

    n_pillars: int = 3
    """Number of state pillars to target (>= 1)."""

    alpha_gdp: float = 0.05
    """GDP growth rate (> 0)."""

    beta_gdp: float = 0.20
    """Instability/Volatility drag on GDP (> 0)."""

    wealth_extraction: float = 0.10
    """Wealth extraction rate per power share (> 0)."""

    # ------------------------------------------------------------------ #
    # Phase 3: Hierarchical (Provinces / Districts)                       #
    # ------------------------------------------------------------------ #
    use_hierarchy: bool = False
    """If True, enable district-level topology and dynamics."""

    n_provinces: int = 7
    """Number of provinces (5–10 when use_hierarchy)."""

    districts_per_province: int = 10
    """Districts per province (5–20), or use list in layout for variable."""

    # District dynamics
    nu_diffusion: float = 0.15
    """Unrest diffusion rate (≤ diffusion_rate_bound(adjacency) for stability)."""

    mu_capital_flow: float = 0.05
    """Capital flow rate between adjacent districts."""

    delta_unrest: float = 0.10
    """Decay rate for local unrest (boundedness)."""

    alpha_local_gdp: float = 0.08
    beta_local_gdp: float = 0.08
    gamma_local_gdp: float = 0.15
    alpha_local_unrest: float = 0.12
    beta_local_unrest: float = 0.10
    alpha_admin: float = 0.05
    zeta_admin: float = 0.20
    alpha_factional_dom: float = 0.10
    alpha_impl_base: float = 0.05
    psi_impl: float = 0.15
    eta_exh_impl: float = 0.30
    eta_frag_impl: float = 0.25
    alpha_local_mem: float = 0.08
    beta_local_mem: float = 1.20
    """Local memory decay (beta_local_mem > 1 for boundedness)."""

    # Admin lag
    tau_delay_base: float = 2.0
    """Base delay time constant; τ_i = tau_delay_base / max(ε, admin_capacity_i)."""

    tau_delay_eps: float = 0.05
    """Lower bound on admin_capacity for delay to avoid singularity."""

    # Multi-scale coupling (district -> regime)
    kappa_var_volatility: float = 0.50
    """Weight for district unrest variance in global volatility."""

    alpha_exh_district_stress: float = 0.03
    """Weight for district stress in exhaustion accumulation."""

    def __post_init__(self) -> None:
        """Validate every parameter against its stability constraint."""
        if not 2 <= self.n_factions <= 6:
            raise ValueError(
                f"n_factions must be in [2, 6], got {self.n_factions}"
            )

        strictly_positive_scalars = {
            "alpha_power": self.alpha_power,
            "beta_power": self.beta_power,
            "gamma_power": self.gamma_power,
            "alpha_rad": self.alpha_rad,
            "beta_rad": self.beta_rad,
            "gamma_rad": self.gamma_rad,
            "alpha_coh": self.alpha_coh,
            "beta_coh": self.beta_coh,
            "alpha_mem": self.alpha_mem,
            "beta_mem": self.beta_mem,
            "lambda_frag": self.lambda_frag,
            "kappa_v": self.kappa_v,
            "alpha_exh": self.alpha_exh,
            "beta_exh": self.beta_exh,
            "alpha_gdp": self.alpha_gdp,
            "beta_gdp": self.beta_gdp,
            "wealth_extraction": self.wealth_extraction,
        }
        for name, value in strictly_positive_scalars.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if self.n_pillars < 1:
            raise ValueError(f"n_pillars must be >= 1, got {self.n_pillars}")
            
        if self.sigma_noise < 0.0:
            raise ValueError(f"sigma_noise must be >= 0, got {self.sigma_noise}")

        # Stability constraint: radicalization self-limits at Rad=1
        if self.beta_rad + self.gamma_rad <= 1.0:
            raise ValueError(
                f"Stability violated: beta_rad ({self.beta_rad}) + "
                f"gamma_rad ({self.gamma_rad}) must exceed 1.0"
            )

        # Stability constraint: memory steady-state cannot reach 1
        if self.beta_mem <= 1.0:
            raise ValueError(
                f"Stability violated: beta_mem ({self.beta_mem}) must exceed 1.0"
            )

        if not 0.0 < self.dt <= 0.05:
            raise ValueError(
                f"dt must be in (0, 0.05], got {self.dt}"
            )

        if self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be > 0, got {self.max_steps}"
            )

        if self.seed < 0:
            raise ValueError(
                f"seed must be >= 0, got {self.seed}"
            )

        # Phase 3: hierarchical constraints
        if self.use_hierarchy:
            if not 5 <= self.n_provinces <= 10:
                raise ValueError(
                    f"n_provinces must be in [5, 10] when use_hierarchy, got {self.n_provinces}"
                )
            dpp = self.districts_per_province
            if isinstance(dpp, int):
                if not 5 <= dpp <= 20:
                    raise ValueError(
                        f"districts_per_province must be in [5, 20], got {dpp}"
                    )
            if self.beta_local_mem <= 1.0:
                raise ValueError(
                    f"beta_local_mem must be > 1 for boundedness, got {self.beta_local_mem}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parameters to a plain dictionary."""
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemParameters":
        """Deserialize parameters from a plain dictionary."""
        return cls(**data)