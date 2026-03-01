"""
GravitasParams — Immutable hyperparameter pack for the GRAVITAS environment.

All dynamics constants, stability constraints, and curriculum knobs live here.
Changing one number changes the physics of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class GravitasParams:
    """
    Complete parameter specification for GRAVITAS.

    Organized by subsystem:
      - Episode structure
      - Cluster topology
      - Stability / hazard dynamics
      - Exhaustion dynamics
      - Polarization dynamics
      - Institutional trust dynamics
      - Resource dynamics
      - Military dynamics
      - Media bias dynamics
      - Shock (Hawkes) process
      - Reward weights
      - Observation noise
    """

    # ── Episode ────────────────────────────────────────────────────────── #
    max_steps: int = 600
    dt: float = 0.01
    seed: int = 0

    # ── Topology ───────────────────────────────────────────────────────── #
    n_clusters: int = 8
    """Number of clusters (regions/factions). Randomized in [n_min, n_max] per episode."""
    n_clusters_min: int = 5
    n_clusters_max: int = 12
    between_link_prob: float = 0.35
    """Probability of an inter-cluster edge in the random topology."""
    conflict_link_prob: float = 0.20
    """Probability that an inter-cluster edge also carries conflict linkage c_ij."""

    # ── Stability ODE  dσ/dt ──────────────────────────────────────────── #
    alpha_sigma: float = 0.12
    """Recovery rate toward structural stability."""
    beta_sigma: float = 0.25
    """Hazard drag on stability."""
    nu_sigma: float = 0.08
    """Spatial diffusion rate for stability across edges."""
    kappa_h: float = 1.8
    """Superlinear exponent on (1-σ) in hazard — tipping behaviour."""
    kappa_p: float = 1.5
    """Superlinear exponent on polarization in hazard."""

    # ── Hazard index ──────────────────────────────────────────────────── #
    gamma_h1: float = 0.40
    """Weight of local instability + polarization in hazard."""
    gamma_h2: float = 0.25
    """Weight of trust deficit × systemic polarization."""
    gamma_h3: float = 0.35
    """Weight of cascade term (neighbour hazard diffusion via c_ij)."""

    # ── Exhaustion ODE  dE/dt ─────────────────────────────────────────── #
    alpha_exh: float = 0.06
    beta_exh: float = 0.28
    """Recovery rate; must satisfy β_exh > 0 for boundedness."""
    military_exh_coeff: float = 0.10
    """Extra exhaustion accumulation per unit military load."""

    # ── Fragmentation  Φ ──────────────────────────────────────────────── #
    alpha_phi: float = 0.04
    """Growth rate of fragmentation probability."""
    beta_phi: float = 0.12
    """Natural decay of fragmentation probability."""
    military_phi_coeff: float = 0.06
    """Military load contribution to fragmentation."""

    # ── Polarization ODE  dΠ/dt ──────────────────────────────────────── #
    alpha_pol: float = 0.10
    beta_pol: float = 0.15
    """Trust-driven depolarization rate."""
    propaganda_pol_coeff: float = 0.08
    """True polarization increase from propaganda (hidden cost)."""
    media_pol_coeff: float = 0.12
    """Bias-driven polarization amplification."""

    # ── Institutional trust ODE  dτ/dt ───────────────────────────────── #
    alpha_tau: float = 0.07
    """Reform-driven trust growth rate."""
    military_tau_cost: float = 0.05
    """Trust erosion per unit military presence in unstable clusters."""
    deprivation_tau_cost: float = 0.08
    """Trust erosion from low resources × high hazard."""
    tau_decay: float = 0.02
    """Natural decay of institutional trust."""

    # ── Resource ODE  dr/dt ──────────────────────────────────────────── #
    alpha_res: float = 0.04
    """Natural resource recovery rate."""
    hazard_res_cost: float = 0.10
    """Resource drain per unit hazard."""
    invest_res_boost: float = 0.15
    """Resource injection coefficient for INVEST action."""

    # ── Military dynamics ─────────────────────────────────────────────── #
    military_hazard_reduction: float = 0.30
    """Immediate hazard reduction per unit military deployment."""
    military_sigma_boost: float = 0.20
    """Immediate stability boost (short-term, exhaustion-gated)."""
    military_decay: float = 0.05
    """Natural decay of military presence per step."""
    max_military_total: float = 0.80
    """Cap on total military deployment (sum over clusters)."""

    # ── Information coherence  Ψ ─────────────────────────────────────── #
    psi_recovery: float = 0.03
    """Coherence recovery toward 1 when no propaganda."""
    psi_propaganda_cost: float = 0.06
    """Coherence reduction from propaganda actions."""
    psi_shock_cost: float = 0.10
    """Coherence reduction per shock event."""

    # ── Media bias dynamics ───────────────────────────────────────────── #
    rho_bias: float = 0.85
    """Bias auto-regression coefficient (memory). Must be in (0,1)."""
    phi_bias_prop: float = 0.15
    """Propaganda effect on bias drift."""
    phi_bias_incoherence: float = 0.10
    """Incoherence noise injection into bias."""
    phi_bias_shock: float = 0.20
    """Shock amplification of bias."""
    beta_max_bias: float = 0.60
    """Maximum absolute bias per cluster."""
    media_autonomy: float = 0.40
    """
    In [0,1]. At 1.0 the agent cannot override bias through propaganda.
    Raised by the curriculum.
    """

    # ── Hawkes shock process ──────────────────────────────────────────── #
    hawkes_base_rate: float = 0.005
    """Baseline shock arrival rate per step."""
    hawkes_alpha: float = 0.30
    """Self-excitation coefficient."""
    hawkes_beta: float = 0.50
    """Decay rate of excitation kernel."""
    shock_pareto_alpha: float = 2.5
    """Shape of Pareto magnitude distribution (heavy tail)."""
    shock_pareto_xmin: float = 0.10
    """Minimum shock magnitude."""

    # ── Reward weights ────────────────────────────────────────────────── #
    w_stability: float = 1.20
    w_fragmentation: float = 0.80
    w_polarization: float = 1.00
    w_exhaustion: float = 1.50
    w_resilience: float = 0.60
    w_smoothness: float = 0.40
    w_unsustainable: float = 2.00
    exhaustion_threshold: float = 0.60
    """Exhaustion above this triggers quadratic penalty."""

    # ── Observation noise ─────────────────────────────────────────────── #
    sigma_obs_base: float = 0.03
    """Base observation noise std."""

    # ── Diplomacy ─────────────────────────────────────────────────────── #
    nu_alliance:       float = 0.05
    """Alliance stability diffusion coefficient (allied σ bleeds across borders)."""
    nu_res_alliance:   float = 0.03
    """Allied resource sharing rate."""
    alpha_hostility:   float = 0.02
    """Hostile-border trust erosion coefficient."""
    alliance_decay:    float = 0.005
    """Per-step drift of alliance values back toward neutral (0)."""
    diplomacy_shift:   float = 0.10
    """Max alliance change per DIPLOMACY action (before weight scaling)."""

    # ── Population dynamics ───────────────────────────────────────────── #
    r_pop:               float = 0.05
    """Intrinsic logistic population growth rate."""
    mu_pop:              float = 0.08
    """Hazard-driven population mortality rate."""
    nu_pop:              float = 0.04
    """Military-driven depopulation (occupation / collateral damage)."""
    gamma_pop:           float = 0.03
    """Polarization-driven emigration rate."""
    delta_pop_migration: float = 0.02
    """Allied-corridor migration inflow coefficient."""
    pop_sigma_lift:      float = 0.02
    """Per-step stability boost from populated clusters."""
    pop_resource_boost:  float = 0.01
    """Per-step resource recovery boost from labour."""
    pop_trust_boost:     float = 0.01
    """Per-step trust building from stable population."""
    pop_hazard_cost:     float = 0.01
    """Per-step hazard signal from depopulated zones."""

    # ── Economy dynamics ──────────────────────────────────────────────── #
    eco_gdp_recovery:       float = 0.06
    """GDP recovery rate from stable+resourced base."""
    eco_gdp_hazard_cost:    float = 0.12
    """GDP loss per unit hazard (normalized)."""
    eco_gdp_debt_drag:      float = 0.05
    """GDP growth drag per unit debt ratio."""
    eco_trade_gain:         float = 0.03
    """Trade-driven GDP growth from allied neighbours."""
    eco_trust_gdp:          float = 0.04
    """Institutional trust boosts GDP convergence."""
    eco_unemp_hazard_rate:  float = 0.08
    """Rate at which hazard creates unemployment."""
    eco_unemp_military_draft: float = 0.03
    """Conscription removes workers, raising unemployment under instability."""
    eco_unemp_gdp_recovery: float = 0.10
    """GDP growth absorbs unemployment."""
    eco_debt_war_rate:      float = 0.04
    """Debt accumulation rate from war spending (military × instability)."""
    eco_debt_crisis_rate:   float = 0.03
    """Emergency borrowing rate during hazard."""
    eco_debt_repay_rate:    float = 0.05
    """Debt repayment enabled by GDP growth."""
    eco_industry_invest:    float = 0.05
    """Investment rate: rebuilds industrial capacity from resource+stability."""
    eco_industry_war_damage: float = 0.08
    """Industrial capacity damaged by conflict."""
    eco_industry_hazard:    float = 0.04
    """Industrial disruption from hazard."""
    # Economy → Cluster feedback
    eco_gdp_resource_boost:  float = 0.015
    """Per-step resource recovery boost from GDP."""
    eco_unemp_hazard_boost:  float = 0.025
    """Per-step hazard increase from unemployment."""
    eco_unemp_trust_cost:    float = 0.018
    """Per-step trust erosion from unemployment."""
    eco_debt_trust_cost:     float = 0.010
    """Per-step trust erosion from debt burden."""
    eco_debt_resource_cost:  float = 0.008
    """Per-step resource drain from debt service."""
    eco_industry_sigma:      float = 0.010
    """Per-step stability boost from industrial base."""
    eco_unemp_sigma_cost:    float = 0.008
    """Per-step stability drain from unemployment."""
    eco_unemp_polar_boost:   float = 0.015
    """Per-step polarization from unemployment+debt."""
    # Economy → Military (via political_interface)
    eco_industry_supply_bonus: float = 0.30
    """Fractional supply refill bonus at full industrial capacity."""
    eco_debt_reinf_penalty:    float = 0.25
    """Fraction of reinforcement pool lost at full debt."""

    # ── Population demographics ───────────────────────────────────────── #
    working_age_frac:    float = 0.62
    """Fraction of population that is working-age (labour force base)."""
    draft_eligible_frac: float = 0.15
    """Fraction of working-age population eligible for conscription."""

    # ── Termination thresholds ────────────────────────────────────────── #
    collapse_exhaustion: float = 0.95
    collapse_hazard_mean: float = 1.50
    collapse_polarization: float = 0.97
    collapse_fragmentation: float = 0.95

    def __post_init__(self) -> None:
        assert 0.0 < self.dt <= 0.05
        assert self.beta_exh_check(), "beta_exh must be > 0"
        assert 0.0 < self.rho_bias < 1.0, "rho_bias must be in (0,1)"
        assert self.n_clusters_min >= 3
        assert self.n_clusters_max <= 50  # Increased for testing
        assert self.hawkes_beta > self.hawkes_alpha, \
            "Hawkes process requires beta > alpha for stationarity"

    def beta_exh_check(self) -> bool:
        return self.beta_exh > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GravitasParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
