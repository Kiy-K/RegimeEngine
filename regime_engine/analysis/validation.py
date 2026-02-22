"""
Extended validation suite for the Adaptive Memory-Driven Regime Engine.

All 10 tests use assert statements exclusively — no print statements.
Run directly:

    python -m regime_engine.analysis.validation

Exit code 0 means all tests passed.

Tests:
  1.  Determinism consistency
  2.  Parameter sensitivity smoothness
  3.  Exhaustion freeze stability
  4.  Faction symmetry invariance
  5.  Memory boundedness proof test
  6.  Fragmentation smooth growth test
  7.  Cascade saturation test
  8.  1000-step stability test
  9.  Multi-agent neutrality baseline
  10. Performance under 10k steps
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from ..core.factions import (
    create_balanced_factions,
    create_dominant_factions,
    recompute_system_state,
)
from ..core.fragmentation import compute_gini
from ..core.integrator import multi_step, step as rk4_step
from ..core.parameters import SystemParameters
from ..core.state import FactionState, RegimeState, SystemState
from ..core.volatility import volatility_upper_bound
from ..agents.action_space import Action, ActionType, apply_actions, null_action
from ..simulation.runner import SimulationRunner
from ..systems.crisis_classifier import ClassifierThresholds, CrisisLevel, classify


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #


def _build_state(
    n_factions: int = 3,
    params: SystemParameters | None = None,
    power_shares: list[float] | None = None,
    radicalization: list[float] | None = None,
    cohesion: list[float] | None = None,
    memory: list[float] | None = None,
    wealth: list[float] | None = None,
    exhaustion: float = 0.0,
    state_gdp: float = 1.0,
) -> RegimeState:
    """Build a fully consistent RegimeState from optional overrides."""
    if params is None:
        params = SystemParameters(n_factions=n_factions)

    factions: List[FactionState] = []
    for i in range(n_factions):
        p = (1.0 / n_factions) if power_shares is None else power_shares[i]
        rad = 0.0 if radicalization is None else radicalization[i]
        coh = 0.5 if cohesion is None else cohesion[i]
        mem = 0.0 if memory is None else memory[i]
        w = 0.5 if wealth is None else wealth[i]
        factions.append(
            FactionState(power=p, radicalization=rad, cohesion=coh, memory=mem, wealth=w)
        )

    placeholder = SystemState(
        legitimacy=0.5, cohesion=0.25, fragmentation=0.0,
        instability=0.0, mobilization=0.0, repression=0.5,
        elite_alignment=0.5, volatility=0.0, exhaustion=exhaustion,
        state_gdp=state_gdp, pillars=tuple(1.0 for _ in range(params.n_pillars))
    )
    aff_mat = tuple(tuple(1.0 if i==j else 0.0 for j in range(n_factions)) for i in range(n_factions))
    state = RegimeState(factions=factions, system=placeholder, affinity_matrix=aff_mat, step=0)
    return recompute_system_state(state, params)


def _all_in_unit(state: RegimeState) -> bool:
    """Return True iff every variable in state is in [0, 1] (or [-1, 1] for aff)."""
    for f in state.factions:
        if not (0.0 <= f.power <= 1.0): return False
        if not (0.0 <= f.radicalization <= 1.0): return False
        if not (0.0 <= f.cohesion <= 1.0): return False
        if not (0.0 <= f.memory <= 1.0): return False
        if not (0.0 <= f.wealth <= 1.0): return False
    sys = state.system
    for val in [sys.legitimacy, sys.cohesion, sys.fragmentation,
                sys.instability, sys.mobilization, sys.repression,
                sys.elite_alignment, sys.volatility, sys.exhaustion,
                sys.state_gdp] + list(sys.pillars):
        if not (0.0 <= val <= 1.0): return False
    for row in state.affinity_matrix:
        for val in row:
            if not (-1.0 <= val <= 1.0): return False
    return True


# --------------------------------------------------------------------------- #
# Test 1 — Determinism consistency                                              #
# --------------------------------------------------------------------------- #


def test_determinism() -> None:
    """Two runs from identical initial state must produce identical trajectories."""
    params = SystemParameters(n_factions=3, seed=42, sigma_noise=0.0)
    s0 = _build_state(n_factions=3, params=params)

    traj_a = [s0]
    traj_b = [s0]
    for _ in range(100):
        traj_a.append(rk4_step(traj_a[-1], params))
        traj_b.append(rk4_step(traj_b[-1], params))

    for sa, sb in zip(traj_a, traj_b):
        assert sa.to_dict() == sb.to_dict(), (
            "Determinism failure: identical inputs produced different outputs"
        )


# --------------------------------------------------------------------------- #
# Test 2 — Parameter sensitivity smoothness                                     #
# --------------------------------------------------------------------------- #


def test_parameter_sensitivity_smoothness() -> None:
    """Smooth parameter variation must produce smooth output variation."""
    base_alpha = 0.05
    n_steps = 50

    prev_instability = None
    for k in range(11):
        alpha = base_alpha + k * 0.01
        params = SystemParameters(n_factions=3, alpha_rad=alpha, sigma_noise=0.0)
        s = _build_state(
            n_factions=3,
            params=params,
            radicalization=[0.3, 0.3, 0.3],
            memory=[0.4, 0.4, 0.4],
        )
        final = multi_step(s, params, n_steps)
        inst = final.system.instability

        if prev_instability is not None:
            delta = abs(inst - prev_instability)
            assert delta < 0.30, (
                f"Parameter sensitivity non-smooth: Δalpha_rad=0.01 caused "
                f"Δinstability={delta:.4f} > 0.30"
            )
        prev_instability = inst


# --------------------------------------------------------------------------- #
# Test 3 — Exhaustion freeze stability                                          #
# --------------------------------------------------------------------------- #


def test_exhaustion_freeze() -> None:
    """At Exh ≈ 1, faction micro-state must be effectively frozen."""
    params = SystemParameters(n_factions=3, sigma_noise=0.0)
    s = _build_state(
        n_factions=3,
        params=params,
        radicalization=[0.5, 0.5, 0.5],
        memory=[0.5, 0.5, 0.5],
        exhaustion=0.999,
    )

    tolerance = 1e-6  # max allowed change per variable per step at near-freeze
    for _ in range(10):
        s_next = rk4_step(s, params)
        for f_before, f_after in zip(s.factions, s_next.factions):
            assert abs(f_after.power - f_before.power) < tolerance, (
                "Exhaustion freeze violated: power changed significantly"
            )
            assert abs(f_after.radicalization - f_before.radicalization) < tolerance, (
                "Exhaustion freeze violated: radicalization changed significantly"
            )
            assert abs(f_after.cohesion - f_before.cohesion) < tolerance, (
                "Exhaustion freeze violated: cohesion changed significantly"
            )
        s = s_next


# --------------------------------------------------------------------------- #
# Test 4 — Faction symmetry invariance                                          #
# --------------------------------------------------------------------------- #


def test_faction_symmetry() -> None:
    """Perfectly symmetric initial conditions must preserve symmetry."""
    params = SystemParameters(n_factions=4, sigma_noise=0.0)
    s = _build_state(
        n_factions=4,
        params=params,
        radicalization=[0.3, 0.3, 0.3, 0.3],
        cohesion=[0.5, 0.5, 0.5, 0.5],
        memory=[0.2, 0.2, 0.2, 0.2],
    )

    for _ in range(50):
        s = rk4_step(s, params)
        powers = s.get_faction_powers()
        rads = s.get_faction_radicalizations()
        cohs = s.get_faction_cohesions()
        mems = s.get_faction_memories()

        tol = 1e-10
        assert float(np.max(powers) - np.min(powers)) < tol, (
            "Symmetry violated: power asymmetry emerged"
        )
        assert float(np.max(rads) - np.min(rads)) < tol, (
            "Symmetry violated: radicalization asymmetry emerged"
        )
        assert float(np.max(cohs) - np.min(cohs)) < tol, (
            "Symmetry violated: cohesion asymmetry emerged"
        )
        assert float(np.max(mems) - np.min(mems)) < tol, (
            "Symmetry violated: memory asymmetry emerged"
        )


# --------------------------------------------------------------------------- #
# Test 5 — Memory boundedness proof test                                        #
# --------------------------------------------------------------------------- #


def test_memory_boundedness() -> None:
    """All Mem_i must remain in [0, 1] under adversarial high-instability forcing."""
    params = SystemParameters(n_factions=3, alpha_mem=0.5, beta_mem=1.05, sigma_noise=0.0)
    # Start with near-max stress: high instability inputs achieved via low cohesion & high rad
    s = _build_state(
        n_factions=3,
        params=params,
        radicalization=[0.9, 0.9, 0.9],
        cohesion=[0.05, 0.05, 0.05],
        memory=[0.0, 0.0, 0.0],
    )

    for step_num in range(1000):
        s = rk4_step(s, params)
        for f in s.factions:
            assert 0.0 <= f.memory <= 1.0, (
                f"Memory out of bounds at step {step_num}: {f.memory}"
            )


# --------------------------------------------------------------------------- #
# Test 6 — Fragmentation smooth growth test                                     #
# --------------------------------------------------------------------------- #


def test_fragmentation_smooth_growth() -> None:
    """Fragmentation F must grow monotonically as Gini increases."""
    from ..core.fragmentation import fragmentation_from_gini

    params = SystemParameters()
    prev_f = 0.0
    for k in range(20):
        gini = k * 0.04  # 0.0, 0.04, ..., 0.76
        f = fragmentation_from_gini(gini, params.lambda_frag)
        assert 0.0 <= f < 1.0, f"Fragmentation out of [0,1) at gini={gini}: {f}"
        assert f >= prev_f, (
            f"Fragmentation non-monotone: gini={gini:.2f} gave f={f:.4f} < prev={prev_f:.4f}"
        )
        prev_f = f


# --------------------------------------------------------------------------- #
# Test 7 — Cascade saturation test                                              #
# --------------------------------------------------------------------------- #


def test_cascade_saturation() -> None:
    """Volatility V must stay strictly below 1.0 at all times."""
    params = SystemParameters(n_factions=3, sigma_noise=0.0)
    upper = volatility_upper_bound(params.kappa_v)
    assert upper < 1.0, f"Theoretical upper bound must be < 1, got {upper}"

    # Max-stress scenario: high mem, high rad, low cohesion, low exhaustion
    s = _build_state(
        n_factions=3,
        params=params,
        radicalization=[0.95, 0.95, 0.95],
        cohesion=[0.05, 0.05, 0.05],
        memory=[0.95, 0.95, 0.95],
        exhaustion=0.0,
    )

    for step_num in range(500):
        s = rk4_step(s, params)
        v = s.system.volatility
        assert v < 1.0, f"Volatility reached 1.0 at step {step_num}: {v}"
        assert v >= 0.0, f"Volatility went negative at step {step_num}: {v}"


# --------------------------------------------------------------------------- #
# Test 8 — 1000-step stability test                                             #
# --------------------------------------------------------------------------- #


def test_1000_step_stability() -> None:
    """All variables must remain in [0, 1] across 1000 steps."""
    params = SystemParameters(n_factions=3)
    s = _build_state(
        n_factions=3,
        params=params,
        radicalization=[0.4, 0.2, 0.6],
        cohesion=[0.6, 0.8, 0.3],
        memory=[0.3, 0.1, 0.5],
    )

    for step_num in range(1000):
        s = rk4_step(s, params)
        assert _all_in_unit(s), (
            f"Bounds violation at step {step_num}: {s.to_dict()}"
        )


# --------------------------------------------------------------------------- #
# Test 9 — Multi-agent neutrality baseline                                      #
# --------------------------------------------------------------------------- #


def test_multi_agent_neutrality() -> None:
    """Balanced opposing actions should produce bounded net drift."""
    params = SystemParameters(n_factions=3, sigma_noise=0.0)
    s = _build_state(n_factions=3, params=params)

    # Two opposing agents: one stabilises, one provokes (same faction, same magnitude)
    stabiliise = Action(
        action_type=ActionType.STABILITY_OPERATION,
        actor_idx=1,
        target_idx=0,
        intensity=0.10,
        agent_id="agent_stabilise",
    )
    provoke = Action(
        action_type=ActionType.PROVOKE,
        actor_idx=2,
        target_idx=0,
        intensity=0.10,
        agent_id="agent_provoke",
    )

    for step_num in range(200):
        s = apply_actions(s, [stabiliise, provoke])
        s = recompute_system_state(s, params)
        s = rk4_step(s, params)
        assert _all_in_unit(s), (
            f"Neutrality test: bounds violated at step {step_num}"
        )


# --------------------------------------------------------------------------- #
# Test 10 — Performance under 10k steps                                         #
# --------------------------------------------------------------------------- #


def test_performance_10k_steps() -> None:
    """10,000 steps must complete within 10 seconds on a single core."""
    params = SystemParameters(n_factions=3, max_steps=10_000)
    s = _build_state(n_factions=3, params=params)

    start = time.perf_counter()
    final = multi_step(s, params, 10_000)
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0, (
        f"Performance test failed: 10k steps took {elapsed:.2f}s (limit: 10s)"
    )
    assert _all_in_unit(final), "Bounds violated at end of 10k-step run"


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #


def run_all_tests() -> None:
    """Execute all 10 validation tests; raises AssertionError on first failure."""
    tests = [
        ("1. Determinism consistency", test_determinism),
        ("2. Parameter sensitivity smoothness", test_parameter_sensitivity_smoothness),
        ("3. Exhaustion freeze stability", test_exhaustion_freeze),
        ("4. Faction symmetry invariance", test_faction_symmetry),
        ("5. Memory boundedness proof test", test_memory_boundedness),
        ("6. Fragmentation smooth growth test", test_fragmentation_smooth_growth),
        ("7. Cascade saturation test", test_cascade_saturation),
        ("8. 1000-step stability test", test_1000_step_stability),
        ("9. Multi-agent neutrality baseline", test_multi_agent_neutrality),
        ("10. Performance under 10k steps", test_performance_10k_steps),
    ]
    for name, fn in tests:
        fn()


if __name__ == "__main__":
    run_all_tests()
