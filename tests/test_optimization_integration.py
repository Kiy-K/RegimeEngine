#!/usr/bin/env python3
"""
Integration test — verify pop dynamics + Cython kernels work correctly.
No Numba dependency. Pure NumPy vectorized + optional Cython hot paths.
"""

import sys
import os
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_pop_dynamics():
    """Test that pop dynamics produce valid output."""
    print("1. Testing pop dynamics imports...")
    from extensions.pop.pop_dynamics import step_pop_vector, compute_gini
    from extensions.pop.pop_params import PopParams
    from extensions.pop.pop_state import initialize_world_pop
    print("   OK — no Numba, pure NumPy vectorized")

    print("2. Testing Gini computation...")
    sizes = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.04, 0.03, 0.03])
    income = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    gini = compute_gini(sizes, income)
    assert 0.0 <= gini <= 1.0, f"Gini out of range: {gini}"
    print(f"   OK — Gini = {gini:.4f}")

    print("3. Testing step_pop_vector...")
    params = PopParams()
    rng = np.random.default_rng(42)
    pop = initialize_world_pop(
        n_clusters=1, params=params, rng=rng,
        cluster_resources=np.array([0.6]),
        cluster_trusts=np.array([0.5]),
    ).pops[0]

    result = step_pop_vector(
        pop=pop, cluster_sigma=0.3, cluster_hazard=0.2,
        cluster_trust=0.7, cluster_resource=0.6,
        sys_polarization=0.4, sys_exhaustion=0.2, sys_coherence=0.6,
        cultural_dist=np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.4], [0.2, 0.4, 0.0]]),
        params=params, dt=0.01, military_load=0.1,
    )
    assert result.satisfaction is not None
    print(f"   OK — satisfaction range [{result.satisfaction.min():.3f}, {result.satisfaction.max():.3f}]")

    print("4. Performance benchmark (1000 steps)...")
    t0 = time.perf_counter()
    for _ in range(1000):
        result = step_pop_vector(
            pop=pop, cluster_sigma=0.3, cluster_hazard=0.2,
            cluster_trust=0.7, cluster_resource=0.6,
            sys_polarization=0.4, sys_exhaustion=0.2, sys_coherence=0.6,
            cultural_dist=np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.4], [0.2, 0.4, 0.0]]),
            params=params, dt=0.01, military_load=0.1,
        )
    elapsed = time.perf_counter() - t0
    print(f"   OK — {elapsed*1000:.1f}ms for 1000 steps ({elapsed:.4f}ms/step)")


def test_cython_kernels():
    """Test Cython compiled kernels if available."""
    print("5. Testing Cython kernels...")
    try:
        from gravitas_engine.core._kernels import compute_hazard, compute_gini as cy_gini
        sizes = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.04, 0.03, 0.03])
        income = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
        gini = cy_gini(sizes, income)
        print(f"   OK — Cython Gini = {gini:.4f}")
    except ImportError:
        print("   SKIP — Cython kernels not compiled (run ./build_cython.sh)")


def test_llm_game_import():
    """Test that llm_game.py imports cleanly with all systems."""
    print("6. Testing llm_game.py import (all 12 systems)...")
    import gravitas.llm_game
    print("   OK — all systems loaded")


def main():
    print("=" * 60)
    print("  GRAVITAS Engine — Integration Test (no Numba)")
    print("=" * 60)
    try:
        test_pop_dynamics()
        test_cython_kernels()
        test_llm_game_import()
        print("\nAll tests passed.")
        return 0
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())