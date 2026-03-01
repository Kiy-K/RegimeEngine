#!/usr/bin/env python3
"""
Simple benchmark to test the optimized population dynamics.
"""

import time
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gravitas_engine.extensions.pop.pop_params import PopParams
from gravitas_engine.extensions.pop.pop_state import initialize_world_pop, PopVector
from gravitas_engine.extensions.pop.pop_dynamics import step_pop_vector, step_pop_vector_optimized

def create_mock_world():
    """Create a simple mock world object for testing."""
    class MockGlobalState:
        def __init__(self):
            self.exhaustion = 0.2
            self.fragmentation = 0.3
            self.polarization = 0.4
            self.coherence = 0.6
            self.military_str = 0.5
            self.trust = 0.7
            self.step = 0

    class MockWorld:
        def __init__(self):
            self.n_clusters = 8
            self.global_state = MockGlobalState()

        def cluster_array(self):
            # Return (N, 6) array: σ, h, r, m, τ, p
            return np.array([
                [0.3, 0.2, 0.6, 0.1, 0.7, 0.4],  # cluster 0
                [0.4, 0.1, 0.5, 0.2, 0.6, 0.3],  # cluster 1
                [0.2, 0.3, 0.7, 0.0, 0.8, 0.5],  # cluster 2
                [0.5, 0.2, 0.4, 0.3, 0.5, 0.2],  # cluster 3
                [0.3, 0.4, 0.5, 0.1, 0.6, 0.4],  # cluster 4
                [0.4, 0.3, 0.6, 0.2, 0.7, 0.3],  # cluster 5
                [0.2, 0.2, 0.8, 0.0, 0.9, 0.5],  # cluster 6
                [0.3, 0.1, 0.7, 0.1, 0.8, 0.4],  # cluster 7
            ], dtype=np.float64)

    return MockWorld()

def benchmark_single_pop_step():
    """Benchmark a single population vector step."""
    print("🚀 Simple Population Dynamics Benchmark")
    print("=" * 50)

    # Create test data
    params = PopParams()
    rng = np.random.default_rng(42)

    # Create a single pop vector
    pop = initialize_world_pop(
        n_clusters=1,
        params=params,
        rng=rng,
        cluster_resources=np.array([0.6]),
        cluster_trusts=np.array([0.5]),
    ).pops[0]

    # Test parameters
    cluster_sigma = 0.3
    cluster_hazard = 0.2
    cluster_trust = 0.7
    cluster_resource = 0.6
    sys_polarization = 0.4
    sys_exhaustion = 0.2
    sys_coherence = 0.6
    cultural_dist = np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.4], [0.2, 0.4, 0.0]])
    dt = 0.01
    military_load = 0.1

    print("📊 Test configuration:")
    print(f"   - Archetypes: {pop.n_archetypes}")
    print(f"   - Ethnic groups: {pop.n_ethnic}")
    print()

    # Benchmark original version
    print("⏳ Benchmarking original implementation...")

    start_time = time.perf_counter()
    for _ in range(1000):
        result_orig = step_pop_vector(
            pop=pop,
            cluster_sigma=cluster_sigma,
            cluster_hazard=cluster_hazard,
            cluster_trust=cluster_trust,
            cluster_resource=cluster_resource,
            sys_polarization=sys_polarization,
            sys_exhaustion=sys_exhaustion,
            sys_coherence=sys_coherence,
            cultural_dist=cultural_dist,
            params=params,
            dt=dt,
            military_load=military_load,
        )
    original_time = time.perf_counter() - start_time

    print(f"   Original: {original_time*1000:.3f} ms for 1000 steps")
    print(f"   Average: {original_time*1000/1000:.3f} ms per step")

    # Benchmark optimized version
    print("⏳ Benchmarking optimized implementation...")

    start_time = time.perf_counter()
    for _ in range(1000):
        result_opt = step_pop_vector_optimized(
            pop=pop,
            cluster_sigma=cluster_sigma,
            cluster_hazard=cluster_hazard,
            cluster_trust=cluster_trust,
            cluster_resource=cluster_resource,
            sys_polarization=sys_polarization,
            sys_exhaustion=sys_exhaustion,
            sys_coherence=sys_coherence,
            cultural_dist=cultural_dist,
            params=params,
            dt=dt,
            military_load=military_load,
        )
    optimized_time = time.perf_counter() - start_time

    print(f"   Optimized: {optimized_time*1000:.3f} ms for 1000 steps")
    print(f"   Average: {optimized_time*1000/1000:.3f} ms per step")

    # Calculate speedup
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"   Speedup: {speedup:.2f}x faster")
    print()

    # Test numerical equivalence
    print("🔬 NUMERICAL VALIDATION")
    print("=" * 50)

    # Compare results
    max_diff = 0.0
    all_close = True

    for field in ['satisfaction', 'radicalization', 'income', 'ethnic_tension']:
        if field == 'ethnic_tension':
            orig_val = result_orig.ethnic_tension
            opt_val = result_opt.ethnic_tension
        else:
            orig_val = getattr(result_orig, field)
            opt_val = getattr(result_opt, field)

        if hasattr(orig_val, '__iter__'):
            diff = np.max(np.abs(np.array(orig_val) - np.array(opt_val)))
        else:
            diff = abs(orig_val - opt_val)

        if diff > max_diff:
            max_diff = diff

        if diff > 1e-6:  # Allow small numerical differences
            all_close = False

    if all_close:
        print("   ✅ NUMERICALLY EQUIVALENT: Results match within tolerance")
        print(f"   Max difference: {max_diff:.2e}")
    else:
        print("   ⚠️  NUMERICAL DIFFERENCES DETECTED")
        print(f"   Max difference: {max_diff:.2e}")

    print()

    # Performance assessment
    print("🎯 PERFORMANCE ASSESSMENT")
    print("=" * 50)

    if speedup > 1.5 and all_close:
        print("   ✅ EXCELLENT: Optimization successful!")
        print("   The Numba-optimized version provides significant performance improvement")
        print("   while maintaining numerical accuracy.")
    elif speedup > 1.2 and all_close:
        print("   👍 GOOD: Optimization shows promising results.")
        print("   Consider further optimizations for even better performance.")
    elif all_close:
        print("   🤷 MODERATE: Some improvement, but more optimization potential exists.")
    else:
        print("   ⚠️  ISSUES DETECTED: Check numerical accuracy or optimization implementation.")

    return {
        'original_time_ms': original_time * 1000,
        'optimized_time_ms': optimized_time * 1000,
        'speedup': speedup,
        'numerically_equivalent': all_close,
        'max_numerical_difference': max_diff,
        'archetypes': pop.n_archetypes,
        'ethnic_groups': pop.n_ethnic
    }

def main():
    """Main benchmark entry point."""
    try:
        results = benchmark_single_pop_step()
        print("\n🎉 Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())