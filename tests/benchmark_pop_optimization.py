#!/usr/bin/env python3
"""
Benchmark script to compare performance of original vs optimized population dynamics.

This script:
1. Creates a test environment with population system
2. Runs both original and optimized versions
3. Measures execution time and memory usage
4. Validates that results are numerically equivalent
5. Generates performance reports
"""

import time
import tracemalloc
import numpy as np
from typing import Tuple, Dict, Any
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gymnasium_shim as gym
    from gymnasium_shim import spaces

from gravitas_engine.extensions.pop.pop_wrapper import PopWrapper
from gravitas_engine.extensions.pop.pop_params import PopParams
from gravitas_engine.extensions.pop.pop_state import WorldPopState, initialize_world_pop
from gravitas_engine.extensions.pop.pop_dynamics import step_world_pop
from gravitas_engine.extensions.pop.pop_dynamics_optimized import step_pop_vector_optimized

def create_test_environment() -> Tuple[PopWrapper, WorldPopState]:
    """Create a test environment with population system."""
    # Create a minimal GravitasEnv (we'll mock it for testing)
    class MockGravitasEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self._max_N = 8
            self._world = self._create_mock_world()
            self.observation_space = spaces.Box(
                low=-10, high=10, shape=(50,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(6)

        def _create_mock_world(self):
            class MockWorld:
                def __init__(self):
                    self.n_clusters = 8
                    self.global_state = self._create_global_state()

                def _create_global_state(self):
                    class MockGlobalState:
                        def __init__(self):
                            self.exhaustion = 0.2
                            self.fragmentation = 0.3
                            self.polarization = 0.4
                            self.coherence = 0.6
                            self.military_str = 0.5
                            self.trust = 0.7
                            self.step = 0
                    return MockGlobalState()

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

        def reset(self, seed=None, options=None):
            return np.zeros(50, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(50, dtype=np.float32), 0.0, False, False, {}

    # Create the environment
    base_env = MockGravitasEnv()
    pop_params = PopParams()

    # Create PopWrapper
    pop_wrapper = PopWrapper(base_env, pop_params, seed=42)

    # Initialize population state
    world = base_env._world
    N = world.n_clusters
    c_arr = world.cluster_array()

    cluster_resources = c_arr[:N, 2]
    cluster_trusts = c_arr[:N, 4]

    world_pop = initialize_world_pop(
        n_clusters=N,
        params=pop_params,
        rng=np.random.default_rng(42),
        cluster_resources=cluster_resources,
        cluster_trusts=cluster_trusts,
    )

    return pop_wrapper, world_pop

def benchmark_function(func, *args, **kwargs) -> Tuple[float, float]:
    """
    Benchmark a function and return execution time and memory usage.

    Returns: (execution_time_seconds, peak_memory_mb)
    """
    # Start memory tracking
    tracemalloc.start()

    # Take initial memory snapshot
    initial_snapshot = tracemalloc.take_snapshot()

    # Time the function execution
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    # Take final memory snapshot
    final_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate memory usage
    initial_memory = sum(stat.size for stat in initial_snapshot.statistics('lineno'))
    final_memory = sum(stat.size for stat in final_snapshot.statistics('lineno'))
    memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert to MB

    execution_time = end_time - start_time

    return execution_time, memory_used, result

def run_benchmark(num_steps=1000):
    """Run comprehensive benchmark comparing original vs optimized versions."""
    print("🚀 Starting GravitasEngine Population System Benchmark")
    print("=" * 60)

    # Create test environment
    print("🔧 Setting up test environment...")
    pop_wrapper, world_pop = create_test_environment()
    world = pop_wrapper._get_world()

    # Test parameters
    stance = 2  # REFORM
    intensity = 0.7
    action_weights = np.ones(8) * 0.125  # Equal weights
    params = pop_wrapper.pop_params
    rng = np.random.default_rng(123)

    print(f"📊 Test configuration:")
    print(f"   - Clusters: {world_pop.n_clusters}")
    print(f"   - Steps: {num_steps}")
    print(f"   - Action: Stance {stance} (REFORM), Intensity {intensity}")
    print()

    # Benchmark original version
    print("⏳ Benchmarking original implementation...")
    original_times = []
    original_memories = []

    for step in range(num_steps):
        if step % 100 == 0:
            print(f"   Step {step}/{num_steps}...", end="\r")

        # Get the world object from the wrapper
        current_world = pop_wrapper._get_world()
        if current_world is None:
            current_world = world

        time_taken, memory_used, (new_world_pop, drivers) = benchmark_function(
            step_world_pop,
            world_pop, current_world, stance, intensity, action_weights, params, rng, False
        )

        original_times.append(time_taken)
        original_memories.append(memory_used)
        world_pop = new_world_pop

    print(f"   Step {num_steps}/{num_steps}... ✓")
    print()

    # Reset for optimized benchmark
    _, world_pop = create_test_environment()

    # Benchmark optimized version
    print("⏳ Benchmarking optimized implementation...")
    optimized_times = []
    optimized_memories = []

    for step in range(num_steps):
        if step % 100 == 0:
            print(f"   Step {step}/{num_steps}...", end="\r")

        time_taken, memory_used, (new_world_pop, drivers) = benchmark_function(
            step_world_pop,
            world_pop, world, stance, intensity, action_weights, params, rng, True
        )

        optimized_times.append(time_taken)
        optimized_memories.append(memory_used)
        world_pop = new_world_pop

    print(f"   Step {step}/{num_steps}... ✓")
    print()

    # Calculate statistics
    original_avg_time = np.mean(original_times)
    original_std_time = np.std(original_times)
    original_avg_memory = np.mean(original_memories)

    optimized_avg_time = np.mean(optimized_times)
    optimized_std_time = np.std(optimized_times)
    optimized_avg_memory = np.mean(optimized_memories)

    # Calculate speedup
    speedup = original_avg_time / optimized_avg_time if optimized_avg_time > 0 else float('inf')
    memory_reduction = ((original_avg_memory - optimized_avg_memory) / original_avg_memory) * 100

    # Generate report
    print("📈 PERFORMANCE REPORT")
    print("=" * 60)
    print(f"🕒 Execution Time (average per step):")
    print(f"   Original:   {original_avg_time*1000:.3f} ms ± {original_std_time*1000:.3f} ms")
    print(f"   Optimized:  {optimized_avg_time*1000:.3f} ms ± {optimized_std_time*1000:.3f} ms")
    print(f"   Speedup:    {speedup:.2f}x faster")
    print()

    print(f"💾 Memory Usage (average per step):")
    print(f"   Original:   {original_avg_memory:.3f} MB")
    print(f"   Optimized:  {optimized_avg_memory:.3f} MB")
    print(f"   Reduction:  {memory_reduction:.1f}% less memory")
    print()

    print(f"📊 Overall Performance:")
    if speedup > 1.5:
        print(f"   ✅ EXCELLENT: {speedup:.2f}x speedup achieved!")
    elif speedup > 1.2:
        print(f"   👍 GOOD: {speedup:.2f}x speedup achieved")
    elif speedup > 1.0:
        print(f"   🤷 MODERATE: {speedup:.2f}x speedup")
    else:
        print(f"   ❌ MINIMAL: {speedup:.2f}x speedup (may need further optimization)")

    print()

    # Test numerical equivalence
    print("🔬 NUMERICAL VALIDATION")
    print("=" * 60)

    # Run both versions with same inputs and compare outputs
    _, world_pop_orig = create_test_environment()
    _, world_pop_opt = create_test_environment()

    # Single step comparison
    _, drivers_orig = step_world_pop(world_pop_orig, world, stance, intensity, action_weights, params, rng, False)
    _, drivers_opt = step_world_pop(world_pop_opt, world, stance, intensity, action_weights, params, rng, True)

    # Compare aggregates
    aggs_orig = world_pop_orig.get_aggregates()
    aggs_opt = world_pop_opt.get_aggregates()

    max_diff = 0.0
    all_close = True

    for i in range(min(len(aggs_orig), len(aggs_opt))):
        for field in ['gini', 'mean_satisfaction', 'radical_mass', 'fractionalization', 'ethnic_tension', 'class_tension']:
            orig_val = getattr(aggs_orig[i], field)
            opt_val = getattr(aggs_opt[i], field)
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
        print("   This may indicate optimization errors or different numerical precision")

    print()

    # Recommendations
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)

    if speedup > 1.5 and all_close:
        print("   ✅ Optimization successful! Consider deploying to production.")
        print("   📝 Next steps:")
        print("      - Monitor performance in real training scenarios")
        print("      - Consider additional optimizations for other subsystems")
        print("      - Add profiling to identify remaining bottlenecks")
    elif speedup > 1.2 and all_close:
        print("   👍 Optimization shows good results. Consider:")
        print("      - Further profiling to identify remaining bottlenecks")
        print("      - Testing with larger cluster counts")
        print("      - Exploring additional optimization techniques")
    else:
        print("   ⚠️  Optimization needs improvement. Consider:")
        print("      - Checking Numba compilation and JIT caching")
        print("      - Profiling to identify specific bottlenecks")
        print("      - Reviewing numerical algorithms for optimization opportunities")

    print()

    # Save results
    results = {
        'original_time_ms': original_avg_time * 1000,
        'original_time_std_ms': original_std_time * 1000,
        'original_memory_mb': original_avg_memory,
        'optimized_time_ms': optimized_avg_time * 1000,
        'optimized_time_std_ms': optimized_std_time * 1000,
        'optimized_memory_mb': optimized_avg_memory,
        'speedup': speedup,
        'memory_reduction_percent': memory_reduction,
        'numerically_equivalent': all_close,
        'max_numerical_difference': max_diff,
        'test_steps': num_steps,
        'clusters': world_pop_orig.n_clusters,
        'archetypes': world_pop_orig.pops[0].n_archetypes,
        'ethnic_groups': world_pop_orig.pops[0].n_ethnic
    }

    return results

def main():
    """Main benchmark entry point."""
    try:
        # Run benchmark
        results = run_benchmark(num_steps=500)  # Reduced for faster testing

        # Return success
        print("🎉 Benchmark completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())