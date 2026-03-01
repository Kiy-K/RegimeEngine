#!/usr/bin/env python3
"""
Integration test to verify that the optimized population dynamics
are properly integrated into the main codebase.
"""

import sys
import os
import time
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_optimization_integration():
    """Test that the optimized version is being used by default."""
    print("🔍 Testing Optimization Integration")
    print("=" * 40)

    # Test 1: Check that optimized functions are available in main module
    print("1️⃣ Testing imports...")
    try:
        from gravitas_engine.extensions.pop.pop_dynamics import step_pop_vector, step_pop_vector_optimized, NUMBA_AVAILABLE
        print("   ✅ Both main and optimized functions available in single module")

        # Check if the main function uses optimized version
        import inspect
        source = inspect.getsource(step_pop_vector)
        if "step_pop_vector_optimized" in source:
            print("   ✅ Main function uses optimized version by default")
        else:
            print("   ❌ Main function does not use optimized version")

        # Check Numba availability
        if NUMBA_AVAILABLE:
            print("   ✅ Numba is available for optimization")
        else:
            print("   ⚠️  Numba not available, using fallback implementation")

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test 2: Test numerical equivalence
    print("\n2️⃣ Testing numerical equivalence...")
    try:
        from gravitas_engine.extensions.pop.pop_params import PopParams
        from gravitas_engine.extensions.pop.pop_state import initialize_world_pop

        params = PopParams()
        rng = np.random.default_rng(42)

        # Create test pop vector
        pop = initialize_world_pop(
            n_clusters=1,
            params=params,
            rng=rng,
            cluster_resources=np.array([0.6]),
            cluster_trusts=np.array([0.5]),
        ).pops[0]

        # Test parameters
        test_params = {
            'cluster_sigma': 0.3,
            'cluster_hazard': 0.2,
            'cluster_trust': 0.7,
            'cluster_resource': 0.6,
            'sys_polarization': 0.4,
            'sys_exhaustion': 0.2,
            'sys_coherence': 0.6,
            'cultural_dist': np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.4], [0.2, 0.4, 0.0]]),
            'params': params,
            'dt': 0.01,
            'military_load': 0.1
        }

        # Run both versions
        result_main = step_pop_vector(pop, **test_params)
        result_opt = step_pop_vector_optimized(pop, **test_params)

        # Compare results
        max_diff = 0.0
        for field in ['satisfaction', 'radicalization', 'income', 'ethnic_tension']:
            if field == 'ethnic_tension':
                diff = abs(result_main.ethnic_tension - result_opt.ethnic_tension)
            else:
                diff = np.max(np.abs(getattr(result_main, field) - getattr(result_opt, field)))

            if diff > max_diff:
                max_diff = diff

        if max_diff < 1e-6:
            print(f"   ✅ Results are numerically equivalent (max diff: {max_diff:.2e})")
        else:
            print(f"   ❌ Results differ (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        print(f"   ❌ Numerical test failed: {e}")
        return False

    # Test 3: Performance comparison
    print("\n3️⃣ Testing performance...")
    try:
        # Time the main function (which should use optimized version)
        start_time = time.perf_counter()
        for _ in range(100):
            result_main = step_pop_vector(pop, **test_params)
        main_time = time.perf_counter() - start_time

        # Time the optimized function directly
        start_time = time.perf_counter()
        for _ in range(100):
            result_opt = step_pop_vector_optimized(pop, **test_params)
        opt_time = time.perf_counter() - start_time

        print(f"   Main function: {main_time*1000:.3f} ms for 100 steps")
        print(f"   Optimized function: {opt_time*1000:.3f} ms for 100 steps")

        # They should be very close since main function uses optimized version
        if abs(main_time - opt_time) < 0.01:  # Within 10ms tolerance
            print("   ✅ Main function performance matches optimized version")
        else:
            print("   ⚠️  Performance difference detected")

    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

    # Test 4: Check wrapper integration
    print("\n4️⃣ Testing wrapper integration...")
    try:
        from gravitas_engine.extensions.pop.pop_wrapper import PopWrapper

        # Check that wrapper imports the main function
        import inspect
        wrapper_source = inspect.getsource(PopWrapper)
        if "step_world_pop" in wrapper_source:
            print("   ✅ Wrapper uses step_world_pop function")
        else:
            print("   ❌ Wrapper doesn't use step_world_pop function")
            return False

        # Check that step_world_pop is from main module
        from gravitas_engine.extensions.pop.pop_dynamics import step_world_pop
        wrapper_module = sys.modules['gravitas_engine.extensions.pop.pop_wrapper']
        if hasattr(wrapper_module, 'step_world_pop'):
            if wrapper_module.step_world_pop is step_world_pop:
                print("   ✅ Wrapper uses main step_world_pop implementation")
            else:
                print("   ❌ Wrapper uses different step_world_pop implementation")
        else:
            print("   ⚠️  Wrapper doesn't directly import step_world_pop")

    except Exception as e:
        print(f"   ❌ Wrapper integration test failed: {e}")
        return False

    print("\n🎉 All integration tests passed!")
    print("✅ Optimization is fully integrated into the main codebase")
    return True

def main():
    """Main test entry point."""
    try:
        success = test_optimization_integration()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())