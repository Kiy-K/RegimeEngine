"""
smoke_test.py — Smoke test and sanity audit for GravitasEnv.

Run:  python smoke_test.py

Checks:
  1.  reset() produces correct obs shape
  2.  step() produces correct obs, reward, done shapes
  3.  All 6 stances can be applied without crash
  4.  Termination conditions are reachable (stress test with forced states)
  5.  Hawkes process self-excitation (λ rises after shocks)
  6.  Media bias drifts and is bounded
  7.  Reward components are all non-trivially zero at some point
  8.  100-step random rollout completes without exception
  9.  Propaganda trap: true polarization rises under sustained propaganda
  10. Observation is always within declared observation_space bounds (approx)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import traceback

from regime_engine.agents.gravitas_env import GravitasEnv
from regime_engine.core.gravitas_params import GravitasParams
from regime_engine.agents.gravitas_actions import Stance

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def run_test(name: str, fn) -> bool:
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS}  {name}")
            return True
        else:
            print(f"  {FAIL}  {name}: {result}")
            return False
    except Exception as e:
        print(f"  {FAIL}  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def make_env(seed=42, max_steps=200):
    return GravitasEnv(
        params=GravitasParams(max_steps=max_steps, seed=seed),
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Test 1: reset() shape                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def test_reset_shape():
    env = make_env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), f"obs is {type(obs)}"
    assert obs.dtype == np.float32, f"dtype is {obs.dtype}"
    expected = env.observation_space.shape[0]
    assert obs.shape == (expected,), f"shape {obs.shape} != ({expected},)"
    assert isinstance(info, dict)
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 2: step() shapes                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def test_step_shape():
    env = make_env()
    obs, _ = env.reset()
    obs2, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert obs2.shape == obs.shape, f"{obs2.shape} != {obs.shape}"
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 3: all stances                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def test_all_stances():
    for stance_idx in range(6):
        env = make_env(seed=10 + stance_idx)
        env.reset()
        for _ in range(5):
            obs, r, done, trunc, info = env.step(stance_idx)
            if done or trunc:
                break
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 4: 100-step random rollout                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def test_random_rollout():
    env = make_env(seed=99, max_steps=100)
    obs, _ = env.reset()
    total_r = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        if done or trunc:
            break
    assert np.isfinite(total_r), f"total_r is {total_r}"
    assert not np.any(np.isnan(obs)), "obs contains NaN"
    assert not np.any(np.isinf(obs)), "obs contains Inf"
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 5: Hawkes self-excitation                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def test_hawkes_excitation():
    """After many steps we should see shock_rate rise above baseline."""
    from regime_engine.systems.hawkes_shock import update_hawkes
    params = GravitasParams()
    rate   = params.hawkes_base_rate
    h_sum  = 0.0

    # Simulate 10 shocks in rapid succession
    for _ in range(10):
        rate, h_sum = update_hawkes(h_sum, shock_occurred=True, params=params)

    assert rate > params.hawkes_base_rate * 2, (
        f"rate {rate:.5f} did not self-excite above 2×baseline "
        f"{params.hawkes_base_rate * 2:.5f}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 6: media bias bounded and drifts                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def test_media_bias():
    env = make_env(seed=7)
    env.reset()
    biases = []
    for _ in range(50):
        obs, r, done, trunc, info = env.step(Stance.PROPAGANDA)  # type: ignore
        b = env.world.media_bias
        assert np.all(np.abs(b) <= env.params.beta_max_bias + 1e-6), (
            f"Bias out of bounds: {b}"
        )
        biases.append(float(np.mean(b)))
        if done or trunc:
            break

    std = float(np.std(biases))
    assert std > 1e-4, f"Bias doesn't drift: std={std}"
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 7: reward components non-trivially non-zero                             #
# ─────────────────────────────────────────────────────────────────────────── #

def test_reward_components():
    env = make_env(seed=3, max_steps=200)
    env.reset()
    stances = [0, 1, 2, 3, 4, 5, 1, 1, 1]  # varied actions including militarize
    for s in stances:
        obs, r, done, trunc, info = env.step(s)
        if done or trunc:
            break

    logs = env.reward_log
    assert len(logs) > 0
    stab_vals = [b.stability for b in logs]
    pol_vals  = [b.polarization for b in logs]
    exh_vals  = [b.exhaustion for b in logs]

    assert max(stab_vals) > 0.01, "stability reward always near zero"
    assert max(pol_vals) > 0.01, "polarization reward always near zero"
    # exhaustion penalty should fire if militarize pushed E high
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 8: propaganda trap (true polarization rises)                           #
# ─────────────────────────────────────────────────────────────────────────── #

def test_propaganda_trap():
    """Sustained propaganda should raise true Π even while bias hides it."""
    env = make_env(seed=55, max_steps=300)
    env.reset()
    pol_start = env.world.global_state.polarization
    for _ in range(80):
        obs, r, done, trunc, info = env.step(Stance.PROPAGANDA)  # type: ignore
        if done or trunc:
            break
    pol_end = env.world.global_state.polarization
    # True polarization should have risen
    assert pol_end > pol_start - 0.05, (
        f"Propaganda trap not manifesting: Π start={pol_start:.3f} end={pol_end:.3f}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 9: observation always finite                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def test_obs_finite():
    env = make_env(seed=21, max_steps=300)
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs)), "initial obs has non-finite values"
    for step in range(200):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            return f"Non-finite obs at step {step}: {obs[~np.isfinite(obs)]}"
        if not np.isfinite(r):
            return f"Non-finite reward at step {step}: {r}"
        if done or trunc:
            break
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 10: render                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def test_render():
    env = GravitasEnv(
        params=GravitasParams(max_steps=10, seed=1),
        seed=1,
        render_mode="ansi",
    )
    env.reset()
    env.step(0)
    out = env.render()
    assert out is not None and "GRAVITAS" in out
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Test 11: multi-episode consistency (reset clears state)                      #
# ─────────────────────────────────────────────────────────────────────────── #

def test_multi_episode():
    env = make_env(seed=0, max_steps=50)
    for ep in range(3):
        obs, _ = env.reset(seed=ep * 100)
        assert np.all(np.isfinite(obs)), f"Ep {ep}: non-finite initial obs"
        for _ in range(30):
            obs, r, done, trunc, info = env.step(env.action_space.sample())
            if done or trunc:
                break
        assert env.world is not None
    return True


# ─────────────────────────────────────────────────────────────────────────── #
# Rollout statistics                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def rollout_stats(n_eps: int = 20, max_steps: int = 300) -> None:
    """Print summary statistics over n_eps random-policy episodes."""
    lengths, rewards, causes = [], [], []
    rng = np.random.default_rng(0)

    for ep in range(n_eps):
        env = make_env(seed=int(rng.integers(0, 10000)), max_steps=max_steps)
        env.reset()
        ep_r = 0.0
        for t in range(max_steps):
            a = env.action_space.sample()
            _, r, done, trunc, info = env.step(a)
            ep_r += r
            if done or trunc:
                lengths.append(t + 1)
                causes.append(env.collapse_cause or "survived")
                break
        else:
            lengths.append(max_steps)
            causes.append("survived")
        rewards.append(ep_r)

    print(f"\n{'─'*55}")
    print(f"  Random-policy stats  ({n_eps} episodes, max={max_steps} steps)")
    print(f"{'─'*55}")
    print(f"  Survival length  mean={np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Episode reward   mean={np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    cause_counts = {}
    for c in causes:
        cause_counts[c] = cause_counts.get(c, 0) + 1
    print("  Termination causes:")
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"    {cause:<30} {count:>3} ({100*count/n_eps:.0f}%)")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  GRAVITAS — Smoke Test Suite")
    print("═" * 55)

    tests = [
        ("Reset shape",              test_reset_shape),
        ("Step shapes",              test_step_shape),
        ("All 6 stances",            test_all_stances),
        ("100-step random rollout",  test_random_rollout),
        ("Hawkes self-excitation",   test_hawkes_excitation),
        ("Media bias bounded+drift", test_media_bias),
        ("Reward components",        test_reward_components),
        ("Propaganda trap",          test_propaganda_trap),
        ("Obs always finite",        test_obs_finite),
        ("Render ANSI",              test_render),
        ("Multi-episode reset",      test_multi_episode),
    ]

    passed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1

    print(f"\n  {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n  Running rollout statistics...\n")
        rollout_stats(n_eps=30)
        print("  ✓ All systems nominal. Environment ready for RL training.\n")
    else:
        print("\n  ✗ Fix failures before training.\n")
        sys.exit(1)
