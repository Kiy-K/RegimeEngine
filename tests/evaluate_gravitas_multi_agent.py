"""
Multi-Agent evaluation harness for GRAVITAS.

Simulates decentralized governance where multiple agents (sharing the same 
base PPO policy) interact with the environment. Each agent has a slightly 
different 'perception' of the state due to injected observation noise, 
leading to divergent actions.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import PPO

from gravitas_engine.agents.gravitas_env import GravitasEnv
from gravitas_engine.core.gravitas_params import GravitasParams as GravitasConfig
from gravitas_engine.agents.gravitas_actions import apply_action, HierarchicalAction

def run_multi_agent_evaluation(
    model_path: str,
    n_agents: int = 3,
    n_episodes: int = 20,
    perception_noise: float = 0.05,
    output_file: str = "multi_agent_results.json"
) -> None:
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # Use standard audit-level difficulty
    config = GravitasConfig(hawkes_base_rate=0.02, max_steps=500)
    env = GravitasEnv(params=config)
    
    survival_times = []
    collapse_causes = []
    total_resource_costs = []

    print(f"Running {n_episodes} multi-agent episodes (N_AGENTS={n_agents})...")
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep + 30000)
            done = False
            truncated = False
            steps = 0
            ep_cost = 0.0
            
            while not (done or truncated):
                # 1. Each agent generates an action based on its noisy perception
                for agent_idx in range(n_agents):
                    # Inject perception noise to ensure agents don't take identical actions
                    # We only perturb the cluster-specific observations (first 3*N elements)
                    # and the bias estimates.
                    noisy_obs = obs.copy()
                    noise = np.random.normal(0, perception_noise, size=obs.shape).astype(np.float32)
                    noisy_obs += noise
                    
                    # Predict action
                    action_int, _ = model.predict(noisy_obs, deterministic=True)
                    
                    # Decode to hierarchical action using internal env logic
                    # We use the env's current world state for decoding heuristics
                    hier_action = env._decode_action(action_int)
                    
                    # Apply action to the world state IN THE ENV
                    # This simulates sequential application of agent impulses in one tick
                    new_world, cost = apply_action(
                        env._world, 
                        hier_action, 
                        env.params, 
                        env._rng
                    )
                    env._world = new_world
                    ep_cost += cost
                
                # 2. Advance the physics of the world by one step
                # We've already applied all agent actions, so we just run the physics part 
                # (RK4, shocks, bias update) manually or by calling a gated step.
                # Here we use the env's internal logic but skip the duplicated action application.
                
                # RK4 integration
                env._world = env._rk4_step_physics(env._world)
                
                # Shocks
                from gravitas_engine.systems.hawkes_shock import apply_shock, sample_shock, update_hawkes
                shock = sample_shock(env._world, env.params, env._rng)
                shock_info = {}
                if shock is not None:
                    env._world, shock_info = apply_shock(env._world, shock, env.params, env._rng)
                
                new_hawkes_rate, new_hawkes_sum = update_hawkes(
                    env._world.hawkes_sum,
                    shock_occurred=(shock is not None),
                    params=env.params,
                )
                env._world = env._world.copy_with_shock(new_hawkes_rate, new_hawkes_sum)
                
                # Media Bias
                from gravitas_engine.systems.media_bias import update_media_bias
                # Use the last agent's action for bias update (simplification)
                new_bias = update_media_bias(
                    world=env._world,
                    propaganda_weights=hier_action.weights[:env._cur_N],
                    propaganda_intensity=(
                        hier_action.propaganda_load if hier_action.stance == 3 else 0.0
                    ),
                    shock_occurred=(shock is not None),
                    shock_cluster=(shock.cluster_idx if shock is not None else -1),
                    params=env.params,
                    rng=env._rng,
                )
                env._world = env._world.copy_with_bias(new_bias)
                
                # Advance step
                env._world = env._world.advance_step()
                
                # Check termination
                done, cause = env._check_termination(env._world)
                truncated = (env._world.global_state.step >= env.params.max_steps)
                
                # Update obs for next tick
                obs = env._make_observation()
                steps += 1

            survival_times.append(steps)
            total_resource_costs.append(ep_cost)
            if done:
                collapse_causes.append(cause)
            
            if (ep + 1) % 5 == 0:
                print(f"  ep {ep+1}/{n_episodes} | avg_survival={np.mean(survival_times):.1f}")

    # Summary
    print("\n" + "="*50)
    print("MULTI-AGENT EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Survival: {np.mean(survival_times):.1f} steps")
    print(f"Average Resource Cost: {np.mean(total_resource_costs):.3f}")
    if collapse_causes:
        print("Collapse Distribution:")
        from collections import Counter
        counts = Counter(collapse_causes)
        for c, count in counts.items():
            print(f"  {c}: {count} ({count/n_episodes*100:.1f}%)")
    else:
        print("No collapses observed.")
    print("="*50)

# We need to add a helper to GravitasEnv to expose physics-only step if not present
# After checking the file, it doesn't have a clean physics-only step. I'll need to 
# add it or refactor the evaluation script to handle it.
# Actually, I'll just refactor the evaluation script to be more self-contained 
# or I can add a method to GravitasEnv.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logs/ppo_gravitas/best_model.zip")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()
    
    # Monkey-patch GravitasEnv to add a physics-only step
    from gravitas_engine.core.gravitas_dynamics import rk4_step
    def _rk4_step_physics(self, world):
        return rk4_step(
            world=world,
            params=self.params,
            military_load=self._prev_action.military_load,
            propaganda_load=self._prev_action.propaganda_load,
            sigma_noise=self.params.sigma_obs_base,
            rng=self._rng,
        )
    GravitasEnv._rk4_step_physics = _rk4_step_physics

    run_multi_agent_evaluation(
        model_path=args.model,
        n_agents=args.agents,
        n_episodes=args.episodes,
        perception_noise=args.noise
    )
