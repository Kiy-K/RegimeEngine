"""
10,000 Episode Monte Carlo Audit.

Runs 5,000 baseline (unperturbed) episodes and 5,000 random-agent perturbed
episodes across multiple CPU cores to validate engine stability and outcome
distributions at scale.
"""

import time
import multiprocessing
import numpy as np
from collections import Counter
import os

from stable_baselines3 import PPO

from regime_engine.core.parameters import SystemParameters
from regime_engine.core.state import RegimeState
from regime_engine.core.factions import create_balanced_factions
from regime_engine.agents.rl_env import RegimeEnv
from regime_engine.systems.crisis_classifier import ClassifierThresholds

EPISODES_BASELINE = 5000
EPISODES_AGENT = 5000
MAX_STEPS = 500


global_model = None

def init_worker(model_path):
    """Initialize worker process by loading the model once."""
    global global_model
    if model_path and os.path.exists(model_path):
        global_model = PPO.load(model_path, device="cpu")

def run_episode(args):
    """Run a single episode."""
    env_seed, use_agent = args
    
    # Use the process-global model if available
    model = global_model
    
    # Randomise parameters slightly for robust auditing
    rng = np.random.default_rng(env_seed)
    params = SystemParameters(
        n_factions=3,
        seed=env_seed,
        max_steps=MAX_STEPS
    )
    
    env = RegimeEnv(params=params, intensity=0.05, agent_id="auditor")
    obs, info = env.reset(seed=env_seed)
    
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        if use_agent:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                # Random discrete action fallback
                action = int(rng.integers(0, env.action_space.n))
        else:
            # Provide an action but with 0.0 intensity essentially (null action equivalent)
            env.intensity = 0.0
            action = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    return {
        "steps": steps,
        "crisis_level": info["crisis_level"],
        "legitimacy": info["legitimacy"],
        "instability": info["instability"],
        "exhaustion": info["exhaustion"],
        "volatility": info["volatility"],
    }


def main():
    print(f"Starting 10k Episode Audit (Max Steps: {MAX_STEPS})...")
    start_time = time.time()
    
    # Prepare arguments
    # Check for best_model first, otherwise fallback to final
    best_model_path = "./logs/ppo_regime_training/best_model.zip"
    if not os.path.exists(best_model_path):
        best_model_path = "./logs/ppo_regime_training/ppo_regime_final.zip"
        
    # Enumerate unique seeds for independent runs
    args_baseline = [(i, False) for i in range(EPISODES_BASELINE)]
    args_agent = [(EPISODES_BASELINE + i, True) for i in range(EPISODES_AGENT)]
    
    print(f"Running {EPISODES_BASELINE} Unperturbed Baseline Episodes...")
    with multiprocessing.Pool() as pool:
        results_baseline = pool.map(run_episode, args_baseline)
        
    print(f"Running {EPISODES_AGENT} PPO-Agent Perturbed Episodes...")
    # For the agent phase, we initialize the pool with the model
    with multiprocessing.Pool(initializer=init_worker, initargs=(best_model_path,)) as pool:
        results_agent = pool.map(run_episode, args_agent)
        
    elapsed = time.time() - start_time
    total_eps = len(results_baseline) + len(results_agent)
    
    print("\n" + "="*50)
    print(f"AUDIT COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"Episodes: {total_eps}")
    print(f"Simulation Speed: {total_eps / elapsed:.2f} eps/sec")
    print("="*50 + "\n")
    
    def summarize(name, results):
        print(f"--- {name} ({len(results)} episodes) ---")
        crises = Counter(r["crisis_level"] for r in results)
        avg_steps = np.mean([r["steps"] for r in results])
        avg_leg = np.mean([r["legitimacy"] for r in results])
        avg_exh = np.mean([r["exhaustion"] for r in results])
        avg_vol = np.mean([r["volatility"] for r in results])
        
        print(f"Outcomes:")
        for k, v in crises.most_common():
            print(f"  {k:<10}: {v:>5} ({v/len(results)*100:.1f}%)")
        print(f"Average Steps Survived: {avg_steps:.1f} / {MAX_STEPS}")
        print(f"Average Final State:")
        print(f"  Legitimacy:  {avg_leg:.3f}")
        print(f"  Volatility:  {avg_vol:.3f}")
        print(f"  Exhaustion:  {avg_exh:.3f}\n")

    summarize("Baseline (Unperturbed)", results_baseline)
    summarize("PPO Agent (Perturbed)", results_agent)


if __name__ == "__main__":
    main()
