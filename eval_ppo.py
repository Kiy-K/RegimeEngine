"""
Phase 3 Multi-Agent RL Evaluation Harness

Evaluates a trained Proximal Policy Optimization (PPO) agent against the
RegimeEnv baseline, recording average survival lengths and GDP.
"""

import os
import numpy as np
import time

from stable_baselines3 import PPO
from regime_engine.agents.rl_env import RegimeEnv
from regime_engine.core.parameters import SystemParameters

def evaluate_model(model_path: str, n_episodes: int = 100) -> None:
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}!")
        return
        
    print(f"Loading Model from {model_path}...")
    model = PPO.load(model_path)
    
    params = SystemParameters(max_steps=400)
    env = RegimeEnv(params=params, intensity=0.05, agent_id="ppo_evaluator")
    
    survival_lengths = []
    final_gdps = []
    
    print(f"Running {n_episodes} evaluation episodes...")
    start_time = time.time()
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=42000 + ep)
        done = False
        steps = 0
        
        while not done:
            # Deterministic actions for stable evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            
        survival_lengths.append(steps)
        # Using the engine's current system state from info dictionary if we had mapped it
        final_gdps.append(info.get("state_gdp", 0.0))
        
        if (ep + 1) % 25 == 0:
            print(f"  Finished {ep + 1}/{n_episodes} episodes...")
            
    elapsed = time.time() - start_time
    
    avg_survival = np.mean(survival_lengths)
    avg_gdp = np.mean(final_gdps)
    
    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE ({elapsed:.2f}s)")
    print("="*50)
    print(f"Average Survival Length: {avg_survival:.1f} steps")
    print(f"Average Final State GDP: {avg_gdp:.3f}")

if __name__ == "__main__":
    # Check for best_model first, otherwise fallback to final
    best_model_path = "./logs/ppo_regime_training/best_model.zip"
    if not os.path.exists(best_model_path):
        best_model_path = "./logs/ppo_regime_training/ppo_regime_final.zip"
        
    evaluate_model(best_model_path, n_episodes=100)
