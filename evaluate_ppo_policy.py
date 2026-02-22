"""
Evaluation Mode for trained PPO policy.
Runs a trained agent on unseen random topologies without updating parameters.
"""

import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from regime_engine.agents.survival_env import SurvivalRegimeEnv, SurvivalConfig

def run_evaluation(model_path: str, n_episodes: int = 50, output_file: str = "evaluation_results.json"):
    print(f"Loading trained policy from {model_path}...")
    
    # Load the trained PPO policy
    model = PPO.load(model_path)
    
    # Set up environment
    config = SurvivalConfig(max_steps=500)
    env = SurvivalRegimeEnv(config=config)
    
    # Metrics containers
    raw_metrics = []
    
    print(f"Starting evaluation of {n_episodes} episodes on random topologies...")
    
    # Evaluation loop
    with torch.no_grad(): # Disable gradient computation
        for ep in range(n_episodes):
            # Sample a new random topology (internal to reset if using SurvivalRegimeEnv correctly)
            # Actually resetting with a different seed each time as the environment handles randomization
            obs, info = env.reset(seed=ep + 1000) # Use a specific seed range for evaluation
            
            terminated = False
            truncated = False
            steps = 0
            peak_hazard = 0.0
            clustering_indices = []
            
            while not (terminated or truncated):
                # Deterministic action selection
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(int(action))
                
                steps += 1
                peak_hazard = max(peak_hazard, info.get("peak_hazard", 0.0))
                clustering_indices.append(info.get("clustering_index", 0.0))
            
            # Record episode metrics
            episode_data = {
                "episode": ep + 1,
                "survival_time": steps,
                "peak_hazard": float(peak_hazard),
                "final_exhaustion": float(info.get("exhaustion", 0.0)),
                "mean_cluster_intensity": float(np.mean(clustering_indices)) if clustering_indices else 0.0,
                "collapse_flag": bool(terminated) # True if terminated early (hazard > 1.2, etc.)
            }
            raw_metrics.append(episode_data)
            
            if (ep + 1) % 10 == 0:
                print(f"  Completed {ep + 1}/{n_episodes} episodes...")
    
    # Compute summary statistics
    survival_times = [m["survival_time"] for m in raw_metrics]
    peak_hazards = [m["peak_hazard"] for m in raw_metrics]
    final_exhaustions = [m["final_exhaustion"] for m in raw_metrics]
    collapse_flags = [m["collapse_flag"] for m in raw_metrics]
    
    summary = {
        "mean_survival": float(np.mean(survival_times)),
        "std_survival": float(np.std(survival_times)),
        "mean_peak_hazard": float(np.mean(peak_hazards)),
        "mean_final_exhaustion": float(np.mean(final_exhaustions)),
        "collapse_rate": float(np.mean(collapse_flags))
    }
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Mean Survival Time:  {summary['mean_survival']:.2f} ± {summary['std_survival']:.2f}")
    print(f"Mean Peak Hazard:    {summary['mean_peak_hazard']:.4f}")
    print(f"Mean Final Exh:      {summary['mean_final_exhaustion']:.4f}")
    print(f"Collapse Rate:       {summary['collapse_rate']*100:.1f}%")
    print("="*50)
    
    # Save raw metrics to JSON
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "episodes": raw_metrics}, f, indent=4)
    print(f"\nRaw metrics saved to {output_file}")

if __name__ == "__main__":
    final_model_path = "./logs/ppo_survival/ppo_survival_final.zip"
    if os.path.exists(final_model_path):
        run_evaluation(final_model_path, n_episodes=50)
    else:
        print(f"Error: Final model not found at {final_model_path}")
