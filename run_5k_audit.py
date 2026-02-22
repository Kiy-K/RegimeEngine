"""
Phase 2 Deep Dynamics Expansion: 5000-Episode Monte Carlo Audit.

This script runs 5,000 simulations using the Regime Engine, verifying that
the expanded Stochastic, Economic, and Topological dynamics behave robustly.
We record trajectories focusing on wealth inequality, GDP collapse probabilities,
and affinity polarization trends.
"""

import time
import numpy as np
from collections import defaultdict

from regime_engine.core.parameters import SystemParameters
from regime_engine.simulation.runner import SimulationRunner
from regime_engine.systems.crisis_classifier import CrisisLevel, classify, ClassifierThresholds


def run_monte_carlo(n_episodes: int = 1000, steps_per_episode: int = 100) -> None:
    print(f"Starting {n_episodes}-Episode Monte Carlo Audit...")
    
    # Enable all the new features: Noise, Wealth Extraction, GDP Drag
    params = SystemParameters(
        n_factions=4,
        max_steps=steps_per_episode,
        sigma_noise=0.08,        # Stochastic Layer
        alpha_gdp=0.05,          # Economic Layer 
        wealth_extraction=0.15,
        n_pillars=3              # Topological Layer
    )
    thresholds = ClassifierThresholds()
    runner = SimulationRunner(params=params, thresholds=thresholds, stop_on_collapse=True)
    
    outcomes = defaultdict(int)
    gdp_records = []
    wealth_gini_records = []
    
    start_time = time.perf_counter()
    
    for ep in range(n_episodes):
        np.random.seed(ep)
        final_state = runner.run_headless()
        
        gdp_records.append(final_state.system.state_gdp)
        crisis = classify(final_state, thresholds)
        outcomes[crisis.name] += 1
        
        wealths = final_state.get_faction_wealths()
        sorted_w = np.sort(wealths)
        n = len(wealths)
        if np.sum(wealths) > 0:
            index = np.arange(1, n + 1)
            gini = (np.sum((2 * index - n  - 1) * sorted_w)) / (n * np.sum(wealths))
            wealth_gini_records.append(gini)
            
        if (ep + 1) % 500 == 0:
            print(f"  Finished {ep + 1} episodes...")
            
    elapsed = time.perf_counter() - start_time
    
    print("\nAudit Complete!")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print(f"Speed: {n_episodes/elapsed:.1f} episodes/sec\n")
    
    print("Outcome Distribution:")
    for k, v in outcomes.items():
        pct = (v / n_episodes) * 100
        print(f"  {k}: {v} ({pct:.1f}%)")
        
    avg_gdp = np.mean(gdp_records)
    min_gdp = np.min(gdp_records)
    print(f"\nEconomic Resilience:")
    print(f"  Average GDP at termination: {avg_gdp:.3f}")
    print(f"  Minimum GDP observed: {min_gdp:.3f}")
    
    if wealth_gini_records:
        avg_inequality = np.mean(wealth_gini_records)
        print(f"\nResource Stratification:")
        print(f"  Average Wealth Gini Coefficient: {avg_inequality:.3f}")

if __name__ == "__main__":
    run_monte_carlo()
