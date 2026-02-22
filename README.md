# RegimeEngine

High-fidelity simulation and optimization of hierarchical political systems and systemic survival under spatial instability.

## Overview

RegimeEngine is a simulation framework designed to model the dynamics of political regimes, focusing on systemic survival rather than fairness. It incorporates hierarchical district and province modeling, allowing for complex spatial interactions and the study of how unrest and instability propagate through a system.

The engine uses Proximal Policy Optimization (PPO) reinforcement learning agents to learn stabilization strategies in environments characterized by randomized topologies and stochastic spatial shocks.

## Key Features

- Hierarchical Modeling: Districts nested within provinces, with custom adjacency and diffusion rates.
- Reinforcement Learning: Trained PPO agents for regime stabilization and hazard mitigation.
- Survival Metrics: Tracking of peak hazard, exhaustion growth, cluster intensity, and early warning indices.
- Spatial Dynamics: Modeling of domino effects and cascade failures across geographical units.
- Robust Evaluation: Automated evaluation harnesses for testing policies across unseen random topologies.

## Installation

```bash
git clone https://github.com/Kiy-K/RegimeEngine.git
cd RegimeEngine
pip install -r requirements.txt
```

## Usage

### Training

To train a new PPO policy:

```bash
python train_ppo.py
```

### Evaluation

To evaluate a trained policy on random topologies:

```bash
python evaluate_ppo_policy.py
```

### Auditing

To run large-scale stability audits:

```bash
python run_10k_audit.py
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
