from typing import List, Dict
import numpy as np

def audit_collapse_risk(trajectories: List[Dict], threshold: float = 0.9) -> float:
    """Compute fraction of trajectories that collapsed (hazard > threshold)."""
    collapses = 0
    for traj in trajectories:
        if "hazard" in traj:
            if any(h > threshold for h in traj["hazard"][-3:]):  # 3-step window
                collapses += 1
    return collapses / len(trajectories) if trajectories else 0.0

def audit_polarization(trajectories: List[Dict]) -> Dict[str, float]:
    """Return mean/max polarization across trajectories."""
    polarizations = []
    for traj in trajectories:
        if "polarization" in traj:
            polarizations.append(max(traj["polarization"]))
    if not polarizations:
        return {"mean": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(polarizations)),
        "max": float(np.max(polarizations)),
    }