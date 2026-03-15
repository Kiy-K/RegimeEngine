"""
Governance System — Budget Distribution + Corruption for GRAVITAS Engine.

Forces strategic tradeoffs in resource allocation. Unexploitable by design.
"""

from .budget_system import (
    BudgetCategory, FactionBudget, CorruptionState, GovernanceWorld,
    BureaucracyState, PendingOrder, ACTION_DELAYS,
    initialize_governance, step_governance, apply_budget_action,
    queue_action, governance_summary,
)

__all__ = [
    "BudgetCategory", "FactionBudget", "CorruptionState", "GovernanceWorld",
    "BureaucracyState", "PendingOrder", "ACTION_DELAYS",
    "initialize_governance", "step_governance", "apply_budget_action",
    "queue_action", "governance_summary",
]
