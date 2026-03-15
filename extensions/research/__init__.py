"""
Research & Technology System — HOI4-style tech trees for GRAVITAS Engine.

Each faction researches technologies across 6 branches that improve
production, military effectiveness, naval/air capabilities, and unlock
new unit types. Research takes time and resources.
"""

from .research_system import (
    TechBranch, TechProject, FactionResearch, ResearchWorld,
    initialize_research, step_research, apply_research_action,
    research_summary, TECH_TREE,
)

__all__ = [
    "TechBranch", "TechProject", "FactionResearch", "ResearchWorld",
    "initialize_research", "step_research", "apply_research_action",
    "research_summary", "TECH_TREE",
]
