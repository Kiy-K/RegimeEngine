"""
Exhaustion Monitoring Extension for GravitasEngine

This extension tracks and monitors exhaustion levels in the simulation,
providing warnings and metrics related to system fatigue.
"""

from .monitor import ExhaustionMonitor

__all__ = [
    "ExhaustionMonitor",
]
