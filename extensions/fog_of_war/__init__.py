"""
Fog of War Extension for GravitasEngine

This extension adds partial observability to the GravitasEngine simulation,
limiting the agent's knowledge of the world state based on various factors.
"""

from .wrapper import FogOfWarWrapper

__all__ = [
    "FogOfWarWrapper",
]
