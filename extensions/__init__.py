"""
Gravitas Extensions Package.

Extensions to the core Gravitas simulation engine, including:
- Population dynamics (pop)
- Military systems (military)
- Fog of war (fog_of_war)
- Exhaustion monitoring (exhaustion)
- Audits (audits)
- Topology visualization (topology)
"""

# Import all extension modules to make them available
from . import pop
from . import military
from . import fog_of_war
from . import exhaustion
from . import audits
from . import topology

__all__ = [
    "pop",
    "military",
    "fog_of_war",
    "exhaustion",
    "audits",
    "topology",
]