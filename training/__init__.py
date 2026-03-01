"""
Gravitas Training Package.

Training infrastructure for Gravitas, including:
- Regime configurations
- Policy implementations
"""

# Import all training modules to make them available
from . import regimes
from . import policies

__all__ = [
    "regimes",
    "policies",
]