"""
GRAVITAS — Governance Under Recursive And Volatile Instability Through Adaptive Simulation
Research-grade RL environment package.
"""

from setuptools import setup, find_packages

setup(
    name="gravitas-env",
    version="2.0.0",
    description="Research-grade governance RL environment with partial observability, "
                "media bias, hierarchical actions, and non-linear dynamics.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "gymnasium>=0.29",
        "scipy>=1.10",
    ],
    extras_require={
        "train": [
            "stable-baselines3>=2.2",
            "torch>=2.0",
            "tqdm",
            "rich",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "matplotlib",
            "pandas",
        ],
    },
)
