"""
Hierarchical state for Phase 3: districts and provinces.

DistrictState holds the 6 local variables (all in [0,1]). HierarchicalState
holds the full district array, pipeline buffers for admin lag, and topology
(adjacency matrix and province layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .topology import (
    build_adjacency_matrix,
    build_province_district_layout,
    district_to_province,
)


# District state: 6 variables, all [0, 1]
DISTRICT_STATE_NAMES = (
    "local_gdp",
    "local_unrest",
    "admin_capacity",
    "factional_dominance",
    "implementation_efficiency",
    "local_memory",
)
N_DISTRICT_VARS = 6


@dataclass(frozen=True)
class DistrictState:
    """Immutable state for one district. All fields in [0, 1]."""

    local_gdp: float
    local_unrest: float
    admin_capacity: float
    factional_dominance: float
    implementation_efficiency: float
    local_memory: float

    def __post_init__(self) -> None:
        for name in DISTRICT_STATE_NAMES:
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"DistrictState.{name} must be in [0, 1], got {val}")

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.local_gdp,
                self.local_unrest,
                self.admin_capacity,
                self.factional_dominance,
                self.implementation_efficiency,
                self.local_memory,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "DistrictState":
        if arr.size != N_DISTRICT_VARS:
            raise ValueError(
                f"DistrictState.from_array expects {N_DISTRICT_VARS} elements, got {arr.size}"
            )
        arr = np.clip(arr.flat[:N_DISTRICT_VARS], 0.0, 1.0)
        return cls(
            local_gdp=float(arr[0]),
            local_unrest=float(arr[1]),
            admin_capacity=float(arr[2]),
            factional_dominance=float(arr[3]),
            implementation_efficiency=float(arr[4]),
            local_memory=float(arr[5]),
        )

    def to_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in DISTRICT_STATE_NAMES}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DistrictState":
        return cls(**{k: data[k] for k in DISTRICT_STATE_NAMES})


def districts_to_array(districts: List[DistrictState]) -> NDArray[np.float64]:
    """Stack district states into (N_D, N_DISTRICT_VARS)."""
    if not districts:
        return np.zeros((0, N_DISTRICT_VARS), dtype=np.float64)
    return np.array([d.to_array() for d in districts], dtype=np.float64)


def array_to_districts(arr: NDArray[np.float64]) -> List[DistrictState]:
    """Unstack (N_D, N_DISTRICT_VARS) into list of DistrictState."""
    if arr.size == 0:
        return []
    arr = np.clip(arr, 0.0, 1.0)
    return [DistrictState.from_array(arr[i]) for i in range(len(arr))]


@dataclass(frozen=True)
class HierarchicalState:
    """Immutable hierarchical state: districts + pipeline buffers + topology.

    Attributes:
        district_states: List of DistrictState (length N_D).
        pipeline_buffers: (N_D, n_policy_dims) float64 — admin lag buffers.
        adjacency: (N_D, N_D) symmetric, zero diagonal.
        counts_per_province: List of district counts per province.
        province_of_district: (N_D,) int — province index per district.
    """

    district_states: Tuple[DistrictState, ...] = field(default_factory=tuple)
    pipeline_buffers: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    adjacency: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    counts_per_province: Tuple[int, ...] = field(default_factory=tuple)
    province_of_district: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    def __post_init__(self) -> None:
        n_d = len(self.district_states)
        if n_d == 0:
            return
        if self.adjacency.shape != (n_d, n_d):
            raise ValueError(
                f"adjacency shape must be ({n_d}, {n_d}), got {self.adjacency.shape}"
            )
        if self.province_of_district.shape != (n_d,):
            raise ValueError(
                f"province_of_district length must be {n_d}, got {len(self.province_of_district)}"
            )
        if self.pipeline_buffers.shape[0] != n_d:
            raise ValueError(
                f"pipeline_buffers first dim must be {n_d}, got {self.pipeline_buffers.shape[0]}"
            )

    @property
    def n_districts(self) -> int:
        return len(self.district_states)

    @property
    def n_provinces(self) -> int:
        return len(self.counts_per_province)

    def get_district_array(self) -> NDArray[np.float64]:
        """(N_D, N_DISTRICT_VARS)."""
        return districts_to_array(list(self.district_states))

    def copy_with_districts(
        self,
        district_states: Tuple[DistrictState, ...] | List[DistrictState],
    ) -> "HierarchicalState":
        return HierarchicalState(
            district_states=tuple(district_states),
            pipeline_buffers=self.pipeline_buffers,
            adjacency=self.adjacency,
            counts_per_province=self.counts_per_province,
            province_of_district=self.province_of_district,
        )

    def copy_with_pipeline_buffers(self, buffers: NDArray[np.float64]) -> "HierarchicalState":
        return HierarchicalState(
            district_states=self.district_states,
            pipeline_buffers=buffers.copy(),
            adjacency=self.adjacency,
            counts_per_province=self.counts_per_province,
            province_of_district=self.province_of_district,
        )


def create_hierarchical_state(
    n_provinces: int,
    districts_per_province: int | List[int],
    n_policy_dims: int,
    connect_between_provinces: bool = False,
    between_weight: float = 0.0,
    initial_unrest: float = 0.05,
    initial_admin_capacity: float = 0.7,
) -> HierarchicalState:
    """Create initial hierarchical state with neutral districts and topology."""
    counts, n_d = build_province_district_layout(n_provinces, districts_per_province)
    province_of = district_to_province(counts)
    A = build_adjacency_matrix(
        counts,
        connect_within_province=True,
        connect_between_provinces=connect_between_provinces,
        between_weight=between_weight,
    )
    districts = [
        DistrictState(
            local_gdp=0.5,
            local_unrest=initial_unrest,
            admin_capacity=initial_admin_capacity,
            factional_dominance=0.1,
            implementation_efficiency=0.7,
            local_memory=0.1,
        )
        for _ in range(n_d)
    ]
    pipeline = np.zeros((n_d, n_policy_dims), dtype=np.float64)
    return HierarchicalState(
        district_states=tuple(districts),
        pipeline_buffers=pipeline,
        adjacency=A,
        counts_per_province=tuple(counts),
        province_of_district=province_of,
    )
