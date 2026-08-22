"""Narrow backend protocol for exact planned resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy.typing as npt

from zadu.engine.resources import PairOrder, ResourceKey

if TYPE_CHECKING:
    from zadu.engine.planner import PairExecutionPlan, TopographicExecutionPlan


@dataclass(frozen=True, slots=True)
class BuiltResource:
    value: Any
    implementation: str
    details: dict[str, Any] = field(default_factory=dict)


class ExactResourceProvider(Protocol):
    name: str
    device: str
    exact: bool

    def build(
        self,
        key: ResourceKey,
        points: npt.NDArray,
        *,
        distance_matrix: npt.NDArray | None,
        condensed_pairs: npt.NDArray | None,
        working_memory_bytes: int | None,
        geodesic: bool,
    ) -> BuiltResource: ...

    def build_pair_statistics(
        self,
        plan: PairExecutionPlan,
        orig: npt.NDArray,
        emb: npt.NDArray,
        *,
        orig_distance_matrix: npt.NDArray | None,
        emb_distance_matrix: npt.NDArray | None,
        orig_condensed: npt.NDArray | None,
        emb_condensed: npt.NDArray | None,
        geodesic: bool,
    ) -> BuiltResource: ...

    def build_ordered_pair_statistics(
        self,
        plan: PairExecutionPlan,
        pair_order: PairOrder,
        *,
        emb_distance_matrix: npt.NDArray | None,
        emb_condensed: npt.NDArray | None,
    ) -> BuiltResource: ...

    def build_topographic_product_statistics(
        self,
        plan: TopographicExecutionPlan,
        orig: npt.NDArray,
        emb: npt.NDArray,
        *,
        orig_knn: npt.NDArray,
        emb_knn: npt.NDArray,
    ) -> BuiltResource: ...
