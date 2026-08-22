"""Narrow backend protocol for exact planned resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy.typing as npt

from zadu.engine.resources import NeighborRanking, PairOrder, ResourceKey, Space

if TYPE_CHECKING:
    from zadu.engine.planner import (
        NeighborStatisticsExecutionPlan,
        PairExecutionPlan,
        RankComparisonExecutionPlan,
        TopographicExecutionPlan,
    )


@dataclass(frozen=True, slots=True)
class BuiltResource:
    value: Any
    implementation: str
    details: dict[str, Any] = field(default_factory=dict)


class BatchResourceError(RuntimeError):
    """A provider-native batch failed for one input in that batch."""

    def __init__(self, batch_index: int, message: str) -> None:
        self.batch_index = batch_index
        super().__init__(message)


class ExactResourceProvider(Protocol):
    name: str
    device: str
    dtype: str
    exact: bool
    supports_embedding_batching: bool

    def fork(self) -> ExactResourceProvider:
        """Return an isolated provider context for one concurrent embedding."""

        ...

    def invalidate(self, space: Space) -> None:
        """Release provider-private state for one invalidated resource space."""

        ...

    def can_batch(self, key: ResourceKey) -> bool:
        """Return whether one resource can use provider-native batching."""

        ...

    def build_batch(
        self,
        key: ResourceKey,
        points_batch: list[npt.NDArray],
        *,
        distance_matrices: list[npt.NDArray | None],
        condensed_pairs: list[npt.NDArray | None],
        working_memory_bytes: int | None,
        geodesic: bool,
    ) -> list[BuiltResource]:
        """Build one resource for each embedding, optionally as one device batch."""

        ...

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

    def build_rank_comparisons(
        self,
        plan: RankComparisonExecutionPlan,
        *,
        orig_ranking: NeighborRanking,
        emb_ranking: NeighborRanking,
    ) -> BuiltResource: ...

    def build_neighbor_statistics(
        self,
        plan: NeighborStatisticsExecutionPlan,
        *,
        orig_knn: npt.NDArray,
        emb_knn: npt.NDArray,
    ) -> BuiltResource: ...
