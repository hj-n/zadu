"""Typed resource contracts and the exact resource cache."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from zadu.backends.base import ExactResourceProvider
    from zadu.engine.planner import ExecutionPlan, MetricPlan


class Space(str, Enum):
    ORIGINAL = "orig"
    EMBEDDED = "emb"
    PAIRED = "pair"


class ResourceKind(str, Enum):
    DISTANCE_MATRIX = "distance_matrix"
    CONDENSED_PAIRS = "condensed_pairs"
    KNN = "knn"
    NEIGHBOR_RANKING = "neighbor_ranking"
    PAIR_STATISTICS = "pair_statistics"


class PairStrategy(str, Enum):
    DENSE = "dense"
    CONDENSED = "condensed"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """One metric argument backed by one resource from each listed space."""

    argument: str
    kind: ResourceKind
    spaces: tuple[Space, ...]
    uses_k: bool = False


DISTANCE_MATRICES = ResourceRequirement(
    "distance_matrices",
    ResourceKind.DISTANCE_MATRIX,
    (Space.ORIGINAL, Space.EMBEDDED),
)
KNN_INFO = ResourceRequirement(
    "knn_info",
    ResourceKind.KNN,
    (Space.ORIGINAL, Space.EMBEDDED),
    uses_k=True,
)
KNN_RANKING_INFO = ResourceRequirement(
    "knn_ranking_info",
    ResourceKind.NEIGHBOR_RANKING,
    (Space.ORIGINAL, Space.EMBEDDED),
    uses_k=True,
)
KNN_EMB_INFO = ResourceRequirement(
    "knn_emb_info",
    ResourceKind.KNN,
    (Space.EMBEDDED,),
    uses_k=True,
)
PAIR_STATISTICS = ResourceRequirement(
    "pair_statistics",
    ResourceKind.PAIR_STATISTICS,
    (Space.PAIRED,),
)


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    kind: ResourceKind
    space: Space
    k: int | None = None


@dataclass(frozen=True, slots=True)
class ResourceKey:
    """Canonical resource allocated by an execution plan."""

    kind: ResourceKind
    space: Space
    k: int | None = None


@dataclass(frozen=True, slots=True)
class NeighborRanking:
    indices: npt.NDArray
    ranking: npt.NDArray


@dataclass(frozen=True, slots=True)
class PairStatistics:
    """Stable sufficient statistics over every unique off-diagonal pair."""

    count: int
    mean_orig: float
    mean_emb: float
    m2_orig: float
    m2_emb: float
    co_moment: float
    sum_orig_squared: float
    sum_emb_squared: float
    sum_product: float
    sum_squared_difference: float
    min_orig: float
    max_orig: float
    min_emb: float
    max_emb: float
    strategy: PairStrategy
    block_count: int
    block_rows: int | None
    chunk_pairs: int | None


@dataclass(slots=True)
class ResourceRecord:
    key: ResourceKey
    provider: str
    dtype: str | dict[str, str] | None
    build_seconds: float
    bytes: int
    generation: int
    details: dict[str, Any]
    released: bool = False


def resource_nbytes(value: Any) -> int:
    if isinstance(value, NeighborRanking):
        return int(value.indices.nbytes + value.ranking.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    return 0


def resource_dtype(value: Any) -> str | dict[str, str] | None:
    if isinstance(value, NeighborRanking):
        return {
            "indices": value.indices.dtype.name,
            "ranking": value.ranking.dtype.name,
        }
    if isinstance(value, np.ndarray):
        return value.dtype.name
    if isinstance(value, PairStatistics):
        return "float64"
    return None


class ResourceCache:
    """Build canonical resources once and provide sliced metric arguments."""

    def __init__(
        self,
        plan: ExecutionPlan,
        provider: ExactResourceProvider,
        *,
        geodesic: bool,
    ) -> None:
        self.plan = plan
        self.provider = provider
        self.geodesic = geodesic
        self._values: dict[ResourceKey, Any] = {}
        self._records: dict[ResourceKey, ResourceRecord] = {}
        self.generation = 0

    @property
    def records(self) -> dict[ResourceKey, ResourceRecord]:
        return dict(self._records)

    def prepare_original(self, points: npt.NDArray) -> None:
        self._prepare(Space.ORIGINAL, points)

    def begin_run(self) -> None:
        self.generation += 1
        self._invalidate(Space.EMBEDDED)
        self._invalidate(Space.PAIRED)

    def prepare_embedded(self, points: npt.NDArray) -> None:
        self._prepare(Space.EMBEDDED, points)

    def prepare_paired(
        self,
        orig: npt.NDArray,
        emb: npt.NDArray,
    ) -> None:
        pair_plan = self.plan.pair_plan
        if pair_plan is None:
            return
        key = pair_plan.statistics_key
        start = perf_counter()
        built = self.provider.build_pair_statistics(
            pair_plan,
            orig,
            emb,
            orig_distance_matrix=self.distance_matrix(Space.ORIGINAL),
            emb_distance_matrix=self.distance_matrix(Space.EMBEDDED),
            orig_condensed=self.condensed_pairs(Space.ORIGINAL),
            emb_condensed=self.condensed_pairs(Space.EMBEDDED),
            geodesic=self.geodesic,
        )
        self._store(key, built, perf_counter() - start)
        if pair_plan.strategy is PairStrategy.CONDENSED:
            self._release(pair_plan.embedded_source_key)

    def _invalidate(self, space: Space) -> None:
        for key in tuple(self._records):
            if key.space is space:
                self._values.pop(key, None)
                self._records.pop(key)

    def _prepare(self, space: Space, points: npt.NDArray) -> None:
        for key in self.plan.resources_for(space):
            if key in self._values:
                continue
            distance_matrix = self.distance_matrix(space)
            start = perf_counter()
            built = self.provider.build(
                key,
                points,
                distance_matrix=distance_matrix,
                geodesic=self.geodesic and space is Space.ORIGINAL,
            )
            self._store(key, built, perf_counter() - start)

    def _store(self, key, built, elapsed: float) -> None:
        self._values[key] = built.value
        self._records[key] = ResourceRecord(
            key=key,
            provider=built.implementation,
            dtype=resource_dtype(built.value),
            build_seconds=elapsed,
            bytes=resource_nbytes(built.value),
            generation=self.generation,
            details=dict(built.details),
        )

    def _release(self, key: ResourceKey | None) -> None:
        if key is None or key not in self._records:
            return
        self._values.pop(key, None)
        self._records[key].released = True

    def release_after(self, metric_index: int) -> None:
        """Release ephemeral resources after their final metric consumer."""

        for key, consumers in self.plan.consumers.items():
            if not consumers or consumers[-1] != metric_index:
                continue
            if key.kind is ResourceKind.PAIR_STATISTICS:
                self._release(key)

    def get(self, request: ResourceRequest) -> Any:
        key = self.plan.resolve(request)
        value = self._values[key]
        if request.kind in {
            ResourceKind.DISTANCE_MATRIX,
            ResourceKind.CONDENSED_PAIRS,
            ResourceKind.PAIR_STATISTICS,
        }:
            return value
        if request.kind is ResourceKind.KNN:
            indices = value.indices if isinstance(value, NeighborRanking) else value
            return indices[:, : request.k]
        if not isinstance(value, NeighborRanking):
            raise RuntimeError("A neighbor-ranking request resolved to invalid data")
        return NeighborRanking(value.indices[:, : request.k], value.ranking)

    def arguments_for(self, metric_plan: MetricPlan) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for binding in metric_plan.bindings:
            values = [self.get(request) for request in binding.requests]
            if binding.requirement.kind is ResourceKind.NEIGHBOR_RANKING:
                flattened = []
                for value in values:
                    flattened.extend((value.indices, value.ranking))
                arguments[binding.requirement.argument] = tuple(flattened)
            elif len(values) == 1:
                arguments[binding.requirement.argument] = values[0]
            else:
                arguments[binding.requirement.argument] = tuple(values)
        return arguments

    def distance_matrix(self, space: Space) -> npt.NDArray | None:
        for key, value in self._values.items():
            if key.space is space and key.kind is ResourceKind.DISTANCE_MATRIX:
                return value
        return None

    def condensed_pairs(self, space: Space) -> npt.NDArray | None:
        for key, value in self._values.items():
            if key.space is space and key.kind is ResourceKind.CONDENSED_PAIRS:
                return value
        return None

    def neighbor_indices(self, space: Space) -> npt.NDArray | None:
        for key, value in self._values.items():
            if key.space is not space:
                continue
            if key.kind is ResourceKind.NEIGHBOR_RANKING:
                return value.indices
            if key.kind is ResourceKind.KNN:
                return value
        return None

    def ranking(self, space: Space) -> npt.NDArray | None:
        for key, value in self._values.items():
            if key.space is space and key.kind is ResourceKind.NEIGHBOR_RANKING:
                return value.ranking
        return None
