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


class ResourceKind(str, Enum):
    DISTANCE_MATRIX = "distance_matrix"
    KNN = "knn"
    NEIGHBOR_RANKING = "neighbor_ranking"


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


@dataclass(slots=True)
class ResourceRecord:
    key: ResourceKey
    value: Any
    provider: str
    build_seconds: float
    bytes: int
    generation: int


def resource_nbytes(value: Any) -> int:
    if isinstance(value, NeighborRanking):
        return int(value.indices.nbytes + value.ranking.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    return 0


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

    def prepare_embedded(self, points: npt.NDArray) -> None:
        self._prepare(Space.EMBEDDED, points)

    def _invalidate(self, space: Space) -> None:
        for key in tuple(self._values):
            if key.space is space:
                self._values.pop(key)
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
            elapsed = perf_counter() - start
            self._values[key] = built.value
            self._records[key] = ResourceRecord(
                key=key,
                value=built.value,
                provider=built.implementation,
                build_seconds=elapsed,
                bytes=resource_nbytes(built.value),
                generation=self.generation,
            )

    def get(self, request: ResourceRequest) -> Any:
        key = self.plan.resolve(request)
        value = self._values[key]
        if request.kind is ResourceKind.DISTANCE_MATRIX:
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
