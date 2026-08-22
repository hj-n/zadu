"""Narrow backend protocol for exact planned resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy.typing as npt

from zadu.engine.resources import ResourceKey

if TYPE_CHECKING:
    from zadu.engine.planner import PairExecutionPlan


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
