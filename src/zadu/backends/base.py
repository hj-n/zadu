"""Narrow backend protocol for exact planned resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy.typing as npt

from zadu.engine.resources import ResourceKey


@dataclass(frozen=True, slots=True)
class BuiltResource:
    value: Any
    implementation: str


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
