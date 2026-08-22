"""Gap Index for measuring distortion in empty regions of 2D projections.

The algorithm and original implementation were created by Jaume Ros,
Alessio Arleo, and Fernando Paulovich:

    Measuring Distortion in the Empty Regions of Dimensionality Reduction
    Scatterplots with the Gap Index. arXiv:2607.28324 (2026).

This module is adapted from the authors' MIT-licensed implementation at
https://codeberg.org/jros/gap-index, revision
0a11e4887864fe5d41526d8487eea33685b8f0b4. See
LICENSES/gap-index-MIT.txt and THIRD_PARTY_NOTICES.md for attribution.
"""

# Copyright (c) 2026 Jaume Ros, Alessio Arleo, Fernando Paulovich
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay, QhullError, distance

DistanceMetric = str | Callable[[npt.NDArray, npt.NDArray], float]
_DEFORMATION_EPSILON = 1e-6
_TRIANGLE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class GapIndexResult:
    """Detailed Gap Index result shared by scoring and regional analysis."""

    score: float
    triangles: npt.NDArray[np.integer]
    deformations: npt.NDArray[np.float64]
    original_relative_areas: npt.NDArray[np.float64]
    embedded_relative_areas: npt.NDArray[np.float64]


def _validate_inputs(
    orig: npt.NDArray, emb: npt.NDArray, metric: DistanceMetric
) -> tuple[npt.NDArray, npt.NDArray]:
    orig_array = np.asarray(orig)
    emb_array = np.asarray(emb, dtype=float)

    if orig_array.ndim != 2:
        raise ValueError(f"orig must be a 2D array, got shape {orig_array.shape}")
    if emb_array.ndim != 2 or emb_array.shape[1] != 2:
        raise ValueError(
            "Gap Index requires a 2D embedding with shape (n, 2), "
            f"got {emb_array.shape}"
        )
    if orig_array.shape[0] != emb_array.shape[0]:
        raise ValueError(
            "orig and emb must have the same number of rows "
            f"(orig={orig_array.shape[0]}, emb={emb_array.shape[0]})"
        )
    if emb_array.shape[0] < 3:
        raise ValueError("Gap Index requires at least 3 points")
    if not np.all(np.isfinite(emb_array)):
        raise ValueError("emb must contain only finite values")

    if metric == "precomputed":
        n = emb_array.shape[0]
        if orig_array.shape != (n, n):
            raise ValueError(
                "For metric='precomputed', orig must be a square (n, n) "
                f"distance matrix, got {orig_array.shape}"
            )
        orig_array = np.asarray(orig_array, dtype=float)
        if not np.all(np.isfinite(orig_array)):
            raise ValueError("The precomputed distance matrix must be finite")
        if np.any(orig_array < 0):
            raise ValueError("The precomputed distance matrix must be non-negative")
        if not np.allclose(orig_array, orig_array.T):
            raise ValueError("The precomputed distance matrix must be symmetric")
        if not np.allclose(np.diag(orig_array), 0):
            raise ValueError("The precomputed distance matrix diagonal must be zero")
    elif not np.all(np.isfinite(orig_array)):
        raise ValueError("orig must contain only finite values")

    return orig_array, emb_array


def _resolve_metric(metric: DistanceMetric) -> Callable:
    if callable(metric):
        return metric
    if not isinstance(metric, str):
        raise TypeError("metric must be a scipy distance function or a metric name")

    metric_fn = getattr(distance, metric, None)
    if not callable(metric_fn):
        raise ValueError(
            f"Unknown scipy distance metric '{metric}'. Pass a callable distance "
            "function or use 'precomputed'."
        )
    return metric_fn


def _triangle_area_from_sides(a: float, b: float, c: float) -> float:
    if not np.all(np.isfinite([a, b, c])) or min(a, b, c) < 0:
        raise ValueError("Triangle edge lengths must be finite and non-negative")

    semiperimeter = (a + b + c) / 2.0
    radicand = (
        semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c)
    )
    scale = max(a, b, c, 1.0) ** 4
    if radicand < -_TRIANGLE_TOLERANCE * scale:
        raise ValueError("Triangle edge lengths do not satisfy the triangle inequality")
    return math.sqrt(max(radicand, 0.0))


def _compute_areas(
    points: npt.NDArray,
    triangles: npt.NDArray,
    metric: DistanceMetric,
) -> npt.NDArray[np.float64]:
    if metric == "precomputed":
        areas = np.array(
            [
                _triangle_area_from_sides(points[a, b], points[a, c], points[b, c])
                for a, b, c in triangles
            ],
            dtype=float,
        )
    else:
        metric_fn = _resolve_metric(metric)
        areas = np.array(
            [
                _triangle_area_from_sides(
                    metric_fn(points[a], points[b]),
                    metric_fn(points[a], points[c]),
                    metric_fn(points[b], points[c]),
                )
                for a, b, c in triangles
            ],
            dtype=float,
        )

    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Gap Index is undefined when all triangle areas are zero")
    return areas / total_area


def compute(
    orig: npt.NDArray,
    emb: npt.NDArray,
    metric: DistanceMetric = "euclidean",
) -> GapIndexResult:
    """Compute the Gap Index score and per-triangle regional distortions.

    ``metric`` may be a scipy distance function, the name of one, or
    ``"precomputed"`` when ``orig`` is an ``(n, n)`` distance matrix.
    The embedding is always measured with Euclidean distance, matching the
    original formulation.
    """

    orig_array, emb_array = _validate_inputs(orig, emb, metric)

    try:
        triangles = Delaunay(emb_array).simplices
    except QhullError as exc:
        raise ValueError(
            "Gap Index requires at least three non-collinear embedded points"
        ) from exc

    original_areas = _compute_areas(orig_array, triangles, metric)
    embedded_areas = _compute_areas(emb_array, triangles, "euclidean")
    max_areas = np.maximum(original_areas, embedded_areas)
    deformations = (embedded_areas - original_areas) / np.maximum(
        max_areas, _DEFORMATION_EPSILON
    )
    score = float(np.sum(np.abs(deformations) * max_areas) / np.sum(max_areas))

    return GapIndexResult(
        score=float(np.clip(score, 0.0, 1.0)),
        triangles=triangles,
        deformations=np.clip(deformations, -1.0, 1.0),
        original_relative_areas=original_areas,
        embedded_relative_areas=embedded_areas,
    )


def gap_index(
    orig: npt.NDArray,
    emb: npt.NDArray,
    metric: DistanceMetric = "euclidean",
) -> float:
    """Return the scalar Gap Index using the original API naming."""

    return compute(orig, emb, metric=metric).score


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    metric: DistanceMetric = "euclidean",
) -> dict[str, float]:
    """Return the Gap Index through ZADU's standard measure contract."""

    return {"gap_index": gap_index(orig, emb, metric=metric)}
