"""Exact fused kernels shared by resource providers and metric wrappers."""

from .pairwise import (
    PairAccumulator,
    pearson_from_statistics,
    scale_normalized_stress_from_statistics,
    stress_from_statistics,
)

__all__ = [
    "PairAccumulator",
    "pearson_from_statistics",
    "scale_normalized_stress_from_statistics",
    "stress_from_statistics",
]
