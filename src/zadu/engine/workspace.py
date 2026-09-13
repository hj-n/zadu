"""Scratch-space contracts for metrics that do not need new shared resources."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True)
class MetricWorkspace:
    """Retained/validation allowance and a conservative row-block estimate."""

    fixed_bytes: int
    bytes_per_row: int
    rows: int

    def plan(self, budget: int | None = None) -> int:
        if budget is None:
            budget = self.fixed_bytes + 64 * 1024**2
        if isinstance(budget, bool) or not isinstance(budget, Integral):
            raise TypeError("working_memory_bytes must be an integer")
        minimum = self.fixed_bytes + self.bytes_per_row
        if budget < minimum:
            raise MemoryError(
                "Metric workspace needs at least one exact row and retained "
                f"storage ({minimum} > {budget} bytes)"
            )
        rows = min(self.rows, (int(budget) - self.fixed_bytes) // self.bytes_per_row)
        return self.fixed_bytes + rows * self.bytes_per_row

    def block_bytes(self, budget: int | None = None) -> int:
        return self.plan(budget) - self.fixed_bytes


def procrustes_workspace(n: int, d: int, e: int, k: int, itemsize: int = 8):
    """Include gathers, centered copies, SVD matrices and residual temporaries."""
    width = max(8, itemsize)
    return MetricWorkspace(
        fixed_bytes=64 * 1024 + n * (width + d + e),
        bytes_per_row=width
        * (6 * k * (d + e) + 8 * d * e + 8 * min(d, e) ** 2 + 8 * (d + e)),
        rows=min(n, 256),
    )


def gap_workspace(
    n: int, d: int, e: int = 2, itemsize: int = 8, *, precomputed: bool = False
):
    """Allow for linear-size 2D triangulation, results, validation and blocks.

    Qhull's native allocator is estimated, not intercepted. User-defined
    distance functions may allocate additional memory outside this contract.
    """
    return MetricWorkspace(
        fixed_bytes=(
            64 * 1024 + n * (1024 + d + e) + (32 * n * n if precomputed else 0)
        ),
        bytes_per_row=max(8, itemsize) * (6 * max(2 if precomputed else d, e) + 64),
        rows=max(1, 2 * n),
    )
