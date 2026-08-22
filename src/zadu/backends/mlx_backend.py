"""Optional MLX provider for exact-algorithm dense Euclidean resources."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from zadu.engine.resources import ResourceKey, ResourceKind

from .base import BuiltResource
from .numpy_backend import NumpyResourceProvider


class MlxResourceProvider(NumpyResourceProvider):
    """Route supported pairwise resources to MLX and fall back resource-wise."""

    name = "mlx"
    exact = True

    def __init__(self, *, device: str, dtype: str) -> None:
        try:
            mx = import_module("mlx.core")
        except ModuleNotFoundError as exc:
            if exc.name not in {"mlx", "mlx.core"}:
                raise
            raise ImportError(
                "The MLX preview is optional and packaged for Apple Silicon. "
                "Install it on a supported platform with "
                "`pip install 'zadu[mlx]'`."
            ) from exc

        if dtype not in {"float32", "float64"}:
            raise ValueError("MLX dtype must be 'float32' or 'float64'")
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("MLX device must be 'auto', 'cpu', or 'gpu'")
        if device == "auto":
            device = (
                "cpu"
                if dtype == "float64"
                else ("gpu" if mx.metal.is_available() else "cpu")
            )
        if device == "gpu" and not mx.metal.is_available():
            raise RuntimeError(
                "The requested MLX GPU is unavailable; choose device='cpu' explicitly"
            )
        if device == "gpu" and dtype != "float32":
            raise ValueError("The MLX GPU requires dtype='float32'")

        self._mx = mx
        self.device = device
        self.dtype = dtype
        self._device = mx.gpu if device == "gpu" else mx.cpu
        self._mlx_dtype = mx.float32 if dtype == "float32" else mx.float64
        self._numpy_dtype = np.dtype(dtype)
        self._compiled_ready = False

        def pairwise(left, right):
            left_squared = mx.sum(left * left, axis=1, keepdims=True)
            right_squared = mx.sum(right * right, axis=1, keepdims=True)
            squared = left_squared + right_squared.T - 2.0 * (left @ right.T)
            return mx.sqrt(mx.maximum(squared, 0.0))

        self._compiled_pairwise = mx.compile(pairwise, shapeless=True)

    def fork(self) -> MlxResourceProvider:
        return type(self)(device=self.device, dtype=self.dtype)

    def build(
        self,
        key: ResourceKey,
        points: npt.NDArray,
        *,
        distance_matrix: npt.NDArray | None,
        condensed_pairs: npt.NDArray | None,
        working_memory_bytes: int | None,
        geodesic: bool,
    ) -> BuiltResource:
        if (
            key.kind
            in {
                ResourceKind.DISTANCE_MATRIX,
                ResourceKind.CONDENSED_PAIRS,
            }
            and not geodesic
        ):
            if working_memory_bytes is None:
                raise RuntimeError("MLX pairwise resources require a memory plan")
            return self._build_euclidean(
                key,
                points,
                working_memory_bytes=working_memory_bytes,
            )

        built = super().build(
            key,
            points,
            distance_matrix=distance_matrix,
            condensed_pairs=condensed_pairs,
            working_memory_bytes=working_memory_bytes,
            geodesic=geodesic,
        )
        reason = "geodesic_not_supported" if geodesic else "unsupported_resource"
        return self._fallback(built, reason)

    def _build_euclidean(
        self,
        key: ResourceKey,
        points: npt.NDArray,
        *,
        working_memory_bytes: int,
    ) -> BuiltResource:
        mx = self._mx
        with np.errstate(over="ignore", invalid="ignore"):
            points = np.asarray(points, dtype=self._numpy_dtype)
        if not np.all(np.isfinite(points)):
            raise OverflowError(
                f"Input values cannot be represented safely as MLX {self.dtype}"
            )
        n_samples = points.shape[0]
        bytes_per_row = n_samples * self._numpy_dtype.itemsize * 4
        if working_memory_bytes < bytes_per_row:
            raise MemoryError(
                "MLX pairwise execution needs enough memory for one distance row"
            )
        block_rows = max(
            1,
            min(n_samples, working_memory_bytes // bytes_per_row),
        )

        transfer_started = perf_counter()
        mlx_points = mx.array(points, dtype=self._mlx_dtype)
        with mx.stream(self._device):
            mx.eval(mlx_points)
            mx.synchronize(self._device)
        input_transfer_seconds = perf_counter() - transfer_started

        if key.kind is ResourceKind.DISTANCE_MATRIX:
            value = np.empty((n_samples, n_samples), dtype=self._numpy_dtype)
        else:
            value = np.empty(
                n_samples * (n_samples - 1) // 2,
                dtype=self._numpy_dtype,
            )

        compile_seconds = 0.0
        execution_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        condensed_offset = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            left = mlx_points[start:stop]
            if not self._compiled_ready:
                cold_started = perf_counter()
                with mx.stream(self._device):
                    cold = self._compiled_pairwise(left, mlx_points)
                    mx.eval(cold)
                    mx.synchronize(self._device)
                compile_seconds += perf_counter() - cold_started
                self._compiled_ready = True

            execution_started = perf_counter()
            with mx.stream(self._device):
                distances = self._compiled_pairwise(left, mlx_points)
                mx.eval(distances)
                mx.synchronize(self._device)
            execution_seconds += perf_counter() - execution_started

            output_started = perf_counter()
            block = np.array(distances, dtype=self._numpy_dtype, copy=True)
            output_transfer_seconds += perf_counter() - output_started
            if key.kind is ResourceKind.DISTANCE_MATRIX:
                value[start:stop] = block
            else:
                for local_row, row in enumerate(range(start, stop)):
                    count = n_samples - row - 1
                    if count:
                        value[condensed_offset : condensed_offset + count] = block[
                            local_row, row + 1 :
                        ]
                        condensed_offset += count
            block_count += 1

        if key.kind is ResourceKind.DISTANCE_MATRIX:
            for row in range(n_samples):
                value[row + 1 :, row] = value[row, row + 1 :]
            np.fill_diagonal(value, 0)

        try:
            mlx_version = version("mlx")
        except PackageNotFoundError:  # pragma: no cover - nonstandard install
            mlx_version = "unknown"
        details: dict[str, Any] = {
            "algorithm": "compiled_blockwise_squared_euclidean",
            "device": self.device,
            "compute_dtype": self.dtype,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "mlx_version": mlx_version,
            "provider_fallback": False,
            "timings": {
                "input_transfer_seconds": float(input_transfer_seconds),
                "compile_and_first_execution_seconds": float(compile_seconds),
                "warm_execution_seconds": float(execution_seconds),
                "output_transfer_seconds": float(output_transfer_seconds),
            },
        }
        return BuiltResource(value, "mlx", details)

    def build_pair_statistics(self, *args, **kwargs) -> BuiltResource:
        return self._fallback(
            super().build_pair_statistics(*args, **kwargs),
            "unsupported_resource",
        )

    def build_ordered_pair_statistics(self, *args, **kwargs) -> BuiltResource:
        return self._fallback(
            super().build_ordered_pair_statistics(*args, **kwargs),
            "unsupported_resource",
        )

    def build_topographic_product_statistics(self, *args, **kwargs) -> BuiltResource:
        return self._fallback(
            super().build_topographic_product_statistics(*args, **kwargs),
            "unsupported_resource",
        )

    def build_rank_comparisons(self, *args, **kwargs) -> BuiltResource:
        return self._fallback(
            super().build_rank_comparisons(*args, **kwargs),
            "unsupported_resource",
        )

    def build_neighbor_statistics(self, *args, **kwargs) -> BuiltResource:
        return self._fallback(
            super().build_neighbor_statistics(*args, **kwargs),
            "unsupported_resource",
        )

    @staticmethod
    def _fallback(built: BuiltResource, reason: str) -> BuiltResource:
        return BuiltResource(
            built.value,
            built.implementation,
            {
                **built.details,
                "requested_provider": "mlx",
                "provider_fallback": True,
                "fallback_reason": reason,
            },
        )
