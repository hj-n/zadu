"""Optional MLX provider for exact-algorithm dense Euclidean resources."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from zadu.engine.resources import (
    NeighborRanking,
    ResourceKey,
    ResourceKind,
    Space,
    compact_index_dtype,
)

from .base import BuiltResource
from .numpy_backend import NumpyResourceProvider


@dataclass(slots=True)
class _MlxWorkspace:
    """One zero-copy MLX view over a validated resource-space input."""

    source: npt.NDArray
    cast_points: npt.NDArray
    mlx_points: Any
    input_seconds: float
    cast_copy: bool
    mlx_copy: bool


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
        self._compiled_ready: set[str] = set()
        self._workspaces: dict[Space, _MlxWorkspace] = {}

        def pairwise(left, right):
            left_squared = mx.sum(left * left, axis=1, keepdims=True)
            right_squared = mx.sum(right * right, axis=1, keepdims=True)
            squared = left_squared + right_squared.T - 2.0 * (left @ right.T)
            return mx.sqrt(mx.maximum(squared, 0.0))

        def stable_order(distances, self_indices):
            columns = mx.arange(distances.shape[1], dtype=self_indices.dtype)
            sortable = mx.where(
                columns[None, :] == self_indices[:, None],
                -float("inf"),
                distances,
            )
            return mx.argsort(sortable, axis=1)

        def ranking_from_order(order):
            positions = mx.zeros_like(order) + mx.arange(
                order.shape[1], dtype=order.dtype
            )
            return mx.put_along_axis(
                mx.zeros_like(order),
                order,
                positions,
                axis=1,
            )

        def order_from_distances(distances, self_indices):
            return stable_order(distances, self_indices)

        def ranking_from_distances(distances, self_indices):
            order = stable_order(distances, self_indices)
            return order, ranking_from_order(order)

        def order_from_points(left, right, self_indices):
            return stable_order(pairwise(left, right), self_indices)

        def ranking_from_points(left, right, self_indices):
            order = stable_order(pairwise(left, right), self_indices)
            return order, ranking_from_order(order)

        self._compiled_pairwise = mx.compile(pairwise, shapeless=True)
        self._compiled_order_from_distances = mx.compile(
            order_from_distances,
            shapeless=True,
        )
        self._compiled_ranking_from_distances = mx.compile(
            ranking_from_distances,
            shapeless=True,
        )
        self._compiled_order_from_points = mx.compile(
            order_from_points,
            shapeless=True,
        )
        self._compiled_ranking_from_points = mx.compile(
            ranking_from_points,
            shapeless=True,
        )

    def fork(self) -> MlxResourceProvider:
        return type(self)(device=self.device, dtype=self.dtype)

    def invalidate(self, space: Space) -> None:
        self._workspaces.pop(space, None)

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
        if not geodesic and key.kind in {
            ResourceKind.KNN,
            ResourceKind.STABLE_KNN,
            ResourceKind.NEIGHBOR_RANKING,
        }:
            if working_memory_bytes is None:
                raise RuntimeError("MLX neighbor resources require a memory plan")
            return self._build_neighbors(
                key,
                points,
                distance_matrix=distance_matrix,
                working_memory_bytes=working_memory_bytes,
            )

        if not geodesic and key.kind in {
            ResourceKind.DISTANCE_MATRIX,
            ResourceKind.CONDENSED_PAIRS,
        }:
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
        workspace, input_reused = self._workspace(key.space, points)
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
            left = workspace.mlx_points[start:stop]
            distances, cold_seconds, warm_seconds = self._execute_compiled(
                "pairwise",
                self._compiled_pairwise,
                left,
                workspace.mlx_points,
            )
            compile_seconds += cold_seconds
            execution_seconds += warm_seconds

            output_started = perf_counter()
            block = np.array(distances, dtype=self._numpy_dtype, copy=False)
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
            output_transfer_seconds += perf_counter() - output_started
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
            "unified_memory": True,
            "input_zero_copy": not workspace.mlx_copy,
            "input_cast_copy": workspace.cast_copy,
            "input_reused": input_reused,
            "provider_fallback": False,
            "timings": {
                "input_transfer_seconds": float(
                    0.0 if input_reused else workspace.input_seconds
                ),
                "compile_and_first_execution_seconds": float(compile_seconds),
                "warm_execution_seconds": float(execution_seconds),
                "output_transfer_seconds": float(output_transfer_seconds),
            },
        }
        return BuiltResource(value, "mlx", details)

    def _build_neighbors(
        self,
        key: ResourceKey,
        points: npt.NDArray,
        *,
        distance_matrix: npt.NDArray | None,
        working_memory_bytes: int,
    ) -> BuiltResource:
        """Build stable exact neighbor prefixes and optional inverse ranks."""

        assert key.k is not None
        mx = self._mx
        n_samples = points.shape[0]
        index_dtype = compact_index_dtype(n_samples)
        ranking_requested = key.kind is ResourceKind.NEIGHBOR_RANKING
        index_arrays = 4 if ranking_requested else 2
        bytes_per_row = n_samples * (
            4 * self._numpy_dtype.itemsize + index_arrays * index_dtype.itemsize
        )
        if working_memory_bytes < bytes_per_row:
            raise MemoryError(
                "MLX neighbor execution needs enough memory for one distance row"
            )
        block_rows = max(
            1,
            min(n_samples, working_memory_bytes // bytes_per_row),
        )
        expected_blocks = (n_samples + block_rows - 1) // block_rows
        mlx_index_dtype = mx.int32 if index_dtype.itemsize == 4 else mx.int64

        workspace = None
        input_reused = False
        input_seconds = 0.0
        input_cast_copy = False
        distance_zero_copy = False
        if distance_matrix is None:
            workspace, input_reused = self._workspace(key.space, points)
            input_seconds = 0.0 if input_reused else workspace.input_seconds
            input_cast_copy = workspace.cast_copy
            input_zero_copy = not workspace.mlx_copy
            mlx_distances = None
            distance_source = "fused_blockwise_pairwise"
        else:
            distance_started = perf_counter()
            raw_distances = np.asarray(distance_matrix)
            distances = np.ascontiguousarray(
                raw_distances,
                dtype=self._numpy_dtype,
            )
            distance_zero_copy = np.shares_memory(raw_distances, distances)
            try:
                mlx_distances = mx.asarray(
                    distances,
                    dtype=self._mlx_dtype,
                    copy=False,
                )
            except ValueError:
                mlx_distances = mx.array(distances, dtype=self._mlx_dtype)
                distance_zero_copy = False
            with mx.stream(self._device):
                mx.eval(mlx_distances)
                mx.synchronize(self._device)
            input_seconds = perf_counter() - distance_started
            input_cast_copy = not distance_zero_copy
            input_reused = distance_zero_copy
            input_zero_copy = distance_zero_copy
            distance_source = "shared_distance_matrix"

        indices_result = (
            None
            if expected_blocks == 1
            else np.empty((n_samples, key.k), dtype=index_dtype)
        )
        ranking_result = (
            None
            if not ranking_requested or expected_blocks == 1
            else np.empty((n_samples, n_samples), dtype=index_dtype)
        )
        compile_seconds = 0.0
        execution_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            self_indices = mx.arange(start, stop, dtype=mx.uint32)
            if distance_matrix is None:
                assert workspace is not None
                arguments = (
                    workspace.mlx_points[start:stop],
                    workspace.mlx_points,
                    self_indices,
                )
                if ranking_requested:
                    compiled_name = "ranking_from_points"
                    compiled = self._compiled_ranking_from_points
                else:
                    compiled_name = "order_from_points"
                    compiled = self._compiled_order_from_points
            else:
                assert mlx_distances is not None
                arguments = (mlx_distances[start:stop], self_indices)
                if ranking_requested:
                    compiled_name = "ranking_from_distances"
                    compiled = self._compiled_ranking_from_distances
                else:
                    compiled_name = "order_from_distances"
                    compiled = self._compiled_order_from_distances

            output, cold_seconds, warm_seconds = self._execute_compiled(
                compiled_name,
                compiled,
                *arguments,
            )
            compile_seconds += cold_seconds
            execution_seconds += warm_seconds
            if ranking_requested:
                order, inverse = output
            else:
                order = output
                inverse = None

            output_started = perf_counter()
            mlx_indices = order[:, 1 : key.k + 1].astype(mlx_index_dtype)
            outputs = [mlx_indices]
            if inverse is not None:
                mlx_ranking = inverse.astype(mlx_index_dtype)
                outputs.append(mlx_ranking)
            with mx.stream(self._device):
                mx.eval(*outputs)
                mx.synchronize(self._device)
            indices_block = np.array(mlx_indices, dtype=index_dtype, copy=False)
            ranking_block = (
                None
                if inverse is None
                else np.array(mlx_ranking, dtype=index_dtype, copy=False)
            )
            if expected_blocks == 1:
                indices_result = indices_block
                if ranking_requested:
                    ranking_result = ranking_block
            else:
                assert indices_result is not None
                indices_result[start:stop] = indices_block
                if ranking_requested:
                    assert ranking_result is not None and ranking_block is not None
                    ranking_result[start:stop] = ranking_block
            output_transfer_seconds += perf_counter() - output_started
            block_count += 1

        assert indices_result is not None
        value: npt.NDArray | NeighborRanking
        if ranking_requested:
            assert ranking_result is not None
            value = NeighborRanking(indices_result, ranking_result)
            algorithm = "compiled_blockwise_stable_full_ranking"
        else:
            value = indices_result
            algorithm = "compiled_blockwise_stable_exact_topk"

        details: dict[str, Any] = {
            "algorithm": algorithm,
            "device": self.device,
            "compute_dtype": self.dtype,
            "index_dtype": index_dtype.name,
            "k": key.k,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "mlx_version": self._mlx_version(),
            "distance_source": distance_source,
            "distance_zero_copy": distance_zero_copy,
            "unified_memory": True,
            "input_zero_copy": input_zero_copy,
            "input_cast_copy": input_cast_copy,
            "input_reused": input_reused,
            "output_zero_copy": expected_blocks == 1,
            "self_exclusion": "forced_rank_zero_then_removed",
            "tie_break": "stable_column_index",
            "top_k_algorithm": "stable_full_order_prefix",
            "provider_fallback": False,
            "timings": {
                "input_transfer_seconds": float(input_seconds),
                "compile_and_first_execution_seconds": float(compile_seconds),
                "warm_execution_seconds": float(execution_seconds),
                "output_transfer_seconds": float(output_transfer_seconds),
            },
        }
        return BuiltResource(value, "mlx", details)

    def _workspace(
        self,
        space: Space,
        points: npt.NDArray,
    ) -> tuple[_MlxWorkspace, bool]:
        existing = self._workspaces.get(space)
        if existing is not None and existing.source is points:
            return existing, True

        with np.errstate(over="ignore", invalid="ignore"):
            cast_points = np.ascontiguousarray(points, dtype=self._numpy_dtype)
        if not np.all(np.isfinite(cast_points)):
            raise OverflowError(
                f"Input values cannot be represented safely as MLX {self.dtype}"
            )
        started = perf_counter()
        mlx_copy = False
        try:
            mlx_points = self._mx.asarray(
                cast_points,
                dtype=self._mlx_dtype,
                copy=False,
            )
        except ValueError:
            mlx_points = self._mx.array(cast_points, dtype=self._mlx_dtype)
            mlx_copy = True
        with self._mx.stream(self._device):
            self._mx.eval(mlx_points)
            self._mx.synchronize(self._device)
        workspace = _MlxWorkspace(
            source=points,
            cast_points=cast_points,
            mlx_points=mlx_points,
            input_seconds=perf_counter() - started,
            cast_copy=not np.shares_memory(points, cast_points),
            mlx_copy=mlx_copy,
        )
        self._workspaces[space] = workspace
        return workspace, False

    def _execute_compiled(self, name: str, function, *args):
        mx = self._mx
        compile_seconds = 0.0
        if name not in self._compiled_ready:
            started = perf_counter()
            with mx.stream(self._device):
                cold = function(*args)
                mx.eval(*(cold if isinstance(cold, tuple) else (cold,)))
                mx.synchronize(self._device)
            compile_seconds = perf_counter() - started
            self._compiled_ready.add(name)

        started = perf_counter()
        with mx.stream(self._device):
            output = function(*args)
            mx.eval(*(output if isinstance(output, tuple) else (output,)))
            mx.synchronize(self._device)
        execution_seconds = perf_counter() - started
        return output, compile_seconds, execution_seconds

    @staticmethod
    def _mlx_version() -> str:
        try:
            return version("mlx")
        except PackageNotFoundError:  # pragma: no cover - nonstandard install
            return "unknown"

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
