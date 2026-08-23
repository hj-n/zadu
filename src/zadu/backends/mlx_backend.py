"""Optional MLX provider for exact-algorithm dense Euclidean resources."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zadu.engine.resources import (
    NeighborRanking,
    RankComparisons,
    ResourceKey,
    ResourceKind,
    Space,
    compact_index_dtype,
)

from .base import BatchResourceError, BuiltResource
from .numpy_backend import NumpyResourceProvider

if TYPE_CHECKING:
    from zadu.engine.planner import RankComparisonExecutionPlan


@dataclass(slots=True)
class _MlxWorkspace:
    """One zero-copy MLX view over a validated resource-space input."""

    source: npt.NDArray
    cast_points: npt.NDArray
    mlx_points: Any
    input_seconds: float
    cast_copy: bool
    mlx_copy: bool


@dataclass(slots=True)
class _MlxBatchWorkspace:
    """One stacked MLX tensor reused across resources for an embedding batch."""

    sources: tuple[npt.NDArray, ...]
    cast_points: npt.NDArray
    mlx_points: Any
    input_seconds: float
    mlx_copy: bool


class MlxResourceProvider(NumpyResourceProvider):
    """Route supported pairwise resources to MLX and fall back resource-wise."""

    name = "mlx"
    exact = True
    supports_embedding_batching = True

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
        self._batch_workspaces: dict[Space, _MlxBatchWorkspace] = {}

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

        def batched_pairwise(left, right):
            left_squared = mx.sum(left * left, axis=2, keepdims=True)
            right_squared = mx.sum(right * right, axis=2, keepdims=True)
            squared = (
                left_squared
                + mx.swapaxes(right_squared, 1, 2)
                - 2.0 * (left @ mx.swapaxes(right, 1, 2))
            )
            return mx.sqrt(mx.maximum(squared, 0.0))

        def batched_stable_order(distances, self_indices):
            columns = mx.arange(distances.shape[2], dtype=self_indices.dtype)
            sortable = mx.where(
                columns[None, None, :] == self_indices[None, :, None],
                -float("inf"),
                distances,
            )
            return mx.argsort(sortable, axis=2)

        def batched_ranking_from_order(order):
            positions = mx.zeros_like(order) + mx.arange(
                order.shape[2], dtype=order.dtype
            )
            return mx.put_along_axis(
                mx.zeros_like(order),
                order,
                positions,
                axis=2,
            )

        def batched_order_from_distances(distances, self_indices):
            return batched_stable_order(distances, self_indices)

        def batched_ranking_from_distances(distances, self_indices):
            order = batched_stable_order(distances, self_indices)
            return order, batched_ranking_from_order(order)

        def batched_order_from_points(left, right, self_indices):
            return batched_stable_order(batched_pairwise(left, right), self_indices)

        def batched_ranking_from_points(left, right, self_indices):
            order = batched_stable_order(batched_pairwise(left, right), self_indices)
            return order, batched_ranking_from_order(order)

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
        self._compiled_batched_pairwise = mx.compile(
            batched_pairwise,
            shapeless=True,
        )
        self._compiled_batched_order_from_distances = mx.compile(
            batched_order_from_distances,
            shapeless=True,
        )
        self._compiled_batched_ranking_from_distances = mx.compile(
            batched_ranking_from_distances,
            shapeless=True,
        )
        self._compiled_batched_order_from_points = mx.compile(
            batched_order_from_points,
            shapeless=True,
        )
        self._compiled_batched_ranking_from_points = mx.compile(
            batched_ranking_from_points,
            shapeless=True,
        )

    def fork(self) -> MlxResourceProvider:
        return type(self)(device=self.device, dtype=self.dtype)

    def invalidate(self, space: Space) -> None:
        self._workspaces.pop(space, None)
        self._batch_workspaces.pop(space, None)

    def can_batch(self, key: ResourceKey) -> bool:
        return key.space is Space.EMBEDDED and key.kind in {
            ResourceKind.DISTANCE_MATRIX,
            ResourceKind.CONDENSED_PAIRS,
            ResourceKind.KNN,
            ResourceKind.STABLE_KNN,
            ResourceKind.NEIGHBOR_RANKING,
        }

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
        if not points_batch:
            return []
        if not geodesic and self.can_batch(key):
            if working_memory_bytes is None:
                raise RuntimeError("MLX batched resources require a memory plan")
            if key.kind in {
                ResourceKind.DISTANCE_MATRIX,
                ResourceKind.CONDENSED_PAIRS,
            }:
                return self._build_euclidean_batch(
                    key,
                    points_batch,
                    working_memory_bytes=working_memory_bytes,
                )
            return self._build_neighbors_batch(
                key,
                points_batch,
                distance_matrices=distance_matrices,
                working_memory_bytes=working_memory_bytes,
            )

        built_batch = super().build_batch(
            key,
            points_batch,
            distance_matrices=distance_matrices,
            condensed_pairs=condensed_pairs,
            working_memory_bytes=working_memory_bytes,
            geodesic=geodesic,
        )
        reason = "geodesic_not_supported" if geodesic else "unsupported_resource"
        return [self._fallback(built, reason) for built in built_batch]

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
            with self._mx.stream(self._device):
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
            with mx.stream(self._device):
                try:
                    mlx_distances = mx.asarray(
                        distances,
                        dtype=self._mlx_dtype,
                        copy=False,
                    )
                except ValueError:
                    mlx_distances = mx.array(distances, dtype=self._mlx_dtype)
                    distance_zero_copy = False
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
            with mx.stream(self._device):
                self_indices = mx.arange(start, stop, dtype=mx.uint32)
                if distance_matrix is None:
                    assert workspace is not None
                    arguments = (
                        workspace.mlx_points[start:stop],
                        workspace.mlx_points,
                        self_indices,
                    )
                else:
                    assert mlx_distances is not None
                    arguments = (mlx_distances[start:stop], self_indices)
            if distance_matrix is None:
                if ranking_requested:
                    compiled_name = "ranking_from_points"
                    compiled = self._compiled_ranking_from_points
                else:
                    compiled_name = "order_from_points"
                    compiled = self._compiled_order_from_points
            else:
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
            with mx.stream(self._device):
                mlx_indices = order[:, 1 : key.k + 1].astype(mlx_index_dtype)
                outputs = [mlx_indices]
                if inverse is not None:
                    mlx_ranking = inverse.astype(mlx_index_dtype)
                    outputs.append(mlx_ranking)
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

    def _build_euclidean_batch(
        self,
        key: ResourceKey,
        points_batch: list[npt.NDArray],
        *,
        working_memory_bytes: int,
    ) -> list[BuiltResource]:
        """Build exact pairwise resources with an explicit batch dimension."""

        workspace, input_reused = self._batch_workspace(key.space, points_batch)
        batch_size = len(points_batch)
        n_samples = points_batch[0].shape[0]
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
            values = [
                np.empty((n_samples, n_samples), dtype=self._numpy_dtype)
                for _ in points_batch
            ]
        else:
            pair_count = n_samples * (n_samples - 1) // 2
            values = [
                np.empty(pair_count, dtype=self._numpy_dtype) for _ in points_batch
            ]

        compile_seconds = 0.0
        execution_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        condensed_offset = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            with self._mx.stream(self._device):
                left = workspace.mlx_points[:, start:stop]
            distances, cold_seconds, warm_seconds = self._execute_compiled(
                "batched_pairwise",
                self._compiled_batched_pairwise,
                left,
                workspace.mlx_points,
            )
            compile_seconds += cold_seconds
            execution_seconds += warm_seconds

            output_started = perf_counter()
            block = np.array(distances, dtype=self._numpy_dtype, copy=False)
            if key.kind is ResourceKind.DISTANCE_MATRIX:
                for batch_index, value in enumerate(values):
                    value[start:stop] = block[batch_index]
            else:
                for local_row, row in enumerate(range(start, stop)):
                    count = n_samples - row - 1
                    if count:
                        for batch_index, value in enumerate(values):
                            value[condensed_offset : condensed_offset + count] = block[
                                batch_index, local_row, row + 1 :
                            ]
                        condensed_offset += count
            output_transfer_seconds += perf_counter() - output_started
            block_count += 1

        if key.kind is ResourceKind.DISTANCE_MATRIX:
            for value in values:
                for row in range(n_samples):
                    value[row + 1 :, row] = value[row, row + 1 :]
                np.fill_diagonal(value, 0)

        timings = {
            "input_transfer_seconds": float(
                0.0 if input_reused else workspace.input_seconds
            ),
            "compile_and_first_execution_seconds": float(compile_seconds),
            "warm_execution_seconds": float(execution_seconds),
            "output_transfer_seconds": float(output_transfer_seconds),
        }
        base_details: dict[str, Any] = {
            "algorithm": "compiled_batched_blockwise_squared_euclidean",
            "device": self.device,
            "compute_dtype": self.dtype,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "batch_working_bytes": batch_size * block_rows * bytes_per_row,
            "mlx_version": self._mlx_version(),
            "unified_memory": True,
            "input_zero_copy": not workspace.mlx_copy,
            "input_cast_copy": True,
            "input_reused": input_reused,
            "provider_fallback": False,
        }
        return self._batched_resources(values, base_details, timings)

    def _build_neighbors_batch(
        self,
        key: ResourceKey,
        points_batch: list[npt.NDArray],
        *,
        distance_matrices: list[npt.NDArray | None],
        working_memory_bytes: int,
    ) -> list[BuiltResource]:
        """Build stable exact neighbors for a native embedding batch."""

        assert key.k is not None
        mx = self._mx
        batch_size = len(points_batch)
        n_samples = points_batch[0].shape[0]
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
        mlx_index_dtype = mx.int32 if index_dtype.itemsize == 4 else mx.int64

        has_distances = [matrix is not None for matrix in distance_matrices]
        if any(has_distances) and not all(has_distances):
            raise RuntimeError("MLX batches require uniform distance resources")
        workspace = None
        input_reused = False
        input_seconds = 0.0
        input_zero_copy = False
        if not any(has_distances):
            workspace, input_reused = self._batch_workspace(key.space, points_batch)
            input_seconds = 0.0 if input_reused else workspace.input_seconds
            input_zero_copy = not workspace.mlx_copy
            distance_source = "fused_batched_blockwise_pairwise"
        else:
            distance_source = "shared_distance_matrix_batch"

        indices_results = [
            np.empty((n_samples, key.k), dtype=index_dtype) for _ in points_batch
        ]
        ranking_results = (
            [np.empty((n_samples, n_samples), dtype=index_dtype) for _ in points_batch]
            if ranking_requested
            else None
        )
        compile_seconds = 0.0
        execution_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            with mx.stream(self._device):
                self_indices = mx.arange(start, stop, dtype=mx.uint32)
            if workspace is not None:
                with mx.stream(self._device):
                    arguments = (
                        workspace.mlx_points[:, start:stop],
                        workspace.mlx_points,
                        self_indices,
                    )
                if ranking_requested:
                    compiled_name = "batched_ranking_from_points"
                    compiled = self._compiled_batched_ranking_from_points
                else:
                    compiled_name = "batched_order_from_points"
                    compiled = self._compiled_batched_order_from_points
            else:
                input_started = perf_counter()
                distance_block = np.ascontiguousarray(
                    np.stack(
                        [np.asarray(matrix)[start:stop] for matrix in distance_matrices]
                    ),
                    dtype=self._numpy_dtype,
                )
                with mx.stream(self._device):
                    try:
                        mlx_distances = mx.asarray(
                            distance_block,
                            dtype=self._mlx_dtype,
                            copy=False,
                        )
                    except ValueError:
                        mlx_distances = mx.array(
                            distance_block,
                            dtype=self._mlx_dtype,
                        )
                    mx.eval(mlx_distances)
                    mx.synchronize(self._device)
                input_seconds += perf_counter() - input_started
                arguments = (mlx_distances, self_indices)
                if ranking_requested:
                    compiled_name = "batched_ranking_from_distances"
                    compiled = self._compiled_batched_ranking_from_distances
                else:
                    compiled_name = "batched_order_from_distances"
                    compiled = self._compiled_batched_order_from_distances

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
            with mx.stream(self._device):
                mlx_indices = order[:, :, 1 : key.k + 1].astype(mlx_index_dtype)
                outputs = [mlx_indices]
                if inverse is not None:
                    mlx_ranking = inverse.astype(mlx_index_dtype)
                    outputs.append(mlx_ranking)
                mx.eval(*outputs)
                mx.synchronize(self._device)
            indices_block = np.array(mlx_indices, dtype=index_dtype, copy=False)
            ranking_block = (
                None
                if inverse is None
                else np.array(mlx_ranking, dtype=index_dtype, copy=False)
            )
            for batch_index in range(batch_size):
                indices_results[batch_index][start:stop] = indices_block[batch_index]
                if ranking_results is not None and ranking_block is not None:
                    ranking_results[batch_index][start:stop] = ranking_block[
                        batch_index
                    ]
            output_transfer_seconds += perf_counter() - output_started
            block_count += 1

        if ranking_results is None:
            values: list[npt.NDArray | NeighborRanking] = indices_results
            algorithm = "compiled_batched_blockwise_stable_exact_topk"
        else:
            values = [
                NeighborRanking(indices, ranking)
                for indices, ranking in zip(
                    indices_results,
                    ranking_results,
                    strict=True,
                )
            ]
            algorithm = "compiled_batched_blockwise_stable_full_ranking"

        timings = {
            "input_transfer_seconds": float(input_seconds),
            "compile_and_first_execution_seconds": float(compile_seconds),
            "warm_execution_seconds": float(execution_seconds),
            "output_transfer_seconds": float(output_transfer_seconds),
        }
        base_details: dict[str, Any] = {
            "algorithm": algorithm,
            "device": self.device,
            "compute_dtype": self.dtype,
            "index_dtype": index_dtype.name,
            "k": key.k,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "batch_working_bytes": batch_size * block_rows * bytes_per_row,
            "mlx_version": self._mlx_version(),
            "distance_source": distance_source,
            "distance_zero_copy": False,
            "unified_memory": True,
            "input_zero_copy": input_zero_copy,
            "input_cast_copy": True,
            "input_reused": input_reused,
            "output_zero_copy": False,
            "self_exclusion": "forced_rank_zero_then_removed",
            "tie_break": "stable_column_index",
            "top_k_algorithm": "stable_full_order_prefix",
            "provider_fallback": False,
        }
        return self._batched_resources(values, base_details, timings)

    def _batch_workspace(
        self,
        space: Space,
        points_batch: list[npt.NDArray],
    ) -> tuple[_MlxBatchWorkspace, bool]:
        sources = tuple(points_batch)
        existing = self._batch_workspaces.get(space)
        if (
            existing is not None
            and len(existing.sources) == len(sources)
            and all(
                previous is current
                for previous, current in zip(existing.sources, sources, strict=True)
            )
        ):
            return existing, True

        stacked_points = np.empty(
            (len(points_batch), *points_batch[0].shape),
            dtype=self._numpy_dtype,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            for batch_index, points in enumerate(points_batch):
                np.copyto(stacked_points[batch_index], points, casting="unsafe")
                if not np.all(np.isfinite(stacked_points[batch_index])):
                    raise BatchResourceError(
                        batch_index,
                        f"Input values cannot be represented safely as MLX {self.dtype}",
                    )
        started = perf_counter()
        mlx_copy = False
        with self._mx.stream(self._device):
            try:
                mlx_points = self._mx.asarray(
                    stacked_points,
                    dtype=self._mlx_dtype,
                    copy=False,
                )
            except ValueError:
                mlx_points = self._mx.array(stacked_points, dtype=self._mlx_dtype)
                mlx_copy = True
            self._mx.eval(mlx_points)
            self._mx.synchronize(self._device)
        workspace = _MlxBatchWorkspace(
            sources=sources,
            cast_points=stacked_points,
            mlx_points=mlx_points,
            input_seconds=perf_counter() - started,
            mlx_copy=mlx_copy,
        )
        self._batch_workspaces[space] = workspace
        return workspace, False

    @staticmethod
    def _batched_resources(values, details, timings) -> list[BuiltResource]:
        batch_size = len(values)
        timing_share = {
            name: float(seconds / batch_size) for name, seconds in timings.items()
        }
        return [
            BuiltResource(
                value,
                "mlx",
                {
                    **details,
                    "provider_batching": True,
                    "batch_size": batch_size,
                    "batch_index": batch_index,
                    "timings": timing_share,
                    "batch_timings": dict(timings),
                },
            )
            for batch_index, value in enumerate(values)
        ]

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
        with self._mx.stream(self._device):
            try:
                mlx_points = self._mx.asarray(
                    cast_points,
                    dtype=self._mlx_dtype,
                    copy=False,
                )
            except ValueError:
                mlx_points = self._mx.array(cast_points, dtype=self._mlx_dtype)
                mlx_copy = True
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

    def build_rank_comparisons(
        self,
        plan: RankComparisonExecutionPlan,
        orig: npt.NDArray,
        emb: npt.NDArray,
        *,
        orig_knn: npt.NDArray,
        orig_distance_matrix: npt.NDArray | None,
        emb_distance_matrix: npt.NDArray | None,
    ) -> BuiltResource:
        if plan.geodesic:
            return self._fallback(
                super().build_rank_comparisons(
                    plan,
                    orig,
                    emb,
                    orig_knn=orig_knn,
                    orig_distance_matrix=orig_distance_matrix,
                    emb_distance_matrix=emb_distance_matrix,
                ),
                "geodesic_not_supported",
            )
        return self._build_rank_comparisons(
            plan,
            orig,
            emb,
            orig_knn=orig_knn,
            orig_distance_matrix=orig_distance_matrix,
            emb_distance_matrix=emb_distance_matrix,
        )

    def _build_rank_comparisons(
        self,
        plan: RankComparisonExecutionPlan,
        orig: npt.NDArray,
        emb: npt.NDArray,
        *,
        orig_knn: npt.NDArray,
        orig_distance_matrix: npt.NDArray | None,
        emb_distance_matrix: npt.NDArray | None,
    ) -> BuiltResource:
        """Build exact paired selected ranks with MLX block workspaces."""

        mx = self._mx
        n_samples = orig.shape[0]
        largest_membership_k = max(plan.membership_ks, default=0)
        bytes_per_row = max(1, n_samples * 24 + largest_membership_k**2)
        if plan.work_budget_bytes < bytes_per_row:
            raise MemoryError(
                "MLX selected-rank execution needs enough memory for one row"
            )
        block_rows = max(
            1,
            min(
                n_samples,
                plan.block_rows,
                plan.work_budget_bytes // bytes_per_row,
            ),
        )
        index_dtype = compact_index_dtype(n_samples)
        mlx_index_dtype = mx.int32 if index_dtype.itemsize == 4 else mx.int64
        orig_indices = np.asarray(orig_knn)[:, : plan.k]
        emb_indices = np.empty((n_samples, plan.k), dtype=index_dtype)
        orig_ranks_of_emb = np.empty((n_samples, plan.k), dtype=index_dtype)
        emb_ranks_of_orig = np.empty((n_samples, plan.k), dtype=index_dtype)
        emb_in_orig = {
            k: np.empty((n_samples, k), dtype=np.bool_) for k in plan.membership_ks
        }
        orig_in_emb = {
            k: np.empty((n_samples, k), dtype=np.bool_) for k in plan.membership_ks
        }

        def matrix_view(matrix):
            started = perf_counter()
            raw = np.asarray(matrix)
            distances = np.ascontiguousarray(raw, dtype=self._numpy_dtype)
            zero_copy = np.shares_memory(raw, distances)
            with mx.stream(self._device):
                try:
                    tensor = mx.asarray(
                        distances,
                        dtype=self._mlx_dtype,
                        copy=False,
                    )
                except ValueError:
                    tensor = mx.array(distances, dtype=self._mlx_dtype)
                    zero_copy = False
                mx.eval(tensor)
                mx.synchronize(self._device)
            return tensor, zero_copy, perf_counter() - started

        orig_workspace = None
        emb_workspace = None
        orig_distances = None
        emb_distances = None
        input_seconds = 0.0
        if orig_distance_matrix is None:
            orig_workspace, orig_input_reused = self._workspace(
                Space.ORIGINAL,
                orig,
            )
            if not orig_input_reused:
                input_seconds += orig_workspace.input_seconds
            orig_input_zero_copy = not orig_workspace.mlx_copy
            orig_input_cast_copy = orig_workspace.cast_copy
            orig_distance_zero_copy = False
            orig_distance_source = "fused_blockwise_pairwise"
        else:
            orig_distances, orig_distance_zero_copy, elapsed = matrix_view(
                orig_distance_matrix
            )
            input_seconds += elapsed
            orig_input_reused = orig_distance_zero_copy
            orig_input_zero_copy = orig_distance_zero_copy
            orig_input_cast_copy = not orig_distance_zero_copy
            orig_distance_source = "shared_distance_matrix"

        if emb_distance_matrix is None:
            emb_workspace, emb_input_reused = self._workspace(
                Space.EMBEDDED,
                emb,
            )
            if not emb_input_reused:
                input_seconds += emb_workspace.input_seconds
            emb_input_zero_copy = not emb_workspace.mlx_copy
            emb_input_cast_copy = emb_workspace.cast_copy
            emb_distance_zero_copy = False
            emb_distance_source = "fused_blockwise_pairwise"
        else:
            emb_distances, emb_distance_zero_copy, elapsed = matrix_view(
                emb_distance_matrix
            )
            input_seconds += elapsed
            emb_input_reused = emb_distance_zero_copy
            emb_input_zero_copy = emb_distance_zero_copy
            emb_input_cast_copy = not emb_distance_zero_copy
            emb_distance_source = "shared_distance_matrix"

        compile_seconds = 0.0
        execution_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            target_started = perf_counter()
            target_block = np.ascontiguousarray(orig_indices[start:stop])
            with mx.stream(self._device):
                self_indices = mx.arange(start, stop, dtype=mx.uint32)
                try:
                    mlx_orig_targets = mx.asarray(
                        target_block,
                        dtype=mlx_index_dtype,
                        copy=False,
                    )
                except ValueError:
                    mlx_orig_targets = mx.array(
                        target_block,
                        dtype=mlx_index_dtype,
                    )
                mx.eval(mlx_orig_targets)
                mx.synchronize(self._device)
            input_seconds += perf_counter() - target_started

            if emb_distances is None:
                assert emb_workspace is not None
                with mx.stream(self._device):
                    emb_arguments = (
                        emb_workspace.mlx_points[start:stop],
                        emb_workspace.mlx_points,
                        self_indices,
                    )
                emb_compiled_name = "ranking_from_points"
                emb_compiled = self._compiled_ranking_from_points
            else:
                with mx.stream(self._device):
                    emb_arguments = (emb_distances[start:stop], self_indices)
                emb_compiled_name = "ranking_from_distances"
                emb_compiled = self._compiled_ranking_from_distances
            emb_output, cold_seconds, warm_seconds = self._execute_compiled(
                emb_compiled_name,
                emb_compiled,
                *emb_arguments,
            )
            compile_seconds += cold_seconds
            execution_seconds += warm_seconds
            emb_order, emb_inverse = emb_output
            with mx.stream(self._device):
                mlx_emb_indices = emb_order[:, 1 : plan.k + 1].astype(mlx_index_dtype)
                mlx_emb_ranks = mx.take_along_axis(
                    emb_inverse,
                    mlx_orig_targets,
                    axis=1,
                ).astype(mlx_index_dtype)
            compact_started = perf_counter()
            with mx.stream(self._device):
                mx.eval(mlx_emb_indices, mlx_emb_ranks)
                mx.synchronize(self._device)
            execution_seconds += perf_counter() - compact_started
            del emb_output, emb_order, emb_inverse

            if orig_distances is None:
                assert orig_workspace is not None
                with mx.stream(self._device):
                    orig_arguments = (
                        orig_workspace.mlx_points[start:stop],
                        orig_workspace.mlx_points,
                        self_indices,
                    )
                orig_compiled_name = "ranking_from_points"
                orig_compiled = self._compiled_ranking_from_points
            else:
                with mx.stream(self._device):
                    orig_arguments = (orig_distances[start:stop], self_indices)
                orig_compiled_name = "ranking_from_distances"
                orig_compiled = self._compiled_ranking_from_distances
            orig_output, cold_seconds, warm_seconds = self._execute_compiled(
                orig_compiled_name,
                orig_compiled,
                *orig_arguments,
            )
            compile_seconds += cold_seconds
            execution_seconds += warm_seconds
            orig_order, orig_inverse = orig_output
            with mx.stream(self._device):
                mlx_orig_ranks = mx.take_along_axis(
                    orig_inverse,
                    mlx_emb_indices,
                    axis=1,
                ).astype(mlx_index_dtype)
                mlx_emb_memberships = []
                mlx_orig_memberships = []
                for requested_k in plan.membership_ks:
                    selected_emb = mlx_emb_indices[:, :requested_k]
                    selected_orig = mlx_orig_targets[:, :requested_k]
                    mlx_emb_memberships.append(
                        mx.any(
                            selected_emb[:, :, None] == selected_orig[:, None, :],
                            axis=2,
                        )
                    )
                    mlx_orig_memberships.append(
                        mx.any(
                            selected_orig[:, :, None] == selected_emb[:, None, :],
                            axis=2,
                        )
                    )
            compact_started = perf_counter()
            compact_outputs = (
                mlx_orig_ranks,
                *mlx_emb_memberships,
                *mlx_orig_memberships,
            )
            with mx.stream(self._device):
                mx.eval(*compact_outputs)
                mx.synchronize(self._device)
            execution_seconds += perf_counter() - compact_started
            del orig_output, orig_order, orig_inverse

            output_started = perf_counter()
            emb_indices[start:stop] = np.array(
                mlx_emb_indices,
                dtype=index_dtype,
                copy=False,
            )
            emb_ranks_of_orig[start:stop] = np.array(
                mlx_emb_ranks,
                dtype=index_dtype,
                copy=False,
            )
            orig_ranks_of_emb[start:stop] = np.array(
                mlx_orig_ranks,
                dtype=index_dtype,
                copy=False,
            )
            for requested_k, emb_membership, orig_membership in zip(
                plan.membership_ks,
                mlx_emb_memberships,
                mlx_orig_memberships,
                strict=True,
            ):
                emb_in_orig[requested_k][start:stop] = np.array(
                    emb_membership,
                    dtype=np.bool_,
                    copy=False,
                )
                orig_in_emb[requested_k][start:stop] = np.array(
                    orig_membership,
                    dtype=np.bool_,
                    copy=False,
                )
            output_transfer_seconds += perf_counter() - output_started
            block_count += 1

        value = RankComparisons(
            orig_ranks_of_emb=orig_ranks_of_emb,
            emb_ranks_of_orig=emb_ranks_of_orig,
            orig_indices=orig_indices,
            emb_indices=emb_indices,
            emb_in_orig=emb_in_orig,
            orig_in_emb=orig_in_emb,
        )
        return BuiltResource(
            value,
            "mlx",
            {
                "algorithm": "compiled_blockwise_selected_ranks",
                "device": self.device,
                "compute_dtype": self.dtype,
                "index_dtype": index_dtype.name,
                "k": plan.k,
                "requested_ks": list(plan.requested_ks),
                "membership_ks": list(plan.membership_ks),
                "block_rows": block_rows,
                "block_count": block_count,
                "work_budget_bytes": plan.work_budget_bytes,
                "working_bytes": block_rows * bytes_per_row,
                "mlx_version": self._mlx_version(),
                "unified_memory": True,
                "input_zero_copy": (orig_input_zero_copy and emb_input_zero_copy),
                "input_cast_copy": (orig_input_cast_copy or emb_input_cast_copy),
                "input_reused": orig_input_reused and emb_input_reused,
                "original_input_reused": orig_input_reused,
                "embedded_input_reused": emb_input_reused,
                "original_distance_zero_copy": orig_distance_zero_copy,
                "embedded_distance_zero_copy": emb_distance_zero_copy,
                "output_zero_copy": False,
                "original_neighbor_source": "cached_stable_knn",
                "original_rank_algorithm": "stable_sort_inverse_scatter",
                "embedded_rank_algorithm": "stable_sort_inverse_scatter",
                "original_distance_source": orig_distance_source,
                "embedded_distance_source": emb_distance_source,
                "tie_break": "stable_column_index",
                "self_exclusion": "forced_rank_zero_then_removed",
                "fused_metrics": list(plan.metric_ids),
                "provider_fallback": False,
                "timings": {
                    "input_transfer_seconds": float(input_seconds),
                    "compile_and_first_execution_seconds": float(compile_seconds),
                    "warm_execution_seconds": float(execution_seconds),
                    "output_transfer_seconds": float(output_transfer_seconds),
                },
            },
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
