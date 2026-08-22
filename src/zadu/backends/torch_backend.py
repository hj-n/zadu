"""Optional PyTorch provider for exact dense Euclidean resources."""

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
class _TorchWorkspace:
    """One validated tensor input retained for a resource space."""

    source: npt.NDArray
    cast_points: npt.NDArray
    tensor: Any
    input_seconds: float
    cast_copy: bool
    device_copy: bool


class TorchResourceProvider(NumpyResourceProvider):
    """Route supported exact resources to PyTorch and fall back individually."""

    name = "torch"
    exact = True
    supports_embedding_batching = False

    def __init__(self, *, device: str, dtype: str) -> None:
        try:
            torch = import_module("torch")
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                "The PyTorch preview is optional. Install it with "
                "`pip install 'zadu[torch]'`."
            ) from exc

        if dtype not in {"float32", "float64"}:
            raise ValueError("PyTorch dtype must be 'float32' or 'float64'")
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("PyTorch device must be 'auto', 'cpu', 'mps', or 'cuda'")
        if device == "auto":
            if torch.backends.mps.is_available() and dtype == "float32":
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "The requested PyTorch MPS device is unavailable; choose "
                "device='cpu' explicitly"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "The requested PyTorch CUDA device is unavailable; choose "
                "device='cpu' explicitly"
            )
        if device == "mps" and dtype != "float32":
            raise ValueError("PyTorch MPS requires dtype='float32'")

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._device = torch.device(device)
        self._torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self._numpy_dtype = np.dtype(dtype)
        self._workspaces: dict[Space, _TorchWorkspace] = {}
        self._executed: set[str] = set()

    def fork(self) -> TorchResourceProvider:
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
            ResourceKind.DISTANCE_MATRIX,
            ResourceKind.CONDENSED_PAIRS,
        }:
            if working_memory_bytes is None:
                raise RuntimeError("PyTorch pairwise resources require a memory plan")
            return self._build_euclidean(
                key,
                points,
                working_memory_bytes=working_memory_bytes,
            )
        if not geodesic and key.kind in {
            ResourceKind.KNN,
            ResourceKind.STABLE_KNN,
            ResourceKind.NEIGHBOR_RANKING,
        }:
            if working_memory_bytes is None:
                raise RuntimeError("PyTorch neighbor resources require a memory plan")
            return self._build_neighbors(
                key,
                points,
                distance_matrix=distance_matrix,
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
                "PyTorch pairwise execution needs enough memory for one distance row"
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

        cold_seconds = 0.0
        warm_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        condensed_offset = 0
        with self._torch.inference_mode():
            for start in range(0, n_samples, block_rows):
                stop = min(start + block_rows, n_samples)
                started = perf_counter()
                distances = self._torch.cdist(
                    workspace.tensor[start:stop],
                    workspace.tensor,
                    p=2.0,
                    compute_mode="use_mm_for_euclid_dist",
                )
                self._synchronize()
                elapsed = perf_counter() - started
                if "pairwise" in self._executed:
                    warm_seconds += elapsed
                else:
                    cold_seconds += elapsed
                    self._executed.add("pairwise")

                output_started = perf_counter()
                block = distances.to("cpu").numpy()
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

        details: dict[str, Any] = {
            "algorithm": "torch_cdist_blockwise_euclidean",
            "device": self.device,
            "compute_dtype": self.dtype,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "torch_version": self._torch_version(),
            "input_zero_copy": self.device == "cpu" and not workspace.device_copy,
            "input_cast_copy": workspace.cast_copy,
            "input_reused": input_reused,
            "provider_fallback": False,
            "timings": {
                "input_transfer_seconds": float(
                    0.0 if input_reused else workspace.input_seconds
                ),
                "compile_and_first_execution_seconds": float(cold_seconds),
                "warm_execution_seconds": float(warm_seconds),
                "output_transfer_seconds": float(output_transfer_seconds),
            },
        }
        return BuiltResource(value, "torch", details)

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
        n_samples = points.shape[0]
        index_dtype = compact_index_dtype(n_samples)
        ranking_requested = key.kind is ResourceKind.NEIGHBOR_RANKING
        index_arrays = 4 if ranking_requested else 2
        bytes_per_row = n_samples * (
            4 * self._numpy_dtype.itemsize + index_arrays * np.dtype(np.int64).itemsize
        )
        if working_memory_bytes < bytes_per_row:
            raise MemoryError(
                "PyTorch neighbor execution needs enough memory for one distance row"
            )
        block_rows = max(
            1,
            min(n_samples, working_memory_bytes // bytes_per_row),
        )

        workspace = None
        input_reused = False
        input_seconds = 0.0
        input_cast_copy = False
        input_zero_copy = False
        distance_zero_copy = False
        distances_array = None
        if distance_matrix is None:
            workspace, input_reused = self._workspace(key.space, points)
            input_seconds = 0.0 if input_reused else workspace.input_seconds
            input_cast_copy = workspace.cast_copy
            input_zero_copy = self.device == "cpu" and not workspace.device_copy
            distance_source = "fused_blockwise_pairwise"
        else:
            raw_distances = np.asarray(distance_matrix)
            if raw_distances.ndim != 2 or raw_distances.shape != (
                n_samples,
                n_samples,
            ):
                raise ValueError(
                    "distance_matrix must be square and match the point count"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                distances_array = np.ascontiguousarray(
                    raw_distances,
                    dtype=self._numpy_dtype,
                )
            if not np.all(np.isfinite(distances_array)):
                raise ValueError("distance_matrix must contain only finite values")
            if np.any(distances_array < 0):
                raise ValueError("distance_matrix must be non-negative")
            input_cast_copy = not np.shares_memory(raw_distances, distances_array)
            distance_source = "shared_distance_matrix"

        indices_result = np.empty((n_samples, key.k), dtype=index_dtype)
        ranking_result = (
            np.empty((n_samples, n_samples), dtype=index_dtype)
            if ranking_requested
            else None
        )
        cold_seconds = 0.0
        warm_seconds = 0.0
        output_transfer_seconds = 0.0
        block_count = 0
        with self._torch.inference_mode():
            for start in range(0, n_samples, block_rows):
                stop = min(start + block_rows, n_samples)
                if workspace is not None:
                    execution_started = perf_counter()
                    sortable = self._torch.cdist(
                        workspace.tensor[start:stop],
                        workspace.tensor,
                        p=2.0,
                        compute_mode="use_mm_for_euclid_dist",
                    )
                else:
                    assert distances_array is not None
                    transfer_started = perf_counter()
                    sortable = self._torch.from_numpy(distances_array[start:stop]).to(
                        self._device, dtype=self._torch_dtype, copy=True
                    )
                    self._synchronize()
                    input_seconds += perf_counter() - transfer_started
                    execution_started = perf_counter()

                local_rows = self._torch.arange(stop - start, device=self._device)
                self_columns = self._torch.arange(start, stop, device=self._device)
                sortable[local_rows, self_columns] = -self._torch.inf
                order = self._torch.argsort(
                    sortable,
                    dim=1,
                    stable=True,
                )
                inverse = None
                if ranking_requested:
                    inverse = self._torch.empty_like(order)
                    positions = self._torch.arange(
                        n_samples,
                        device=self._device,
                    ).expand_as(order)
                    inverse.scatter_(1, order, positions)
                self._synchronize()
                elapsed = perf_counter() - execution_started
                if "neighbors" in self._executed:
                    warm_seconds += elapsed
                else:
                    cold_seconds += elapsed
                    self._executed.add("neighbors")

                output_started = perf_counter()
                indices_result[start:stop] = (
                    order[:, 1 : key.k + 1]
                    .to("cpu")
                    .numpy()
                    .astype(index_dtype, copy=False)
                )
                if inverse is not None:
                    assert ranking_result is not None
                    ranking_result[start:stop] = (
                        inverse.to("cpu").numpy().astype(index_dtype, copy=False)
                    )
                output_transfer_seconds += perf_counter() - output_started
                block_count += 1

        value: npt.NDArray | NeighborRanking
        if ranking_requested:
            assert ranking_result is not None
            value = NeighborRanking(indices_result, ranking_result)
            algorithm = "torch_blockwise_stable_full_ranking"
        else:
            value = indices_result
            algorithm = "torch_blockwise_stable_exact_topk"

        details: dict[str, Any] = {
            "algorithm": algorithm,
            "device": self.device,
            "compute_dtype": self.dtype,
            "index_dtype": index_dtype.name,
            "k": key.k,
            "block_rows": block_rows,
            "block_count": block_count,
            "working_bytes": block_rows * bytes_per_row,
            "torch_version": self._torch_version(),
            "distance_source": distance_source,
            "distance_zero_copy": distance_zero_copy,
            "input_zero_copy": input_zero_copy,
            "input_cast_copy": input_cast_copy,
            "input_reused": input_reused,
            "output_zero_copy": False,
            "self_exclusion": "forced_rank_zero_then_removed",
            "tie_break": "stable_column_index",
            "top_k_algorithm": "stable_full_order_prefix",
            "provider_fallback": False,
            "timings": {
                "input_transfer_seconds": float(input_seconds),
                "compile_and_first_execution_seconds": float(cold_seconds),
                "warm_execution_seconds": float(warm_seconds),
                "output_transfer_seconds": float(output_transfer_seconds),
            },
        }
        return BuiltResource(value, "torch", details)

    def _workspace(
        self,
        space: Space,
        points: npt.NDArray,
    ) -> tuple[_TorchWorkspace, bool]:
        existing = self._workspaces.get(space)
        if existing is not None and existing.source is points:
            return existing, True

        with np.errstate(over="ignore", invalid="ignore"):
            cast_points = np.ascontiguousarray(points, dtype=self._numpy_dtype)
        if not np.all(np.isfinite(cast_points)):
            raise OverflowError(
                f"Input values cannot be represented safely as PyTorch {self.dtype}"
            )
        started = perf_counter()
        cpu_tensor = self._torch.from_numpy(cast_points)
        tensor = cpu_tensor if self.device == "cpu" else cpu_tensor.to(self._device)
        self._synchronize()
        workspace = _TorchWorkspace(
            source=points,
            cast_points=cast_points,
            tensor=tensor,
            input_seconds=perf_counter() - started,
            cast_copy=not np.shares_memory(points, cast_points),
            device_copy=self.device != "cpu",
        )
        self._workspaces[space] = workspace
        return workspace, False

    def _synchronize(self) -> None:
        if self.device == "mps":
            self._torch.mps.synchronize()
        elif self.device == "cuda":  # pragma: no cover - requires CUDA hardware
            self._torch.cuda.synchronize(self._device)

    @staticmethod
    def _torch_version() -> str:
        try:
            return version("torch")
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
                "requested_provider": "torch",
                "provider_fallback": True,
                "fallback_reason": reason,
            },
        )
