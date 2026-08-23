"""NumPy/SciPy/FAISS provider preserving ZADU's exact 0.5.0 behavior."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from zadu.engine.resources import (
    NeighborRanking,
    NeighborStatistics,
    OrderedPairStatistics,
    PairOrder,
    PairStrategy,
    RankComparisons,
    ResourceKey,
    ResourceKind,
    Space,
    TopographicProductStatistics,
    compact_index_dtype,
)
from zadu.kernels import PairAccumulator
from zadu.measures.utils import knn
from zadu.measures.utils import pairwise_dist as pdist
from zadu.measures.utils.vectorized import (
    rowwise_intersection_count,
    rowwise_membership,
)

from .base import BuiltResource

if TYPE_CHECKING:
    from zadu.engine.planner import (
        NeighborStatisticsExecutionPlan,
        PairExecutionPlan,
        RankComparisonExecutionPlan,
        TopographicExecutionPlan,
    )


class NumpyResourceProvider:
    name = "numpy"
    device = "cpu"
    dtype = "float64"
    exact = True
    supports_embedding_batching = False

    def fork(self) -> NumpyResourceProvider:
        """Return a stateless provider context for one embedding worker."""

        return type(self)()

    def invalidate(self, space: Space) -> None:
        """The stateless NumPy provider has no private resources to release."""

        del space

    def can_batch(self, key: ResourceKey) -> bool:
        del key
        return False

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
        """Default batch adapter used by providers for unsupported resources."""

        return [
            self.build(
                key,
                points,
                distance_matrix=distance_matrix,
                condensed_pairs=condensed,
                working_memory_bytes=working_memory_bytes,
                geodesic=geodesic,
            )
            for points, distance_matrix, condensed in zip(
                points_batch,
                distance_matrices,
                condensed_pairs,
                strict=True,
            )
        ]

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
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            value = (
                self.pairwise_geodesic_distance_matrix(points)
                if geodesic
                else pdist.pairwise_distance_matrix(points)
            )
            return BuiltResource(value, "numpy")
        if key.kind is ResourceKind.DENSITY:
            if distance_matrix is None or key.parameter is None:
                raise RuntimeError("Density requires a distance matrix and sigma")
            value = pdist.distance_matrix_to_density(
                distance_matrix,
                key.parameter,
            )
            return BuiltResource(
                value,
                "numpy",
                {"sigma": key.parameter, "source": "distance_matrix"},
            )
        if key.kind is ResourceKind.CONDENSED_PAIRS:
            value = self.condensed_distances(points, geodesic=geodesic)
            return BuiltResource(value, "scipy")
        if key.kind is ResourceKind.PAIR_ORDER:
            distances = (
                self.condensed_distances(points, geodesic=geodesic)
                if distance_matrix is None and condensed_pairs is None
                else self._pair_distances(distance_matrix, condensed_pairs)
            )
            index_dtype = compact_index_dtype(distances.size)
            indices = np.argsort(distances).astype(index_dtype, copy=False)
            value = PairOrder(
                indices=indices,
                sorted_ranks=rankdata(distances)[indices],
                min_distance=float(np.min(distances)),
                max_distance=float(np.max(distances)),
            )
            return BuiltResource(
                value,
                "numpy",
                {
                    "pair_count": distances.size,
                    "ordering": "numpy.argsort",
                    "ranks": "scipy.stats.rankdata",
                },
            )
        if key.kind is ResourceKind.NEIGHBOR_RANKING:
            assert key.k is not None
            indices, ranking = knn.knn_with_ranking(
                points, key.k, distance_matrix=distance_matrix
            )
            index_dtype = compact_index_dtype(points.shape[0])
            value = NeighborRanking(
                indices.astype(index_dtype, copy=False),
                ranking.astype(index_dtype, copy=False),
            )
            return BuiltResource(
                value,
                "numpy",
                {"index_dtype": index_dtype.name},
            )
        if key.kind is ResourceKind.STABLE_KNN:
            assert key.k is not None
            if working_memory_bytes is None:
                raise RuntimeError("Stable kNN requires a working-memory plan")
            value, block_count, block_rows = self.stable_knn(
                points,
                key.k,
                working_memory_bytes=working_memory_bytes,
                geodesic=geodesic,
                distance_matrix=distance_matrix,
            )
            return BuiltResource(
                value,
                "scipy",
                {
                    "algorithm": "blockwise_stable_argpartition",
                    "block_count": block_count,
                    "block_rows": block_rows,
                    "working_bytes": working_memory_bytes,
                },
            )
        if key.kind is not ResourceKind.KNN:
            raise RuntimeError(f"Unsupported NumPy resource kind: {key.kind.value}")
        assert key.k is not None
        if distance_matrix is not None:
            value = knn.knn_from_distance_matrix(distance_matrix, key.k)
            return BuiltResource(
                value.astype(compact_index_dtype(points.shape[0]), copy=False),
                "numpy",
            )
        value = knn.knn(points, key.k)
        return BuiltResource(
            value.astype(compact_index_dtype(points.shape[0]), copy=False),
            "faiss",
        )

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
    ) -> BuiltResource:
        accumulator = PairAccumulator(
            needs_stress="stress" in plan.metric_ids,
            needs_scale="scale_normalized_stress" in plan.metric_ids,
            needs_pearson="pearson_r" in plan.metric_ids,
        )
        if plan.strategy is PairStrategy.DENSE:
            if orig_distance_matrix is None or emb_distance_matrix is None:
                raise RuntimeError(
                    "Dense pair statistics require two distance matrices"
                )
            assert plan.block_rows is not None
            blocks = self._matrix_pair_blocks(
                orig_distance_matrix,
                emb_distance_matrix,
                plan.block_rows,
            )
        elif plan.strategy is PairStrategy.CONDENSED:
            if orig_condensed is None or emb_condensed is None:
                raise RuntimeError(
                    "Condensed pair statistics require two condensed arrays"
                )
            assert plan.chunk_pairs is not None
            blocks = self._condensed_pair_blocks(
                orig_condensed,
                emb_condensed,
                plan.chunk_pairs,
            )
        else:
            assert plan.block_rows is not None
            blocks = self._stream_pair_blocks(
                orig,
                emb,
                plan.block_rows,
                geodesic=geodesic,
            )

        for orig_distances, emb_distances in blocks:
            accumulator.update(orig_distances, emb_distances)
        statistics = accumulator.finalize(
            strategy=plan.strategy,
            block_rows=plan.block_rows,
            chunk_pairs=plan.chunk_pairs,
        )
        if statistics.count != plan.pair_count:
            raise RuntimeError(
                "Pair provider produced an unexpected number of distances "
                f"({statistics.count} != {plan.pair_count})"
            )
        return BuiltResource(
            statistics,
            "numpy",
            {
                "strategy": plan.strategy.value,
                "pair_count": statistics.count,
                "block_count": statistics.block_count,
                "block_rows": plan.block_rows,
                "chunk_pairs": plan.chunk_pairs,
                "working_bytes": plan.working_bytes,
                "fused_metrics": list(plan.metric_ids),
            },
        )

    def build_ordered_pair_statistics(
        self,
        plan: PairExecutionPlan,
        pair_order: PairOrder,
        *,
        emb_distance_matrix: npt.NDArray | None,
        emb_condensed: npt.NDArray | None,
    ) -> BuiltResource:
        emb_distances = self._pair_distances(
            emb_distance_matrix,
            emb_condensed,
        )
        if (
            emb_distances.size != plan.pair_count
            or pair_order.indices.size != plan.pair_count
            or pair_order.sorted_ranks.size != plan.pair_count
        ):
            raise RuntimeError(
                "Ordered pair provider produced an unexpected number of distances"
            )

        spearman_rho = None
        if "spearman_rho" in plan.ordered_metric_ids:
            if pair_order.min_distance == pair_order.max_distance or float(
                np.min(emb_distances)
            ) == float(np.max(emb_distances)):
                raise ValueError(
                    "Spearman correlation is undefined for constant distances"
                )
            embedded_ranks = rankdata(emb_distances)
            spearman_rho = float(
                np.corrcoef(
                    pair_order.sorted_ranks,
                    embedded_ranks[pair_order.indices],
                )[0, 1]
            )

        non_metric_stress = None
        if "non_metric_stress" in plan.ordered_metric_ids:
            if pair_order.max_distance <= 0 or float(np.max(emb_distances)) <= 0:
                raise ValueError(
                    "Non-metric stress is undefined when all pairwise distances "
                    "are zero"
                )
            emb_sorted = emb_distances[pair_order.indices]
            isotonic = IsotonicRegression().fit(
                pair_order.sorted_ranks,
                emb_sorted,
            )
            d_hat = isotonic.predict(pair_order.sorted_ranks)
            residual = emb_sorted - d_hat
            raw_stress = float(np.vdot(residual, residual))
            normalization_factor = float(np.vdot(emb_sorted, emb_sorted))
            non_metric_stress = math.sqrt(raw_stress / normalization_factor)

        statistics = OrderedPairStatistics(
            spearman_rho=spearman_rho,
            non_metric_stress=non_metric_stress,
            strategy=plan.strategy,
            pair_count=plan.pair_count,
        )
        return BuiltResource(
            statistics,
            "numpy",
            {
                "strategy": plan.strategy.value,
                "pair_count": plan.pair_count,
                "working_bytes": plan.working_bytes,
                "fused_metrics": list(plan.ordered_metric_ids),
                "ordering_reused": True,
            },
        )

    def build_topographic_product_statistics(
        self,
        plan: TopographicExecutionPlan,
        orig: npt.NDArray,
        emb: npt.NDArray,
        *,
        orig_knn: npt.NDArray,
        emb_knn: npt.NDArray,
    ) -> BuiltResource:
        orig_indices = np.asarray(orig_knn)[:, : plan.k]
        emb_indices = np.asarray(emb_knn)[:, : plan.k]
        expected_shape = (orig.shape[0], plan.k)
        if orig_indices.shape != expected_shape or emb_indices.shape != expected_shape:
            raise RuntimeError(
                "Topographic Product requires matching planned neighbor tables"
            )

        maximum_dimension = max(orig.shape[1], emb.shape[1])
        bytes_per_row = 16 * plan.k * (maximum_dimension + 5)
        if plan.work_budget_bytes < bytes_per_row:
            raise MemoryError(
                "Topographic Product selected distances exceed the planned "
                "working-memory budget"
            )
        block_rows = min(
            plan.block_rows,
            max(1, plan.work_budget_bytes // bytes_per_row),
        )
        totals = np.full(plan.k, np.nan, dtype=np.float64)
        for requested_k in plan.requested_ks:
            totals[requested_k - 1] = 0.0
        prefix_lengths = np.arange(1, plan.k + 1, dtype=np.float64)
        block_count = 0
        for start in range(0, orig.shape[0], block_rows):
            stop = min(start + block_rows, orig.shape[0])
            orig_neighbors = orig_indices[start:stop]
            emb_neighbors = emb_indices[start:stop]

            orig_to_emb = self.selected_distances(
                orig,
                emb_neighbors,
                start,
                stop,
                geodesic=plan.geodesic,
            )
            orig_to_orig = self.selected_distances(
                orig,
                orig_neighbors,
                start,
                stop,
                geodesic=plan.geodesic,
            )
            if np.any(orig_to_orig <= 0):
                raise ValueError(
                    "Topographic Product is undefined for zero-distance "
                    "original-space neighbors"
                )
            orig_to_emb /= orig_to_orig

            emb_to_emb = self.selected_distances(
                emb,
                emb_neighbors,
                start,
                stop,
                geodesic=False,
            )
            emb_to_orig = self.selected_distances(
                emb,
                orig_neighbors,
                start,
                stop,
                geodesic=False,
            )
            if np.any(emb_to_orig <= 0):
                raise ValueError(
                    "Topographic Product is undefined for zero-distance "
                    "embedded-space neighbors"
                )
            emb_to_emb /= emb_to_orig
            orig_to_emb *= emb_to_emb
            if np.any(orig_to_emb <= 0) or not np.all(np.isfinite(orig_to_emb)):
                raise ValueError(
                    "Topographic Product is undefined for coincident points"
                )

            np.log(orig_to_emb, out=orig_to_emb)
            np.cumsum(orig_to_emb, axis=1, out=orig_to_emb)
            orig_to_emb /= 2 * prefix_lengths
            for requested_k in plan.requested_ks:
                totals[requested_k - 1] += float(np.sum(orig_to_emb[:, :requested_k]))
            block_count += 1

        for requested_k in plan.requested_ks:
            totals[requested_k - 1] /= orig.shape[0] * requested_k
        statistics = TopographicProductStatistics(
            scores=totals,
            block_count=block_count,
            block_rows=block_rows,
        )
        return BuiltResource(
            statistics,
            "numpy",
            {
                "algorithm": "blockwise_selected_distances",
                "k": plan.k,
                "requested_ks": list(plan.requested_ks),
                "block_count": block_count,
                "block_rows": block_rows,
                "working_bytes": block_rows * bytes_per_row,
                "fused_metrics": list(plan.metric_ids),
            },
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
        """Build exact cross-space ranks without retaining full inverse rankings."""

        n_samples = orig.shape[0]
        bytes_per_sort_row = n_samples * 24
        largest_membership_k = max(plan.membership_ks, default=0)
        bytes_per_row = max(1, bytes_per_sort_row, largest_membership_k**2)
        if plan.work_budget_bytes < bytes_per_row:
            raise MemoryError(
                "Selected-rank sorting requires enough working memory for one row"
            )

        index_dtype = compact_index_dtype(n_samples)
        orig_indices = np.asarray(orig_knn)[:, : plan.k]
        emb_indices = np.empty((n_samples, plan.k), dtype=index_dtype)
        orig_ranks_of_emb = np.empty((n_samples, plan.k), dtype=index_dtype)
        emb_ranks_of_orig = np.empty((n_samples, plan.k), dtype=index_dtype)
        positions = np.arange(n_samples, dtype=np.intp)
        block_count = 0

        for start in range(0, n_samples, plan.block_rows):
            stop = min(start + plan.block_rows, n_samples)
            local_rows = np.arange(stop - start)[:, None]
            global_rows = np.arange(start, stop)

            emb_distances = self._rank_distance_block(
                emb,
                emb_distance_matrix,
                start,
                stop,
                geodesic=False,
            )
            emb_distances[np.arange(stop - start), global_rows] = -np.inf
            emb_order = np.argsort(emb_distances, axis=1, kind="stable")
            emb_indices[start:stop] = emb_order[:, 1 : plan.k + 1]
            del emb_distances

            inverse = np.empty_like(emb_order)
            inverse[local_rows, emb_order] = positions
            emb_ranks_of_orig[start:stop] = inverse[
                local_rows, orig_indices[start:stop]
            ]
            del emb_order, inverse

            orig_distances = self._rank_distance_block(
                orig,
                orig_distance_matrix,
                start,
                stop,
                geodesic=plan.geodesic,
            )
            orig_distances[np.arange(stop - start), global_rows] = -np.inf
            selected_targets = emb_indices[start:stop]
            columns = positions[None, :]
            for rank_column in range(plan.k):
                target_indices = selected_targets[:, rank_column]
                target_distances = orig_distances[
                    np.arange(stop - start), target_indices
                ][:, None]
                selected_ranks = np.count_nonzero(
                    orig_distances < target_distances,
                    axis=1,
                )
                tied_before_target = columns < target_indices[:, None]
                np.logical_and(
                    orig_distances == target_distances,
                    tied_before_target,
                    out=tied_before_target,
                )
                selected_ranks += np.count_nonzero(tied_before_target, axis=1)
                orig_ranks_of_emb[start:stop, rank_column] = selected_ranks
            block_count += 1

        emb_in_orig = {}
        orig_in_emb = {}
        for k in plan.membership_ks:
            emb_in_orig[k] = rowwise_membership(
                emb_indices[:, :k],
                orig_indices[:, :k],
                max_block_bytes=plan.work_budget_bytes,
            )
            orig_in_emb[k] = rowwise_membership(
                orig_indices[:, :k],
                emb_indices[:, :k],
                max_block_bytes=plan.work_budget_bytes,
            )
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
            "numpy",
            {
                "algorithm": "blockwise_selected_ranks",
                "k": plan.k,
                "requested_ks": list(plan.requested_ks),
                "membership_ks": list(plan.membership_ks),
                "block_count": block_count,
                "block_rows": plan.block_rows,
                "work_budget_bytes": plan.work_budget_bytes,
                "working_bytes": plan.working_bytes,
                "index_dtype": index_dtype.name,
                "original_neighbor_source": "cached_stable_knn",
                "original_rank_algorithm": "stable_distance_count",
                "embedded_rank_algorithm": "stable_sort_inverse_scatter",
                "original_distance_source": (
                    "shared_distance_matrix"
                    if orig_distance_matrix is not None
                    else (
                        "blockwise_geodesic"
                        if plan.geodesic
                        else "blockwise_scipy_cdist"
                    )
                ),
                "embedded_distance_source": (
                    "shared_distance_matrix"
                    if emb_distance_matrix is not None
                    else "blockwise_scipy_cdist"
                ),
                "fused_metrics": list(plan.metric_ids),
            },
        )

    @classmethod
    def _rank_distance_block(
        cls,
        points: npt.NDArray,
        distance_matrix: npt.NDArray | None,
        start: int,
        stop: int,
        *,
        geodesic: bool,
    ) -> npt.NDArray:
        if distance_matrix is not None:
            return np.array(distance_matrix[start:stop], copy=True)
        if geodesic:
            if points.shape[1] < 2:
                raise ValueError(
                    "geodesic=True requires orig[:, 0] = longitude and "
                    "orig[:, 1] = latitude in radians"
                )
            return cls.geodesic_distance_block(points[start:stop], points)
        return cdist(points[start:stop], points)

    def build_neighbor_statistics(
        self,
        plan: NeighborStatisticsExecutionPlan,
        *,
        orig_knn: npt.NDArray,
        emb_knn: npt.NDArray,
    ) -> BuiltResource:
        """Compute shared exact LCMC and neighbor-dissimilarity statistics."""

        original_value = (
            orig_knn.indices if isinstance(orig_knn, NeighborRanking) else orig_knn
        )
        embedded_value = (
            emb_knn.indices if isinstance(emb_knn, NeighborRanking) else emb_knn
        )
        original = np.asarray(original_value)[:, : plan.k]
        embedded = np.asarray(embedded_value)[:, : plan.k]
        n_samples = original.shape[0]
        local_lcmc = {}
        for k in plan.lcmc_ks:
            intersections = rowwise_intersection_count(
                original[:, :k],
                embedded[:, :k],
                max_block_bytes=plan.work_budget_bytes,
            )
            local_lcmc[k] = (intersections - (k * k) / (n_samples - 1)) / k

        neighbor_dissimilarity = {}
        nd_block_count = 0
        for k in plan.nd_ks:
            orig_graph = self._symmetric_neighbor_graph(original[:, :k])
            emb_graph = self._symmetric_neighbor_graph(embedded[:, :k])
            orig_transpose = orig_graph.T.tocsr()
            emb_transpose = emb_graph.T.tocsr()
            positive_squared = 0.0
            negative_squared = 0.0
            for start in range(0, n_samples, plan.block_rows):
                stop = min(start + plan.block_rows, n_samples)
                difference = (
                    orig_graph[start:stop] @ orig_transpose
                    - emb_graph[start:stop] @ emb_transpose
                ).tocsr()
                row_indices = np.repeat(
                    np.arange(stop - start),
                    np.diff(difference.indptr),
                )
                off_diagonal = difference.indices != start + row_indices
                values = difference.data[off_diagonal].astype(np.float64) / k
                positive_values = values[values > 0]
                negative_values = values[values < 0]
                positive_squared += float(np.vdot(positive_values, positive_values))
                negative_squared += float(np.vdot(negative_values, negative_values))
                nd_block_count += 1
            neighbor_dissimilarity[k] = max(
                math.sqrt(positive_squared),
                math.sqrt(negative_squared),
            )

        value = NeighborStatistics(
            local_lcmc=local_lcmc,
            neighbor_dissimilarity=neighbor_dissimilarity,
        )
        return BuiltResource(
            value,
            "scipy",
            {
                "algorithm": "fused_neighbor_statistics",
                "k": plan.k,
                "lcmc_ks": list(plan.lcmc_ks),
                "neighbor_dissimilarity_ks": list(plan.nd_ks),
                "block_count": nd_block_count,
                "block_rows": plan.block_rows,
                "working_bytes": plan.working_bytes,
                "fused_metrics": list(plan.metric_ids),
            },
        )

    @staticmethod
    def _symmetric_neighbor_graph(indices: npt.NDArray) -> csr_matrix:
        n_samples, k = indices.shape
        rows = np.repeat(np.arange(n_samples), k)
        directed = csr_matrix(
            (
                np.ones(rows.size, dtype=np.int32),
                (rows, indices.reshape(-1)),
            ),
            shape=(n_samples, n_samples),
        )
        return ((directed + directed.T) > 0).astype(np.int32).tocsr()

    @classmethod
    def stable_knn(
        cls,
        points: npt.NDArray,
        k: int,
        *,
        working_memory_bytes: int,
        geodesic: bool,
        distance_matrix: npt.NDArray | None = None,
    ) -> tuple[npt.NDArray, int, int]:
        n_samples = points.shape[0]
        bytes_per_row = n_samples * 32
        if working_memory_bytes < bytes_per_row:
            raise MemoryError("Stable kNN requires at least one distance row")
        block_rows = max(
            1,
            min(n_samples, working_memory_bytes // bytes_per_row),
        )
        result = np.empty(
            (n_samples, k),
            dtype=compact_index_dtype(n_samples),
        )
        block_count = 0
        for start in range(0, n_samples, block_rows):
            stop = min(start + block_rows, n_samples)
            if distance_matrix is not None:
                distances = np.array(distance_matrix[start:stop], copy=True)
            else:
                distances = (
                    cls.geodesic_distance_block(points[start:stop], points)
                    if geodesic
                    else cdist(points[start:stop], points)
                )
            distances[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            candidates = np.argpartition(distances, k, axis=1)[:, : k + 1]
            candidate_distances = np.take_along_axis(
                distances,
                candidates,
                axis=1,
            )
            candidate_order = np.lexsort(
                (candidates, candidate_distances),
                axis=1,
            )
            candidates = np.take_along_axis(candidates, candidate_order, axis=1)
            candidate_distances = np.take_along_axis(
                candidate_distances,
                candidate_order,
                axis=1,
            )
            thresholds = candidate_distances[:, -1]
            tie_counts = np.count_nonzero(
                distances == thresholds[:, None],
                axis=1,
            )
            for row in np.flatnonzero(tie_counts > 1):
                below_threshold = candidates[
                    row,
                    candidate_distances[row] < thresholds[row],
                ]
                needed = k + 1 - below_threshold.size
                stable_boundary = np.flatnonzero(distances[row] == thresholds[row])[
                    :needed
                ]
                candidates[row] = np.concatenate((below_threshold, stable_boundary))
            result[start:stop] = candidates[:, 1 : k + 1]
            block_count += 1
        return result, block_count, block_rows

    @classmethod
    def selected_distances(
        cls,
        points: npt.NDArray,
        indices: npt.NDArray,
        start: int,
        stop: int,
        *,
        geodesic: bool,
    ) -> npt.NDArray:
        centers = np.asarray(points[start:stop], dtype=np.float64)
        selected = np.asarray(points[indices], dtype=np.float64)
        if geodesic:
            if points.shape[1] < 2:
                raise ValueError(
                    "geodesic=True requires orig[:, 0] = longitude and "
                    "orig[:, 1] = latitude in radians"
                )
            center_longitude = centers[:, None, 0]
            center_latitude = centers[:, None, 1]
            selected_longitude = selected[:, :, 0]
            selected_latitude = selected[:, :, 1]
            cosine = np.sin(center_latitude) * np.sin(selected_latitude) + np.cos(
                center_latitude
            ) * np.cos(selected_latitude) * np.cos(
                np.abs(selected_longitude - center_longitude)
            )
            return np.arccos(np.clip(cosine, -1.0, 1.0))
        selected -= centers[:, None, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", selected, selected))

    @staticmethod
    def _pair_distances(
        distance_matrix: npt.NDArray | None,
        condensed_pairs: npt.NDArray | None,
    ) -> npt.NDArray:
        if condensed_pairs is not None:
            return np.asarray(condensed_pairs)
        if distance_matrix is None:
            raise RuntimeError(
                "Ordered pair resources require a dense or condensed distance source"
            )
        matrix = np.asarray(distance_matrix)
        return matrix[np.triu_indices(matrix.shape[0], k=1)]

    @staticmethod
    def _matrix_pair_blocks(
        orig_distance_matrix: npt.NDArray,
        emb_distance_matrix: npt.NDArray,
        block_rows: int,
    ) -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        n_samples = orig_distance_matrix.shape[0]
        for left_start in range(0, n_samples, block_rows):
            left_stop = min(left_start + block_rows, n_samples)
            for right_start in range(left_start, n_samples, block_rows):
                right_stop = min(right_start + block_rows, n_samples)
                if left_start == right_start:
                    for row in range(left_start, left_stop):
                        if row + 1 < right_stop:
                            yield (
                                orig_distance_matrix[row, row + 1 : right_stop],
                                emb_distance_matrix[row, row + 1 : right_stop],
                            )
                else:
                    yield (
                        orig_distance_matrix[
                            left_start:left_stop, right_start:right_stop
                        ],
                        emb_distance_matrix[
                            left_start:left_stop, right_start:right_stop
                        ],
                    )

    @staticmethod
    def _condensed_pair_blocks(
        orig_condensed: npt.NDArray,
        emb_condensed: npt.NDArray,
        chunk_pairs: int,
    ) -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        for start in range(0, orig_condensed.size, chunk_pairs):
            stop = min(start + chunk_pairs, orig_condensed.size)
            yield orig_condensed[start:stop], emb_condensed[start:stop]

    @classmethod
    def _stream_pair_blocks(
        cls,
        orig: npt.NDArray,
        emb: npt.NDArray,
        block_rows: int,
        *,
        geodesic: bool,
    ) -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        n_samples = orig.shape[0]
        for left_start in range(0, n_samples, block_rows):
            left_stop = min(left_start + block_rows, n_samples)
            orig_left = orig[left_start:left_stop]
            emb_left = emb[left_start:left_stop]
            for right_start in range(left_start, n_samples, block_rows):
                right_stop = min(right_start + block_rows, n_samples)
                if left_start == right_start:
                    orig_distances = cls.condensed_distances(
                        orig_left, geodesic=geodesic
                    )
                    emb_distances = scipy_pdist(emb_left)
                else:
                    orig_right = orig[right_start:right_stop]
                    emb_right = emb[right_start:right_stop]
                    orig_distances = (
                        cls.geodesic_distance_block(orig_left, orig_right)
                        if geodesic
                        else cdist(orig_left, orig_right)
                    )
                    emb_distances = cdist(emb_left, emb_right)
                yield orig_distances, emb_distances

    @classmethod
    def condensed_distances(
        cls,
        points: npt.NDArray,
        *,
        geodesic: bool,
    ) -> npt.NDArray:
        if not geodesic:
            return scipy_pdist(points)
        if points.shape[1] < 2:
            raise ValueError(
                "geodesic=True requires orig[:, 0] = longitude and "
                "orig[:, 1] = latitude in radians"
            )
        result = np.empty(points.shape[0] * (points.shape[0] - 1) // 2)
        offset = 0
        for left in range(points.shape[0] - 1):
            for right in range(left + 1, points.shape[0]):
                result[offset] = cls.geodesic_distance(
                    points[left, 1],
                    points[left, 0],
                    points[right, 1],
                    points[right, 0],
                )
                offset += 1
        return result

    @classmethod
    def geodesic_distance_block(
        cls,
        left_points: npt.NDArray,
        right_points: npt.NDArray,
    ) -> npt.NDArray:
        result = np.empty((left_points.shape[0], right_points.shape[0]))
        for left in range(left_points.shape[0]):
            for right in range(right_points.shape[0]):
                result[left, right] = cls.geodesic_distance(
                    left_points[left, 1],
                    left_points[left, 0],
                    right_points[right, 1],
                    right_points[right, 0],
                )
        return result

    @staticmethod
    def geodesic_distance(phi1, lambda1, phi2, lambda2) -> float:
        cosine = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(
            phi2
        ) * math.cos(abs(lambda2 - lambda1))
        return math.acos(float(np.clip(cosine, -1.0, 1.0)))

    @classmethod
    def pairwise_geodesic_distance_matrix(cls, points: npt.NDArray) -> npt.NDArray:
        if points.shape[1] < 2:
            raise ValueError(
                "geodesic=True requires orig[:, 0] = longitude and "
                "orig[:, 1] = latitude in radians"
            )
        data_len = len(points)
        distance_matrix = np.zeros((data_len, data_len))
        for left in range(data_len):
            for right in range(left + 1, data_len):
                distance_matrix[left, right] = distance_matrix[right, left] = (
                    cls.geodesic_distance(
                        points[left, 1],
                        points[left, 0],
                        points[right, 1],
                        points[right, 0],
                    )
                )
        return distance_matrix
