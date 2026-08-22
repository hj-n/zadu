"""NumPy/SciPy/FAISS provider preserving ZADU's exact 0.5.0 behavior."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from zadu.engine.resources import (
    NeighborRanking,
    OrderedPairStatistics,
    PairOrder,
    PairStrategy,
    ResourceKey,
    ResourceKind,
)
from zadu.kernels import PairAccumulator
from zadu.measures.utils import knn
from zadu.measures.utils import pairwise_dist as pdist

from .base import BuiltResource

if TYPE_CHECKING:
    from zadu.engine.planner import PairExecutionPlan


class NumpyResourceProvider:
    name = "numpy"
    device = "cpu"
    exact = True

    def build(
        self,
        key: ResourceKey,
        points: npt.NDArray,
        *,
        distance_matrix: npt.NDArray | None,
        condensed_pairs: npt.NDArray | None,
        geodesic: bool,
    ) -> BuiltResource:
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            value = (
                self.pairwise_geodesic_distance_matrix(points)
                if geodesic
                else pdist.pairwise_distance_matrix(points)
            )
            return BuiltResource(value, "numpy")
        if key.kind is ResourceKind.CONDENSED_PAIRS:
            value = self.condensed_distances(points, geodesic=geodesic)
            return BuiltResource(value, "scipy")
        if key.kind is ResourceKind.PAIR_ORDER:
            distances = (
                self.condensed_distances(points, geodesic=geodesic)
                if distance_matrix is None and condensed_pairs is None
                else self._pair_distances(distance_matrix, condensed_pairs)
            )
            indices = np.argsort(distances)
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
            return BuiltResource(NeighborRanking(indices, ranking), "numpy")
        if key.kind is not ResourceKind.KNN:
            raise RuntimeError(f"Unsupported NumPy resource kind: {key.kind.value}")
        assert key.k is not None
        if distance_matrix is not None:
            return BuiltResource(
                knn.knn_from_distance_matrix(distance_matrix, key.k), "numpy"
            )
        return BuiltResource(knn.knn(points, key.k), "faiss")

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
