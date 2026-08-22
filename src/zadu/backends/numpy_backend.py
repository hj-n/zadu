"""NumPy/SciPy/FAISS provider preserving ZADU's exact 0.5.0 behavior."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from zadu.engine.resources import NeighborRanking, ResourceKey, ResourceKind
from zadu.measures.utils import knn
from zadu.measures.utils import pairwise_dist as pdist

from .base import BuiltResource


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
        geodesic: bool,
    ) -> BuiltResource:
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            value = (
                self.pairwise_geodesic_distance_matrix(points)
                if geodesic
                else pdist.pairwise_distance_matrix(points)
            )
            return BuiltResource(value, "numpy")
        if key.kind is ResourceKind.NEIGHBOR_RANKING:
            indices, ranking = knn.knn_with_ranking(
                points, key.k, distance_matrix=distance_matrix
            )
            return BuiltResource(NeighborRanking(indices, ranking), "numpy")
        if distance_matrix is not None:
            return BuiltResource(
                knn.knn_from_distance_matrix(distance_matrix, key.k), "numpy"
            )
        return BuiltResource(knn.knn(points, key.k), "faiss")

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
