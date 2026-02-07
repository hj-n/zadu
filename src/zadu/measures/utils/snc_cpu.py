from __future__ import annotations

from collections import deque

import hdbscan
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .pairwise_dist import pairwise_distance_matrix


class SNCCPU:
    """CPU implementation of Steadiness & Cohesiveness (SNC).

    This mirrors the original `snc` package behavior while removing the external
    runtime dependency from zadu.
    """

    def __init__(
        self,
        raw: npt.NDArray,
        emb: npt.NDArray,
        iteration: int = 150,
        walk_num_ratio: float = 0.3,
        alpha: float = 0.1,
        k: int | None = None,
        cluster_strategy: str = "dbscan",
    ) -> None:
        self.raw = np.asarray(raw, dtype=np.float64)
        self.emb = np.asarray(emb, dtype=np.float64)

        if self.raw.shape[0] != self.emb.shape[0]:
            raise ValueError("raw and emb must have the same number of rows")
        if self.raw.shape[0] < 2:
            raise ValueError("At least 2 points are required for SNC")

        self.n = self.raw.shape[0]
        self.iteration = int(iteration)
        self.walk_num = max(1, int(self.n * walk_num_ratio))
        self.alpha = float(alpha)
        self.k = int(np.sqrt(self.n)) if k is None else int(k)
        self.cluster_strategy = cluster_strategy

        if self.k < 1 or self.k >= self.n:
            raise ValueError(f"k must satisfy 1 <= k < n (n={self.n}), got k={self.k}")

        self.max_compress: float | None = None
        self.min_compress: float | None = None
        self.max_stretch: float | None = None
        self.min_stretch: float | None = None

        self.raw_knn: npt.NDArray | None = None
        self.emb_knn: npt.NDArray | None = None
        self.raw_snn: npt.NDArray | None = None
        self.emb_snn: npt.NDArray | None = None
        self.raw_dist_matrix: npt.NDArray | None = None
        self.emb_dist_matrix: npt.NDArray | None = None

        self.record = False
        self.stead_log = [dict() for _ in range(self.n)]
        self.cohev_log = [dict() for _ in range(self.n)]

        self.stead_score: float | None = None
        self.cohev_score: float | None = None

    def fit(self, record_vis_info: bool = False) -> None:
        self.record = record_vis_info

        raw_dist_matrix = pairwise_distance_matrix(self.raw)
        emb_dist_matrix = pairwise_distance_matrix(self.emb)

        raw_max = float(np.max(raw_dist_matrix))
        emb_max = float(np.max(emb_dist_matrix))

        if raw_max > 0:
            raw_dist_matrix = raw_dist_matrix / raw_max
        if emb_max > 0:
            emb_dist_matrix = emb_dist_matrix / emb_max

        self.raw_knn = self._knn_info(raw_dist_matrix, self.k)
        self.emb_knn = self._knn_info(emb_dist_matrix, self.k)

        self.raw_snn = self._weighted_snn(self.raw_knn, self.k)
        self.emb_snn = self._weighted_snn(self.emb_knn, self.k)

        raw_snn_max = float(np.max(self.raw_snn))
        emb_snn_max = float(np.max(self.emb_snn))

        if raw_snn_max > 0:
            self.raw_snn = self.raw_snn / raw_snn_max
        if emb_snn_max > 0:
            self.emb_snn = self.emb_snn / emb_snn_max

        self.raw_dist_matrix = 1.0 / (self.raw_snn + self.alpha)
        self.emb_dist_matrix = 1.0 / (self.emb_snn + self.alpha)

        dissim = self.raw_dist_matrix - self.emb_dist_matrix
        dissim_max = float(np.max(dissim))
        dissim_min = float(np.min(dissim))

        self.max_compress = dissim_max if dissim_max > 0 else 0.0
        self.min_compress = dissim_min if dissim_min > 0 else 0.0
        self.max_stretch = -dissim_min if dissim_min < 0 else 0.0
        self.min_stretch = -dissim_max if dissim_max < 0 else 0.0

    def steadiness(self) -> float:
        if self.max_compress is None or self.min_compress is None:
            raise RuntimeError("Call fit() before computing steadiness")
        self.stead_score = self._measure("steadiness", self.max_compress, self.min_compress)
        return self.stead_score

    def cohesiveness(self) -> float:
        if self.max_stretch is None or self.min_stretch is None:
            raise RuntimeError("Call fit() before computing cohesiveness")
        self.cohev_score = self._measure("cohesiveness", self.max_stretch, self.min_stretch)
        return self.cohev_score

    def local_scores(self) -> tuple[npt.NDArray, npt.NDArray]:
        if not self.record:
            raise RuntimeError("record_vis_info=False; local scores are unavailable")
        if self.stead_score is None or self.cohev_score is None:
            raise RuntimeError("Compute steadiness() and cohesiveness() first")

        self._finalize_logs()

        vertices_stead = self._vertices_info(self.stead_log)
        vertices_cohev = self._vertices_info(self.cohev_log)

        stead_ratio = max((1 - self.stead_score) * 2, 1)
        cohev_ratio = max((1 - self.cohev_score) * 2, 1)

        false_val = vertices_stead * cohev_ratio
        missing_val = vertices_cohev * stead_ratio

        local_stead = 1 - false_val
        local_cohev = 1 - missing_val
        return local_stead, local_cohev

    @staticmethod
    def _knn_info(dist_matrix: npt.NDArray, k: int) -> npt.NDArray:
        nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed")
        nbrs.fit(dist_matrix)
        return nbrs.kneighbors(return_distance=False)

    @staticmethod
    def _weighted_snn(knn_info: npt.NDArray, k: int) -> npt.NDArray:
        n = knn_info.shape[0]
        rows = np.repeat(np.arange(n), k)
        cols = knn_info.reshape(-1)
        # Keep the original weighting convention: k+1, k, ..., 2.
        vals = np.tile(np.arange(k + 1, 1, -1, dtype=np.float64), n)
        knn_graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
        return (knn_graph @ knn_graph.T).toarray()

    def _extract_cluster(self, mode: str, walk_num: int) -> npt.NDArray:
        if mode == "steadiness":
            knn_info = self.emb_knn
            snn_matrix = self.emb_snn
        else:
            knn_info = self.raw_knn
            snn_matrix = self.raw_snn

        if knn_info is None or snn_matrix is None:
            raise RuntimeError("fit() must run before cluster extraction")

        seed_idx = int(np.random.randint(self.n))
        cluster_member: set[int] = {seed_idx}
        current_queue: deque[int] = deque([seed_idx])

        visit_num = 0
        while visit_num < walk_num:
            if not current_queue:
                break
            i = current_queue.popleft()
            for j in knn_info[i]:
                probability = 1 - snn_matrix[i, j]
                if np.random.rand() > probability:
                    jj = int(j)
                    current_queue.append(jj)
                    cluster_member.add(jj)
                    visit_num += 1

        return np.array(list(cluster_member), dtype=np.int64)

    def _clustering(self, mode: str, indices: npt.NDArray) -> npt.NDArray:
        if mode == "steadiness":
            dist_matrix = self.raw_dist_matrix
            data = self.raw
        else:
            dist_matrix = self.emb_dist_matrix
            data = self.emb

        if dist_matrix is None:
            raise RuntimeError("fit() must run before clustering")

        if self.cluster_strategy == "dbscan":
            cluster_dist = dist_matrix[np.ix_(indices, indices)].copy()
            np.fill_diagonal(cluster_dist, 0)
            clusterer = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True)
            clusterer.fit(cluster_dist)
            return clusterer.labels_

        if self.cluster_strategy == "kmeans":
            k_val = max(2, int(np.sqrt(len(indices))))
            clusterer = KMeans(n_clusters=min(k_val, len(indices)), n_init="auto")
            clusterer.fit(data[indices])
            return clusterer.labels_

        if self.cluster_strategy.endswith("-means"):
            k_val = int(self.cluster_strategy.split("-")[0])
            clusterer = KMeans(n_clusters=min(k_val, len(indices)), n_init="auto")
            clusterer.fit(data[indices])
            return clusterer.labels_

        raise ValueError(f"Unsupported clustering_strategy: {self.cluster_strategy}")

    @staticmethod
    def _separate_cluster_labels(
        cluster_indices: npt.NDArray, clustering_result: npt.NDArray
    ) -> list[list[int]]:
        cluster_num = int(np.max(clustering_result)) + 1
        clusters = [[] for _ in range(cluster_num)]
        for idx, cluster_idx in enumerate(clustering_result):
            point_idx = int(cluster_indices[idx])
            if cluster_idx >= 0:
                clusters[int(cluster_idx)].append(point_idx)
            else:
                clusters.append([point_idx])
        return clusters

    def _compute_distance(
        self, cluster_a: npt.NDArray, cluster_b: npt.NDArray
    ) -> tuple[float, float]:
        if self.raw_snn is None or self.emb_snn is None:
            raise RuntimeError("fit() must run before distance computation")

        pair_num = cluster_a.size * cluster_b.size
        raw_sim = float(np.sum(self.raw_snn[np.ix_(cluster_a, cluster_b)]) / pair_num)
        emb_sim = float(np.sum(self.emb_snn[np.ix_(cluster_a, cluster_b)]) / pair_num)

        raw_dist = 1.0 / (raw_sim + self.alpha)
        emb_dist = 1.0 / (emb_sim + self.alpha)
        return raw_dist, emb_dist

    def _measure(self, mode: str, max_val: float, min_val: float) -> float:
        distortion_sum = 0.0
        weight_sum = 0.0

        for _ in range(self.iteration):
            part_distortion, part_weight = self._measure_single_iter(mode, max_val, min_val)
            distortion_sum += part_distortion
            weight_sum += part_weight

        if weight_sum == 0:
            return 1.0
        return float(1 - (distortion_sum / weight_sum))

    def _measure_single_iter(self, mode: str, max_val: float, min_val: float) -> tuple[float, float]:
        cluster_indices = self._extract_cluster(mode, self.walk_num)
        while cluster_indices.size <= 1:
            cluster_indices = self._extract_cluster(mode, self.walk_num)

        clustering_result = self._clustering(mode, cluster_indices)
        separated_clusters = self._separate_cluster_labels(cluster_indices, clustering_result)

        partial_distortion_sum = 0.0
        partial_weight_sum = 0.0

        scale = max_val - min_val
        if scale <= 0:
            return partial_distortion_sum, partial_weight_sum

        for i in range(len(separated_clusters)):
            cluster_i = np.array(separated_clusters[i], dtype=np.int64)
            for j in range(i):
                cluster_j = np.array(separated_clusters[j], dtype=np.int64)
                raw_dist, emb_dist = self._compute_distance(cluster_i, cluster_j)

                if mode == "steadiness":
                    distance = raw_dist - emb_dist
                else:
                    distance = emb_dist - raw_dist

                if distance <= 0:
                    continue

                distortion = (distance - min_val) / scale
                weight = cluster_i.size * cluster_j.size

                partial_distortion_sum += distortion * weight
                partial_weight_sum += weight

                if self.record:
                    self._record_log(mode, distortion, weight, cluster_i, cluster_j)

        return partial_distortion_sum, partial_weight_sum

    def _record_log(
        self,
        mode: str,
        distortion: float,
        weight: int,
        cluster_a: npt.NDArray,
        cluster_b: npt.NDArray,
    ) -> None:
        log = self.stead_log if mode == "steadiness" else self.cohev_log
        dval = distortion * weight

        for i in cluster_a:
            row_i = log[int(i)]
            for j in cluster_b:
                jj = int(j)

                if jj not in row_i:
                    row_i[jj] = [dval, 1]
                else:
                    row_i[jj] = [row_i[jj][0] + dval, row_i[jj][1] + 1]

                row_j = log[jj]
                ii = int(i)
                if ii not in row_j:
                    row_j[ii] = [dval, 1]
                else:
                    row_j[ii] = [row_j[ii][0] + dval, row_j[ii][1] + 1]

    def _finalize_logs(self) -> None:
        for datum_log in self.stead_log:
            for key_idx in list(datum_log.keys()):
                acc, cnt = datum_log[key_idx]
                datum_log[key_idx] = acc / cnt

        for datum_log in self.cohev_log:
            for key_idx in list(datum_log.keys()):
                acc, cnt = datum_log[key_idx]
                datum_log[key_idx] = acc / cnt

    @staticmethod
    def _vertices_info(log: list[dict[int, float]]) -> npt.NDArray:
        vertices = np.zeros(len(log), dtype=np.float64)
        for i, datum in enumerate(log):
            if datum:
                vertices[i] = float(sum(datum.values()))

        max_val = float(np.max(vertices))
        if max_val > 0:
            vertices /= max_val
        return vertices
