from __future__ import annotations

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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
        random_state: int | np.random.Generator | None = None,
        n_jobs: int = 1,
        working_memory_bytes: int | None = None,
    ) -> None:
        self.raw = np.asarray(raw, dtype=np.float64)
        self.emb = np.asarray(emb, dtype=np.float64)

        if self.raw.shape[0] != self.emb.shape[0]:
            raise ValueError("raw and emb must have the same number of rows")
        if self.raw.shape[0] < 2:
            raise ValueError("At least 2 points are required for SNC")

        self.n = self.raw.shape[0]
        if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
            raise TypeError("iteration must be an integer")
        if iteration < 1:
            raise ValueError("iteration must be at least 1")
        if not np.isfinite(walk_num_ratio) or walk_num_ratio <= 0:
            raise ValueError("walk_num_ratio must be finite and greater than zero")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and greater than zero")
        if not isinstance(cluster_strategy, str):
            raise TypeError("cluster_strategy must be a string")
        if cluster_strategy not in {
            "dbscan",
            "kmeans",
        } and not cluster_strategy.endswith("-means"):
            raise ValueError(f"Unsupported clustering_strategy: {cluster_strategy}")

        if k is not None and (
            isinstance(k, bool) or not isinstance(k, (int, np.integer))
        ):
            raise TypeError("k must be an integer or None")
        if cluster_strategy.endswith("-means"):
            prefix = cluster_strategy.removesuffix("-means")
            if not prefix.isdigit() or int(prefix) < 1:
                raise ValueError(
                    "Custom KMeans strategies must use a positive '<k>-means' prefix"
                )
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
            raise TypeError("n_jobs must be an integer")
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if working_memory_bytes is not None:
            if isinstance(working_memory_bytes, bool) or not isinstance(
                working_memory_bytes, (int, np.integer)
            ):
                raise TypeError("working_memory_bytes must be an integer or None")
            if working_memory_bytes < 1:
                raise ValueError("working_memory_bytes must be at least 1")

        self.iteration = int(iteration)
        self.walk_num = max(1, int(self.n * walk_num_ratio))
        self.alpha = float(alpha)
        self.k = math.isqrt(self.n) if k is None else int(k)
        self.cluster_strategy = cluster_strategy
        self.n_jobs = min(int(n_jobs), self.iteration)
        self.working_memory_bytes = (
            None if working_memory_bytes is None else int(working_memory_bytes)
        )
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is None:
            # Preserve the legacy package's global NumPy RNG behavior.
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(random_state)

        if self.k < 1 or self.k >= self.n:
            raise ValueError(f"k must satisfy 1 <= k < n (n={self.n}), got k={self.k}")

        self.max_compress: float | None = None
        self.min_compress: float | None = None
        self.max_stretch: float | None = None
        self.min_stretch: float | None = None

        self.raw_knn: npt.NDArray | None = None
        self.emb_knn: npt.NDArray | None = None
        self.raw_snn: csr_matrix | None = None
        self.emb_snn: csr_matrix | None = None
        self.raw_knn_similarity: npt.NDArray | None = None
        self.emb_knn_similarity: npt.NDArray | None = None

        self.record = False
        self.stead_log = [dict() for _ in range(self.n)]
        self.cohev_log = [dict() for _ in range(self.n)]

        self.stead_score: float | None = None
        self.cohev_score: float | None = None

    def fit(
        self,
        record_vis_info: bool = False,
        knn_info: tuple[npt.NDArray, npt.NDArray] | None = None,
    ) -> None:
        self.record = record_vis_info

        if knn_info is None:
            raw_dist_matrix = pairwise_distance_matrix(self.raw)
            emb_dist_matrix = pairwise_distance_matrix(self.emb)
            self.raw_knn = self._knn_info(raw_dist_matrix, self.k)
            self.emb_knn = self._knn_info(emb_dist_matrix, self.k)
        else:
            if not isinstance(knn_info, tuple) or len(knn_info) != 2:
                raise TypeError("knn_info must be a (raw_knn, emb_knn) tuple")
            raw_knn, emb_knn = (np.asarray(item) for item in knn_info)
            expected = (self.n, self.k)
            if (
                raw_knn.ndim != 2
                or raw_knn.shape[0] != self.n
                or raw_knn.shape[1] < self.k
            ):
                raise ValueError(
                    f"raw knn_info must have shape (n, >=k), expected {expected}"
                )
            if (
                emb_knn.ndim != 2
                or emb_knn.shape[0] != self.n
                or emb_knn.shape[1] < self.k
            ):
                raise ValueError(
                    f"emb knn_info must have shape (n, >=k), expected {expected}"
                )
            self.raw_knn = raw_knn[:, : self.k]
            self.emb_knn = emb_knn[:, : self.k]

        if self.working_memory_bytes is not None:
            required = self.estimate_graph_bytes(
                self.n, self.k
            ) + self.n_jobs * self.estimate_iteration_bytes(
                self.n, self.k, self.walk_num
            )
            if required > self.working_memory_bytes:
                raise MemoryError(
                    "SNC sparse graph working set exceeds its memory budget "
                    f"({required} > {self.working_memory_bytes})"
                )

        self.raw_snn = self._weighted_snn(self.raw_knn, self.k)
        self.emb_snn = self._weighted_snn(self.emb_knn, self.k)

        raw_snn_max = float(self.raw_snn.max())
        emb_snn_max = float(self.emb_snn.max())

        if raw_snn_max > 0:
            self.raw_snn.data /= raw_snn_max
        if emb_snn_max > 0:
            self.emb_snn.data /= emb_snn_max

        self.raw_knn_similarity = self._neighbor_similarities(
            self.raw_snn, self.raw_knn
        )
        self.emb_knn_similarity = self._neighbor_similarities(
            self.emb_snn, self.emb_knn
        )

        dissim_min, dissim_max = self._distance_difference_extrema()

        self.max_compress = dissim_max if dissim_max > 0 else 0.0
        self.min_compress = dissim_min if dissim_min > 0 else 0.0
        self.max_stretch = -dissim_min if dissim_min < 0 else 0.0
        self.min_stretch = -dissim_max if dissim_max < 0 else 0.0

    def steadiness(self) -> float:
        if self.max_compress is None or self.min_compress is None:
            raise RuntimeError("Call fit() before computing steadiness")
        self.stead_score = self._measure(
            "steadiness", self.max_compress, self.min_compress
        )
        return self.stead_score

    def cohesiveness(self) -> float:
        if self.max_stretch is None or self.min_stretch is None:
            raise RuntimeError("Call fit() before computing cohesiveness")
        self.cohev_score = self._measure(
            "cohesiveness", self.max_stretch, self.min_stretch
        )
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
    def _weighted_snn(knn_info: npt.NDArray, k: int) -> csr_matrix:
        n = knn_info.shape[0]
        rows = np.repeat(np.arange(n), k)
        cols = knn_info.reshape(-1)
        # Keep the original weighting convention: k+1, k, ..., 2.
        vals = np.tile(np.arange(k + 1, 1, -1, dtype=np.float64), n)
        knn_graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
        snn = (knn_graph @ knn_graph.T).tocsr()
        # The reference CUDA kernel defines self-similarity as zero rather
        # than the weighted dot product of a row with itself.
        snn.setdiag(0.0)
        snn.eliminate_zeros()
        return snn

    @staticmethod
    def estimate_graph_bytes(n_samples: int, k: int) -> int:
        """Conservative storage estimate for two weighted sparse SNN graphs."""

        entries_per_graph = n_samples * n_samples
        return int(2 * entries_per_graph * 16 + 2 * n_samples * k * 8)

    @staticmethod
    def estimate_iteration_bytes(n_samples: int, k: int, walk_num: int) -> int:
        """Conservative private working set for one independent iteration."""

        cluster_size = min(n_samples, walk_num + k + 1)
        return int(max(1, 24 * cluster_size * cluster_size))

    @staticmethod
    def _neighbor_similarities(
        snn: csr_matrix,
        indices: npt.NDArray,
    ) -> npt.NDArray:
        rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
        values = np.asarray(snn[rows, indices.reshape(-1)]).reshape(-1)
        return values.reshape(indices.shape)

    def _distance_difference_extrema(self) -> tuple[float, float]:
        if self.raw_snn is None or self.emb_snn is None:
            raise RuntimeError("fit() must run before distance preprocessing")
        union = (self.raw_snn + self.emb_snn).tocsr()
        rows = np.repeat(np.arange(self.n), np.diff(union.indptr))
        columns = union.indices
        raw_values = np.asarray(self.raw_snn[rows, columns]).reshape(-1)
        emb_values = np.asarray(self.emb_snn[rows, columns]).reshape(-1)
        differences = 1.0 / (raw_values + self.alpha) - 1.0 / (emb_values + self.alpha)
        return (
            min(0.0, float(np.min(differences, initial=0.0))),
            max(0.0, float(np.max(differences, initial=0.0))),
        )

    def _extract_cluster(
        self,
        mode: str,
        walk_num: int,
        seed_idx: int | None = None,
        rng=None,
    ) -> npt.NDArray:
        if mode == "steadiness":
            knn_info = self.emb_knn
            knn_similarity = self.emb_knn_similarity
        else:
            knn_info = self.raw_knn
            knn_similarity = self.raw_knn_similarity

        if knn_info is None or knn_similarity is None:
            raise RuntimeError("fit() must run before cluster extraction")

        if seed_idx is None:
            seed_idx = self._random_integer(self.n, rng)
        cluster_member: set[int] = {seed_idx}
        current_queue: deque[int] = deque([seed_idx])

        visit_num = 0
        while visit_num < walk_num:
            if not current_queue:
                break
            i = current_queue.popleft()
            for neighbor_position, j in enumerate(knn_info[i]):
                probability = 1 - knn_similarity[i, neighbor_position]
                if self._random_unit(rng) > probability:
                    jj = int(j)
                    current_queue.append(jj)
                    cluster_member.add(jj)
                    visit_num += 1

        return np.array(list(cluster_member), dtype=np.int64)

    def _seed_can_expand(self, mode: str, seed_idx: int) -> bool:
        if mode == "steadiness":
            knn_similarity = self.emb_knn_similarity
        else:
            knn_similarity = self.raw_knn_similarity

        if knn_similarity is None:
            raise RuntimeError("fit() must run before cluster extraction")
        return bool(np.any(knn_similarity[seed_idx] > 0))

    def _clustering(
        self,
        mode: str,
        indices: npt.NDArray,
        random_state: int | None = None,
    ) -> npt.NDArray:
        if mode == "steadiness":
            snn_matrix = self.raw_snn
            data = self.raw
        else:
            snn_matrix = self.emb_snn
            data = self.emb

        if snn_matrix is None:
            raise RuntimeError("fit() must run before clustering")

        if self.cluster_strategy == "dbscan":
            cluster_similarity = snn_matrix[indices][:, indices].toarray()
            cluster_dist = 1.0 / (cluster_similarity + self.alpha)
            np.fill_diagonal(cluster_dist, 0)
            clusterer = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True)
            clusterer.fit(cluster_dist)
            return clusterer.labels_

        if self.cluster_strategy == "kmeans":
            k_val = max(2, int(np.sqrt(len(indices))))
            clusterer = KMeans(
                n_clusters=min(k_val, len(indices)),
                n_init="auto",
                random_state=random_state,
            )
            clusterer.fit(data[indices])
            return clusterer.labels_

        if self.cluster_strategy.endswith("-means"):
            k_val = int(self.cluster_strategy.split("-")[0])
            clusterer = KMeans(
                n_clusters=min(k_val, len(indices)),
                n_init="auto",
                random_state=random_state,
            )
            clusterer.fit(data[indices])
            return clusterer.labels_

        raise ValueError(f"Unsupported clustering_strategy: {self.cluster_strategy}")

    def _random_integer(self, high: int, rng=None) -> int:
        source = self.rng if rng is None else rng
        if isinstance(source, np.random.Generator):
            return int(source.integers(high))
        return int(source.randint(high))

    def _random_unit(self, rng=None) -> float:
        source = self.rng if rng is None else rng
        if isinstance(source, np.random.Generator):
            return float(source.random())
        return float(source.rand())

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

    def _cluster_distance_matrices(
        self,
        clusters: list[npt.NDArray],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if self.raw_snn is None or self.emb_snn is None:
            raise RuntimeError("fit() must run before distance computation")
        cluster_count = len(clusters)
        sizes = np.fromiter(
            (cluster.size for cluster in clusters),
            dtype=np.int64,
            count=cluster_count,
        )
        rows = np.repeat(np.arange(cluster_count), sizes)
        columns = np.concatenate(clusters)
        membership = csr_matrix(
            (np.ones(columns.size), (rows, columns)),
            shape=(cluster_count, self.n),
        )
        normalizer = sizes[:, None] * sizes[None, :]
        raw_similarity = (membership @ self.raw_snn @ membership.T).toarray()
        emb_similarity = (membership @ self.emb_snn @ membership.T).toarray()
        raw_similarity /= normalizer
        emb_similarity /= normalizer
        return (
            1.0 / (raw_similarity + self.alpha),
            1.0 / (emb_similarity + self.alpha),
        )

    def _measure(self, mode: str, max_val: float, min_val: float) -> float:
        distortion_sum = 0.0
        weight_sum = 0.0

        def run_iteration(iteration_input):
            return self._measure_single_iter(
                mode,
                max_val,
                min_val,
                iteration_input,
            )

        def accumulate(results):
            nonlocal distortion_sum, weight_sum
            for part_distortion, part_weight, record_events in results:
                if self.record:
                    for event in record_events:
                        self._record_log(mode, *event)
                distortion_sum += part_distortion
                weight_sum += part_weight

        if self.n_jobs == 1:
            for _ in range(self.iteration):
                accumulate((run_iteration(self._prepare_iteration(mode)),))
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                remaining = self.iteration
                while remaining:
                    batch_size = min(self.n_jobs, remaining)
                    iteration_inputs = [
                        self._prepare_iteration(mode) for _ in range(batch_size)
                    ]
                    accumulate(executor.map(run_iteration, iteration_inputs))
                    remaining -= batch_size

        if weight_sum == 0:
            return 1.0
        return float(1 - (distortion_sum / weight_sum))

    def _prepare_iteration(self, mode: str) -> tuple[npt.NDArray, int | None]:
        """Consume legacy RNG serially and return immutable iteration inputs."""

        for _ in range(max(100, self.n * 10)):
            seed_idx = self._random_integer(self.n)
            if not self._seed_can_expand(mode, seed_idx):
                continue
            for _ in range(1_000):
                cluster_indices = self._extract_cluster(
                    mode,
                    self.walk_num,
                    seed_idx=seed_idx,
                )
                if cluster_indices.size > 1:
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(
                "SNC could not extract a multi-point cluster; increase k or "
                "walk_num_ratio"
            )

        cluster_random_state = None
        if self.cluster_strategy != "dbscan":
            cluster_random_state = self._random_integer(np.iinfo(np.int32).max)
        return cluster_indices, cluster_random_state

    def _measure_single_iter(
        self,
        mode: str,
        max_val: float,
        min_val: float,
        iteration_input: tuple[npt.NDArray, int | None],
    ) -> tuple[float, float, list[tuple]]:
        cluster_indices, cluster_random_state = iteration_input
        clustering_result = self._clustering(
            mode,
            cluster_indices,
            cluster_random_state,
        )
        separated_clusters = self._separate_cluster_labels(
            cluster_indices, clustering_result
        )
        clusters = [
            np.asarray(cluster, dtype=np.int64) for cluster in separated_clusters
        ]
        raw_distances, emb_distances = self._cluster_distance_matrices(clusters)

        partial_distortion_sum = 0.0
        partial_weight_sum = 0.0
        record_events = []

        scale = max_val - min_val
        if scale <= 0:
            return partial_distortion_sum, partial_weight_sum, record_events

        for i, cluster_i in enumerate(clusters):
            for j in range(i):
                cluster_j = clusters[j]
                raw_dist = float(raw_distances[i, j])
                emb_dist = float(emb_distances[i, j])

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
                    record_events.append((distortion, weight, cluster_i, cluster_j))

        return partial_distortion_sum, partial_weight_sum, record_events

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
