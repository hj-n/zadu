"""Benchmark exact sparse/batched SNC against the pre-PR dense CPU path."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import hdbscan
import numpy as np

from zadu import ZADU
from zadu.measures.utils.snc_cpu import SNCCPU

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


class _DenseScalarSNCCPU(SNCCPU):
    """Review baseline matching the dense, scalar cluster-pair implementation."""

    def fit(self, record_vis_info=False, knn_info=None):
        self.record = record_vis_info
        raw_distances = self._euclidean_distances(self.raw)
        emb_distances = self._euclidean_distances(self.emb)
        self.raw_knn = self._knn_info(raw_distances, self.k)
        self.emb_knn = self._knn_info(emb_distances, self.k)
        self.raw_snn = self._dense_weighted_snn(self.raw_knn, self.k)
        self.emb_snn = self._dense_weighted_snn(self.emb_knn, self.k)
        raw_max = float(np.max(self.raw_snn))
        emb_max = float(np.max(self.emb_snn))
        if raw_max > 0:
            self.raw_snn /= raw_max
        if emb_max > 0:
            self.emb_snn /= emb_max
        self.raw_knn_similarity = np.take_along_axis(self.raw_snn, self.raw_knn, axis=1)
        self.emb_knn_similarity = np.take_along_axis(self.emb_snn, self.emb_knn, axis=1)
        raw_graph_distances = 1 / (self.raw_snn + self.alpha)
        emb_graph_distances = 1 / (self.emb_snn + self.alpha)
        difference = raw_graph_distances - emb_graph_distances
        maximum = float(np.max(difference))
        minimum = float(np.min(difference))
        self.max_compress = maximum if maximum > 0 else 0.0
        self.min_compress = minimum if minimum > 0 else 0.0
        self.max_stretch = -minimum if minimum < 0 else 0.0
        self.min_stretch = -maximum if maximum < 0 else 0.0

    @staticmethod
    def _euclidean_distances(points):
        from zadu.measures.utils.pairwise_dist import pairwise_distance_matrix

        return pairwise_distance_matrix(points)

    @staticmethod
    def _dense_weighted_snn(indices, k):
        from scipy.sparse import csr_matrix

        n_samples = indices.shape[0]
        rows = np.repeat(np.arange(n_samples), k)
        values = np.tile(np.arange(k + 1, 1, -1, dtype=np.float64), n_samples)
        graph = csr_matrix(
            (values, (rows, indices.reshape(-1))),
            shape=(n_samples, n_samples),
        )
        snn = (graph @ graph.T).toarray()
        np.fill_diagonal(snn, 0)
        return snn

    def _clustering(self, mode, indices, random_state=None):
        if self.cluster_strategy != "dbscan":
            return super()._clustering(mode, indices, random_state)
        graph = self.raw_snn if mode == "steadiness" else self.emb_snn
        distances = 1 / (graph[np.ix_(indices, indices)] + self.alpha)
        np.fill_diagonal(distances, 0)
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            allow_single_cluster=True,
        )
        return clusterer.fit_predict(distances)

    def _cluster_distance_matrices(self, clusters):
        cluster_count = len(clusters)
        raw_distances = np.empty((cluster_count, cluster_count))
        emb_distances = np.empty((cluster_count, cluster_count))
        for row, cluster_a in enumerate(clusters):
            for column, cluster_b in enumerate(clusters):
                pair_count = cluster_a.size * cluster_b.size
                raw_similarity = float(
                    np.sum(self.raw_snn[np.ix_(cluster_a, cluster_b)]) / pair_count
                )
                emb_similarity = float(
                    np.sum(self.emb_snn[np.ix_(cluster_a, cluster_b)]) / pair_count
                )
                raw_distances[row, column] = 1 / (raw_similarity + self.alpha)
                emb_distances[row, column] = 1 / (emb_similarity + self.alpha)
        return raw_distances, emb_distances


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--walk-num-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker",
        choices=("dense", "planned-single", "planned-parallel"),
        help=argparse.SUPPRESS,
    )
    return parser


def _peak_rss_mib():
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / (1024**2)


def _median(function, repeat):
    durations = []
    value = None
    for _ in range(repeat):
        gc.collect()
        started = perf_counter()
        value = function()
        durations.append(perf_counter() - started)
    return float(np.median(durations)), value


def _dense_scores(args, orig, emb):
    obj = _DenseScalarSNCCPU(
        orig,
        emb,
        iteration=args.iterations,
        walk_num_ratio=args.walk_num_ratio,
        k=args.neighbors,
        random_state=args.seed,
    )
    obj.fit()
    return {"steadiness": obj.steadiness(), "cohesiveness": obj.cohesiveness()}


def _specs(args, workers):
    return [
        {
            "id": "snc",
            "params": {
                "iteration": args.iterations,
                "walk_num_ratio": args.walk_num_ratio,
                "k": args.neighbors,
                "random_state": args.seed,
                "n_jobs": workers,
            },
        }
    ]


def _worker(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = orig @ rng.normal(
        size=(args.dimension, args.embedding_dimension)
    ) + 0.05 * rng.normal(size=(args.samples, args.embedding_dimension))

    if args.worker == "dense":
        seconds, score = _median(lambda: _dense_scores(args, orig, emb), args.repeat)
        cache_bytes = 4 * args.samples**2 * 8
        planned_peak_bytes = None
        effective_workers = 1
    else:
        workers = 1 if args.worker == "planned-single" else args.workers
        specs = _specs(args, workers)
        runner = ZADU(specs, orig)
        seconds, scores = _median(lambda: runner.measure(emb), args.repeat)
        score = scores[0]
        cache_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]
        effective_workers = runner.last_run_info["snc_strategy"]["effective_workers"][0]

    return {
        "mode": args.worker,
        "seconds": seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "estimated_cache_bytes": cache_bytes,
        "planned_peak_bytes": planned_peak_bytes,
        "effective_workers": effective_workers,
        "score": score,
    }


def _run_worker(args, mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--samples",
        str(args.samples),
        "--dimension",
        str(args.dimension),
        "--embedding-dimension",
        str(args.embedding_dimension),
        "--neighbors",
        str(args.neighbors),
        "--iterations",
        str(args.iterations),
        "--walk-num-ratio",
        str(args.walk_num_ratio),
        "--workers",
        str(args.workers),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--worker",
        mode,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _score_delta(left, right):
    return max(abs(left[key] - right[key]) for key in left)


def main():
    args = _parser().parse_args()
    if args.samples < 3 or args.neighbors < 1 or args.neighbors >= args.samples:
        raise ValueError("samples and neighbors are outside the supported range")
    if args.iterations < 1 or args.repeat < 1 or args.workers < 1:
        raise ValueError("iterations, repeat, and workers must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    dense = _run_worker(args, "dense")
    planned_single = _run_worker(args, "planned-single")
    planned_parallel = _run_worker(args, "planned-parallel")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "neighbors": args.neighbors,
            "iterations": args.iterations,
            "walk_num_ratio": args.walk_num_ratio,
            "workers": args.workers,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "dense": dense,
        "planned_single": planned_single,
        "planned_parallel": planned_parallel,
        "single_speedup": dense["seconds"] / planned_single["seconds"],
        "parallel_speedup": dense["seconds"] / planned_parallel["seconds"],
        "parallel_over_single": (
            planned_single["seconds"] / planned_parallel["seconds"]
        ),
        "single_score_delta": _score_delta(dense["score"], planned_single["score"]),
        "parallel_score_delta": _score_delta(
            planned_single["score"], planned_parallel["score"]
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
