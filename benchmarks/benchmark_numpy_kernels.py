"""Benchmark exact NumPy metric kernels against simple reference formulas."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist

from zadu.measures import (
    class_aware_trustworthiness_continuity as ca_tnc,
)
from zadu.measures import (
    local_continuity_meta_criteria as lcmc,
)
from zadu.measures import (
    mean_relative_rank_error as mrre,
)
from zadu.measures import (
    neighborhood_hit,
    procrustes,
    topographic_product,
)
from zadu.measures import (
    trustworthiness_continuity as tnc,
)
from zadu.measures.utils.knn import knn_with_ranking

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


def _reference_tnc(base_knn, base_ranking, target_knn, k):
    local = []
    n = base_knn.shape[0]
    for row in range(n):
        missing = np.setdiff1d(target_knn[row], base_knn[row])
        distortion = 0.0
        for index in missing:
            distortion += base_ranking[row, index] - k
        local.append(distortion)
    local = 1 - np.asarray(local) * (2 / (k * (2 * n - 3 * k - 1)))
    return float(np.mean(local))


def _reference_ca_tnc(base_knn, base_ranking, target_knn, labels, k):
    local = []
    n = base_knn.shape[0]
    for row in range(n):
        missing = np.setdiff1d(target_knn[row], base_knn[row])
        distortion = 0.0
        for index in missing:
            if labels[row] != labels[index]:
                distortion += base_ranking[row, index] - k
        local.append(distortion)
    local = 1 - np.asarray(local) * (2 / (k * (2 * n - 3 * k - 1)))
    return float(np.mean(local))


def _reference_mrre(base_ranking, target_ranking, target_knn, k):
    local = []
    n = target_knn.shape[0]
    for row in range(n):
        base_rank = base_ranking[row][target_knn[row]]
        target_rank = target_ranking[row][target_knn[row]]
        local.append(np.sum(np.abs(base_rank - target_rank) / target_rank))
    normalizer = sum(abs(n - 2 * rank + 1) / rank for rank in range(1, k + 1))
    return float(np.mean(1 - np.asarray(local) / normalizer))


def _reference_lcmc(orig_knn, emb_knn, n, k):
    local = []
    for row in range(n):
        overlap = np.intersect1d(orig_knn[row], emb_knn[row]).shape[0]
        local.append((overlap - (k * k) / (n - 1)) / k)
    return float(np.mean(local))


def _reference_nh(emb_knn, labels, k):
    local = []
    for row in range(len(labels)):
        local.append(np.sum(labels[emb_knn[row]] == labels[row]) / k)
    return float(np.mean(local))


def _reference_topo(orig_dist, emb_dist, orig_knn, emb_knn, k):
    total = 0.0
    n = orig_knn.shape[0]
    for row in range(n):
        for prefix_end in range(k):
            q1_product = 1.0
            q2_product = 1.0
            for rank in range(prefix_end + 1):
                q1_product *= (
                    orig_dist[row, emb_knn[row, rank]]
                    / orig_dist[row, orig_knn[row, rank]]
                )
                q2_product *= (
                    emb_dist[row, emb_knn[row, rank]]
                    / emb_dist[row, orig_knn[row, rank]]
                )
            total += np.log((q1_product * q2_product) ** (1 / (2 * (prefix_end + 1))))
    return total / (n * k)


def _reference_procrustes(orig, emb, orig_knn, emb_knn):
    scores = []
    for row in range(orig.shape[0]):
        orig_neighbors = orig[orig_knn[row]]
        emb_neighbors = emb[emb_knn[row]]
        k = orig_neighbors.shape[0]
        centering = np.eye(k) - np.ones((k, k)) / k
        u, _, vh = np.linalg.svd(
            orig_neighbors.T @ centering @ emb_neighbors, full_matrices=False
        )
        rotation = u @ vh
        residual = centering @ (orig_neighbors - emb_neighbors @ rotation.T)
        numerator = np.linalg.norm(residual, ord="fro") ** 2
        denominator = np.linalg.norm(centering @ orig_neighbors, ord="fro") ** 2
        scores.append(numerator / denominator)
    return float(np.mean(scores))


def _median_seconds(function: Callable[[], float], repeat: int) -> tuple[float, float]:
    durations = []
    value = 0.0
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        value = float(function())
        durations.append(perf_counter() - start)
    return float(np.median(durations)), value


def _peak_rss_mib() -> float:
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / (1024**2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=50)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if not 1 <= args.neighbors < args.samples / 2:
        raise ValueError("--neighbors must satisfy 1 <= k < n / 2")
    if args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError("dimensions must be positive")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension)).astype(dtype)
    projection = rng.normal(size=(args.dimension, args.embedding_dimension)).astype(
        dtype
    )
    emb = (
        orig @ projection
        + 0.05 * rng.normal(size=(args.samples, args.embedding_dimension))
    ).astype(dtype)
    labels = np.arange(args.samples) % 10

    orig_dist = cdist(orig, orig)
    emb_dist = cdist(emb, emb)
    orig_knn, orig_ranking = knn_with_ranking(orig, args.neighbors, orig_dist)
    emb_knn, emb_ranking = knn_with_ranking(emb, args.neighbors, emb_dist)
    k = args.neighbors

    cases = [
        (
            "T&C",
            lambda: _reference_tnc(orig_knn, orig_ranking, emb_knn, k),
            lambda: tnc.tnc_computation(orig_knn, orig_ranking, emb_knn, k),
        ),
        (
            "Class-aware T&C",
            lambda: _reference_ca_tnc(orig_knn, orig_ranking, emb_knn, labels, k),
            lambda: ca_tnc.ca_tnc_computation(
                orig_knn, orig_ranking, emb_knn, labels, k, "false"
            ),
        ),
        (
            "MRRE",
            lambda: _reference_mrre(orig_ranking, emb_ranking, emb_knn, k),
            lambda: mrre.mrre_computation(orig_ranking, emb_ranking, emb_knn, k),
        ),
        (
            "LCMC",
            lambda: _reference_lcmc(orig_knn, emb_knn, args.samples, k),
            lambda: lcmc.measure(orig, emb, k, knn_info=(orig_knn, emb_knn))["lcmc"],
        ),
        (
            "Neighborhood Hit",
            lambda: _reference_nh(emb_knn, labels, k),
            lambda: neighborhood_hit.measure(emb, labels, k, knn_info=emb_knn)[
                "neighborhood_hit"
            ],
        ),
        (
            "Topographic Product",
            lambda: _reference_topo(orig_dist, emb_dist, orig_knn, emb_knn, k),
            lambda: topographic_product.measure(
                orig,
                emb,
                k,
                distance_matrices=(orig_dist, emb_dist),
                knn_info=(orig_knn, emb_knn),
            )["topographic_product"],
        ),
        (
            "Procrustes",
            lambda: _reference_procrustes(orig, emb, orig_knn, emb_knn),
            lambda: procrustes.measure(orig, emb, k, knn_info=(orig_knn, emb_knn))[
                "procrustes"
            ],
        ),
    ]

    metadata = {
        "samples": args.samples,
        "dimension": args.dimension,
        "embedding_dimension": args.embedding_dimension,
        "neighbors": k,
        "dtype": dtype.name,
        "repeat": args.repeat,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    results = []
    for name, reference, current in cases:
        current()  # Warm imports and reusable library state.
        reference_seconds, reference_value = _median_seconds(reference, args.repeat)
        current_seconds, current_value = _median_seconds(current, args.repeat)
        results.append(
            {
                "metric": name,
                "reference_seconds": reference_seconds,
                "current_seconds": current_seconds,
                "speedup": reference_seconds / current_seconds,
                "absolute_delta": abs(reference_value - current_value),
                "peak_rss_mib": _peak_rss_mib(),
            }
        )

    print(json.dumps(metadata, indent=2))
    print(
        "\n| Metric | Reference (s) | Current (s) | Speedup | Abs. delta | Peak RSS (MiB) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['metric']} | {result['reference_seconds']:.6f} | "
            f"{result['current_seconds']:.6f} | {result['speedup']:.2f}x | "
            f"{result['absolute_delta']:.3e} | {result['peak_rss_mib']:.1f} |"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "results": results}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
