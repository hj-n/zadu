"""Benchmark PR 3-B3 compact/fused resources against the pre-fusion CPU path."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from zadu import ZADU
from zadu.measures import (
    class_aware_trustworthiness_continuity,
    distance_to_measure,
    kl_divergence,
    local_continuity_meta_criteria,
    mean_relative_rank_error,
    neighbor_dissimilarity,
    neighborhood_hit,
    trustworthiness_continuity,
)
from zadu.measures.utils import knn
from zadu.measures.utils import pairwise_dist as pdist

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker", choices=("baseline", "planned"), help=argparse.SUPPRESS
    )
    return parser


def _specs(k, sigma):
    return [
        {"id": "tnc", "params": {"k": k}},
        {"id": "ca_tnc", "params": {"k": k}},
        {"id": "mrre", "params": {"k": k}},
        {"id": "lcmc", "params": {"k": k}},
        {"id": "nh", "params": {"k": k}},
        {"id": "nd", "params": {"k": k}},
        {"id": "dtm", "params": {"sigma": sigma}},
        {"id": "kl_div", "params": {"sigma": sigma}},
    ]


def _peak_rss_mib() -> float:
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
        start = perf_counter()
        value = function()
        durations.append(perf_counter() - start)
    return float(np.median(durations)), value


def _original_resources(orig, k):
    distances = pdist.pairwise_distance_matrix(orig)
    indices, ranking = knn.knn_with_ranking(orig, k, distances)
    return distances, indices, ranking


def _baseline_scores(orig, emb, labels, k, sigma, original_resources=None):
    if original_resources is None:
        original_resources = _original_resources(orig, k)
    orig_distances, orig_indices, orig_ranking = original_resources
    emb_distances = pdist.pairwise_distance_matrix(emb)
    emb_indices, emb_ranking = knn.knn_with_ranking(emb, k, emb_distances)
    ranking_info = orig_indices, orig_ranking, emb_indices, emb_ranking
    neighbor_info = orig_indices, emb_indices
    distance_matrices = orig_distances, emb_distances
    return [
        trustworthiness_continuity.measure(
            orig, emb, k=k, knn_ranking_info=ranking_info
        ),
        class_aware_trustworthiness_continuity.measure(
            orig, emb, labels, k=k, knn_ranking_info=ranking_info
        ),
        mean_relative_rank_error.measure(orig, emb, k=k, knn_ranking_info=ranking_info),
        local_continuity_meta_criteria.measure(orig, emb, k=k, knn_info=neighbor_info),
        neighborhood_hit.measure(emb, labels, k=k, knn_emb_info=emb_indices),
        neighbor_dissimilarity.measure(orig, emb, k=k, knn_info=neighbor_info),
        distance_to_measure.measure(
            orig, emb, sigma=sigma, distance_matrices=distance_matrices
        ),
        kl_divergence.measure(
            orig, emb, sigma=sigma, distance_matrices=distance_matrices
        ),
    ]


def _worker(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = rng.normal(size=(args.samples, args.embedding_dimension))
    labels = np.arange(args.samples) % 10
    specs = _specs(args.neighbors, args.sigma)

    if args.worker == "baseline":
        _baseline_scores(
            orig[:30],
            emb[:30],
            labels[:30],
            min(args.neighbors, 14),
            args.sigma,
        )
        cold_seconds, scores = _median(
            lambda: _baseline_scores(orig, emb, labels, args.neighbors, args.sigma),
            args.repeat,
        )
        original_resources = _original_resources(orig, args.neighbors)
        warm_seconds, scores = _median(
            lambda: _baseline_scores(
                orig,
                emb,
                labels,
                args.neighbors,
                args.sigma,
                original_resources,
            ),
            args.repeat,
        )
        index_bytes = np.dtype(np.intp).itemsize
        estimated_cache_bytes = 2 * args.samples**2 * 8 + 2 * (
            args.samples**2 * index_bytes + args.samples * args.neighbors * index_bytes
        )
        planned_peak_bytes = None
        strategies = {"rank": "per_metric", "neighbor": "per_metric"}
    else:
        warmup_k = min(args.neighbors, 14)
        ZADU(_specs(warmup_k, args.sigma), orig[:30]).measure(emb[:30], labels[:30])
        cold_seconds, scores = _median(
            lambda: ZADU(specs, orig).measure(emb, labels),
            args.repeat,
        )
        runner = ZADU(specs, orig)
        warm_seconds, scores = _median(lambda: runner.measure(emb, labels), args.repeat)
        estimated_cache_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]
        strategies = {
            "rank": runner.last_run_info["rank_comparison_strategy"],
            "neighbor": runner.last_run_info["neighbor_statistics_strategy"],
        }

    return {
        "mode": args.worker,
        "strategies": strategies,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "estimated_cache_bytes": estimated_cache_bytes,
        "planned_peak_bytes": planned_peak_bytes,
        "scores": scores,
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
        "--sigma",
        str(args.sigma),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--worker",
        mode,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _maximum_score_delta(left, right):
    return max(
        abs(left_value - right[index][name])
        for index, score in enumerate(left)
        for name, left_value in score.items()
    )


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3 or args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError(
            "samples must be at least three and dimensions must be positive"
        )
    if args.neighbors < 1 or args.neighbors >= args.samples / 2:
        raise ValueError("neighbors must satisfy 1 <= neighbors < samples / 2")
    if args.sigma <= 0 or args.repeat < 1:
        raise ValueError("sigma and repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    baseline = _run_worker(args, "baseline")
    planned = _run_worker(args, "planned")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "neighbors": args.neighbors,
            "sigma": args.sigma,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "baseline": baseline,
        "planned": planned,
        "cold_speedup": baseline["cold_seconds"] / planned["cold_seconds"],
        "warm_speedup": baseline["warm_seconds"] / planned["warm_seconds"],
        "maximum_score_delta": _maximum_score_delta(
            baseline["scores"], planned["scores"]
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
