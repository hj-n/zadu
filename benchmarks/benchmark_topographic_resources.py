"""Benchmark selected-distance Topographic Product against dense resources."""

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
from scipy.spatial.distance import cdist

from zadu import ZADU
from zadu.measures import topographic_product
from zadu.measures.utils.knn import knn_from_distance_matrix

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker", choices=("dense", "planned"), help=argparse.SUPPRESS
    )
    return parser


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


def _dense_score(orig, emb, k, original_resources=None):
    if original_resources is None:
        orig_distances = cdist(orig, orig)
        orig_knn = knn_from_distance_matrix(orig_distances, k)
    else:
        orig_distances, orig_knn = original_resources
    emb_distances = cdist(emb, emb)
    emb_knn = knn_from_distance_matrix(emb_distances, k)
    return topographic_product.measure(
        orig,
        emb,
        k,
        distance_matrices=(orig_distances, emb_distances),
        knn_info=(orig_knn, emb_knn),
    )


def _worker(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = rng.normal(size=(args.samples, args.embedding_dimension))
    specs = [{"id": "topo", "params": {"k": args.neighbors}}]

    if args.worker == "dense":
        _dense_score(orig[:20], emb[:20], min(args.neighbors, 19))
        cold_seconds, score = _median(
            lambda: _dense_score(orig, emb, args.neighbors),
            args.repeat,
        )
        orig_distances = cdist(orig, orig)
        original_resources = (
            orig_distances,
            knn_from_distance_matrix(orig_distances, args.neighbors),
        )
        warm_seconds, score = _median(
            lambda: _dense_score(orig, emb, args.neighbors, original_resources),
            args.repeat,
        )
        strategy = "dense-reference"
        estimated_cache_bytes = (
            2 * args.samples**2 * 8
            + 2 * args.samples * args.neighbors * np.dtype(np.intp).itemsize
        )
        planned_peak_bytes = None
    else:
        warmup_k = min(args.neighbors, 19)
        ZADU([{"id": "topo", "params": {"k": warmup_k}}], orig[:20]).measure(emb[:20])
        cold_seconds, scores = _median(
            lambda: ZADU(specs, orig).measure(emb),
            args.repeat,
        )
        runner = ZADU(specs, orig)
        warm_seconds, scores = _median(lambda: runner.measure(emb), args.repeat)
        score = scores[0]
        strategy = runner.last_run_info["topographic_strategy"]
        estimated_cache_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]

    return {
        "mode": args.worker,
        "strategy": strategy,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "estimated_cache_bytes": estimated_cache_bytes,
        "planned_peak_bytes": planned_peak_bytes,
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
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--worker",
        mode,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 2 or args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError("samples must be at least two and dimensions must be positive")
    if args.neighbors < 1 or args.neighbors >= args.samples:
        raise ValueError("neighbors must satisfy 1 <= neighbors < samples")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    dense = _run_worker(args, "dense")
    planned = _run_worker(args, "planned")
    score_key = "topographic_product"
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "neighbors": args.neighbors,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "dense": dense,
        "planned": planned,
        "cold_speedup": dense["cold_seconds"] / planned["cold_seconds"],
        "warm_speedup": dense["warm_seconds"] / planned["warm_seconds"],
        "score_delta": abs(dense["score"][score_key] - planned["score"][score_key]),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
