"""Compare legacy dense and memory-aware exact mixed-resource execution."""

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

from zadu import ZADU, ExecutionConfig
from zadu.measures import local_continuity_meta_criteria, pearson_r, stress
from zadu.measures.utils.knn import knn_from_distance_matrix

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
    parser.add_argument("--memory-budget", default="48MiB")
    parser.add_argument("--repeat", type=int, default=3)
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
    return peak / 1024**2


def _dense_scores(orig, emb, k, orig_distances=None, orig_knn=None):
    if orig_distances is None:
        orig_distances = cdist(orig, orig)
    if orig_knn is None:
        orig_knn = knn_from_distance_matrix(orig_distances, k)
    emb_distances = cdist(emb, emb)
    emb_knn = knn_from_distance_matrix(emb_distances, k)
    matrices = orig_distances, emb_distances
    neighbors = orig_knn, emb_knn
    return [
        local_continuity_meta_criteria.measure(orig, emb, k, neighbors),
        stress.measure(orig, emb, matrices),
        pearson_r.measure(orig, emb, matrices),
    ]


def _median(function, repeat):
    durations = []
    value = None
    for _ in range(repeat):
        gc.collect()
        started = perf_counter()
        value = function()
        durations.append(perf_counter() - started)
    return float(np.median(durations)), value


def _worker(args) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = rng.normal(size=(args.samples, args.embedding_dimension))
    specs = [
        {"id": "lcmc", "params": {"k": args.neighbors}},
        {"id": "stress"},
        {"id": "pr"},
    ]
    if args.worker == "dense":
        cold_seconds, scores = _median(
            lambda: _dense_scores(orig, emb, args.neighbors), args.repeat
        )
        orig_distances = cdist(orig, orig)
        orig_knn = knn_from_distance_matrix(orig_distances, args.neighbors)
        warm_seconds, scores = _median(
            lambda: _dense_scores(
                orig,
                emb,
                args.neighbors,
                orig_distances,
                orig_knn,
            ),
            args.repeat,
        )
        retained_bytes = (
            2 * args.samples**2 * 8
            + 2 * args.samples * args.neighbors * np.dtype(np.int32).itemsize
        )
        strategy = "dense-reference"
        planned_peak_bytes = None
        providers = ["numpy", "scipy"]
    else:
        execution = ExecutionConfig(memory_budget=args.memory_budget)
        cold_seconds, scores = _median(
            lambda: ZADU(specs, orig, execution=execution).measure(emb),
            args.repeat,
        )
        runner = ZADU(specs, orig, execution=execution)
        warm_seconds, scores = _median(lambda: runner.measure(emb), args.repeat)
        retained_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]
        strategy = runner.last_run_info["pair_strategy"]
        providers = sorted(
            {resource["provider"] for resource in runner.last_run_info["resources"]}
        )
    return {
        "mode": args.worker,
        "strategy": strategy,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "retained_bytes": retained_bytes,
        "planned_peak_bytes": planned_peak_bytes,
        "providers": providers,
        "scores": scores,
    }


def _run_worker(args, mode: str) -> dict[str, object]:
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
        "--memory-budget",
        args.memory_budget,
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--worker",
        mode,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _maximum_score_delta(left, right) -> float:
    return max(
        abs(left_score[key] - right_score[key])
        for left_score, right_score in zip(left, right, strict=True)
        for key in left_score
    )


def main() -> None:
    args = _parser().parse_args()
    if (
        args.samples < 2
        or args.dimension < 1
        or args.embedding_dimension < 1
        or args.repeat < 1
    ):
        raise ValueError("samples, dimensions, and repeat must be positive")
    if args.neighbors < 1 or args.neighbors >= args.samples:
        raise ValueError("neighbors must satisfy 1 <= neighbors < samples")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    dense = _run_worker(args, "dense")
    planned = _run_worker(args, "planned")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "neighbors": args.neighbors,
            "memory_budget": args.memory_budget,
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
        "retained_memory_reduction": dense["retained_bytes"]
        / planned["retained_bytes"],
        "maximum_score_delta": _maximum_score_delta(
            dense.pop("scores"), planned.pop("scores")
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
