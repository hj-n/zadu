"""Benchmark exact ordered-pair metrics against the legacy dense path."""

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
from zadu.measures import non_metric_stress, spearman_rho

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


ORDERED_SPECS = [{"id": "srho"}, {"id": "nm_stress"}]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--memory-budget", default="64MiB")
    parser.add_argument("--temporary-budget", default="512MiB")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker",
        choices=("dense", "planned", "external"),
        help=argparse.SUPPRESS,
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


def _dense_scores(orig, emb, orig_distances=None):
    if orig_distances is None:
        orig_distances = cdist(orig, orig)
    matrices = orig_distances, cdist(emb, emb)
    return [
        spearman_rho.measure(orig, emb, matrices),
        non_metric_stress.measure(orig, emb, matrices),
    ]


def _worker(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = rng.normal(size=(args.samples, args.embedding_dimension))

    if args.worker == "dense":
        _dense_scores(orig[:20], emb[:20])
        cold_seconds, scores = _median(lambda: _dense_scores(orig, emb), args.repeat)
        orig_distances = cdist(orig, orig)
        warm_seconds, scores = _median(
            lambda: _dense_scores(orig, emb, orig_distances), args.repeat
        )
        strategy = "dense-reference"
        estimated_cache_bytes = 2 * args.samples**2 * 8
        planned_peak_bytes = None
    elif args.worker == "planned":
        ZADU(ORDERED_SPECS, orig[:20]).measure(emb[:20])
        cold_seconds, scores = _median(
            lambda: ZADU(ORDERED_SPECS, orig).measure(emb),
            args.repeat,
        )
        runner = ZADU(ORDERED_SPECS, orig)
        warm_seconds, scores = _median(lambda: runner.measure(emb), args.repeat)
        strategy = runner.last_run_info["pair_strategy"]
        estimated_cache_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]
        planned_temporary_bytes = 0
        temporary_bytes_peak = 0
    else:
        config = ExecutionConfig(
            memory_budget=args.memory_budget,
            pair_order_strategy="external",
            temporary_budget=args.temporary_budget,
        )
        ZADU(ORDERED_SPECS, orig[:20], execution=config).measure(emb[:20])
        cold_seconds, scores = _median(
            lambda: ZADU(ORDERED_SPECS, orig, execution=config).measure(emb),
            args.repeat,
        )
        runner = ZADU(ORDERED_SPECS, orig, execution=config)
        warm_seconds, scores = _median(lambda: runner.measure(emb), args.repeat)
        strategy = runner.last_run_info["pair_strategy"]
        estimated_cache_bytes = runner.estimated_cache_bytes
        planned_peak_bytes = runner.last_run_info["planned_peak_bytes"]
        ordered = next(
            resource
            for resource in runner.last_run_info["resources"]
            if resource["kind"] == "ordered_pair_statistics"
        )
        planned_temporary_bytes = ordered["details"]["planned_temporary_bytes"]
        temporary_bytes_peak = ordered["details"]["temporary_bytes_peak"]

    if args.worker == "dense":
        planned_temporary_bytes = None
        temporary_bytes_peak = None

    return {
        "mode": args.worker,
        "strategy": strategy,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "estimated_cache_bytes": estimated_cache_bytes,
        "planned_peak_bytes": planned_peak_bytes,
        "planned_temporary_bytes": planned_temporary_bytes,
        "temporary_bytes_peak": temporary_bytes_peak,
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
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--memory-budget",
        args.memory_budget,
        "--temporary-budget",
        args.temporary_budget,
        "--worker",
        mode,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _score_delta(left, right):
    deltas = []
    for left_score, right_score in zip(left, right, strict=True):
        key = next(iter(left_score))
        deltas.append(abs(left_score[key] - right_score[key]))
    return max(deltas)


def _revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 2 or args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError("samples must be at least two and dimensions must be positive")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    dense = _run_worker(args, "dense")
    planned = _run_worker(args, "planned")
    external = _run_worker(args, "external")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "repeat": args.repeat,
            "memory_budget": args.memory_budget,
            "temporary_budget": args.temporary_budget,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "revision": _revision(),
        },
        "dense": dense,
        "planned": planned,
        "external": external,
        "cold_speedup": dense["cold_seconds"] / planned["cold_seconds"],
        "warm_speedup": dense["warm_seconds"] / planned["warm_seconds"],
        "maximum_score_delta": _score_delta(dense["scores"], planned["scores"]),
        "external_cold_slowdown": external["cold_seconds"] / planned["cold_seconds"],
        "external_warm_slowdown": external["warm_seconds"] / planned["warm_seconds"],
        "maximum_external_score_delta": _score_delta(
            planned["scores"], external["scores"]
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
