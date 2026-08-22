"""Benchmark exact repeated-embedding execution and original-resource reuse."""

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

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


SPECS = [
    {"id": "srho"},
    {"id": "nm_stress"},
    {"id": "tnc", "params": {"k": 10}},
    {"id": "mrre", "params": {"k": 20}},
    {"id": "proc", "params": {"k": 12}},
    {"id": "proc", "params": {"k": 20}},
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--embeddings", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker",
        choices=("independent", "manual-reuse", "measure-many"),
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


def _dataset(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    embeddings = []
    for _ in range(args.embeddings):
        projection = rng.normal(size=(args.dimension, args.embedding_dimension))
        embeddings.append(
            orig @ projection
            + 0.03 * rng.normal(size=(args.samples, args.embedding_dimension))
        )
    return orig, embeddings


def _execute(mode, orig, embeddings):
    if mode == "independent":
        results = []
        last_info = None
        for embedding in embeddings:
            runner = ZADU(SPECS, orig)
            results.append(runner.measure(embedding))
            last_info = runner.last_run_info
        return results, {
            "planned_peak_bytes": last_info["planned_peak_bytes"],
            "original_resource_reuse_events": 0,
        }

    runner = ZADU(SPECS, orig)
    if mode == "manual-reuse":
        results = [runner.measure(embedding) for embedding in embeddings]
        original_count = sum(
            resource["space"] == "orig"
            for resource in runner.last_run_info["resources"]
        )
        diagnostics = {
            "planned_peak_bytes": runner.last_run_info["planned_peak_bytes"],
            "original_resource_reuse_events": original_count * len(embeddings),
        }
        return results, diagnostics

    results = runner.measure_many(embeddings)
    diagnostics = {
        "planned_peak_bytes": runner.last_run_info["planned_peak_bytes"],
        "original_resource_reuse_events": runner.last_run_info[
            "original_resource_reuse_events"
        ],
    }
    return results, diagnostics


def _worker(args):
    orig, embeddings = _dataset(args)
    warm_n = max(42, 2 * 20 + 2)
    _execute(
        args.worker,
        orig[:warm_n],
        [embedding[:warm_n] for embedding in embeddings[:2]],
    )

    durations = []
    scores = None
    diagnostics = None
    for _ in range(args.repeat):
        gc.collect()
        started = perf_counter()
        scores, diagnostics = _execute(args.worker, orig, embeddings)
        durations.append(perf_counter() - started)

    return {
        "mode": args.worker,
        "seconds": float(np.median(durations)),
        "peak_rss_mib": _peak_rss_mib(),
        "planned_peak_bytes": diagnostics["planned_peak_bytes"],
        "original_resource_reuse_events": diagnostics["original_resource_reuse_events"],
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
        "--embeddings",
        str(args.embeddings),
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
        abs(left_value - right_value)
        for left_embedding, right_embedding in zip(left, right, strict=True)
        for left_metric, right_metric in zip(
            left_embedding, right_embedding, strict=True
        )
        for left_value, right_value in zip(
            left_metric.values(), right_metric.values(), strict=True
        )
    )


def main() -> None:
    args = _parser().parse_args()
    if args.samples <= 2 * 20:
        raise ValueError("--samples must be greater than 40 for the benchmark specs")
    if args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError("dimensions must be positive")
    if args.embeddings < 1 or args.repeat < 1:
        raise ValueError("--embeddings and --repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    independent = _run_worker(args, "independent")
    manual = _run_worker(args, "manual-reuse")
    many = _run_worker(args, "measure-many")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "embeddings": args.embeddings,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "independent": independent,
        "manual_reuse": manual,
        "measure_many": many,
        "measure_many_speedup_over_independent": (
            independent["seconds"] / many["seconds"]
        ),
        "measure_many_speedup_over_manual_reuse": (manual["seconds"] / many["seconds"]),
        "maximum_score_delta_vs_independent": _maximum_score_delta(
            independent["scores"], many["scores"]
        ),
        "maximum_score_delta_vs_manual_reuse": _maximum_score_delta(
            manual["scores"], many["scores"]
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
