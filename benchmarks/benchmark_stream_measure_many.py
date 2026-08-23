"""Benchmark materialized and bounded-stream repeated embedding evaluation."""

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

from zadu import ZADU, ExecutionConfig

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


SPECS = [
    {"id": "tnc", "params": {"k": 10}},
    {"id": "mrre", "params": {"k": 20}},
    {"id": "stress"},
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--embeddings", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker",
        choices=("materialized", "stream", "parallel-stream"),
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


def _original(args) -> np.ndarray:
    return np.random.default_rng(args.seed).normal(size=(args.samples, args.dimension))


def _embedding_source(args, orig):
    rng = np.random.default_rng(args.seed + 1)
    for _ in range(args.embeddings):
        projection = rng.normal(size=(args.dimension, args.embedding_dimension))
        yield orig @ projection + 0.03 * rng.normal(
            size=(args.samples, args.embedding_dimension)
        )


def _score_checksum(result) -> float:
    return float(sum(float(value) for metric in result for value in metric.values()))


def _execute(args, orig):
    workers = args.workers if args.worker == "parallel-stream" else 1
    runner = ZADU(
        SPECS,
        orig,
        execution=ExecutionConfig(embedding_workers=workers),
    )
    source = _embedding_source(args, orig)
    if args.worker == "materialized":
        results = runner.measure_many(list(source))
        checksum = sum(_score_checksum(result) for result in results)
    else:
        checksum = sum(
            _score_checksum(record.result)
            for record in runner.iter_measure_many(source)
        )
    return checksum, runner.last_run_info


def _worker(args):
    orig = _original(args)
    durations = []
    checksum = None
    diagnostics = None
    for _ in range(args.repeat):
        gc.collect()
        started = perf_counter()
        checksum, diagnostics = _execute(args, orig)
        durations.append(perf_counter() - started)
    return {
        "mode": args.worker,
        "seconds": float(np.median(durations)),
        "peak_rss_mib": _peak_rss_mib(),
        "checksum": checksum,
        "planned_peak_bytes": diagnostics["planned_peak_bytes"],
        "effective_workers": diagnostics["effective_workers"],
        "max_in_flight_observed": diagnostics.get("max_in_flight_observed"),
        "runs_retained": diagnostics.get("runs_retained", True),
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


def main() -> None:
    args = _parser().parse_args()
    if args.samples <= 2 * 20:
        raise ValueError("--samples must be greater than 40 for the benchmark specs")
    if args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError("dimensions must be positive")
    if args.embeddings < 1 or args.workers < 1 or args.repeat < 1:
        raise ValueError("--embeddings, --workers, and --repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    materialized = _run_worker(args, "materialized")
    stream = _run_worker(args, "stream")
    parallel_stream = _run_worker(args, "parallel-stream")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "embeddings": args.embeddings,
            "workers": args.workers,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "materialized": materialized,
        "stream": stream,
        "parallel_stream": parallel_stream,
        "stream_speed_ratio": materialized["seconds"] / stream["seconds"],
        "parallel_stream_speed_ratio": (
            materialized["seconds"] / parallel_stream["seconds"]
        ),
        "stream_peak_rss_ratio": materialized["peak_rss_mib"] / stream["peak_rss_mib"],
        "maximum_checksum_delta": max(
            abs(materialized["checksum"] - stream["checksum"]),
            abs(materialized["checksum"] - parallel_stream["checksum"]),
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
