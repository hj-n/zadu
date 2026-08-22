"""Benchmark sequential and native-batched Torch repeated-embedding execution."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from zadu import ZADU, ExecutionConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--embeddings", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


def _measure_many(runner, embeddings):
    started = perf_counter()
    scores = runner.measure_many(embeddings)
    return scores, perf_counter() - started


def _maximum_score_delta(left, right) -> float:
    return max(
        abs(float(left_value) - float(right_value))
        for left_embedding, right_embedding in zip(left, right, strict=True)
        for left_metric, right_metric in zip(
            left_embedding,
            right_embedding,
            strict=True,
        )
        for left_value, right_value in zip(
            left_metric.values(),
            right_metric.values(),
            strict=True,
        )
    )


def main() -> None:
    args = _parser().parse_args()
    if args.samples <= 2 * args.k:
        raise ValueError("--samples must be greater than two times --k")
    if (
        min(
            args.dimension,
            args.embedding_dimension,
            args.embeddings,
            args.batch_size,
            args.repeat,
        )
        < 1
    ):
        raise ValueError("dimensions, counts, batch size, and repeat must be positive")
    if args.device == "mps" and args.dtype != "float32":
        raise ValueError("PyTorch MPS requires --dtype float32")

    rng = np.random.default_rng(args.seed)
    original = rng.normal(size=(args.samples, args.dimension))
    embeddings = [
        original @ rng.normal(size=(args.dimension, args.embedding_dimension))
        + rng.normal(scale=0.03, size=(args.samples, args.embedding_dimension))
        for _ in range(args.embeddings)
    ]
    specs = [
        {"id": "stress"},
        {"id": "tnc", "params": {"k": args.k}},
        {"id": "lcmc", "params": {"k": args.k}},
        {"id": "topo", "params": {"k": args.k}},
    ]

    def runner(workers):
        return ZADU(
            specs,
            original,
            execution=ExecutionConfig(
                backend="torch",
                device=args.device,
                dtype=args.dtype,
                embedding_workers=workers,
            ),
        )

    sequential_runner = runner(1)
    sequential_cold, sequential_cold_seconds = _measure_many(
        sequential_runner,
        embeddings,
    )
    sequential_durations = [
        _measure_many(sequential_runner, embeddings)[1] for _ in range(args.repeat)
    ]

    batched_runner = runner(args.batch_size)
    batched_cold, batched_cold_seconds = _measure_many(batched_runner, embeddings)
    batched_durations = []
    batched_warm = batched_cold
    for _ in range(args.repeat):
        batched_warm, elapsed = _measure_many(batched_runner, embeddings)
        batched_durations.append(elapsed)

    sequential_seconds = statistics.median(sequential_durations)
    batched_seconds = statistics.median(batched_durations)
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "embeddings": args.embeddings,
            "requested_batch_size": args.batch_size,
            "k": args.k,
            "device": args.device,
            "dtype": args.dtype,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": version("torch"),
            "platform": platform.platform(),
        },
        "sequential_cold_seconds": sequential_cold_seconds,
        "sequential_warm_seconds": sequential_seconds,
        "batched_cold_seconds": batched_cold_seconds,
        "batched_warm_seconds": batched_seconds,
        "cold_speedup": sequential_cold_seconds / batched_cold_seconds,
        "warm_speedup": sequential_seconds / batched_seconds,
        "cold_maximum_score_delta": _maximum_score_delta(
            sequential_cold,
            batched_cold,
        ),
        "warm_maximum_score_delta": _maximum_score_delta(
            sequential_cold,
            batched_warm,
        ),
        "batch_info": batched_runner.last_run_info,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
