"""Benchmark the exact neighbor-metric suite through NumPy and MLX resources."""

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
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


def _measure(runner, embedding, labels):
    started = perf_counter()
    result = runner.measure(embedding, labels)
    return result, perf_counter() - started


def _maximum_score_delta(left, right) -> float:
    return max(
        abs(float(left_score[name]) - float(right_score[name]))
        for left_score, right_score in zip(left, right, strict=True)
        for name in left_score
    )


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3 or args.dimension < 1 or args.repeat < 1:
        raise ValueError("samples, dimension, and repeat must be positive")
    if not 1 <= args.k < args.samples / 2:
        raise ValueError("k must satisfy 1 <= k < samples / 2")
    if args.device == "gpu" and args.dtype != "float32":
        raise ValueError("The MLX GPU requires --dtype float32")

    rng = np.random.default_rng(args.seed)
    original = rng.normal(size=(args.samples, args.dimension))
    embedding = original @ rng.normal(size=(args.dimension, 2))
    embedding += rng.normal(scale=0.05, size=embedding.shape)
    labels = np.arange(args.samples) % 5
    specs = [
        {"id": "tnc", "params": {"k": args.k}},
        {"id": "lcmc", "params": {"k": args.k}},
        {"id": "nh", "params": {"k": args.k}},
        {"id": "proc", "params": {"k": args.k}},
        {"id": "topo", "params": {"k": args.k}},
    ]

    numpy_cold_started = perf_counter()
    numpy_runner = ZADU(specs, original)
    numpy_cold = numpy_runner.measure(embedding, labels)
    numpy_cold_seconds = perf_counter() - numpy_cold_started
    numpy_durations = [
        _measure(numpy_runner, embedding, labels)[1] for _ in range(args.repeat)
    ]

    mlx_cold_started = perf_counter()
    mlx_runner = ZADU(
        specs,
        original,
        execution=ExecutionConfig(
            backend="mlx",
            device=args.device,
            dtype=args.dtype,
        ),
    )
    mlx_cold = mlx_runner.measure(embedding, labels)
    mlx_cold_seconds = perf_counter() - mlx_cold_started
    mlx_durations = []
    mlx_warm = mlx_cold
    for _ in range(args.repeat):
        mlx_warm, elapsed = _measure(mlx_runner, embedding, labels)
        mlx_durations.append(elapsed)

    numpy_seconds = statistics.median(numpy_durations)
    mlx_warm_seconds = statistics.median(mlx_durations)
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "k": args.k,
            "device": args.device,
            "dtype": args.dtype,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mlx": version("mlx"),
            "platform": platform.platform(),
        },
        "numpy_construction_and_first_measure_seconds": numpy_cold_seconds,
        "numpy_warm_seconds": numpy_seconds,
        "mlx_construction_and_first_measure_seconds": mlx_cold_seconds,
        "mlx_warm_seconds": mlx_warm_seconds,
        "warm_speedup": numpy_seconds / mlx_warm_seconds,
        "cold_maximum_score_delta": _maximum_score_delta(numpy_cold, mlx_cold),
        "warm_maximum_score_delta": _maximum_score_delta(numpy_cold, mlx_warm),
        "mlx_run_info": mlx_runner.last_run_info,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
