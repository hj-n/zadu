"""Benchmark exact selected-rank metrics through one optional backend."""

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
    parser.add_argument("--backend", choices=("mlx", "torch"), required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-budget-mib", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


def _measure(runner: ZADU, embedding: np.ndarray):
    started = perf_counter()
    result = runner.measure(embedding)
    return result, perf_counter() - started


def _maximum_score_delta(left, right) -> float:
    return max(
        abs(float(left_score[name]) - float(right_score[name]))
        for left_score, right_score in zip(left, right, strict=True)
        for name in left_score
    )


def _rank_record(run_info):
    return next(
        resource
        for resource in run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3 or args.dimension < 1 or args.repeat < 1:
        raise ValueError("samples, dimension, and repeat must be positive")
    if not 1 <= args.k < args.samples / 2:
        raise ValueError("k must satisfy 1 <= k < samples / 2")
    if args.memory_budget_mib < 1:
        raise ValueError("memory-budget-mib must be positive")
    if args.backend == "mlx" and args.device not in {"auto", "cpu", "gpu"}:
        raise ValueError("MLX device must be auto, cpu, or gpu")
    if args.backend == "torch" and args.device == "gpu":
        raise ValueError("PyTorch device must be auto, cpu, mps, or cuda")

    rng = np.random.default_rng(args.seed)
    original = rng.normal(size=(args.samples, args.dimension))
    embedding = original @ rng.normal(size=(args.dimension, 2))
    embedding += rng.normal(scale=0.05, size=embedding.shape)
    specs = [
        {"id": "tnc", "params": {"k": args.k}},
        {"id": "mrre", "params": {"k": args.k}},
    ]
    memory_budget = args.memory_budget_mib * 1024**2

    numpy_started = perf_counter()
    numpy_runner = ZADU(
        specs,
        original,
        execution=ExecutionConfig(memory_budget=memory_budget),
    )
    numpy_cold, numpy_first_seconds = _measure(numpy_runner, embedding)
    numpy_cold_seconds = perf_counter() - numpy_started
    numpy_durations = []
    numpy_rank_durations = []
    for _ in range(args.repeat):
        _, elapsed = _measure(numpy_runner, embedding)
        numpy_durations.append(elapsed)
        numpy_rank_durations.append(
            _rank_record(numpy_runner.last_run_info)["build_seconds"]
        )

    native_started = perf_counter()
    native_runner = ZADU(
        specs,
        original,
        execution=ExecutionConfig(
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            memory_budget=memory_budget,
        ),
    )
    native_cold, native_first_seconds = _measure(native_runner, embedding)
    native_cold_seconds = perf_counter() - native_started
    native_durations = []
    native_rank_durations = []
    native_warm = native_cold
    for _ in range(args.repeat):
        native_warm, elapsed = _measure(native_runner, embedding)
        native_durations.append(elapsed)
        native_rank_durations.append(
            _rank_record(native_runner.last_run_info)["build_seconds"]
        )

    numpy_warm_seconds = statistics.median(numpy_durations)
    native_warm_seconds = statistics.median(native_durations)
    rank_record = _rank_record(native_runner.last_run_info)
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "k": args.k,
            "backend": args.backend,
            "device": native_runner.execution.device,
            "dtype": args.dtype,
            "repeat": args.repeat,
            "memory_budget_bytes": memory_budget,
            "python": platform.python_version(),
            "numpy": np.__version__,
            args.backend: version(args.backend),
            "platform": platform.platform(),
        },
        "numpy_construction_and_first_measure_seconds": numpy_cold_seconds,
        "numpy_first_measure_seconds": numpy_first_seconds,
        "numpy_warm_seconds": numpy_warm_seconds,
        "numpy_warm_samples_seconds": numpy_durations,
        "native_construction_and_first_measure_seconds": native_cold_seconds,
        "native_first_measure_seconds": native_first_seconds,
        "native_warm_seconds": native_warm_seconds,
        "native_warm_samples_seconds": native_durations,
        "warm_speedup": numpy_warm_seconds / native_warm_seconds,
        "cold_maximum_score_delta": _maximum_score_delta(
            numpy_cold,
            native_cold,
        ),
        "warm_maximum_score_delta": _maximum_score_delta(
            numpy_cold,
            native_warm,
        ),
        "numpy_selected_rank_build_seconds": statistics.median(numpy_rank_durations),
        "numpy_selected_rank_build_samples_seconds": numpy_rank_durations,
        "native_selected_rank_build_seconds": statistics.median(native_rank_durations),
        "native_selected_rank_build_samples_seconds": native_rank_durations,
        "native_selected_rank_details": rank_record["details"],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
