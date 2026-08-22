"""Compare NumPy and optional MLX exact neighbor resources."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from zadu.backends.mlx_backend import MlxResourceProvider
from zadu.backends.numpy_backend import NumpyResourceProvider
from zadu.engine.resources import NeighborRanking, ResourceKey, ResourceKind, Space


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--kind",
        choices=("ranking", "knn", "stable-knn"),
        default="ranking",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--working-memory-mib", type=float, default=64.0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


def _value_arrays(value):
    if isinstance(value, NeighborRanking):
        return value.indices, value.ranking
    return (value,)


def _build(provider, key, points, working_memory_bytes):
    started = perf_counter()
    built = provider.build(
        key,
        points,
        distance_matrix=None,
        condensed_pairs=None,
        working_memory_bytes=working_memory_bytes,
        geodesic=False,
    )
    return built, perf_counter() - started


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 2 or args.dimension < 1 or args.repeat < 1:
        raise ValueError("samples, dimension, and repeat must be positive")
    if not 1 <= args.k < args.samples:
        raise ValueError("k must satisfy 1 <= k < samples")
    if args.working_memory_mib <= 0:
        raise ValueError("working memory must be positive")
    if args.device == "gpu" and args.dtype != "float32":
        raise ValueError("The MLX GPU requires --dtype float32")

    rng = np.random.default_rng(args.seed)
    points = rng.normal(size=(args.samples, args.dimension)).astype(args.dtype)
    kind = {
        "ranking": ResourceKind.NEIGHBOR_RANKING,
        "knn": ResourceKind.KNN,
        "stable-knn": ResourceKind.STABLE_KNN,
    }[args.kind]
    key = ResourceKey(kind, Space.ORIGINAL, args.k)
    working_memory_bytes = int(args.working_memory_mib * 1024**2)

    numpy_provider = NumpyResourceProvider()
    numpy_durations = []
    numpy_result = None
    for _ in range(args.repeat):
        built, elapsed = _build(
            numpy_provider,
            key,
            points,
            working_memory_bytes,
        )
        numpy_result = built.value
        numpy_durations.append(elapsed)

    mlx_provider = MlxResourceProvider(device=args.device, dtype=args.dtype)
    mlx_cold, cold_seconds = _build(
        mlx_provider,
        key,
        points,
        working_memory_bytes,
    )
    mlx_durations = []
    mlx_result = mlx_cold.value
    warm_details = None
    for _ in range(args.repeat):
        built, elapsed = _build(
            mlx_provider,
            key,
            points,
            working_memory_bytes,
        )
        mlx_result = built.value
        warm_details = built.details
        mlx_durations.append(elapsed)

    numpy_arrays = _value_arrays(numpy_result)
    mlx_arrays = _value_arrays(mlx_result)
    exact_match = all(
        np.array_equal(numpy_array, mlx_array)
        for numpy_array, mlx_array in zip(numpy_arrays, mlx_arrays, strict=True)
    )
    mismatch_count = sum(
        int(np.count_nonzero(numpy_array != mlx_array))
        for numpy_array, mlx_array in zip(numpy_arrays, mlx_arrays, strict=True)
    )
    compared_values = sum(array.size for array in numpy_arrays)
    numpy_seconds = statistics.median(numpy_durations)
    mlx_warm_seconds = statistics.median(mlx_durations)
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "k": args.k,
            "kind": args.kind,
            "device": args.device,
            "dtype": args.dtype,
            "working_memory_bytes": working_memory_bytes,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mlx": version("mlx"),
            "platform": platform.platform(),
        },
        "numpy_seconds": numpy_seconds,
        "mlx_cold_seconds": cold_seconds,
        "mlx_warm_seconds": mlx_warm_seconds,
        "warm_speedup": numpy_seconds / mlx_warm_seconds,
        "exact_index_match": exact_match,
        "index_mismatch_count": mismatch_count,
        "index_mismatch_rate": mismatch_count / compared_values,
        "cold_details": mlx_cold.details,
        "warm_details": warm_details,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
