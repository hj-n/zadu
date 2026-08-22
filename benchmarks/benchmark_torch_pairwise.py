"""Compare NumPy and optional PyTorch exact pairwise resources."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from zadu.backends.numpy_backend import NumpyResourceProvider
from zadu.backends.torch_backend import TorchResourceProvider
from zadu.engine.resources import ResourceKey, ResourceKind, Space


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument(
        "--kind",
        choices=("distance-matrix", "condensed-pairs"),
        default="distance-matrix",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--working-memory-mib", type=float, default=64.0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    return parser


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
    if args.working_memory_mib <= 0:
        raise ValueError("working memory must be positive")
    if args.device == "mps" and args.dtype != "float32":
        raise ValueError("PyTorch MPS requires --dtype float32")

    rng = np.random.default_rng(args.seed)
    points = rng.normal(size=(args.samples, args.dimension)).astype(args.dtype)
    kind = (
        ResourceKind.DISTANCE_MATRIX
        if args.kind == "distance-matrix"
        else ResourceKind.CONDENSED_PAIRS
    )
    key = ResourceKey(kind, Space.ORIGINAL)
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

    torch_provider = TorchResourceProvider(device=args.device, dtype=args.dtype)
    torch_cold, cold_seconds = _build(
        torch_provider,
        key,
        points,
        working_memory_bytes,
    )
    torch_durations = []
    torch_result = torch_cold.value
    warm_details = None
    for _ in range(args.repeat):
        built, elapsed = _build(
            torch_provider,
            key,
            points,
            working_memory_bytes,
        )
        torch_result = built.value
        warm_details = built.details
        torch_durations.append(elapsed)

    numpy_seconds = statistics.median(numpy_durations)
    torch_warm_seconds = statistics.median(torch_durations)
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "kind": args.kind,
            "device": args.device,
            "dtype": args.dtype,
            "working_memory_bytes": working_memory_bytes,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": version("torch"),
            "platform": platform.platform(),
        },
        "numpy_seconds": numpy_seconds,
        "torch_cold_seconds": cold_seconds,
        "torch_warm_seconds": torch_warm_seconds,
        "warm_speedup": numpy_seconds / torch_warm_seconds,
        "maximum_absolute_delta": float(
            np.max(np.abs(np.asarray(numpy_result) - np.asarray(torch_result)))
        ),
        "cold_details": torch_cold.details,
        "warm_details": warm_details,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
