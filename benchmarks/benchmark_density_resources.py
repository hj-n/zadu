"""Compare dense and exact blockwise multi-sigma density construction."""

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

from zadu.backends import NumpyResourceProvider
from zadu.engine.config import parse_memory_budget
from zadu.measures.utils.pairwise_dist import distance_matrix_to_density

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--sigmas", type=float, nargs="+", default=(0.1, 0.3))
    parser.add_argument("--memory-budget", default="16MiB")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker", choices=("dense", "blockwise"), help=argparse.SUPPRESS
    )
    return parser


def _peak_rss_mib() -> float:
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / 1024**2


def _worker(args) -> dict[str, object]:
    points = np.random.default_rng(args.seed).normal(
        size=(args.samples, args.dimension)
    )
    durations = []
    values = None
    details = {}
    for _ in range(args.repeat):
        gc.collect()
        started = perf_counter()
        if args.worker == "dense":
            matrix = cdist(points, points)
            values = {
                sigma: distance_matrix_to_density(matrix, sigma)
                for sigma in args.sigmas
            }
            details = {
                "block_rows": args.samples,
                "block_count": 1,
                "working_bytes": 2 * args.samples**2 * 8,
            }
        else:
            budget = parse_memory_budget(args.memory_budget)
            assert budget is not None
            values, details = NumpyResourceProvider.blockwise_densities(
                points,
                tuple(args.sigmas),
                working_memory_bytes=budget,
                geodesic=False,
            )
        durations.append(perf_counter() - started)
    assert values is not None
    return {
        "mode": args.worker,
        "seconds": float(np.median(durations)),
        "peak_rss_mib": _peak_rss_mib(),
        "retained_bytes": sum(value.nbytes for value in values.values()),
        "working_bytes": details["working_bytes"],
        "block_rows": details["block_rows"],
        "block_count": details["block_count"],
        "densities": {str(sigma): values[sigma].tolist() for sigma in args.sigmas},
    }


def _run_worker(args, mode: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--samples",
        str(args.samples),
        "--dimension",
        str(args.dimension),
        "--sigmas",
        *(str(sigma) for sigma in args.sigmas),
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


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 2 or args.dimension < 1 or args.repeat < 1:
        raise ValueError("samples, dimension, and repeat must be positive")
    if not args.sigmas or any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("sigmas must contain positive values")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    dense = _run_worker(args, "dense")
    blockwise = _run_worker(args, "blockwise")
    maximum_delta = max(
        np.max(
            np.abs(
                np.asarray(dense["densities"][str(sigma)])
                - np.asarray(blockwise["densities"][str(sigma)])
            )
        )
        for sigma in args.sigmas
    )
    for result in (dense, blockwise):
        result.pop("densities")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "sigmas": args.sigmas,
            "memory_budget": args.memory_budget,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "dense": dense,
        "blockwise": blockwise,
        "runtime_ratio": blockwise["seconds"] / dense["seconds"],
        "working_memory_reduction": dense["working_bytes"] / blockwise["working_bytes"],
        "maximum_density_delta": float(maximum_delta),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
