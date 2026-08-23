"""Compare scalar and exact bounded-vectorized Gap Index triangle kernels."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay, distance

from zadu.measures import gap_index

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker", choices=("scalar", "vectorized"), help=argparse.SUPPRESS
    )
    return parser


def _peak_rss_mib() -> float:
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / 1024**2


def _normalized_scalar_areas(points, triangles):
    areas = gap_index._scalar_triangle_areas(
        points,
        triangles,
        distance.euclidean,
    )
    return areas / np.sum(areas)


def _score(original_areas, embedded_areas):
    max_areas = np.maximum(original_areas, embedded_areas)
    deformations = (embedded_areas - original_areas) / np.maximum(
        max_areas,
        gap_index._DEFORMATION_EPSILON,
    )
    return float(np.sum(np.abs(deformations) * max_areas) / np.sum(max_areas))


def _worker(args) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    emb = orig[:, :2] + 0.05 * rng.normal(size=(args.samples, 2))
    triangles = Delaunay(emb).simplices
    durations = []
    original_areas = None
    embedded_areas = None
    for _ in range(args.repeat):
        gc.collect()
        started = perf_counter()
        if args.worker == "scalar":
            original_areas = _normalized_scalar_areas(orig, triangles)
            embedded_areas = _normalized_scalar_areas(emb, triangles)
        else:
            original_areas = gap_index._compute_areas(orig, triangles, "euclidean")
            embedded_areas = gap_index._compute_areas(emb, triangles, "euclidean")
        durations.append(perf_counter() - started)
    assert original_areas is not None and embedded_areas is not None
    block_rows = (
        1
        if args.worker == "scalar"
        else min(gap_index._area_block_rows(orig), len(triangles))
    )
    return {
        "mode": args.worker,
        "seconds": float(np.median(durations)),
        "peak_rss_mib": _peak_rss_mib(),
        "triangle_count": len(triangles),
        "block_rows": block_rows,
        "block_count": (
            len(triangles)
            if args.worker == "scalar"
            else math.ceil(len(triangles) / block_rows)
        ),
        "retained_area_bytes": original_areas.nbytes + embedded_areas.nbytes,
        "score": _score(original_areas, embedded_areas),
    }


def _run_worker(args, mode: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--samples",
        str(args.samples),
        "--dimension",
        str(args.dimension),
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
    if args.samples < 3 or args.dimension < 2 or args.repeat < 1:
        raise ValueError(
            "samples must be at least three, dimension at least two, and repeat positive"
        )
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    scalar = _run_worker(args, "scalar")
    vectorized = _run_worker(args, "vectorized")
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "scalar": scalar,
        "vectorized": vectorized,
        "speedup": scalar["seconds"] / vectorized["seconds"],
        "score_delta": abs(scalar["score"] - vectorized["score"]),
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
