"""Compare one ZADU source tree with historical releases in isolated processes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

SUITES = {
    "core": [
        {"id": "tnc", "params": {"k": None}},
        {"id": "mrre", "params": {"k": None}},
        {"id": "lcmc", "params": {"k": None}},
        {"id": "nh", "params": {"k": None}},
    ],
    "pair": [
        {"id": "stress"},
        {"id": "pr"},
    ],
    "topology": [
        {"id": "topo", "params": {"k": None}},
        {"id": "proc", "params": {"k": None}},
    ],
    "representative": [
        {"id": "stress"},
        {"id": "pr"},
        {"id": "tnc", "params": {"k": None}},
        {"id": "mrre", "params": {"k": None}},
        {"id": "lcmc", "params": {"k": None}},
        {"id": "nh", "params": {"k": None}},
        {"id": "topo", "params": {"k": None}},
        {"id": "proc", "params": {"k": None}},
    ],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        metavar="LABEL=PATH",
        help="Source checkout to benchmark; repeat in baseline-to-current order",
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(SUITES),
        help="Suite to run; repeat as needed (default: every suite)",
    )
    parser.add_argument("--embeddings", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--accelerated-label",
        help="Source label that should use the requested non-default backend",
    )
    parser.add_argument("--backend", choices=("numpy", "mlx", "torch"), default="numpy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype")
    parser.add_argument("--embedding-workers", type=int, default=1)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-suite", choices=tuple(SUITES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", default="numpy", help=argparse.SUPPRESS)
    return parser


def _parse_sources(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        raise ValueError("At least one --source LABEL=PATH is required")
    sources = []
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--source must use LABEL=PATH")
        path = Path(raw_path).resolve()
        if not (path / "src" / "zadu").is_dir():
            raise ValueError(f"Source checkout has no src/zadu package: {path}")
        sources.append((label, path))
    if len({label for label, _ in sources}) != len(sources):
        raise ValueError("Source labels must be unique")
    return sources


def _specs(suite: str, k: int) -> list[dict]:
    return [
        {
            **spec,
            **(
                {"params": {"k": k}}
                if "params" in spec and spec["params"].get("k") is None
                else {}
            ),
        }
        for spec in SUITES[suite]
    ]


def _as_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    return value


def _peak_rss_bytes() -> int:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _worker(args: argparse.Namespace) -> None:
    try:
        from zadu import ZADU
    except ImportError:
        from zadu.zadu import ZADU

    n_samples = args.worker_samples
    suite = args.worker_suite
    rng = np.random.default_rng(args.seed)
    original = rng.normal(size=(n_samples, args.dimension))
    projections = [
        rng.normal(size=(args.dimension, args.embedding_dimension))
        for _ in range(args.embeddings)
    ]
    embeddings = [
        original @ projection
        + rng.normal(scale=0.03, size=(n_samples, args.embedding_dimension))
        for projection in projections
    ]
    labels = np.arange(n_samples) % 5
    specs = _specs(suite, args.k)

    construction_started = perf_counter()
    if args.worker_backend == "numpy":
        runner = ZADU(specs, original)
    else:
        from zadu import ExecutionConfig

        runner = ZADU(
            specs,
            original,
            execution=ExecutionConfig(
                backend=args.worker_backend,
                device=args.device,
                dtype=args.dtype,
                embedding_workers=args.embedding_workers,
            ),
        )
    construction_seconds = perf_counter() - construction_started

    def evaluate_collection():
        return [runner.measure(embedding, labels) for embedding in embeddings]

    first_started = perf_counter()
    first_scores = evaluate_collection()
    first_measure_seconds = perf_counter() - first_started
    durations = []
    warm_scores = first_scores
    for _ in range(args.repeat):
        started = perf_counter()
        warm_scores = evaluate_collection()
        durations.append(perf_counter() - started)

    payload = {
        "label": args.worker_label,
        "suite": suite,
        "samples": n_samples,
        "dimension": args.dimension,
        "embedding_dimension": args.embedding_dimension,
        "k": args.k,
        "embeddings": args.embeddings,
        "repeat": args.repeat,
        "construction_seconds": construction_seconds,
        "first_measure_seconds": first_measure_seconds,
        "cold_total_seconds": construction_seconds + first_measure_seconds,
        "warm_seconds": statistics.median(durations),
        "warm_samples_seconds": durations,
        "peak_rss_bytes": _peak_rss_bytes(),
        "scores": _as_python(warm_scores),
    }
    print(json.dumps(payload, allow_nan=False))


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_worker(
    args: argparse.Namespace,
    *,
    label: str,
    source: Path,
    suite: str,
    samples: int,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-label",
        label,
        "--worker-suite",
        suite,
        "--worker-samples",
        str(samples),
        "--worker-backend",
        args.backend if label == args.accelerated_label else "numpy",
        "--dimension",
        str(args.dimension),
        "--embedding-dimension",
        str(args.embedding_dimension),
        "--k",
        str(args.k),
        "--embeddings",
        str(args.embeddings),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--embedding-workers",
        str(args.embedding_workers),
    ]
    if args.dtype is not None:
        command.extend(["--dtype", args.dtype])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source / "src")
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {label}/{suite}/n={samples}:\n" f"{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _flatten_numbers(value):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_numbers(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def _score_delta(left, right) -> float | None:
    left_values = list(_flatten_numbers(left))
    right_values = list(_flatten_numbers(right))
    if len(left_values) != len(right_values):
        return None
    return max(
        (
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            )
        ),
        default=0.0,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        _worker(args)
        return
    sources = _parse_sources(args.source)
    suites = args.suite or list(SUITES)
    if (
        min(
            *args.samples,
            args.dimension,
            args.embedding_dimension,
            args.k,
            args.embeddings,
            args.repeat,
        )
        < 1
    ):
        raise ValueError(
            "sizes, dimensions, k, embeddings, and repeat must be positive"
        )
    if any(samples <= 2 * args.k for samples in args.samples):
        raise ValueError("every sample count must be greater than two times k")
    if args.embedding_workers < 1:
        raise ValueError("embedding-workers must be positive")
    labels = {label for label, _ in sources}
    if args.accelerated_label is not None and args.accelerated_label not in labels:
        raise ValueError("accelerated-label must match one source label")
    if args.backend != "numpy" and args.accelerated_label is None:
        raise ValueError("a non-default backend requires --accelerated-label")

    results = []
    for suite in suites:
        for samples in args.samples:
            for label, source in sources:
                results.append(
                    _run_worker(
                        args,
                        label=label,
                        source=source,
                        suite=suite,
                        samples=samples,
                    )
                )

    current_label = sources[-1][0]
    grouped = {}
    for result in results:
        grouped.setdefault((result["suite"], result["samples"]), {})[
            result["label"]
        ] = result
    comparisons = []
    for (suite, samples), group in grouped.items():
        current = group[current_label]
        for baseline_label, _ in sources[:-1]:
            baseline = group[baseline_label]
            comparisons.append(
                {
                    "suite": suite,
                    "samples": samples,
                    "baseline": baseline_label,
                    "current": current_label,
                    "cold_speedup": baseline["cold_total_seconds"]
                    / current["cold_total_seconds"],
                    "warm_speedup": baseline["warm_seconds"] / current["warm_seconds"],
                    "peak_rss_ratio": baseline["peak_rss_bytes"]
                    / current["peak_rss_bytes"],
                    "maximum_score_delta": _score_delta(
                        baseline["scores"],
                        current["scores"],
                    ),
                }
            )

    payload = {
        "metadata": {
            "sources": [
                {
                    "label": label,
                    "path": str(source),
                    "revision": _git_revision(source),
                }
                for label, source in sources
            ],
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "k": args.k,
            "suites": suites,
            "embeddings": args.embeddings,
            "repeat": args.repeat,
            "seed": args.seed,
            "accelerated_label": args.accelerated_label,
            "backend": args.backend,
            "device": args.device,
            "dtype": args.dtype,
            "embedding_workers": args.embedding_workers,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "comparison_scope": (
                "same interpreter and installed dependency environment; "
                "only the ZADU source checkout changes"
            ),
        },
        "results": results,
        "comparisons": comparisons,
    }
    output = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output)


if __name__ == "__main__":
    main()
