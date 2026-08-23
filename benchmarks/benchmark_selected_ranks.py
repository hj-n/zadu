"""Compare full inverse rankings with an exact blockwise selected-rank oracle.

This benchmark is the executable design gate for the post-0.5.1 rank-resource
work.  The blockwise implementation intentionally lives outside the installed
package until its exactness and performance have been reviewed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from zadu.engine.resources import RankComparisons, compact_index_dtype
from zadu.measures import (
    class_aware_trustworthiness_continuity,
    mean_relative_rank_error,
    trustworthiness_continuity,
)
from zadu.measures.utils.knn import knn_with_ranking
from zadu.measures.utils.vectorized import gather_ranks, rowwise_membership

try:
    import resource
except ImportError:  # pragma: no cover - only exercised on Windows
    resource = None

_BYTES_PER_BLOCK_CELL = 24


@dataclass(frozen=True, slots=True)
class OracleResult:
    comparisons: RankComparisons
    persistent_bytes: int
    planned_working_bytes: int | None
    block_count: int
    block_rows: int | None


def _validate_inputs(
    orig: npt.ArrayLike,
    emb: npt.ArrayLike,
    k: int,
    membership_ks: tuple[int, ...],
) -> tuple[npt.NDArray, npt.NDArray, tuple[int, ...]]:
    original = np.asarray(orig)
    embedded = np.asarray(emb)
    if original.ndim != 2 or embedded.ndim != 2:
        raise ValueError("orig and emb must both be 2D arrays")
    if original.shape[0] != embedded.shape[0]:
        raise ValueError("orig and emb must contain the same number of samples")
    if not np.all(np.isfinite(original)) or not np.all(np.isfinite(embedded)):
        raise ValueError("orig and emb must contain only finite values")
    n_samples = original.shape[0]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 1 or k >= n_samples:
        raise ValueError("k must satisfy 1 <= k < n_samples")
    normalized_ks = tuple(sorted(set(membership_ks)))
    if any(value < 1 or value > k for value in normalized_ks):
        raise ValueError("membership_ks must satisfy 1 <= value <= k")
    return original, embedded, normalized_ks


def _memberships(
    orig_indices: npt.NDArray,
    emb_indices: npt.NDArray,
    membership_ks: tuple[int, ...],
    *,
    max_working_bytes: int,
) -> tuple[dict[int, npt.NDArray], dict[int, npt.NDArray]]:
    emb_in_orig = {}
    orig_in_emb = {}
    for requested_k in membership_ks:
        emb_in_orig[requested_k] = rowwise_membership(
            emb_indices[:, :requested_k],
            orig_indices[:, :requested_k],
            max_block_bytes=max_working_bytes,
        )
        orig_in_emb[requested_k] = rowwise_membership(
            orig_indices[:, :requested_k],
            emb_indices[:, :requested_k],
            max_block_bytes=max_working_bytes,
        )
    return emb_in_orig, orig_in_emb


def _comparison_nbytes(comparisons: RankComparisons) -> int:
    arrays = (
        comparisons.orig_ranks_of_emb,
        comparisons.emb_ranks_of_orig,
        comparisons.orig_indices,
        comparisons.emb_indices,
        *comparisons.emb_in_orig.values(),
        *comparisons.orig_in_emb.values(),
    )
    return int(sum(value.nbytes for value in arrays))


def full_ranking_oracle(
    orig: npt.ArrayLike,
    emb: npt.ArrayLike,
    k: int,
    *,
    membership_ks: tuple[int, ...] = (),
    max_working_bytes: int = 64 * 1024**2,
) -> OracleResult:
    """Build the exact 0.5.1 full-ranking representation."""

    original, embedded, membership_ks = _validate_inputs(orig, emb, k, membership_ks)
    n_samples = original.shape[0]
    index_dtype = compact_index_dtype(n_samples)
    orig_indices, orig_ranking = knn_with_ranking(original, k)
    emb_indices, emb_ranking = knn_with_ranking(embedded, k)
    orig_indices = orig_indices.astype(index_dtype, copy=False)
    emb_indices = emb_indices.astype(index_dtype, copy=False)
    orig_ranking = orig_ranking.astype(index_dtype, copy=False)
    emb_ranking = emb_ranking.astype(index_dtype, copy=False)
    emb_in_orig, orig_in_emb = _memberships(
        orig_indices,
        emb_indices,
        membership_ks,
        max_working_bytes=max_working_bytes,
    )
    comparisons = RankComparisons(
        orig_ranks_of_emb=gather_ranks(orig_ranking, emb_indices),
        emb_ranks_of_orig=gather_ranks(emb_ranking, orig_indices),
        orig_indices=orig_indices,
        emb_indices=emb_indices,
        emb_in_orig=emb_in_orig,
        orig_in_emb=orig_in_emb,
    )
    persistent_bytes = (
        _comparison_nbytes(comparisons) + orig_ranking.nbytes + emb_ranking.nbytes
    )
    return OracleResult(
        comparisons=comparisons,
        persistent_bytes=persistent_bytes,
        planned_working_bytes=None,
        block_count=1,
        block_rows=None,
    )


def blockwise_selected_rank_oracle(
    orig: npt.ArrayLike,
    emb: npt.ArrayLike,
    k: int,
    *,
    membership_ks: tuple[int, ...] = (),
    max_working_bytes: int = 64 * 1024**2,
) -> OracleResult:
    """Keep only cross-space ranks while preserving full stable-sort semantics."""

    original, embedded, membership_ks = _validate_inputs(orig, emb, k, membership_ks)
    n_samples = original.shape[0]
    bytes_per_row = n_samples * _BYTES_PER_BLOCK_CELL
    if max_working_bytes < bytes_per_row:
        raise MemoryError(
            "selected-rank sorting requires enough working memory for one row"
        )
    block_rows = max(1, min(n_samples, max_working_bytes // bytes_per_row))
    index_dtype = compact_index_dtype(n_samples)
    orig_indices = np.empty((n_samples, k), dtype=index_dtype)
    emb_indices = np.empty((n_samples, k), dtype=index_dtype)
    orig_ranks_of_emb = np.empty((n_samples, k), dtype=index_dtype)
    emb_ranks_of_orig = np.empty((n_samples, k), dtype=index_dtype)
    positions = np.arange(n_samples, dtype=np.intp)
    block_count = 0

    for start in range(0, n_samples, block_rows):
        stop = min(start + block_rows, n_samples)
        local_rows = np.arange(stop - start)[:, None]
        global_rows = np.arange(start, stop)

        orig_distances = cdist(original[start:stop], original)
        orig_distances[np.arange(stop - start), global_rows] = -np.inf
        orig_order = np.argsort(orig_distances, axis=1, kind="stable")
        orig_indices[start:stop] = orig_order[:, 1 : k + 1]
        del orig_distances

        emb_distances = cdist(embedded[start:stop], embedded)
        emb_distances[np.arange(stop - start), global_rows] = -np.inf
        emb_order = np.argsort(emb_distances, axis=1, kind="stable")
        emb_indices[start:stop] = emb_order[:, 1 : k + 1]
        del emb_distances

        inverse = np.empty_like(orig_order)
        inverse[local_rows, orig_order] = positions
        orig_ranks_of_emb[start:stop] = inverse[local_rows, emb_indices[start:stop]]
        inverse[local_rows, emb_order] = positions
        emb_ranks_of_orig[start:stop] = inverse[local_rows, orig_indices[start:stop]]
        block_count += 1

    emb_in_orig, orig_in_emb = _memberships(
        orig_indices,
        emb_indices,
        membership_ks,
        max_working_bytes=max_working_bytes,
    )
    comparisons = RankComparisons(
        orig_ranks_of_emb=orig_ranks_of_emb,
        emb_ranks_of_orig=emb_ranks_of_orig,
        orig_indices=orig_indices,
        emb_indices=emb_indices,
        emb_in_orig=emb_in_orig,
        orig_in_emb=orig_in_emb,
    )
    return OracleResult(
        comparisons=comparisons,
        persistent_bytes=_comparison_nbytes(comparisons),
        planned_working_bytes=block_rows * bytes_per_row,
        block_count=block_count,
        block_rows=block_rows,
    )


def _scores(result: OracleResult, orig, emb, labels, k):
    comparisons = result.comparisons
    return {
        **trustworthiness_continuity.measure(
            orig, emb, k=k, rank_comparisons=comparisons
        ),
        **class_aware_trustworthiness_continuity.measure(
            orig, emb, labels, k=k, rank_comparisons=comparisons
        ),
        **mean_relative_rank_error.measure(
            orig, emb, k=k, rank_comparisons=comparisons
        ),
    }


def _digest(result: OracleResult) -> str:
    digest = hashlib.sha256()
    comparisons = result.comparisons
    arrays = (
        comparisons.orig_indices,
        comparisons.emb_indices,
        comparisons.orig_ranks_of_emb,
        comparisons.emb_ranks_of_orig,
        *comparisons.emb_in_orig.values(),
        *comparisons.orig_in_emb.values(),
    )
    for value in arrays:
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    if resource is None:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak / (1024**2)


def _median(function, repeat):
    durations = []
    value = None
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        value = function()
        durations.append(perf_counter() - start)
    return float(np.median(durations)), value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--embedding-dimension", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--memory-budget", type=int, default=64 * 1024**2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker", choices=("full", "selected"), help=argparse.SUPPRESS
    )
    return parser


def _worker(args):
    rng = np.random.default_rng(args.seed)
    orig = rng.normal(size=(args.samples, args.dimension))
    projection = rng.normal(size=(args.dimension, args.embedding_dimension))
    emb = orig @ projection + 0.05 * rng.normal(
        size=(args.samples, args.embedding_dimension)
    )
    labels = np.arange(args.samples) % 10
    function = (
        full_ranking_oracle if args.worker == "full" else blockwise_selected_rank_oracle
    )
    warmup_samples = min(args.samples, 30)
    warmup_k = min(args.neighbors, warmup_samples - 1)
    warmup_budget = max(args.memory_budget, warmup_samples * _BYTES_PER_BLOCK_CELL)
    function(
        orig[:warmup_samples],
        emb[:warmup_samples],
        warmup_k,
        membership_ks=(warmup_k,),
        max_working_bytes=warmup_budget,
    )
    seconds, result = _median(
        lambda: function(
            orig,
            emb,
            args.neighbors,
            membership_ks=(args.neighbors,),
            max_working_bytes=args.memory_budget,
        ),
        args.repeat,
    )
    return {
        "mode": args.worker,
        "seconds": seconds,
        "peak_rss_mib": _peak_rss_mib(),
        "persistent_bytes": result.persistent_bytes,
        "planned_working_bytes": result.planned_working_bytes,
        "block_count": result.block_count,
        "block_rows": result.block_rows,
        "digest": _digest(result),
        "scores": _scores(result, orig, emb, labels, args.neighbors),
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
        "--neighbors",
        str(args.neighbors),
        "--memory-budget",
        str(args.memory_budget),
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
    if args.samples < 3 or args.dimension < 1 or args.embedding_dimension < 1:
        raise ValueError(
            "samples must be at least three and dimensions must be positive"
        )
    if args.neighbors < 1 or args.neighbors >= args.samples / 2:
        raise ValueError("neighbors must satisfy 1 <= neighbors < samples / 2")
    if args.memory_budget < 1 or args.repeat < 1:
        raise ValueError("memory-budget and repeat must be positive")
    if args.worker is not None:
        print(json.dumps(_worker(args)))
        return

    full = _run_worker(args, "full")
    selected = _run_worker(args, "selected")
    maximum_score_delta = max(
        abs(value - selected["scores"][name]) for name, value in full["scores"].items()
    )
    payload = {
        "metadata": {
            "samples": args.samples,
            "dimension": args.dimension,
            "embedding_dimension": args.embedding_dimension,
            "neighbors": args.neighbors,
            "memory_budget": args.memory_budget,
            "repeat": args.repeat,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "full_ranking": full,
        "selected_rank": selected,
        "runtime_ratio": selected["seconds"] / full["seconds"],
        "persistent_memory_reduction": (
            full["persistent_bytes"] / selected["persistent_bytes"]
        ),
        "exact_digest_match": full["digest"] == selected["digest"],
        "maximum_score_delta": maximum_score_delta,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
