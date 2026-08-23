"""Exact external-memory ordering for globally ranked pair metrics."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.spatial.distance import cdist

from zadu.engine.resources import (
    OrderedPairStatistics,
    PairStrategy,
    compact_index_dtype,
)

if TYPE_CHECKING:
    from zadu.engine.planner import PairExecutionPlan


@dataclass(frozen=True, slots=True)
class _Run:
    path: Path
    count: int


@dataclass(slots=True)
class _Workspace:
    root: Path
    budget_bytes: int
    peak_bytes: int = 0
    counter: int = 0

    def new_path(self, prefix: str) -> Path:
        self.counter += 1
        return self.root / f"{prefix}-{self.counter:06d}.dat"

    def account(self) -> None:
        current = sum(path.stat().st_size for path in self.root.iterdir())
        self.peak_bytes = max(self.peak_bytes, current)
        if current > self.budget_bytes:
            raise MemoryError(
                "Exact external pair ordering exceeded temporary_budget "
                f"({current} > {self.budget_bytes})"
            )

    def reserve(self, additional_bytes: int) -> None:
        current = sum(path.stat().st_size for path in self.root.iterdir())
        projected = current + additional_bytes
        self.peak_bytes = max(self.peak_bytes, projected)
        if projected > self.budget_bytes:
            raise MemoryError(
                "Exact external pair ordering would exceed temporary_budget "
                f"({projected} > {self.budget_bytes})"
            )

    def remove(self, path: Path) -> None:
        path.unlink(missing_ok=True)


def build_external_ordered_pair_statistics(
    plan: PairExecutionPlan,
    orig: npt.NDArray,
    emb: npt.NDArray,
    *,
    geodesic: bool,
) -> tuple[OrderedPairStatistics, dict[str, object]]:
    """Return exact ordered metrics while bounding RAM and temporary disk."""

    if plan.strategy is not PairStrategy.EXTERNAL:
        raise RuntimeError("External ordering requires an external pair plan")
    if plan.chunk_pairs is None or plan.chunk_pairs < 1:
        raise RuntimeError("External ordering requires a positive run size")
    if plan.temporary_budget_bytes is None:
        raise RuntimeError("External ordering requires a temporary disk budget")

    record_dtype = np.dtype(
        [
            ("original", np.float64),
            ("embedded", np.float64),
            ("pair_index", compact_index_dtype(plan.pair_count)),
        ]
    )
    temporary = TemporaryDirectory(
        prefix="zadu-pairs-",
        dir=plan.temporary_directory,
    )
    root = Path(temporary.name)
    workspace = _Workspace(root, plan.temporary_budget_bytes)
    timings = {
        "run_generation_seconds": 0.0,
        "merge_seconds": 0.0,
        "rank_seconds": 0.0,
        "isotonic_seconds": 0.0,
    }
    initial_original_runs = 0
    initial_embedded_runs = 0
    merge_passes = 0
    tie_group_count = 0
    pava_block_count = 0
    result = None
    try:
        started = perf_counter()
        original_runs, bounds = _generate_runs(
            plan,
            orig,
            emb,
            record_dtype,
            workspace,
            sort_field="original",
            geodesic=geodesic,
        )
        timings["run_generation_seconds"] += perf_counter() - started
        initial_original_runs = len(original_runs)
        _validate_bounds(plan, bounds)

        started = perf_counter()
        original_run, passes = _merge_to_one(
            original_runs,
            record_dtype,
            "original",
            plan.merge_fan_in,
            workspace,
        )
        timings["merge_seconds"] += perf_counter() - started
        merge_passes += passes

        started = perf_counter()
        non_metric_stress, rank_path, tie_group_count, pava_block_count = (
            _scan_original_order(
                plan,
                original_run,
                record_dtype,
                workspace,
            )
        )
        scan_seconds = perf_counter() - started
        timings["rank_seconds"] += scan_seconds
        if "non_metric_stress" in plan.ordered_metric_ids:
            timings["isotonic_seconds"] += scan_seconds
        workspace.remove(original_run.path)

        spearman_rho = None
        if "spearman_rho" in plan.ordered_metric_ids:
            assert rank_path is not None
            started = perf_counter()
            embedded_runs, embedded_bounds = _generate_runs(
                plan,
                orig,
                emb,
                record_dtype,
                workspace,
                sort_field="embedded",
                geodesic=geodesic,
            )
            timings["run_generation_seconds"] += perf_counter() - started
            initial_embedded_runs = len(embedded_runs)
            _validate_bounds(plan, embedded_bounds)

            started = perf_counter()
            embedded_run, passes = _merge_to_one(
                embedded_runs,
                record_dtype,
                "embedded",
                plan.merge_fan_in,
                workspace,
            )
            merge_passes += passes
            spearman_rho = _spearman_from_run(
                embedded_run,
                rank_path,
                plan.pair_count,
                record_dtype,
                workspace,
            )
            workspace.remove(embedded_run.path)
            workspace.remove(rank_path)
            timings["merge_seconds"] += perf_counter() - started

        result = OrderedPairStatistics(
            spearman_rho=spearman_rho,
            non_metric_stress=non_metric_stress,
            strategy=PairStrategy.EXTERNAL,
            pair_count=plan.pair_count,
        )
    finally:
        peak_bytes = workspace.peak_bytes
        temporary.cleanup()
    workspace_removed = not root.exists()
    if result is None:  # pragma: no cover - exceptions propagate through finally
        raise RuntimeError("External pair ordering did not produce statistics")
    return result, {
        "strategy": PairStrategy.EXTERNAL.value,
        "pair_count": plan.pair_count,
        "working_bytes": plan.working_bytes,
        "run_pairs": plan.chunk_pairs,
        "initial_original_run_count": initial_original_runs,
        "initial_embedded_run_count": initial_embedded_runs,
        "merge_fan_in": plan.merge_fan_in,
        "merge_algorithm": (
            "numba_binary" if plan.merge_fan_in == 2 else "python_k_way"
        ),
        "merge_passes": merge_passes,
        "tie_group_count": tie_group_count,
        "pava_block_count": pava_block_count,
        "temporary_budget_bytes": plan.temporary_budget_bytes,
        "planned_temporary_bytes": plan.planned_temporary_bytes,
        "temporary_bytes_peak": peak_bytes,
        "workspace_removed": workspace_removed,
        "ordering_reused": False,
        "fused_metrics": list(plan.ordered_metric_ids),
        "timings": timings,
    }


def _record_bounds() -> dict[str, float]:
    return {
        "original_min": math.inf,
        "original_max": -math.inf,
        "embedded_min": math.inf,
        "embedded_max": -math.inf,
    }


def _generate_runs(
    plan: PairExecutionPlan,
    orig: npt.NDArray,
    emb: npt.NDArray,
    record_dtype: np.dtype,
    workspace: _Workspace,
    *,
    sort_field: str,
    geodesic: bool,
) -> tuple[list[_Run], dict[str, float]]:
    run_pairs = plan.chunk_pairs
    assert run_pairs is not None
    records = np.empty(run_pairs, dtype=record_dtype)
    runs = []
    bounds = _record_bounds()
    filled = 0
    pair_index = 0

    def flush() -> None:
        nonlocal filled
        if filled == 0:
            return
        order = np.lexsort(
            (
                records["pair_index"][:filled],
                records[sort_field][:filled],
            )
        )
        path = workspace.new_path(f"{sort_field}-run")
        workspace.reserve(filled * record_dtype.itemsize)
        output = np.memmap(path, dtype=record_dtype, mode="w+", shape=(filled,))
        output[:] = records[:filled][order]
        output.flush()
        del output
        runs.append(_Run(path, filled))
        filled = 0
        workspace.account()

    for left in range(orig.shape[0] - 1):
        right_start = left + 1
        while right_start < orig.shape[0]:
            take = min(run_pairs - filled, orig.shape[0] - right_start)
            right_stop = right_start + take
            orig_distances = _original_distance_segment(
                orig,
                left,
                right_start,
                right_stop,
                geodesic=geodesic,
            )
            emb_distances = cdist(
                emb[left : left + 1],
                emb[right_start:right_stop],
            ).reshape(-1)
            target = slice(filled, filled + take)
            records["original"][target] = orig_distances
            records["embedded"][target] = emb_distances
            records["pair_index"][target] = np.arange(
                pair_index,
                pair_index + take,
                dtype=records["pair_index"].dtype,
            )
            bounds["original_min"] = min(
                bounds["original_min"], float(np.min(orig_distances))
            )
            bounds["original_max"] = max(
                bounds["original_max"], float(np.max(orig_distances))
            )
            bounds["embedded_min"] = min(
                bounds["embedded_min"], float(np.min(emb_distances))
            )
            bounds["embedded_max"] = max(
                bounds["embedded_max"], float(np.max(emb_distances))
            )
            filled += take
            pair_index += take
            right_start = right_stop
            if filled == run_pairs:
                flush()
    flush()
    if pair_index != plan.pair_count:
        raise RuntimeError(
            "External pair generator produced an unexpected number of distances "
            f"({pair_index} != {plan.pair_count})"
        )
    return runs, bounds


def _original_distance_segment(
    points: npt.NDArray,
    left: int,
    right_start: int,
    right_stop: int,
    *,
    geodesic: bool,
) -> npt.NDArray:
    if not geodesic:
        return cdist(
            points[left : left + 1],
            points[right_start:right_stop],
        ).reshape(-1)
    if points.shape[1] < 2:
        raise ValueError(
            "geodesic=True requires orig[:, 0] = longitude and "
            "orig[:, 1] = latitude in radians"
        )
    center_longitude = points[left, 0]
    center_latitude = points[left, 1]
    selected_longitude = points[right_start:right_stop, 0]
    selected_latitude = points[right_start:right_stop, 1]
    cosine = np.sin(center_latitude) * np.sin(selected_latitude) + np.cos(
        center_latitude
    ) * np.cos(selected_latitude) * np.cos(
        np.abs(selected_longitude - center_longitude)
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _validate_bounds(plan: PairExecutionPlan, bounds: dict[str, float]) -> None:
    if "spearman_rho" in plan.ordered_metric_ids and (
        bounds["original_min"] == bounds["original_max"]
        or bounds["embedded_min"] == bounds["embedded_max"]
    ):
        raise ValueError("Spearman correlation is undefined for constant distances")
    if "non_metric_stress" in plan.ordered_metric_ids and (
        bounds["original_max"] <= 0 or bounds["embedded_max"] <= 0
    ):
        raise ValueError(
            "Non-metric stress is undefined when all pairwise distances are zero"
        )


def _merge_to_one(
    runs: list[_Run],
    record_dtype: np.dtype,
    sort_field: str,
    fan_in: int,
    workspace: _Workspace,
) -> tuple[_Run, int]:
    runs, passes = _reduce_runs(
        runs,
        record_dtype,
        sort_field,
        fan_in,
        workspace,
    )
    if len(runs) == 1:
        return runs[0], passes
    return (
        _merge_group(runs, record_dtype, sort_field, workspace),
        passes + 1,
    )


def _reduce_runs(
    runs: list[_Run],
    record_dtype: np.dtype,
    sort_field: str,
    fan_in: int,
    workspace: _Workspace,
) -> tuple[list[_Run], int]:
    passes = 0
    while len(runs) > fan_in:
        merged = []
        for start in range(0, len(runs), fan_in):
            group = runs[start : start + fan_in]
            merged.append(
                group[0]
                if len(group) == 1
                else _merge_group(
                    group,
                    record_dtype,
                    sort_field,
                    workspace,
                )
            )
        runs = merged
        passes += 1
    return runs, passes


def _merge_group(
    runs: list[_Run],
    record_dtype: np.dtype,
    sort_field: str,
    workspace: _Workspace,
) -> _Run:
    inputs = [
        np.memmap(run.path, dtype=record_dtype, mode="r", shape=(run.count,))
        for run in runs
    ]
    output_count = sum(run.count for run in runs)
    output_path = workspace.new_path(f"{sort_field}-merge")
    workspace.reserve(output_count * record_dtype.itemsize)
    output = np.memmap(
        output_path,
        dtype=record_dtype,
        mode="w+",
        shape=(output_count,),
    )
    if len(runs) == 2:
        _merge_two_numba(
            inputs[0][sort_field],
            inputs[0]["pair_index"],
            inputs[0]["original"],
            inputs[0]["embedded"],
            inputs[1][sort_field],
            inputs[1]["pair_index"],
            inputs[1]["original"],
            inputs[1]["embedded"],
            output["original"],
            output["embedded"],
            output["pair_index"],
        )
    else:
        positions = [0] * len(runs)
        heap = []
        for run_index, values in enumerate(inputs):
            if len(values):
                heapq.heappush(
                    heap,
                    (
                        float(values[sort_field][0]),
                        int(values["pair_index"][0]),
                        run_index,
                    ),
                )
        output_index = 0
        while heap:
            _, _, run_index = heapq.heappop(heap)
            position = positions[run_index]
            output[output_index] = inputs[run_index][position]
            output_index += 1
            position += 1
            positions[run_index] = position
            if position < runs[run_index].count:
                values = inputs[run_index]
                heapq.heappush(
                    heap,
                    (
                        float(values[sort_field][position]),
                        int(values["pair_index"][position]),
                        run_index,
                    ),
                )
    output.flush()
    del output
    del inputs
    workspace.account()
    for run in runs:
        workspace.remove(run.path)
    return _Run(output_path, output_count)


def _scan_original_order(
    plan: PairExecutionPlan,
    run: _Run,
    record_dtype: np.dtype,
    workspace: _Workspace,
) -> tuple[float | None, Path | None, int, int]:
    records = np.memmap(run.path, dtype=record_dtype, mode="r", shape=(run.count,))
    needs_spearman = "spearman_rho" in plan.ordered_metric_ids
    needs_stress = "non_metric_stress" in plan.ordered_metric_ids
    rank_path = None
    ranks = None
    if needs_spearman:
        rank_path = workspace.new_path("original-ranks")
        workspace.reserve(plan.pair_count * np.dtype(np.float64).itemsize)
        ranks = np.memmap(
            rank_path,
            dtype=np.float64,
            mode="w+",
            shape=(plan.pair_count,),
        )
        workspace.account()

    pava_path = None
    pava = None
    pava_dtype = np.dtype(
        [("weight", np.int64), ("sum", np.float64), ("sum_squared", np.float64)]
    )
    if needs_stress:
        pava_path = workspace.new_path("pava")
        workspace.reserve(plan.pair_count * pava_dtype.itemsize)
        pava = np.memmap(
            pava_path,
            dtype=pava_dtype,
            mode="w+",
            shape=(plan.pair_count,),
        )
        workspace.account()

    rank_values = ranks if ranks is not None else np.empty(0, dtype=np.float64)
    pava_weights = pava["weight"] if pava is not None else np.empty(0, dtype=np.int64)
    pava_sums = pava["sum"] if pava is not None else np.empty(0, dtype=np.float64)
    pava_squares = (
        pava["sum_squared"] if pava is not None else np.empty(0, dtype=np.float64)
    )
    group_count, total_emb_squared = _scan_groups_numba(
        records["original"],
        records["embedded"],
        records["pair_index"],
        rank_values,
        ranks is not None,
        pava_weights,
        pava_sums,
        pava_squares,
        pava is not None,
    )
    top = (
        _pava_numba(pava_weights, pava_sums, pava_squares, group_count)
        if pava is not None
        else 0
    )

    if ranks is not None:
        ranks.flush()
        del ranks
    non_metric_stress = None
    if pava is not None:
        raw_stress = _pava_stress_numba(
            pava_weights,
            pava_sums,
            pava_squares,
            top,
        )
        non_metric_stress = math.sqrt(max(0.0, raw_stress) / total_emb_squared)
        del pava
        assert pava_path is not None
        workspace.remove(pava_path)
    del records
    return non_metric_stress, rank_path, group_count, top


def _spearman_from_run(
    run: _Run,
    rank_path: Path,
    pair_count: int,
    record_dtype: np.dtype,
    workspace: _Workspace,
) -> float:
    original_ranks = np.memmap(
        rank_path,
        dtype=np.float64,
        mode="r",
        shape=(pair_count,),
    )
    records = np.memmap(
        run.path,
        dtype=record_dtype,
        mode="r",
        shape=(run.count,),
    )
    result = _spearman_numba(
        records["embedded"],
        records["pair_index"],
        original_ranks,
    )
    del records
    del original_ranks
    del workspace
    return result


@njit
def _merge_two_numba(
    left_keys,
    left_indices,
    left_original,
    left_embedded,
    right_keys,
    right_indices,
    right_original,
    right_embedded,
    output_original,
    output_embedded,
    output_indices,
):
    left = 0
    right = 0
    output = 0
    while left < left_keys.size and right < right_keys.size:
        take_left = left_keys[left] < right_keys[right] or (
            left_keys[left] == right_keys[right]
            and left_indices[left] <= right_indices[right]
        )
        if take_left:
            output_original[output] = left_original[left]
            output_embedded[output] = left_embedded[left]
            output_indices[output] = left_indices[left]
            left += 1
        else:
            output_original[output] = right_original[right]
            output_embedded[output] = right_embedded[right]
            output_indices[output] = right_indices[right]
            right += 1
        output += 1
    while left < left_keys.size:
        output_original[output] = left_original[left]
        output_embedded[output] = left_embedded[left]
        output_indices[output] = left_indices[left]
        left += 1
        output += 1
    while right < right_keys.size:
        output_original[output] = right_original[right]
        output_embedded[output] = right_embedded[right]
        output_indices[output] = right_indices[right]
        right += 1
        output += 1


@njit
def _scan_groups_numba(
    original_distances,
    embedded_distances,
    pair_indices,
    ranks,
    write_ranks,
    pava_weights,
    pava_sums,
    pava_squares,
    write_pava,
):
    start = 0
    group = 0
    total_embedded_squared = 0.0
    while start < original_distances.size:
        stop = start + 1
        while (
            stop < original_distances.size
            and original_distances[stop] == original_distances[start]
        ):
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        embedded_sum = 0.0
        embedded_squared = 0.0
        for index in range(start, stop):
            value = embedded_distances[index]
            embedded_sum += value
            embedded_squared += value * value
            if write_ranks:
                ranks[pair_indices[index]] = average_rank
        total_embedded_squared += embedded_squared
        if write_pava:
            pava_weights[group] = stop - start
            pava_sums[group] = embedded_sum
            pava_squares[group] = embedded_squared
        group += 1
        start = stop
    return group, total_embedded_squared


@njit
def _pava_numba(weights, sums, squares, group_count):
    top = 0
    for group in range(group_count):
        weights[top] = weights[group]
        sums[top] = sums[group]
        squares[top] = squares[group]
        top += 1
        while top > 1:
            left_mean = sums[top - 2] / weights[top - 2]
            right_mean = sums[top - 1] / weights[top - 1]
            if left_mean <= right_mean:
                break
            weights[top - 2] += weights[top - 1]
            sums[top - 2] += sums[top - 1]
            squares[top - 2] += squares[top - 1]
            top -= 1
    return top


@njit
def _pava_stress_numba(weights, sums, squares, block_count):
    result = 0.0
    for block in range(block_count):
        result += squares[block] - sums[block] * sums[block] / weights[block]
    return result


@njit
def _spearman_numba(embedded_distances, pair_indices, original_ranks):
    sum_x = 0.0
    sum_x_squared = 0.0
    sum_y = 0.0
    sum_y_squared = 0.0
    sum_product = 0.0
    start = 0
    while start < embedded_distances.size:
        stop = start + 1
        while (
            stop < embedded_distances.size
            and embedded_distances[stop] == embedded_distances[start]
        ):
            stop += 1
        embedded_rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            original_rank = original_ranks[pair_indices[index]]
            sum_x += original_rank
            sum_x_squared += original_rank * original_rank
            sum_y += embedded_rank
            sum_y_squared += embedded_rank * embedded_rank
            sum_product += original_rank * embedded_rank
        start = stop
    count = float(embedded_distances.size)
    covariance = sum_product - sum_x * sum_y / count
    variance_x = sum_x_squared - sum_x * sum_x / count
    variance_y = sum_y_squared - sum_y * sum_y / count
    return covariance / math.sqrt(variance_x * variance_y)
