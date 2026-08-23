"""Run metadata kept separate from metric score dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .resources import ResourceCache, Space

if TYPE_CHECKING:
    from .batching import BatchExecutionPlan
    from .planner import ExecutionPlan


def _plan_info(
    plan: ExecutionPlan,
    *,
    backend: str,
    device: str,
    dtype: str,
    snc_effective_workers: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Return diagnostics shared by single- and repeated-embedding runs."""

    snc_strategy = None
    if plan.snc_plan is not None:
        effective_workers = (
            plan.snc_plan.effective_workers
            if snc_effective_workers is None
            else snc_effective_workers
        )
        working_by_metric = {
            metric_index: (
                plan.snc_plan.graph_bytes[metric_index]
                + workers * plan.snc_plan.iteration_bytes[metric_index]
            )
            for metric_index, workers in effective_workers.items()
        }
        snc_strategy = {
            "algorithm": "sparse_batched_iterations",
            "requested_workers": plan.snc_plan.requested_workers,
            "effective_workers": effective_workers,
            "working_bytes": max(working_by_metric.values()),
        }

    return {
        "exact": True,
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "estimated_cache_bytes": plan.estimated_cache_bytes,
        "planned_peak_bytes": plan.planned_peak_bytes,
        "memory_budget_bytes": plan.memory_budget_bytes,
        "pair_strategy": (
            plan.pair_plan.strategy.value if plan.pair_plan is not None else None
        ),
        "pair_temporary_budget_bytes": (
            plan.pair_plan.temporary_budget_bytes
            if plan.pair_plan is not None
            and plan.pair_plan.strategy.value == "external"
            else None
        ),
        "pair_planned_temporary_bytes": (
            plan.pair_plan.planned_temporary_bytes
            if plan.pair_plan is not None
            and plan.pair_plan.strategy.value == "external"
            else 0
        ),
        "topographic_strategy": (
            "blockwise_selected_distances"
            if plan.topographic_plan is not None
            else None
        ),
        "rank_comparison_strategy": (
            "blockwise_selected_ranks"
            if plan.rank_comparison_plan is not None
            else None
        ),
        "neighbor_statistics_strategy": (
            "fused_neighbor_statistics"
            if plan.neighbor_statistics_plan is not None
            else None
        ),
        "snc_strategy": snc_strategy,
    }


def build_run_info(
    *,
    plan: ExecutionPlan,
    cache: ResourceCache,
    backend: str,
    device: str,
    dtype: str,
    metric_timings: list[tuple[str, float]],
    total_seconds: float,
    snc_effective_workers: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Create JSON-compatible diagnostics for the most recent measurement."""

    records = cache.records
    resources = []
    for key in plan.resources:
        record = records[key]
        consumer_indices = plan.consumers[key]
        resources.append(
            {
                "kind": key.kind.value,
                "space": key.space.value,
                "k": key.k,
                "parameter": key.parameter,
                "provider": record.provider,
                "dtype": record.dtype,
                "bytes": record.bytes,
                "build_seconds": record.build_seconds,
                "built_in_run": record.generation == cache.generation,
                "reused": record.generation != cache.generation,
                "released": record.released,
                "details": record.details,
                "consumer_count": len(consumer_indices),
                "consumers": [
                    plan.metric_plans[index].metric_id for index in consumer_indices
                ],
                "first_consumer": min(consumer_indices, default=None),
                "last_consumer": max(consumer_indices, default=None),
            }
        )

    return {
        **_plan_info(
            plan,
            backend=backend,
            device=device,
            dtype=dtype,
            snc_effective_workers=snc_effective_workers,
        ),
        "resource_seconds": float(
            sum(
                record.build_seconds
                for record in records.values()
                if record.generation == cache.generation
            )
        ),
        "metric_seconds": float(sum(seconds for _, seconds in metric_timings)),
        "total_seconds": float(total_seconds),
        "provider_timings": _provider_timings(
            record
            for record in records.values()
            if record.generation == cache.generation
        ),
        "resources": resources,
        "metrics": [
            {"id": metric_id, "seconds": float(seconds)}
            for metric_id, seconds in metric_timings
        ],
    }


def build_many_run_info(
    *,
    plan: ExecutionPlan,
    cache: ResourceCache,
    backend: str,
    device: str,
    dtype: str,
    batch_plan: BatchExecutionPlan,
    run_infos: list[dict[str, Any]],
    total_seconds: float,
    snc_effective_workers: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Create JSON-compatible diagnostics for ordered repeated embeddings."""

    original_records = [
        record
        for record in cache.records.values()
        if record.key.space is Space.ORIGINAL
    ]
    indexed_runs = [
        {"embedding_index": index, **run_info}
        for index, run_info in enumerate(run_infos)
    ]
    original_resources_reused = bool(original_records and run_infos) and all(
        resource["reused"]
        for run_info in run_infos
        for resource in run_info["resources"]
        if resource["space"] == "orig"
    )
    metric_timings = []
    for metric_index, metric_plan in enumerate(plan.metric_plans):
        metric_timings.append(
            {
                "id": metric_plan.metric_id,
                "seconds": float(
                    sum(
                        run_info["metrics"][metric_index]["seconds"]
                        for run_info in run_infos
                    )
                ),
            }
        )

    return {
        **_plan_info(
            plan,
            backend=backend,
            device=device,
            dtype=dtype,
            snc_effective_workers=snc_effective_workers,
        ),
        "mode": "many",
        "batch_strategy": batch_plan.strategy,
        "requested_workers": batch_plan.requested_workers,
        "effective_workers": batch_plan.effective_workers,
        "worker_limit_reason": batch_plan.limit_reason,
        "native_threads_per_worker": batch_plan.native_threads_per_worker,
        "provider_batching": batch_plan.provider_batching,
        "native_batch_size": batch_plan.native_batch_size,
        "per_embedding_temporary_bytes": (batch_plan.per_embedding_temporary_bytes),
        "planned_temporary_bytes": batch_plan.planned_temporary_bytes,
        "temporary_budget_bytes": batch_plan.temporary_budget_bytes,
        "embedding_count": len(run_infos),
        "per_embedding_planned_peak_bytes": plan.planned_peak_bytes,
        "shared_original_bytes": batch_plan.shared_original_bytes,
        "per_embedding_peak_bytes": batch_plan.per_embedding_peak_bytes,
        "planned_peak_bytes": batch_plan.planned_peak_bytes,
        "original_resource_count": len(original_records),
        "original_resource_bytes": int(
            sum(record.bytes for record in original_records)
        ),
        "original_resource_seconds": float(
            sum(record.build_seconds for record in original_records)
        ),
        "original_resources_reused": original_resources_reused,
        "original_resource_reuse_events": len(original_records) * len(run_infos),
        "resource_seconds": float(sum(info["resource_seconds"] for info in run_infos)),
        "metric_seconds": float(sum(info["metric_seconds"] for info in run_infos)),
        "total_seconds": float(total_seconds),
        "provider_timings": {
            name: float(
                sum(info["provider_timings"].get(name, 0.0) for info in run_infos)
            )
            for name in _PROVIDER_TIMING_NAMES
        },
        "metrics": metric_timings,
        "runs": indexed_runs,
    }


def build_stream_run_info(
    *,
    plan: ExecutionPlan,
    cache: ResourceCache,
    backend: str,
    device: str,
    dtype: str,
    batch_plan: BatchExecutionPlan,
    embedding_count: int,
    input_consumed_count: int,
    resource_seconds: float,
    metric_seconds: float,
    provider_timings: dict[str, float],
    metric_timings: list[float],
    total_seconds: float,
    original_resources_reused: bool,
    max_in_flight_observed: int,
    stream_complete: bool,
    snc_effective_workers: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Create bounded diagnostics for an iterator-style collection run."""

    original_records = [
        record
        for record in cache.records.values()
        if record.key.space is Space.ORIGINAL
    ]
    return {
        **_plan_info(
            plan,
            backend=backend,
            device=device,
            dtype=dtype,
            snc_effective_workers=snc_effective_workers,
        ),
        "mode": "many_stream",
        "batch_strategy": batch_plan.strategy,
        "requested_workers": batch_plan.requested_workers,
        "effective_workers": batch_plan.effective_workers,
        "worker_limit_reason": batch_plan.limit_reason,
        "native_threads_per_worker": batch_plan.native_threads_per_worker,
        "provider_batching": False,
        "native_batch_size": 1 if embedding_count else 0,
        "per_embedding_temporary_bytes": (batch_plan.per_embedding_temporary_bytes),
        "planned_temporary_bytes": batch_plan.planned_temporary_bytes,
        "temporary_budget_bytes": batch_plan.temporary_budget_bytes,
        "embedding_count": embedding_count,
        "input_consumed_count": input_consumed_count,
        "stream_complete": stream_complete,
        "runs_retained": False,
        "max_in_flight_observed": max_in_flight_observed,
        "per_embedding_planned_peak_bytes": plan.planned_peak_bytes,
        "shared_original_bytes": batch_plan.shared_original_bytes,
        "per_embedding_peak_bytes": batch_plan.per_embedding_peak_bytes,
        "planned_peak_bytes": batch_plan.planned_peak_bytes,
        "original_resource_count": len(original_records),
        "original_resource_bytes": int(
            sum(record.bytes for record in original_records)
        ),
        "original_resource_seconds": float(
            sum(record.build_seconds for record in original_records)
        ),
        "original_resources_reused": original_resources_reused,
        "original_resource_reuse_events": len(original_records) * embedding_count,
        "resource_seconds": float(resource_seconds),
        "metric_seconds": float(metric_seconds),
        "total_seconds": float(total_seconds),
        "provider_timings": {
            name: float(provider_timings.get(name, 0.0))
            for name in _PROVIDER_TIMING_NAMES
        },
        "metrics": [
            {"id": metric_plan.metric_id, "seconds": float(seconds)}
            for metric_plan, seconds in zip(
                plan.metric_plans,
                metric_timings,
                strict=True,
            )
        ],
    }


_PROVIDER_TIMING_NAMES = (
    "input_transfer_seconds",
    "compile_and_first_execution_seconds",
    "warm_execution_seconds",
    "output_transfer_seconds",
)


def _provider_timings(records) -> dict[str, float]:
    totals = {name: 0.0 for name in _PROVIDER_TIMING_NAMES}
    for record in records:
        timings = record.details.get("timings", {})
        for name in totals:
            totals[name] += float(timings.get(name, 0.0))
    return totals
