"""Run metadata kept separate from metric score dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .resources import ResourceCache, Space

if TYPE_CHECKING:
    from .planner import ExecutionPlan


def _plan_info(
    plan: ExecutionPlan,
    *,
    backend: str,
    device: str,
) -> dict[str, Any]:
    """Return diagnostics shared by single- and repeated-embedding runs."""

    return {
        "exact": True,
        "backend": backend,
        "device": device,
        "estimated_cache_bytes": plan.estimated_cache_bytes,
        "planned_peak_bytes": plan.planned_peak_bytes,
        "memory_budget_bytes": plan.memory_budget_bytes,
        "pair_strategy": (
            plan.pair_plan.strategy.value if plan.pair_plan is not None else None
        ),
        "topographic_strategy": (
            "blockwise_selected_distances"
            if plan.topographic_plan is not None
            else None
        ),
        "rank_comparison_strategy": (
            "fused_gathered_ranks_and_membership"
            if plan.rank_comparison_plan is not None
            else None
        ),
        "neighbor_statistics_strategy": (
            "fused_neighbor_statistics"
            if plan.neighbor_statistics_plan is not None
            else None
        ),
        "snc_strategy": (
            {
                "algorithm": "sparse_batched_iterations",
                "requested_workers": plan.snc_plan.requested_workers,
                "effective_workers": plan.snc_plan.effective_workers,
                "working_bytes": plan.snc_plan.working_bytes,
            }
            if plan.snc_plan is not None
            else None
        ),
    }


def build_run_info(
    *,
    plan: ExecutionPlan,
    cache: ResourceCache,
    backend: str,
    device: str,
    metric_timings: list[tuple[str, float]],
    total_seconds: float,
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
        **_plan_info(plan, backend=backend, device=device),
        "resource_seconds": float(
            sum(
                record.build_seconds
                for record in records.values()
                if record.generation == cache.generation
            )
        ),
        "metric_seconds": float(sum(seconds for _, seconds in metric_timings)),
        "total_seconds": float(total_seconds),
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
    run_infos: list[dict[str, Any]],
    total_seconds: float,
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
        **_plan_info(plan, backend=backend, device=device),
        "mode": "many",
        "batch_strategy": "sequential_shared_original",
        "embedding_count": len(run_infos),
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
        "metrics": metric_timings,
        "runs": indexed_runs,
    }
