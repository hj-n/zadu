"""Run metadata kept separate from metric score dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .resources import ResourceCache

if TYPE_CHECKING:
    from .planner import ExecutionPlan


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
