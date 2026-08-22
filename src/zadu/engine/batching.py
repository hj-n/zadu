"""Memory-bounded planning for ordered repeated-embedding execution."""

from __future__ import annotations

from dataclasses import dataclass

from .planner import ExecutionPlan


@dataclass(frozen=True, slots=True)
class BatchExecutionPlan:
    """One exact collection execution strategy."""

    requested_workers: int
    effective_workers: int
    embedding_count: int
    shared_original_bytes: int
    per_embedding_peak_bytes: int
    planned_peak_bytes: int
    strategy: str
    limit_reason: str | None
    native_threads_per_worker: int | None
    provider_batching: bool
    native_batch_size: int


def build_batch_execution_plan(
    plan: ExecutionPlan,
    *,
    embedding_count: int,
    requested_workers: int,
    parallel_fallback_reason: str | None = None,
    provider_batching: bool = False,
    provider_batch_input_bytes: int = 0,
) -> BatchExecutionPlan:
    """Cap exact embedding workers by input count, safety, and memory budget."""

    if embedding_count < 0:
        raise ValueError("embedding_count must be zero or greater")
    if requested_workers < 1:
        raise ValueError("requested_workers must be at least 1")
    if provider_batch_input_bytes < 0:
        raise ValueError("provider_batch_input_bytes must be zero or greater")

    if embedding_count == 0:
        return BatchExecutionPlan(
            requested_workers=requested_workers,
            effective_workers=0,
            embedding_count=0,
            shared_original_bytes=plan.original_cache_bytes,
            per_embedding_peak_bytes=plan.per_embedding_peak_bytes,
            planned_peak_bytes=plan.original_cache_bytes,
            strategy="sequential_shared_original",
            limit_reason=None,
            native_threads_per_worker=None,
            provider_batching=False,
            native_batch_size=0,
        )

    usable_workers = min(requested_workers, embedding_count)
    limit_reason = "embedding_count" if usable_workers < requested_workers else None
    if provider_batching:
        per_embedding_peak_bytes = (
            plan.per_embedding_peak_bytes + provider_batch_input_bytes
        )
        if plan.memory_budget_bytes is not None and per_embedding_peak_bytes > 0:
            available = max(0, plan.memory_budget_bytes - plan.original_cache_bytes)
            memory_capacity = available // per_embedding_peak_bytes
            if memory_capacity < 2:
                return BatchExecutionPlan(
                    requested_workers=requested_workers,
                    effective_workers=1,
                    embedding_count=embedding_count,
                    shared_original_bytes=plan.original_cache_bytes,
                    per_embedding_peak_bytes=plan.per_embedding_peak_bytes,
                    planned_peak_bytes=(
                        plan.original_cache_bytes + plan.per_embedding_peak_bytes
                    ),
                    strategy="sequential_shared_original",
                    limit_reason="memory_budget",
                    native_threads_per_worker=None,
                    provider_batching=False,
                    native_batch_size=1,
                )
            if memory_capacity < usable_workers:
                usable_workers = memory_capacity
                limit_reason = "memory_budget"
        native_batch_size = max(1, usable_workers)
        return BatchExecutionPlan(
            requested_workers=requested_workers,
            effective_workers=1,
            embedding_count=embedding_count,
            shared_original_bytes=plan.original_cache_bytes,
            per_embedding_peak_bytes=per_embedding_peak_bytes,
            planned_peak_bytes=(
                plan.original_cache_bytes + native_batch_size * per_embedding_peak_bytes
            ),
            strategy="provider_native_batch",
            limit_reason=limit_reason,
            native_threads_per_worker=None,
            provider_batching=True,
            native_batch_size=native_batch_size,
        )

    if parallel_fallback_reason is not None and usable_workers > 1:
        usable_workers = 1
        limit_reason = parallel_fallback_reason

    if plan.memory_budget_bytes is not None and plan.per_embedding_peak_bytes > 0:
        available = max(0, plan.memory_budget_bytes - plan.original_cache_bytes)
        memory_capacity = max(1, available // plan.per_embedding_peak_bytes)
        if memory_capacity < usable_workers:
            usable_workers = memory_capacity
            limit_reason = "memory_budget"

    effective_workers = max(1, usable_workers)
    planned_peak_bytes = (
        plan.original_cache_bytes + effective_workers * plan.per_embedding_peak_bytes
    )
    return BatchExecutionPlan(
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        embedding_count=embedding_count,
        shared_original_bytes=plan.original_cache_bytes,
        per_embedding_peak_bytes=plan.per_embedding_peak_bytes,
        planned_peak_bytes=planned_peak_bytes,
        strategy=(
            "threaded_shared_original"
            if effective_workers > 1
            else "sequential_shared_original"
        ),
        limit_reason=limit_reason,
        native_threads_per_worker=1 if effective_workers > 1 else None,
        provider_batching=False,
        native_batch_size=1,
    )
