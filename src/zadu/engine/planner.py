"""Deterministic planning and deduplication of exact metric resources."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .resources import (
    PairStrategy,
    ResourceKey,
    ResourceKind,
    ResourceRequest,
    ResourceRequirement,
    Space,
)

if TYPE_CHECKING:
    from zadu.registry import MetricDefinition


DEFAULT_PAIR_CACHE_BYTES = 256 * 1024**2
DEFAULT_PAIR_WORK_BYTES = 64 * 1024**2
PAIR_WORK_BYTES_PER_CELL = 64
CONDENSED_WORK_BYTES_PER_PAIR = 48
PAIR_ORDER_BYTES_PER_PAIR = np.dtype(np.intp).itemsize + np.dtype(np.float64).itemsize
ORDERED_WORK_BYTES_PER_PAIR = 64
DEFAULT_RESOURCE_WORK_BYTES = 64 * 1024**2
STABLE_KNN_WORK_BYTES_PER_CELL = 32
TOPOGRAPHIC_WORK_BYTES_PER_SELECTED_DISTANCE = 16


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    requirement: ResourceRequirement
    requests: tuple[ResourceRequest, ...]


@dataclass(frozen=True, slots=True)
class MetricPlan:
    metric_id: str
    bindings: tuple[ResourceBinding, ...]


@dataclass(frozen=True, slots=True)
class PairExecutionPlan:
    strategy: PairStrategy
    statistics_key: ResourceKey | None
    ordered_statistics_key: ResourceKey | None
    order_key: ResourceKey | None
    original_source_key: ResourceKey | None
    embedded_source_key: ResourceKey | None
    pair_count: int
    block_rows: int | None
    chunk_pairs: int | None
    working_bytes: int
    metric_ids: tuple[str, ...]
    ordered_metric_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopographicExecutionPlan:
    statistics_key: ResourceKey
    original_knn_key: ResourceKey
    embedded_knn_key: ResourceKey
    k: int
    block_rows: int
    work_budget_bytes: int
    working_bytes: int
    metric_ids: tuple[str, ...]
    requested_ks: tuple[int, ...]
    geodesic: bool


@dataclass(slots=True)
class ExecutionPlan:
    resources: tuple[ResourceKey, ...]
    metric_plans: tuple[MetricPlan, ...]
    request_to_key: dict[ResourceRequest, ResourceKey]
    consumers: dict[ResourceKey, tuple[int, ...]]
    estimated_cache_bytes: int
    planned_peak_bytes: int
    memory_budget_bytes: int | None
    pair_plan: PairExecutionPlan | None
    topographic_plan: TopographicExecutionPlan | None
    resource_working_bytes: dict[ResourceKey, int]

    def resolve(self, request: ResourceRequest) -> ResourceKey:
        return self.request_to_key[request]

    def resources_for(self, space: Space) -> tuple[ResourceKey, ...]:
        return tuple(key for key in self.resources if key.space is space)


def build_execution_plan(
    definitions: list[MetricDefinition],
    specs: list[dict],
    *,
    n_samples: int,
    original_dimension: int,
    default_k: int,
    memory_budget: int | None = None,
    geodesic: bool = False,
) -> ExecutionPlan:
    """Build one deterministic plan and collapse compatible kNN requests."""

    metric_plans = []
    all_requests: list[ResourceRequest] = []
    for definition, spec in zip(definitions, specs, strict=True):
        bindings = []
        for requirement in definition.resources:
            k = spec["params"].get("k", default_k) if requirement.uses_k else None
            requests = tuple(
                ResourceRequest(requirement.kind, space, k)
                for space in requirement.spaces
            )
            bindings.append(ResourceBinding(requirement, requests))
            all_requests.extend(requests)
        metric_plans.append(MetricPlan(definition.id, tuple(bindings)))

    maxima: dict[tuple[ResourceKind, Space], int] = {}
    requested_pairs = {(request.kind, request.space) for request in all_requests}
    for request in all_requests:
        if request.k is not None:
            pair = (request.kind, request.space)
            maxima[pair] = max(maxima.get(pair, -1), request.k)

    pair_count = n_samples * (n_samples - 1) // 2
    has_pair_statistics = (
        ResourceKind.PAIR_STATISTICS,
        Space.PAIRED,
    ) in requested_pairs
    has_ordered_pair_statistics = (
        ResourceKind.ORDERED_PAIR_STATISTICS,
        Space.PAIRED,
    ) in requested_pairs
    has_pair_resources = has_pair_statistics or has_ordered_pair_statistics
    has_other_exact_resources = any(
        request.space is not Space.PAIRED
        and request.kind
        in {
            ResourceKind.DISTANCE_MATRIX,
            ResourceKind.KNN,
            ResourceKind.NEIGHBOR_RANKING,
        }
        for request in all_requests
    )
    pair_strategy = None
    if has_pair_resources:
        dense_for_compatibility = geodesic or has_other_exact_resources
        condensed_bytes = 2 * pair_count * np.dtype(np.float64).itemsize
        if dense_for_compatibility:
            pair_strategy = PairStrategy.DENSE
        elif has_ordered_pair_statistics:
            # Exact rank and isotonic metrics require all pairs to be materialized.
            pair_strategy = PairStrategy.CONDENSED
        elif memory_budget is None:
            pair_strategy = (
                PairStrategy.CONDENSED
                if condensed_bytes <= DEFAULT_PAIR_CACHE_BYTES
                else PairStrategy.STREAMING
            )
        else:
            pair_strategy = (
                PairStrategy.CONDENSED
                if condensed_bytes + CONDENSED_WORK_BYTES_PER_PAIR <= memory_budget
                else PairStrategy.STREAMING
            )

    resources = []
    canonical_by_pair: dict[tuple[ResourceKind, Space], ResourceKey] = {}
    for space in (Space.ORIGINAL, Space.EMBEDDED):
        distance_pair = (ResourceKind.DISTANCE_MATRIX, space)
        if distance_pair in requested_pairs or pair_strategy is PairStrategy.DENSE:
            key = ResourceKey(ResourceKind.DISTANCE_MATRIX, space)
            resources.append(key)
            canonical_by_pair[distance_pair] = key
        elif pair_strategy is PairStrategy.CONDENSED and (
            space is Space.EMBEDDED or has_pair_statistics
        ):
            condensed_pair = (ResourceKind.CONDENSED_PAIRS, space)
            key = ResourceKey(ResourceKind.CONDENSED_PAIRS, space)
            resources.append(key)
            canonical_by_pair[condensed_pair] = key

        ranking_pair = (ResourceKind.NEIGHBOR_RANKING, space)
        knn_pair = (ResourceKind.KNN, space)
        ranking_k = maxima.get(ranking_pair, -1)
        knn_k = maxima.get(knn_pair, -1)
        if ranking_k >= 0:
            key = ResourceKey(
                ResourceKind.NEIGHBOR_RANKING, space, max(ranking_k, knn_k)
            )
            resources.append(key)
            canonical_by_pair[ranking_pair] = key
            canonical_by_pair[knn_pair] = key
        elif knn_k >= 0:
            key = ResourceKey(ResourceKind.KNN, space, knn_k)
            resources.append(key)
            canonical_by_pair[knn_pair] = key

        stable_knn_pair = (ResourceKind.STABLE_KNN, space)
        stable_knn_k = maxima.get(stable_knn_pair, -1)
        if stable_knn_k >= 0:
            key = ResourceKey(ResourceKind.STABLE_KNN, space, stable_knn_k)
            resources.append(key)
            canonical_by_pair[stable_knn_pair] = key

    order_key = None
    if has_ordered_pair_statistics:
        order_key = ResourceKey(ResourceKind.PAIR_ORDER, Space.ORIGINAL)
        resources.append(order_key)

    statistics_key = None
    if has_pair_statistics:
        statistics_key = ResourceKey(ResourceKind.PAIR_STATISTICS, Space.PAIRED)
        resources.append(statistics_key)
        canonical_by_pair[(ResourceKind.PAIR_STATISTICS, Space.PAIRED)] = statistics_key

    ordered_statistics_key = None
    if has_ordered_pair_statistics:
        ordered_statistics_key = ResourceKey(
            ResourceKind.ORDERED_PAIR_STATISTICS,
            Space.PAIRED,
        )
        resources.append(ordered_statistics_key)
        canonical_by_pair[(ResourceKind.ORDERED_PAIR_STATISTICS, Space.PAIRED)] = (
            ordered_statistics_key
        )

    topographic_k = maxima.get(
        (ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS, Space.PAIRED),
        -1,
    )
    topographic_statistics_key = None
    if topographic_k >= 0:
        topographic_statistics_key = ResourceKey(
            ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS,
            Space.PAIRED,
            topographic_k,
        )
        resources.append(topographic_statistics_key)
        canonical_by_pair[
            (ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS, Space.PAIRED)
        ] = topographic_statistics_key

    request_to_key = {
        request: canonical_by_pair[(request.kind, request.space)]
        for request in all_requests
    }
    consumer_sets: dict[ResourceKey, set[int]] = {key: set() for key in resources}
    for metric_index, metric_plan in enumerate(metric_plans):
        for binding in metric_plan.bindings:
            for request in binding.requests:
                consumer_sets[request_to_key[request]].add(metric_index)

    pair_consumers = set() if statistics_key is None else consumer_sets[statistics_key]
    ordered_pair_consumers = (
        set()
        if ordered_statistics_key is None
        else consumer_sets[ordered_statistics_key]
    )
    all_pair_consumers = pair_consumers | ordered_pair_consumers
    original_source_key = None
    embedded_source_key = None
    if pair_strategy is PairStrategy.DENSE:
        original_source_key = canonical_by_pair[
            (ResourceKind.DISTANCE_MATRIX, Space.ORIGINAL)
        ]
        embedded_source_key = canonical_by_pair[
            (ResourceKind.DISTANCE_MATRIX, Space.EMBEDDED)
        ]
    elif pair_strategy is PairStrategy.CONDENSED:
        original_source_key = canonical_by_pair.get(
            (ResourceKind.CONDENSED_PAIRS, Space.ORIGINAL)
        )
        embedded_source_key = canonical_by_pair[
            (ResourceKind.CONDENSED_PAIRS, Space.EMBEDDED)
        ]
    for source_key in (original_source_key, embedded_source_key):
        if source_key is not None:
            consumer_sets[source_key].update(all_pair_consumers)
    if order_key is not None:
        consumer_sets[order_key].update(ordered_pair_consumers)

    topographic_consumers = (
        set()
        if topographic_statistics_key is None
        else consumer_sets[topographic_statistics_key]
    )
    topographic_original_knn_key = None
    topographic_embedded_knn_key = None
    if topographic_statistics_key is not None:
        topographic_original_knn_key = canonical_by_pair[
            (ResourceKind.STABLE_KNN, Space.ORIGINAL)
        ]
        topographic_embedded_knn_key = canonical_by_pair[
            (ResourceKind.STABLE_KNN, Space.EMBEDDED)
        ]
        consumer_sets[topographic_original_knn_key].update(topographic_consumers)
        consumer_sets[topographic_embedded_knn_key].update(topographic_consumers)

    if pair_strategy is PairStrategy.DENSE:
        for metric_index, metric_plan in enumerate(metric_plans):
            for binding in metric_plan.bindings:
                for request in binding.requests:
                    if request.kind not in {
                        ResourceKind.KNN,
                        ResourceKind.NEIGHBOR_RANKING,
                    }:
                        continue
                    distance_key = canonical_by_pair[
                        (ResourceKind.DISTANCE_MATRIX, request.space)
                    ]
                    consumer_sets[distance_key].add(metric_index)
    consumers = {key: tuple(sorted(indices)) for key, indices in consumer_sets.items()}

    estimated_cache_bytes = _estimate_cache_bytes(resources, n_samples)
    pair_plan = None
    peak_working_bytes = 0
    available_work_bytes = (
        DEFAULT_RESOURCE_WORK_BYTES
        if memory_budget is None
        else max(0, memory_budget - estimated_cache_bytes)
    )
    if pair_strategy is not None:
        pair_work_bytes = (
            DEFAULT_PAIR_WORK_BYTES if memory_budget is None else available_work_bytes
        )
        block_rows = None
        chunk_pairs = None
        statistics_working_bytes = 0
        if statistics_key is not None:
            if pair_strategy is PairStrategy.CONDENSED:
                chunk_pairs = max(
                    1,
                    min(
                        pair_count,
                        pair_work_bytes // CONDENSED_WORK_BYTES_PER_PAIR,
                    ),
                )
                statistics_working_bytes = chunk_pairs * CONDENSED_WORK_BYTES_PER_PAIR
            else:
                block_rows = max(
                    1,
                    min(
                        n_samples,
                        math.isqrt(pair_work_bytes // PAIR_WORK_BYTES_PER_CELL),
                    ),
                )
                statistics_working_bytes = PAIR_WORK_BYTES_PER_CELL * block_rows**2
        ordered_working_bytes = (
            pair_count * ORDERED_WORK_BYTES_PER_PAIR
            if ordered_statistics_key is not None
            else 0
        )
        working_bytes = max(statistics_working_bytes, ordered_working_bytes)
        peak_working_bytes = max(peak_working_bytes, working_bytes)
        pair_plan = PairExecutionPlan(
            strategy=pair_strategy,
            statistics_key=statistics_key,
            ordered_statistics_key=ordered_statistics_key,
            order_key=order_key,
            original_source_key=original_source_key,
            embedded_source_key=embedded_source_key,
            pair_count=pair_count,
            block_rows=block_rows,
            chunk_pairs=chunk_pairs,
            working_bytes=working_bytes,
            metric_ids=tuple(
                metric_plans[index].metric_id for index in sorted(pair_consumers)
            ),
            ordered_metric_ids=tuple(
                metric_plans[index].metric_id
                for index in sorted(ordered_pair_consumers)
            ),
        )

    resource_working_bytes = {}
    for key in resources:
        if key.kind is not ResourceKind.STABLE_KNN:
            continue
        bytes_per_row = n_samples * STABLE_KNN_WORK_BYTES_PER_CELL
        block_rows = max(
            1,
            min(n_samples, available_work_bytes // bytes_per_row),
        )
        working_bytes = block_rows * bytes_per_row
        resource_working_bytes[key] = working_bytes
        peak_working_bytes = max(peak_working_bytes, working_bytes)

    topographic_plan = None
    if topographic_statistics_key is not None:
        assert topographic_original_knn_key is not None
        assert topographic_embedded_knn_key is not None
        bytes_per_row = (
            topographic_k
            * (original_dimension + 5)
            * TOPOGRAPHIC_WORK_BYTES_PER_SELECTED_DISTANCE
        )
        block_rows = max(
            1,
            min(n_samples, available_work_bytes // bytes_per_row),
        )
        working_bytes = block_rows * bytes_per_row
        peak_working_bytes = max(peak_working_bytes, working_bytes)
        topographic_plan = TopographicExecutionPlan(
            statistics_key=topographic_statistics_key,
            original_knn_key=topographic_original_knn_key,
            embedded_knn_key=topographic_embedded_knn_key,
            k=topographic_k,
            block_rows=block_rows,
            work_budget_bytes=available_work_bytes,
            working_bytes=working_bytes,
            metric_ids=tuple(
                metric_plans[index].metric_id for index in sorted(topographic_consumers)
            ),
            requested_ks=tuple(
                sorted(
                    {
                        request.k
                        for request in all_requests
                        if request.kind is ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS
                        and request.k is not None
                    }
                )
            ),
            geodesic=geodesic,
        )

    planned_peak_bytes = estimated_cache_bytes + peak_working_bytes

    return ExecutionPlan(
        resources=tuple(resources),
        metric_plans=tuple(metric_plans),
        request_to_key=request_to_key,
        consumers=consumers,
        estimated_cache_bytes=estimated_cache_bytes,
        planned_peak_bytes=planned_peak_bytes,
        memory_budget_bytes=memory_budget,
        pair_plan=pair_plan,
        topographic_plan=topographic_plan,
        resource_working_bytes=resource_working_bytes,
    )


def _estimate_cache_bytes(resources: list[ResourceKey], n_samples: int) -> int:
    total = 0
    for key in resources:
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            total += n_samples * n_samples * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.CONDENSED_PAIRS:
            total += n_samples * (n_samples - 1) // 2 * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.PAIR_ORDER:
            total += n_samples * (n_samples - 1) // 2 * PAIR_ORDER_BYTES_PER_PAIR
        elif key.kind is ResourceKind.NEIGHBOR_RANKING:
            assert key.k is not None
            total += n_samples * n_samples * np.dtype(np.intp).itemsize
            total += n_samples * key.k * np.dtype(np.int64).itemsize
        elif key.kind is ResourceKind.KNN:
            assert key.k is not None
            total += n_samples * key.k * np.dtype(np.int64).itemsize
        elif key.kind is ResourceKind.STABLE_KNN:
            assert key.k is not None
            total += n_samples * key.k * np.dtype(np.intp).itemsize
        elif key.kind is ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS:
            assert key.k is not None
            total += key.k * np.dtype(np.float64).itemsize
    return int(total)
