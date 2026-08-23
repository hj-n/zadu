"""Deterministic planning and deduplication of exact metric resources."""

from __future__ import annotations

import math
from collections.abc import Callable
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
    compact_index_dtype,
)

if TYPE_CHECKING:
    from zadu.registry import MetricDefinition


DEFAULT_PAIR_CACHE_BYTES = 256 * 1024**2
DEFAULT_PAIR_WORK_BYTES = 64 * 1024**2
PAIR_WORK_BYTES_PER_CELL = 64
CONDENSED_WORK_BYTES_PER_PAIR = 48
ORDERED_WORK_BYTES_PER_PAIR = 64
DEFAULT_RESOURCE_WORK_BYTES = 64 * 1024**2
STABLE_KNN_WORK_BYTES_PER_CELL = 32
TOPOGRAPHIC_WORK_BYTES_PER_SELECTED_DISTANCE = 16
SELECTED_RANK_WORK_BYTES_PER_CELL = 24
RANK_WORK_BYTES_PER_COMPARISON = 1
NEIGHBOR_WORK_BYTES_PER_COMPARISON = 1
NEIGHBOR_GRAPH_BYTES_PER_EDGE = 16
NEIGHBOR_PRODUCT_BYTES_PER_CELL = 48
SNC_SPARSE_BYTES_PER_ENTRY = 16
SNC_ITERATION_BYTES_PER_CELL = 24
MLX_PAIRWISE_WORK_ARRAYS = 4
MLX_NEIGHBOR_FLOAT_WORK_ARRAYS = 4
MLX_KNN_INDEX_WORK_ARRAYS = 2
MLX_RANKING_INDEX_WORK_ARRAYS = 4
TORCH_KNN_INDEX_WORK_ARRAYS = 2
TORCH_RANKING_INDEX_WORK_ARRAYS = 4


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


@dataclass(frozen=True, slots=True)
class RankComparisonExecutionPlan:
    statistics_key: ResourceKey
    original_knn_key: ResourceKey
    k: int
    membership_ks: tuple[int, ...]
    requested_ks: tuple[int, ...]
    block_rows: int
    work_budget_bytes: int
    working_bytes: int
    metric_ids: tuple[str, ...]
    geodesic: bool


@dataclass(frozen=True, slots=True)
class NeighborStatisticsExecutionPlan:
    statistics_key: ResourceKey
    original_knn_key: ResourceKey
    embedded_knn_key: ResourceKey
    k: int
    lcmc_ks: tuple[int, ...]
    nd_ks: tuple[int, ...]
    block_rows: int
    work_budget_bytes: int
    working_bytes: int
    metric_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SNCExecutionPlan:
    requested_workers: dict[int, int]
    effective_workers: dict[int, int]
    graph_bytes: dict[int, int]
    iteration_bytes: dict[int, int]
    metric_working_bytes: dict[int, int]
    working_bytes: int


@dataclass(slots=True)
class ExecutionPlan:
    resources: tuple[ResourceKey, ...]
    metric_plans: tuple[MetricPlan, ...]
    request_to_key: dict[ResourceRequest, ResourceKey]
    consumers: dict[ResourceKey, tuple[int, ...]]
    estimated_cache_bytes: int
    original_cache_bytes: int
    per_embedding_peak_bytes: int
    planned_peak_bytes: int
    memory_budget_bytes: int | None
    pair_plan: PairExecutionPlan | None
    topographic_plan: TopographicExecutionPlan | None
    rank_comparison_plan: RankComparisonExecutionPlan | None
    neighbor_statistics_plan: NeighborStatisticsExecutionPlan | None
    snc_plan: SNCExecutionPlan | None
    resource_working_bytes: dict[ResourceKey, int]
    release_after_prepare: dict[Space, tuple[ResourceKey, ...]]

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
    backend: str = "numpy",
    resource_dtype_bytes: int = np.dtype(np.float64).itemsize,
    provider_working_memory: (
        Callable[[ResourceKey, int, int], int | None] | None
    ) = None,
) -> ExecutionPlan:
    """Build one deterministic plan and collapse compatible kNN requests."""

    metric_plans = []
    all_requests: list[ResourceRequest] = []
    for definition, spec in zip(definitions, specs, strict=True):
        bindings = []
        for requirement in definition.resources:
            if requirement.uses_k:
                requirement_default_k = (
                    math.isqrt(n_samples)
                    if requirement.k_default_rule == "sqrt"
                    else default_k
                )
                k = spec["params"].get("k", requirement_default_k)
            else:
                k = None
            parameter = (
                spec["params"].get(
                    requirement.parameter_name,
                    requirement.default_parameter,
                )
                if requirement.parameter_name is not None
                else None
            )
            requests = tuple(
                ResourceRequest(requirement.kind, space, k, parameter)
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

    rank_comparison_k = maxima.get((ResourceKind.RANK_COMPARISONS, Space.PAIRED), -1)
    if rank_comparison_k >= 0:
        pair = (ResourceKind.STABLE_KNN, Space.ORIGINAL)
        requested_pairs.add(pair)
        maxima[pair] = max(maxima.get(pair, -1), rank_comparison_k)

    neighbor_statistics_k = maxima.get(
        (ResourceKind.NEIGHBOR_STATISTICS, Space.PAIRED), -1
    )
    if neighbor_statistics_k >= 0:
        for space in (Space.ORIGINAL, Space.EMBEDDED):
            pair = (ResourceKind.KNN, space)
            requested_pairs.add(pair)
            maxima[pair] = max(maxima.get(pair, -1), neighbor_statistics_k)

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
        (
            request.space is not Space.PAIRED
            and request.kind
            in {
                ResourceKind.DISTANCE_MATRIX,
                ResourceKind.DENSITY,
                ResourceKind.KNN,
                ResourceKind.NEIGHBOR_RANKING,
            }
        )
        or request.kind
        in {ResourceKind.RANK_COMPARISONS, ResourceKind.NEIGHBOR_STATISTICS}
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
    parameterized_keys: dict[ResourceRequest, ResourceKey] = {}
    for space in (Space.ORIGINAL, Space.EMBEDDED):
        distance_pair = (ResourceKind.DISTANCE_MATRIX, space)
        density_pair = (ResourceKind.DENSITY, space)
        if (
            distance_pair in requested_pairs
            or density_pair in requested_pairs
            or pair_strategy is PairStrategy.DENSE
        ):
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

        density_parameters = sorted(
            {
                request.parameter
                for request in all_requests
                if request.kind is ResourceKind.DENSITY
                and request.space is space
                and request.parameter is not None
            }
        )
        for parameter in density_parameters:
            key = ResourceKey(ResourceKind.DENSITY, space, parameter=parameter)
            resources.append(key)
            parameterized_keys[
                ResourceRequest(
                    ResourceKind.DENSITY,
                    space,
                    parameter=parameter,
                )
            ] = key

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

    rank_comparison_statistics_key = None
    if rank_comparison_k >= 0:
        rank_comparison_statistics_key = ResourceKey(
            ResourceKind.RANK_COMPARISONS,
            Space.PAIRED,
            rank_comparison_k,
        )
        resources.append(rank_comparison_statistics_key)
        canonical_by_pair[(ResourceKind.RANK_COMPARISONS, Space.PAIRED)] = (
            rank_comparison_statistics_key
        )

    neighbor_statistics_key = None
    if neighbor_statistics_k >= 0:
        neighbor_statistics_key = ResourceKey(
            ResourceKind.NEIGHBOR_STATISTICS,
            Space.PAIRED,
            neighbor_statistics_k,
        )
        resources.append(neighbor_statistics_key)
        canonical_by_pair[(ResourceKind.NEIGHBOR_STATISTICS, Space.PAIRED)] = (
            neighbor_statistics_key
        )

    request_to_key = {
        request: (
            parameterized_keys[request]
            if request.kind is ResourceKind.DENSITY
            else canonical_by_pair[(request.kind, request.space)]
        )
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

    for density_key in parameterized_keys.values():
        density_consumers = consumer_sets[density_key]
        distance_key = canonical_by_pair[
            (ResourceKind.DISTANCE_MATRIX, density_key.space)
        ]
        consumer_sets[distance_key].update(density_consumers)

    rank_comparison_consumers = (
        set()
        if rank_comparison_statistics_key is None
        else consumer_sets[rank_comparison_statistics_key]
    )
    rank_original_knn_key = None
    if rank_comparison_statistics_key is not None:
        rank_original_knn_key = canonical_by_pair[
            (ResourceKind.STABLE_KNN, Space.ORIGINAL)
        ]
        consumer_sets[rank_original_knn_key].update(rank_comparison_consumers)

    neighbor_statistics_consumers = (
        set()
        if neighbor_statistics_key is None
        else consumer_sets[neighbor_statistics_key]
    )
    neighbor_original_key = None
    neighbor_embedded_key = None
    if neighbor_statistics_key is not None:
        neighbor_original_key = canonical_by_pair[(ResourceKind.KNN, Space.ORIGINAL)]
        neighbor_embedded_key = canonical_by_pair[(ResourceKind.KNN, Space.EMBEDDED)]
        consumer_sets[neighbor_original_key].update(neighbor_statistics_consumers)
        consumer_sets[neighbor_embedded_key].update(neighbor_statistics_consumers)

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

    rank_requested_ks = tuple(
        sorted(
            {
                request.k
                for request in all_requests
                if request.kind is ResourceKind.RANK_COMPARISONS
                and request.k is not None
            }
        )
    )
    rank_membership_ks = tuple(
        sorted(
            {
                spec["params"].get("k", default_k)
                for definition, spec in zip(definitions, specs, strict=True)
                if definition.id
                in {
                    "trustworthiness_continuity",
                    "class_aware_trustworthiness_continuity",
                }
            }
        )
    )
    lcmc_ks = tuple(
        sorted(
            {
                spec["params"].get("k", default_k)
                for definition, spec in zip(definitions, specs, strict=True)
                if definition.id == "local_continuity_meta_criteria"
            }
        )
    )
    nd_ks = tuple(
        sorted(
            {
                spec["params"].get("k", default_k)
                for definition, spec in zip(definitions, specs, strict=True)
                if definition.id == "neighbor_dissimilarity"
            }
        )
    )
    explicitly_requested_distance_spaces = {
        request.space
        for request in all_requests
        if request.kind is ResourceKind.DISTANCE_MATRIX
    }
    release_after_prepare: dict[Space, tuple[ResourceKey, ...]] = {}
    for space in (Space.ORIGINAL, Space.EMBEDDED):
        if (
            (ResourceKind.DENSITY, space) in requested_pairs
            and space not in explicitly_requested_distance_spaces
            and pair_strategy is not PairStrategy.DENSE
        ):
            release_after_prepare[space] = (
                canonical_by_pair[(ResourceKind.DISTANCE_MATRIX, space)],
            )
    transient_keys = {key for keys in release_after_prepare.values() for key in keys}
    estimated_cache_bytes = _estimate_cache_bytes(
        resources,
        n_samples,
        rank_membership_ks=rank_membership_ks,
        lcmc_ks=lcmc_ks,
        nd_ks=nd_ks,
        excluded_keys=transient_keys,
    )
    original_cache_bytes = _estimate_cache_bytes(
        [key for key in resources if key.space is Space.ORIGINAL],
        n_samples,
        rank_membership_ks=rank_membership_ks,
        lcmc_ks=lcmc_ks,
        nd_ks=nd_ks,
        excluded_keys=transient_keys,
    )
    transient_resource_bytes = max(
        (
            _estimate_cache_bytes(
                [key],
                n_samples,
                rank_membership_ks=(),
                lcmc_ks=(),
                nd_ks=(),
            )
            for key in transient_keys
        ),
        default=0,
    )
    pair_plan = None
    density_working_bytes = (
        n_samples * n_samples * np.dtype(np.float64).itemsize
        if any(key.kind is ResourceKind.DENSITY for key in resources)
        else 0
    )
    peak_working_bytes = transient_resource_bytes + density_working_bytes
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
    if backend in {"mlx", "torch"}:
        for key in resources:
            if key.kind in {
                ResourceKind.DISTANCE_MATRIX,
                ResourceKind.CONDENSED_PAIRS,
            }:
                bytes_per_row = (
                    n_samples * resource_dtype_bytes * MLX_PAIRWISE_WORK_ARRAYS
                )
            elif backend == "mlx" and key.kind in {
                ResourceKind.KNN,
                ResourceKind.STABLE_KNN,
            }:
                bytes_per_row = n_samples * (
                    resource_dtype_bytes * MLX_NEIGHBOR_FLOAT_WORK_ARRAYS
                    + compact_index_dtype(n_samples).itemsize
                    * MLX_KNN_INDEX_WORK_ARRAYS
                )
            elif backend == "mlx" and key.kind is ResourceKind.NEIGHBOR_RANKING:
                bytes_per_row = n_samples * (
                    resource_dtype_bytes * MLX_NEIGHBOR_FLOAT_WORK_ARRAYS
                    + compact_index_dtype(n_samples).itemsize
                    * MLX_RANKING_INDEX_WORK_ARRAYS
                )
            elif backend == "torch" and key.kind in {
                ResourceKind.KNN,
                ResourceKind.STABLE_KNN,
            }:
                bytes_per_row = n_samples * (
                    resource_dtype_bytes * MLX_NEIGHBOR_FLOAT_WORK_ARRAYS
                    + np.dtype(np.int64).itemsize * TORCH_KNN_INDEX_WORK_ARRAYS
                )
            elif backend == "torch" and key.kind is ResourceKind.NEIGHBOR_RANKING:
                bytes_per_row = n_samples * (
                    resource_dtype_bytes * MLX_NEIGHBOR_FLOAT_WORK_ARRAYS
                    + np.dtype(np.int64).itemsize * TORCH_RANKING_INDEX_WORK_ARRAYS
                )
            else:
                continue
            block_rows = max(
                1,
                min(n_samples, available_work_bytes // bytes_per_row),
            )
            working_bytes = block_rows * bytes_per_row
            resource_working_bytes[key] = working_bytes
            peak_working_bytes = max(peak_working_bytes, working_bytes)
    if provider_working_memory is not None:
        for key in resources:
            if key in resource_working_bytes:
                continue
            working_bytes = provider_working_memory(
                key,
                n_samples,
                available_work_bytes,
            )
            if working_bytes is None:
                continue
            if isinstance(working_bytes, bool) or not isinstance(
                working_bytes, (int, np.integer)
            ):
                raise TypeError(
                    "Backend working_memory_bytes() must return an integer or None"
                )
            working_bytes = int(working_bytes)
            if working_bytes < 1:
                raise ValueError(
                    "Backend working_memory_bytes() must return a positive value"
                )
            resource_working_bytes[key] = working_bytes
            peak_working_bytes = max(peak_working_bytes, working_bytes)
    for key in resources:
        if key.kind is not ResourceKind.STABLE_KNN or key in resource_working_bytes:
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

    rank_comparison_plan = None
    if rank_comparison_statistics_key is not None:
        assert rank_original_knn_key is not None
        largest_membership_k = max(rank_membership_ks, default=0)
        bytes_per_row = max(
            1,
            n_samples * SELECTED_RANK_WORK_BYTES_PER_CELL,
            largest_membership_k**2 * RANK_WORK_BYTES_PER_COMPARISON,
        )
        block_rows = max(
            1,
            min(n_samples, available_work_bytes // bytes_per_row),
        )
        working_bytes = block_rows * bytes_per_row
        peak_working_bytes = max(peak_working_bytes, working_bytes)
        rank_comparison_plan = RankComparisonExecutionPlan(
            statistics_key=rank_comparison_statistics_key,
            original_knn_key=rank_original_knn_key,
            k=rank_comparison_k,
            membership_ks=rank_membership_ks,
            requested_ks=rank_requested_ks,
            block_rows=block_rows,
            work_budget_bytes=available_work_bytes,
            working_bytes=working_bytes,
            metric_ids=tuple(
                metric_plans[index].metric_id
                for index in sorted(rank_comparison_consumers)
            ),
            geodesic=geodesic,
        )

    neighbor_statistics_plan = None
    if neighbor_statistics_key is not None:
        assert neighbor_original_key is not None
        assert neighbor_embedded_key is not None
        largest_lcmc_k = max(lcmc_ks, default=0)
        lcmc_bytes_per_row = max(
            1,
            largest_lcmc_k**2 * NEIGHBOR_WORK_BYTES_PER_COMPARISON,
        )
        lcmc_block_rows = max(
            1,
            min(n_samples, available_work_bytes // lcmc_bytes_per_row),
        )
        lcmc_working_bytes = lcmc_block_rows * lcmc_bytes_per_row
        graph_bytes = (
            4 * n_samples * max(nd_ks, default=0) * NEIGHBOR_GRAPH_BYTES_PER_EDGE
        )
        product_bytes_per_row = n_samples * NEIGHBOR_PRODUCT_BYTES_PER_CELL
        product_budget = max(0, available_work_bytes - graph_bytes)
        product_block_rows = max(
            1,
            min(n_samples, product_budget // product_bytes_per_row),
        )
        nd_working_bytes = (
            graph_bytes + product_block_rows * product_bytes_per_row if nd_ks else 0
        )
        working_bytes = max(lcmc_working_bytes, nd_working_bytes)
        peak_working_bytes = max(peak_working_bytes, working_bytes)
        neighbor_statistics_plan = NeighborStatisticsExecutionPlan(
            statistics_key=neighbor_statistics_key,
            original_knn_key=neighbor_original_key,
            embedded_knn_key=neighbor_embedded_key,
            k=neighbor_statistics_k,
            lcmc_ks=lcmc_ks,
            nd_ks=nd_ks,
            block_rows=product_block_rows,
            work_budget_bytes=available_work_bytes,
            working_bytes=working_bytes,
            metric_ids=tuple(
                metric_plans[index].metric_id
                for index in sorted(neighbor_statistics_consumers)
            ),
        )

    snc_plan = None
    snc_metric_indices = [
        index
        for index, definition in enumerate(definitions)
        if definition.id == "steadiness_cohesiveness"
    ]
    if snc_metric_indices:
        requested_workers = {}
        effective_workers = {}
        metric_graph_bytes = {}
        metric_iteration_bytes = {}
        metric_working_bytes = {}
        for metric_index in snc_metric_indices:
            params = specs[metric_index]["params"]
            k = params.get("k", math.isqrt(n_samples))
            walk_num = max(1, int(n_samples * params.get("walk_num_ratio", 0.3)))
            workers = params.get("n_jobs", 1)
            usable_workers = min(workers, params.get("iteration", 150))
            graph_entries = n_samples * n_samples
            graph_bytes = (
                2 * graph_entries * SNC_SPARSE_BYTES_PER_ENTRY
                + 2 * n_samples * k * np.dtype(np.float64).itemsize
            )
            cluster_size = min(n_samples, walk_num + k + 1)
            iteration_bytes = max(
                1,
                SNC_ITERATION_BYTES_PER_CELL * cluster_size * cluster_size,
            )
            if memory_budget is None:
                planned_workers = usable_workers
            else:
                worker_capacity = max(
                    0,
                    (available_work_bytes - graph_bytes) // iteration_bytes,
                )
                planned_workers = max(1, min(usable_workers, worker_capacity))
            working_bytes = graph_bytes + planned_workers * iteration_bytes
            requested_workers[metric_index] = workers
            effective_workers[metric_index] = planned_workers
            metric_graph_bytes[metric_index] = graph_bytes
            metric_iteration_bytes[metric_index] = iteration_bytes
            metric_working_bytes[metric_index] = working_bytes
            peak_working_bytes = max(peak_working_bytes, working_bytes)
        snc_plan = SNCExecutionPlan(
            requested_workers=requested_workers,
            effective_workers=effective_workers,
            graph_bytes=metric_graph_bytes,
            iteration_bytes=metric_iteration_bytes,
            metric_working_bytes=metric_working_bytes,
            working_bytes=max(metric_working_bytes.values()),
        )

    planned_peak_bytes = estimated_cache_bytes + peak_working_bytes
    per_embedding_peak_bytes = planned_peak_bytes - original_cache_bytes

    return ExecutionPlan(
        resources=tuple(resources),
        metric_plans=tuple(metric_plans),
        request_to_key=request_to_key,
        consumers=consumers,
        estimated_cache_bytes=estimated_cache_bytes,
        original_cache_bytes=original_cache_bytes,
        per_embedding_peak_bytes=per_embedding_peak_bytes,
        planned_peak_bytes=planned_peak_bytes,
        memory_budget_bytes=memory_budget,
        pair_plan=pair_plan,
        topographic_plan=topographic_plan,
        rank_comparison_plan=rank_comparison_plan,
        neighbor_statistics_plan=neighbor_statistics_plan,
        snc_plan=snc_plan,
        resource_working_bytes=resource_working_bytes,
        release_after_prepare=release_after_prepare,
    )


def _estimate_cache_bytes(
    resources: list[ResourceKey],
    n_samples: int,
    *,
    rank_membership_ks: tuple[int, ...] = (),
    lcmc_ks: tuple[int, ...] = (),
    nd_ks: tuple[int, ...] = (),
    excluded_keys: set[ResourceKey] | None = None,
) -> int:
    index_bytes = compact_index_dtype(n_samples).itemsize
    total = 0
    for key in resources:
        if excluded_keys is not None and key in excluded_keys:
            continue
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            total += n_samples * n_samples * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.DENSITY:
            total += n_samples * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.CONDENSED_PAIRS:
            total += n_samples * (n_samples - 1) // 2 * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.PAIR_ORDER:
            pair_count = n_samples * (n_samples - 1) // 2
            pair_index_bytes = compact_index_dtype(pair_count).itemsize
            total += pair_count * (pair_index_bytes + np.dtype(np.float64).itemsize)
        elif key.kind is ResourceKind.NEIGHBOR_RANKING:
            assert key.k is not None
            total += n_samples * n_samples * index_bytes
            total += n_samples * key.k * index_bytes
        elif key.kind in {ResourceKind.KNN, ResourceKind.STABLE_KNN}:
            assert key.k is not None
            total += n_samples * key.k * index_bytes
        elif key.kind is ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS:
            assert key.k is not None
            total += key.k * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.RANK_COMPARISONS:
            assert key.k is not None
            total += 3 * n_samples * key.k * index_bytes
            total += 2 * n_samples * sum(rank_membership_ks)
        elif key.kind is ResourceKind.NEIGHBOR_STATISTICS:
            total += n_samples * len(lcmc_ks) * np.dtype(np.float64).itemsize
            total += len(nd_ks) * np.dtype(np.float64).itemsize
    return int(total)
