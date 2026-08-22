"""Deterministic planning and deduplication of exact metric resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .resources import (
    ResourceKey,
    ResourceKind,
    ResourceRequest,
    ResourceRequirement,
    Space,
)

if TYPE_CHECKING:
    from zadu.registry import MetricDefinition


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    requirement: ResourceRequirement
    requests: tuple[ResourceRequest, ...]


@dataclass(frozen=True, slots=True)
class MetricPlan:
    metric_id: str
    bindings: tuple[ResourceBinding, ...]


@dataclass(slots=True)
class ExecutionPlan:
    resources: tuple[ResourceKey, ...]
    metric_plans: tuple[MetricPlan, ...]
    request_to_key: dict[ResourceRequest, ResourceKey]
    consumers: dict[ResourceKey, tuple[int, ...]]
    estimated_cache_bytes: int

    def resolve(self, request: ResourceRequest) -> ResourceKey:
        return self.request_to_key[request]

    def resources_for(self, space: Space) -> tuple[ResourceKey, ...]:
        return tuple(key for key in self.resources if key.space is space)


def build_execution_plan(
    definitions: list[MetricDefinition],
    specs: list[dict],
    *,
    n_samples: int,
    default_k: int,
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

    resources = []
    canonical_by_pair: dict[tuple[ResourceKind, Space], ResourceKey] = {}
    for space in (Space.ORIGINAL, Space.EMBEDDED):
        distance_pair = (ResourceKind.DISTANCE_MATRIX, space)
        if distance_pair in requested_pairs:
            key = ResourceKey(ResourceKind.DISTANCE_MATRIX, space)
            resources.append(key)
            canonical_by_pair[distance_pair] = key

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

    request_to_key = {
        request: canonical_by_pair[(request.kind, request.space)]
        for request in all_requests
    }
    consumer_sets: dict[ResourceKey, set[int]] = {key: set() for key in resources}
    for metric_index, metric_plan in enumerate(metric_plans):
        for binding in metric_plan.bindings:
            for request in binding.requests:
                consumer_sets[request_to_key[request]].add(metric_index)
    consumers = {key: tuple(sorted(indices)) for key, indices in consumer_sets.items()}

    return ExecutionPlan(
        resources=tuple(resources),
        metric_plans=tuple(metric_plans),
        request_to_key=request_to_key,
        consumers=consumers,
        estimated_cache_bytes=_estimate_cache_bytes(resources, n_samples),
    )


def _estimate_cache_bytes(resources: list[ResourceKey], n_samples: int) -> int:
    total = 0
    for key in resources:
        if key.kind is ResourceKind.DISTANCE_MATRIX:
            total += n_samples * n_samples * np.dtype(np.float64).itemsize
        elif key.kind is ResourceKind.NEIGHBOR_RANKING:
            total += n_samples * n_samples * np.dtype(np.intp).itemsize
            total += n_samples * key.k * np.dtype(np.int64).itemsize
        else:
            total += n_samples * key.k * np.dtype(np.int64).itemsize
    return int(total)
