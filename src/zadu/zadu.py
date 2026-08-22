"""Scheduled execution of ZADU dimensionality-reduction metrics."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from .backends import NumpyResourceProvider
from .engine.config import ExecutionConfig
from .engine.planner import build_execution_plan
from .engine.resources import ResourceCache, ResourceKind, Space
from .engine.result import build_run_info
from .measures.utils.validation import (
    as_finite_2d,
    validate_labels,
    validate_neighbor_k,
    validate_trustworthiness_k,
)
from .registry import METRIC_BY_ALIAS, METRIC_BY_ID, MetricDefinition


class ZADU:
    """Evaluate one embedding with one or more registered metrics."""

    ABBREVIATIONS: ClassVar[dict[str, str]] = {
        alias: metric.id for alias, metric in METRIC_BY_ALIAS.items()
    }
    DEFAULT_K = 20

    def __init__(
        self,
        spec_list,
        orig,
        return_local: bool = False,
        verbose: bool = False,
        geodesic: bool = False,
        max_memory_bytes: int | None = None,
        execution: ExecutionConfig | None = None,
    ):
        self.spec_list = deepcopy(spec_list)
        self.return_local = bool(return_local)
        self.verbose = bool(verbose)
        self.geodesic = bool(geodesic)
        self.execution = self._resolve_execution(execution, max_memory_bytes)
        self.max_memory_bytes = self.execution.memory_budget_bytes

        self.orig = as_finite_2d(orig, "orig")
        self.emb = None
        self.label = None
        self.last_run_info: dict[str, Any] | None = None

        # Compatibility views backed by the typed resource cache.
        self.distance_matrices_flag = False
        self.ranking_k = -1
        self.knn_both_k = -1
        self.knn_emb_k = -1
        self.orig_distance_matrix = None
        self.emb_distance_matrix = None
        self.orig_knn_ranking = None
        self.emb_knn_ranking = None
        self.orig_knn_indices = None
        self.emb_knn_indices = None

        self._definitions: list[MetricDefinition] = []
        self._validate_and_normalize_specs()
        self._interpret_specs()
        self._execution_plan = build_execution_plan(
            self._definitions,
            self.spec_list,
            n_samples=self.orig.shape[0],
            original_dimension=self.orig.shape[1],
            default_k=self.DEFAULT_K,
            memory_budget=self.max_memory_bytes,
            geodesic=self.geodesic,
        )
        self.distance_matrices_flag = any(
            key.kind is ResourceKind.DISTANCE_MATRIX
            for key in self._execution_plan.resources
        )
        self.estimated_cache_bytes = self._estimate_cache_bytes()
        if (
            self.max_memory_bytes is not None
            and self.estimated_cache_bytes > self.max_memory_bytes
        ):
            raise MemoryError(
                "Estimated ZADU cache size exceeds max_memory_bytes "
                f"({self.estimated_cache_bytes} > {self.max_memory_bytes})"
            )
        if (
            self.max_memory_bytes is not None
            and self._execution_plan.planned_peak_bytes > self.max_memory_bytes
        ):
            raise MemoryError(
                "Estimated ZADU cache size or peak working memory exceeds "
                "max_memory_bytes "
                f"({self._execution_plan.planned_peak_bytes} > "
                f"{self.max_memory_bytes})"
            )

        self._provider = NumpyResourceProvider()
        self._resource_cache = ResourceCache(
            self._execution_plan,
            self._provider,
            geodesic=self.geodesic,
        )
        self._prepare_original_space()

    def measure(self, emb, label=None):
        """Compute every configured metric for *emb* in specification order."""

        emb_array = as_finite_2d(emb, "emb")
        if self.orig.shape[0] != emb_array.shape[0]:
            raise ValueError(
                "orig and emb must have the same number of rows "
                f"(orig={self.orig.shape[0]}, emb={emb_array.shape[0]})"
            )

        if any(definition.needs_label for definition in self._definitions):
            if label is None:
                names = [
                    definition.id
                    for definition in self._definitions
                    if definition.needs_label
                ]
                raise ValueError(
                    f"Label is required for measure(s): {', '.join(names)}"
                )
            label = validate_labels(label, emb_array.shape[0])

        self.last_run_info = None
        self.emb = emb_array
        self.label = label
        self._resource_cache.begin_run()
        run_started = perf_counter()
        self._prepare_embedded_space()
        self._prepare_paired_resources()

        score_results = []
        local_results = []
        metric_timings = []
        for index, (spec, definition) in enumerate(
            zip(self.spec_list, self._definitions, strict=True)
        ):
            exec_params = dict(spec["params"])
            if "orig" in definition.inputs:
                exec_params["orig"] = self.orig
            if "emb" in definition.inputs:
                exec_params["emb"] = self.emb
            if definition.needs_label:
                exec_params["label"] = self.label
            exec_params.update(
                self._resource_cache.arguments_for(
                    self._execution_plan.metric_plans[index]
                )
            )
            if definition.supports_local:
                exec_params["return_local"] = self.return_local

            if self.verbose:
                print(f"Computing {definition.id}")
            metric_started = perf_counter()
            result = definition.load().measure(**exec_params)
            metric_timings.append((definition.id, perf_counter() - metric_started))
            if self.return_local and definition.supports_local:
                score, local = result
                score_results.append(_python_scalars(score))
                local_results.append(local)
            else:
                score_results.append(_python_scalars(result))
                if self.return_local:
                    local_results.append(None)
            self._resource_cache.release_after(index)

        self.last_run_info = build_run_info(
            plan=self._execution_plan,
            cache=self._resource_cache,
            backend=self.execution.resolved_backend,
            device=self.execution.resolved_device,
            metric_timings=metric_timings,
            total_seconds=perf_counter() - run_started,
        )
        if self.return_local:
            return score_results, local_results
        return score_results

    @staticmethod
    def _resolve_execution(
        execution: ExecutionConfig | None,
        max_memory_bytes: int | None,
    ) -> ExecutionConfig:
        if execution is not None and not isinstance(execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig")
        config = execution or ExecutionConfig()
        if max_memory_bytes is None:
            return config
        if isinstance(max_memory_bytes, bool) or not isinstance(
            max_memory_bytes, Integral
        ):
            raise TypeError("max_memory_bytes must be an integer")
        if max_memory_bytes < 1:
            raise ValueError("max_memory_bytes must be greater than zero")
        if config.memory_budget_bytes is not None:
            raise ValueError(
                "Provide only one of max_memory_bytes or execution.memory_budget"
            )
        return replace(config, memory_budget=int(max_memory_bytes))

    def _validate_and_normalize_specs(self) -> None:
        if isinstance(self.spec_list, (str, bytes)) or not isinstance(
            self.spec_list, Sequence
        ):
            raise TypeError("spec_list must be a sequence of measure specifications")

        for spec in self.spec_list:
            if not isinstance(spec, dict):
                raise TypeError("Each measure specification must be a dictionary")
            if "id" not in spec:
                raise ValueError(
                    f"Measure specification missing required key 'id': {spec}"
                )

            raw_id = spec["id"]
            measure_id = raw_id.value if hasattr(raw_id, "value") else str(raw_id)
            definition = METRIC_BY_ID.get(measure_id) or METRIC_BY_ALIAS.get(measure_id)
            if definition is None:
                raise ValueError(f"Invalid measure name: {measure_id}")

            params = spec.get("params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise TypeError(
                    f"Invalid params for measure {measure_id}: params must be a dict"
                )

            unknown = set(params) - definition.user_params
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Invalid parameter(s) {names} for measure {definition.id}"
                )

            spec["id"] = definition.id
            spec["params"] = params
            self._validate_k(definition, params)
            self._definitions.append(definition)

    def _validate_k(self, definition: MetricDefinition, params: dict[str, Any]) -> None:
        if definition.k_rule is None:
            return
        value = params.get("k")
        if definition.k_rule == "optional_neighbor" and value is None:
            return
        if value is None:
            value = self.DEFAULT_K
        if definition.k_rule == "trustworthiness":
            validate_trustworthiness_k(self.orig.shape[0], value)
        else:
            validate_neighbor_k(self.orig.shape[0], value)

    def _interpret_specs(self) -> None:
        for spec, definition in zip(self.spec_list, self._definitions, strict=True):
            params = spec["params"]
            for requirement in definition.resources:
                if requirement.argument == "distance_matrices":
                    self.distance_matrices_flag = True
                elif requirement.argument == "knn_ranking_info":
                    self.ranking_k = max(
                        self.ranking_k, params.get("k", self.DEFAULT_K)
                    )
                elif requirement.argument == "knn_info":
                    self.knn_both_k = max(
                        self.knn_both_k, params.get("k", self.DEFAULT_K)
                    )
                elif requirement.argument == "knn_emb_info":
                    self.knn_emb_k = max(
                        self.knn_emb_k, params.get("k", self.DEFAULT_K)
                    )

    def _prepare_original_space(self) -> None:
        self._resource_cache.prepare_original(self.orig)
        self._sync_resource_views()

    def _estimate_cache_bytes(self) -> int:
        """Estimate persistent original+embedded cache storage."""

        return self._execution_plan.estimated_cache_bytes

    def _prepare_embedded_space(self) -> None:
        self._resource_cache.prepare_embedded(self.emb)
        self._sync_resource_views()

    def _prepare_paired_resources(self) -> None:
        self._resource_cache.prepare_paired(self.orig, self.emb)

    def _sync_resource_views(self) -> None:
        self.orig_distance_matrix = self._resource_cache.distance_matrix(Space.ORIGINAL)
        self.emb_distance_matrix = self._resource_cache.distance_matrix(Space.EMBEDDED)
        self.orig_knn_indices = self._resource_cache.neighbor_indices(Space.ORIGINAL)
        self.emb_knn_indices = self._resource_cache.neighbor_indices(Space.EMBEDDED)
        self.orig_knn_ranking = self._resource_cache.ranking(Space.ORIGINAL)
        self.emb_knn_ranking = self._resource_cache.ranking(Space.EMBEDDED)

    @staticmethod
    def _geodesic_distance(phi1, lambda1, phi2, lambda2) -> float:
        return NumpyResourceProvider.geodesic_distance(phi1, lambda1, phi2, lambda2)

    @classmethod
    def _pairwise_geodesic_distance_matrix(cls, orig):
        return NumpyResourceProvider.pairwise_geodesic_distance_matrix(orig)


def _python_scalars(value):
    """Convert NumPy scalar results while preserving arrays used by local output."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _python_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_python_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_python_scalars(item) for item in value)
    return value
