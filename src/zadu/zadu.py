"""Scheduled execution of ZADU dimensionality-reduction metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from time import perf_counter
from typing import Any, ClassVar

import numpy as np
from threadpoolctl import threadpool_limits

from .backends import NumpyResourceProvider, create_resource_provider
from .engine.batching import BatchExecutionPlan, build_batch_execution_plan
from .engine.config import ExecutionConfig
from .engine.errors import EmbeddingExecutionError
from .engine.planner import build_execution_plan
from .engine.resources import ResourceCache, ResourceKind, Space
from .engine.result import build_many_run_info, build_run_info
from .measures.utils.validation import (
    as_finite_2d,
    validate_labels,
    validate_neighbor_k,
    validate_positive_real,
    validate_trustworthiness_k,
)
from .registry import METRIC_BY_ALIAS, METRIC_BY_ID, MetricDefinition


class ZADU:
    """Evaluate one or more embeddings with registered exact metrics."""

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
        self._provider = create_resource_provider(self.execution)
        provider_working_memory = getattr(
            self._provider,
            "working_memory_bytes",
            None,
        )
        self._execution_plan = build_execution_plan(
            self._definitions,
            self.spec_list,
            n_samples=self.orig.shape[0],
            original_dimension=self.orig.shape[1],
            default_k=self.DEFAULT_K,
            memory_budget=self.max_memory_bytes,
            geodesic=self.geodesic,
            backend=self._provider.name,
            resource_dtype_bytes=np.dtype(self._provider.dtype).itemsize,
            provider_working_memory=provider_working_memory,
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

        self._resource_cache = ResourceCache(
            self._execution_plan,
            self._provider,
            geodesic=self.geodesic,
        )
        self._prepare_original_space()

    def measure(self, emb, label=None):
        """Compute every configured metric for *emb* in specification order."""

        emb_array = self._validate_embedding(emb, "emb")
        label_array = self._validate_measure_label(label)
        self.last_run_info = None
        result, run_info = self._measure_validated(emb_array, label_array)
        self.last_run_info = run_info
        return result

    def measure_many(self, embeddings, labels=None):
        """Compute every metric for an ordered collection of embeddings.

        Immutable original-space resources are shared across every embedding.
        Collection concurrency is opt-in through ``embedding_workers`` and is
        capped by the configured memory budget. ``labels`` is one optional
        label vector shared by the collection.
        """

        if isinstance(embeddings, (str, bytes)):
            raise TypeError("embeddings must be an iterable of 2D arrays")
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            raise ValueError(
                "embeddings must contain multiple 2D arrays; "
                "use measure() for one embedding"
            )
        try:
            raw_embeddings = list(embeddings)
        except TypeError as exc:
            raise TypeError("embeddings must be an iterable of 2D arrays") from exc

        embedding_arrays = [
            self._validate_embedding(embedding, f"embeddings[{index}]")
            for index, embedding in enumerate(raw_embeddings)
        ]
        label_array = self._validate_measure_label(labels)
        provider_batching, provider_batching_reason = self._provider_batching_mode(
            embedding_arrays
        )
        parallel_fallback_reason = self._parallel_fallback_reason()
        if (
            self._provider.supports_embedding_batching
            and self.execution.embedding_workers > 1
            and not provider_batching
        ):
            parallel_fallback_reason = provider_batching_reason
        batch_plan = build_batch_execution_plan(
            self._execution_plan,
            embedding_count=len(embedding_arrays),
            requested_workers=self.execution.embedding_workers,
            parallel_fallback_reason=parallel_fallback_reason,
            provider_batching=provider_batching,
            provider_batch_input_bytes=(
                embedding_arrays[0].size * np.dtype(self._provider.dtype).itemsize
                if provider_batching
                else 0
            ),
        )
        snc_effective_workers = self._snc_workers_for_collection(
            batch_plan.effective_workers
        )
        self.last_run_info = None
        batch_started = perf_counter()
        results = []
        run_infos = []
        final_cache = self._resource_cache
        if batch_plan.provider_batching:
            results, run_infos, final_cache = self._execute_provider_batches(
                embedding_arrays,
                label_array,
                batch_plan,
                snc_effective_workers,
            )
        elif batch_plan.effective_workers <= 1:
            for index, embedding in enumerate(embedding_arrays):
                try:
                    result, run_info = self._execute_embedding(
                        embedding,
                        label_array,
                        self._resource_cache,
                        snc_effective_workers,
                    )
                except Exception as exc:
                    raise EmbeddingExecutionError(index) from exc
                results.append(result)
                run_infos.append(run_info)
        else:
            results, run_infos, final_cache = self._execute_embedding_batches(
                embedding_arrays,
                label_array,
                batch_plan,
                snc_effective_workers,
            )

        if embedding_arrays:
            self._adopt_execution_state(
                embedding_arrays[-1],
                label_array,
                final_cache,
            )

        self.last_run_info = build_many_run_info(
            plan=self._execution_plan,
            cache=self._resource_cache,
            backend=self._provider.name,
            device=self._provider.device,
            dtype=self._provider.dtype,
            batch_plan=batch_plan,
            run_infos=run_infos,
            total_seconds=perf_counter() - batch_started,
            snc_effective_workers=snc_effective_workers,
        )
        return results

    def _execute_provider_batches(
        self,
        embedding_arrays,
        label_array,
        batch_plan: BatchExecutionPlan,
        snc_effective_workers: dict[int, int] | None,
    ):
        """Build embedded resources in native batches, then score in order."""

        results = []
        run_infos = []
        final_cache = self._resource_cache
        batch_size = batch_plan.native_batch_size
        for batch_start in range(0, len(embedding_arrays), batch_size):
            batch = embedding_arrays[batch_start : batch_start + batch_size]
            preparation_started = perf_counter()
            try:
                caches = final_cache.prepare_embedded_batch(batch)
            except Exception as exc:
                failed_index = batch_start + getattr(exc, "batch_index", 0)
                raise EmbeddingExecutionError(failed_index) from exc
            preparation_seconds = (perf_counter() - preparation_started) / len(batch)
            for offset, (embedding, cache) in enumerate(
                zip(batch, caches, strict=True)
            ):
                embedding_index = batch_start + offset
                try:
                    result, run_info = self._execute_prepared_embedding(
                        embedding,
                        label_array,
                        cache,
                        snc_effective_workers,
                        preparation_seconds=preparation_seconds,
                    )
                except Exception as exc:
                    raise EmbeddingExecutionError(embedding_index) from exc
                results.append(result)
                run_infos.append(run_info)
                final_cache = cache
        return results, run_infos, final_cache

    def _execute_embedding_batches(
        self,
        embedding_arrays,
        label_array,
        batch_plan: BatchExecutionPlan,
        snc_effective_workers: dict[int, int] | None,
    ):
        """Run worker-sized batches so completed caches cannot accumulate."""

        results = []
        run_infos = []
        final_cache = self._resource_cache
        label_view = _readonly_view(label_array)
        base_cache = self._resource_cache
        base_cache.freeze_original()
        worker_count = batch_plan.effective_workers
        with (
            threadpool_limits(limits=1),
            ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="zadu-embedding",
            ) as executor,
        ):
            for batch_start in range(0, len(embedding_arrays), worker_count):
                batch = embedding_arrays[batch_start : batch_start + worker_count]
                futures = [
                    executor.submit(
                        self._execute_isolated_embedding,
                        _readonly_view(embedding),
                        label_view,
                        base_cache,
                        snc_effective_workers,
                    )
                    for embedding in batch
                ]
                for offset, future in enumerate(futures):
                    embedding_index = batch_start + offset
                    try:
                        result, run_info, cache = future.result()
                    except Exception as exc:
                        for pending in futures[offset + 1 :]:
                            pending.cancel()
                        raise EmbeddingExecutionError(embedding_index) from exc
                    results.append(result)
                    run_infos.append(run_info)
                    if embedding_index == len(embedding_arrays) - 1:
                        final_cache = cache
                del futures
        return results, run_infos, final_cache

    def _execute_isolated_embedding(
        self,
        embedding,
        label,
        base_cache: ResourceCache,
        snc_effective_workers: dict[int, int] | None,
    ):
        cache = base_cache.fork_original(base_cache.provider.fork())
        result, run_info = self._execute_embedding(
            embedding,
            label,
            cache,
            snc_effective_workers,
        )
        return result, run_info, cache

    def _validate_embedding(self, emb, name: str) -> np.ndarray:
        emb_array = as_finite_2d(emb, name)
        if self.orig.shape[0] != emb_array.shape[0]:
            raise ValueError(
                "orig and emb must have the same number of rows "
                f"(orig={self.orig.shape[0]}, emb={emb_array.shape[0]})"
            )
        return emb_array

    def _validate_measure_label(self, label):
        if not any(definition.needs_label for definition in self._definitions):
            return label
        if label is None:
            names = [
                definition.id
                for definition in self._definitions
                if definition.needs_label
            ]
            raise ValueError(f"Label is required for measure(s): {', '.join(names)}")
        return validate_labels(label, self.orig.shape[0])

    def _measure_validated(self, emb_array, label_array):
        """Measure one validated embedding and return its result and diagnostics."""

        snc_effective_workers = self._snc_workers_for_collection(1)
        result, run_info = self._execute_embedding(
            emb_array,
            label_array,
            self._resource_cache,
            snc_effective_workers,
        )
        self._adopt_execution_state(emb_array, label_array, self._resource_cache)
        return result, run_info

    def _execute_embedding(
        self,
        emb_array,
        label_array,
        cache: ResourceCache,
        snc_effective_workers: dict[int, int] | None,
    ):
        """Execute one embedding without mutating collection-level state."""

        preparation_started = perf_counter()
        cache.begin_run()
        cache.prepare_embedded(emb_array)
        return self._execute_prepared_embedding(
            emb_array,
            label_array,
            cache,
            snc_effective_workers,
            preparation_seconds=perf_counter() - preparation_started,
        )

    def _execute_prepared_embedding(
        self,
        emb_array,
        label_array,
        cache: ResourceCache,
        snc_effective_workers: dict[int, int] | None,
        *,
        preparation_seconds: float = 0.0,
    ):
        """Finish paired resources and metrics after embedded preparation."""

        run_started = perf_counter()
        cache.prepare_paired(self.orig, emb_array)

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
                exec_params["emb"] = emb_array
            if definition.needs_label:
                exec_params["label"] = label_array
            exec_params.update(
                cache.arguments_for(self._execution_plan.metric_plans[index])
            )
            snc_plan = self._execution_plan.snc_plan
            if snc_plan is not None and index in snc_plan.effective_workers:
                workers = snc_effective_workers[index]
                exec_params["n_jobs"] = workers
                exec_params["working_memory_bytes"] = (
                    snc_plan.graph_bytes[index]
                    + workers * snc_plan.iteration_bytes[index]
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
            cache.release_after(index)

        run_info = build_run_info(
            plan=self._execution_plan,
            cache=cache,
            backend=cache.provider.name,
            device=cache.provider.device,
            dtype=cache.provider.dtype,
            metric_timings=metric_timings,
            total_seconds=preparation_seconds + perf_counter() - run_started,
            snc_effective_workers=snc_effective_workers,
        )
        if self.return_local:
            return (score_results, local_results), run_info
        return score_results, run_info

    def _provider_batching_mode(
        self,
        embedding_arrays: list[np.ndarray],
    ) -> tuple[bool, str | None]:
        """Select native batching without enabling unsafe provider threads."""

        if not self._provider.supports_embedding_batching:
            return False, None
        if self.execution.embedding_workers <= 1:
            return False, None
        if len(embedding_arrays) <= 1:
            return False, "embedding_count"
        dimensions = {embedding.shape[1] for embedding in embedding_arrays}
        if len(dimensions) != 1:
            return False, "embedding_shape_mismatch"
        if not any(
            self._provider.can_batch(key)
            for key in self._execution_plan.resources_for(Space.EMBEDDED)
        ):
            return False, "no_batchable_embedded_resources"
        return True, None

    def _snc_workers_for_collection(
        self,
        collection_workers: int,
    ) -> dict[int, int] | None:
        snc_plan = self._execution_plan.snc_plan
        if snc_plan is None:
            return None
        if collection_workers > 1:
            return {index: 1 for index in snc_plan.effective_workers}
        return dict(snc_plan.effective_workers)

    def _parallel_fallback_reason(self) -> str | None:
        for spec, definition in zip(
            self.spec_list,
            self._definitions,
            strict=True,
        ):
            params = spec["params"]
            if _contains_mutable_random_state(params):
                return "mutable_random_state"
            if (
                definition.id == "steadiness_cohesiveness"
                and params.get("random_state") is None
            ):
                return "unseeded_snc"
            if (
                definition.id == "label_trustworthiness_and_continuity"
                and str(params.get("cvm", "dsc")).lower() == "ch_btw"
            ):
                return "global_random_state"
            if definition.id == "clustering_and_external_validation_measure":
                clustering = str(params.get("clustering", "kmeans")).lower()
                clustering_args = params.get("clustering_args")
                if (
                    clustering == "kmeans"
                    and isinstance(clustering_args, dict)
                    and clustering_args.get("random_state", 0) is None
                ):
                    return "unseeded_kmeans"
        return None

    def _adopt_execution_state(
        self,
        emb_array,
        label_array,
        cache: ResourceCache,
    ) -> None:
        self.emb = emb_array
        self.label = label_array
        self._resource_cache = cache
        self._provider = cache.provider
        self._sync_resource_views()

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
            if definition.id in {"distance_to_measure", "kl_divergence"}:
                params["sigma"] = validate_positive_real(
                    params.get("sigma", 0.1),
                    "sigma",
                )
            if definition.id == "steadiness_cohesiveness":
                self._validate_snc_params(params)
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

    @staticmethod
    def _validate_snc_params(params: dict[str, Any]) -> None:
        if "iteration" in params:
            iteration = params["iteration"]
            if isinstance(iteration, bool) or not isinstance(iteration, Integral):
                raise TypeError("iteration must be an integer")
            if iteration < 1:
                raise ValueError("iteration must be at least 1")
            params["iteration"] = int(iteration)
        for name, default in (("walk_num_ratio", 0.3), ("alpha", 0.1)):
            if name in params:
                params[name] = validate_positive_real(params.get(name, default), name)
        if "n_jobs" in params:
            n_jobs = params["n_jobs"]
            if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
                raise TypeError("n_jobs must be an integer")
            if n_jobs < 1:
                raise ValueError("n_jobs must be at least 1")
            params["n_jobs"] = int(n_jobs)

    def _interpret_specs(self) -> None:
        for spec, definition in zip(self.spec_list, self._definitions, strict=True):
            params = spec["params"]
            for requirement in definition.resources:
                if requirement.argument == "distance_matrices":
                    self.distance_matrices_flag = True
                elif (
                    requirement.argument == "knn_ranking_info"
                    or requirement.argument == "rank_comparisons"
                ):
                    self.ranking_k = max(
                        self.ranking_k, params.get("k", self.DEFAULT_K)
                    )
                elif (
                    requirement.argument == "knn_info"
                    or requirement.argument == "neighbor_statistics"
                ):
                    default_k = (
                        math.isqrt(self.orig.shape[0])
                        if requirement.k_default_rule == "sqrt"
                        else self.DEFAULT_K
                    )
                    self.knn_both_k = max(self.knn_both_k, params.get("k", default_k))
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


def _readonly_view(value):
    if value is None:
        return None
    view = np.asarray(value).view()
    view.flags.writeable = False
    return view


def _contains_mutable_random_state(value) -> bool:
    if isinstance(value, (np.random.Generator, np.random.RandomState)):
        return True
    if isinstance(value, dict):
        return any(_contains_mutable_random_state(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_mutable_random_state(item) for item in value)
    return False
