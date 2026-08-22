"""Scheduled execution of ZADU dimensionality-reduction metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np

from .measures.utils import knn
from .measures.utils import pairwise_dist as pdist
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
    ):
        self.spec_list = deepcopy(spec_list)
        self.return_local = bool(return_local)
        self.verbose = bool(verbose)
        self.geodesic = bool(geodesic)
        if max_memory_bytes is not None and max_memory_bytes < 1:
            raise ValueError("max_memory_bytes must be greater than zero")
        self.max_memory_bytes = max_memory_bytes

        self.orig = as_finite_2d(orig, "orig")
        self.emb = None
        self.label = None

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
        self.estimated_cache_bytes = self._estimate_cache_bytes()
        if (
            self.max_memory_bytes is not None
            and self.estimated_cache_bytes > self.max_memory_bytes
        ):
            raise MemoryError(
                "Estimated ZADU cache size exceeds max_memory_bytes "
                f"({self.estimated_cache_bytes} > {self.max_memory_bytes})"
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

        self.emb = emb_array
        self.label = label
        self._prepare_embedded_space()

        score_results = []
        local_results = []
        for spec, definition in zip(self.spec_list, self._definitions, strict=True):
            exec_params = dict(spec["params"])
            if "orig" in definition.inputs:
                exec_params["orig"] = self.orig
            if "emb" in definition.inputs:
                exec_params["emb"] = self.emb
            if definition.needs_label:
                exec_params["label"] = self.label

            k_value = exec_params.get("k", self.DEFAULT_K)
            if "distance_matrices" in definition.cache:
                exec_params["distance_matrices"] = (
                    self.orig_distance_matrix,
                    self.emb_distance_matrix,
                )
            if "knn_ranking_info" in definition.cache:
                exec_params["knn_ranking_info"] = (
                    self.orig_knn_indices[:, :k_value],
                    self.orig_knn_ranking,
                    self.emb_knn_indices[:, :k_value],
                    self.emb_knn_ranking,
                )
            if "knn_info" in definition.cache:
                exec_params["knn_info"] = (
                    self.orig_knn_indices[:, :k_value],
                    self.emb_knn_indices[:, :k_value],
                )
            if "knn_emb_info" in definition.cache:
                exec_params["knn_emb_info"] = self.emb_knn_indices[:, :k_value]
            if definition.supports_local:
                exec_params["return_local"] = self.return_local

            if self.verbose:
                print(f"Computing {definition.id}")
            result = definition.load().measure(**exec_params)
            if self.return_local and definition.supports_local:
                score, local = result
                score_results.append(_python_scalars(score))
                local_results.append(local)
            else:
                score_results.append(_python_scalars(result))
                if self.return_local:
                    local_results.append(None)

        if self.return_local:
            return score_results, local_results
        return score_results

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
            if "distance_matrices" in definition.cache:
                self.distance_matrices_flag = True
            if "knn_ranking_info" in definition.cache:
                self.ranking_k = max(self.ranking_k, params.get("k", self.DEFAULT_K))
            if "knn_info" in definition.cache:
                self.knn_both_k = max(self.knn_both_k, params.get("k", self.DEFAULT_K))
            if "knn_emb_info" in definition.cache:
                self.knn_emb_k = max(self.knn_emb_k, params.get("k", self.DEFAULT_K))

    def _prepare_original_space(self) -> None:
        if self.distance_matrices_flag:
            if self.geodesic:
                self.orig_distance_matrix = self._pairwise_geodesic_distance_matrix(
                    self.orig
                )
            else:
                self.orig_distance_matrix = pdist.pairwise_distance_matrix(self.orig)

        orig_k = max(self.ranking_k, self.knn_both_k)
        if orig_k < 0:
            return
        if self.ranking_k >= 0:
            self.orig_knn_indices, self.orig_knn_ranking = knn.knn_with_ranking(
                self.orig, orig_k, distance_matrix=self.orig_distance_matrix
            )
        elif self.orig_distance_matrix is not None:
            self.orig_knn_indices = knn.knn_from_distance_matrix(
                self.orig_distance_matrix, orig_k
            )
        else:
            self.orig_knn_indices = knn.knn(self.orig, orig_k)

    def _estimate_cache_bytes(self) -> int:
        """Estimate persistent original+embedded cache storage."""

        n_samples = self.orig.shape[0]
        total = 0
        if self.distance_matrices_flag:
            total += 2 * n_samples * n_samples * np.dtype(np.float64).itemsize
        if self.ranking_k >= 0:
            total += 2 * n_samples * n_samples * np.dtype(np.intp).itemsize
        orig_k = max(self.ranking_k, self.knn_both_k, 0)
        emb_k = max(self.ranking_k, self.knn_both_k, self.knn_emb_k, 0)
        total += (orig_k + emb_k) * n_samples * np.dtype(np.int64).itemsize
        return int(total)

    def _prepare_embedded_space(self) -> None:
        if self.distance_matrices_flag:
            self.emb_distance_matrix = pdist.pairwise_distance_matrix(self.emb)

        emb_k = max(self.ranking_k, self.knn_both_k, self.knn_emb_k)
        if emb_k < 0:
            return
        if self.ranking_k >= 0:
            self.emb_knn_indices, self.emb_knn_ranking = knn.knn_with_ranking(
                self.emb, emb_k, distance_matrix=self.emb_distance_matrix
            )
        elif self.emb_distance_matrix is not None:
            self.emb_knn_indices = knn.knn_from_distance_matrix(
                self.emb_distance_matrix, emb_k
            )
        else:
            self.emb_knn_indices = knn.knn(self.emb, emb_k)

    @staticmethod
    def _geodesic_distance(phi1, lambda1, phi2, lambda2) -> float:
        cosine = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(
            phi2
        ) * math.cos(abs(lambda2 - lambda1))
        return math.acos(float(np.clip(cosine, -1.0, 1.0)))

    @classmethod
    def _pairwise_geodesic_distance_matrix(cls, orig):
        if orig.shape[1] < 2:
            raise ValueError(
                "geodesic=True requires orig[:, 0] = longitude and "
                "orig[:, 1] = latitude in radians"
            )
        data_len = len(orig)
        distance_matrix = np.zeros((data_len, data_len))
        for i in range(data_len):
            for j in range(i + 1, data_len):
                distance_matrix[i, j] = distance_matrix[j, i] = cls._geodesic_distance(
                    orig[i, 1], orig[i, 0], orig[j, 1], orig[j, 0]
                )
        return distance_matrix


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
