"""Explicit registry describing every metric exposed by :class:`zadu.ZADU`."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class MetricDefinition:
    """Execution contract for one metric."""

    id: str
    alias: str
    inputs: frozenset[str] = frozenset(("orig", "emb"))
    user_params: frozenset[str] = frozenset()
    needs_label: bool = False
    cache: frozenset[str] = frozenset()
    supports_local: bool = False
    k_rule: str | None = None

    def load(self) -> ModuleType:
        return import_module(f"zadu.measures.{self.id}")


def _metric(
    id: str,
    alias: str,
    *,
    params: tuple[str, ...] = (),
    inputs: tuple[str, ...] = ("orig", "emb"),
    label: bool = False,
    cache: tuple[str, ...] = (),
    local: bool = False,
    k_rule: str | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        id=id,
        alias=alias,
        inputs=frozenset(inputs),
        user_params=frozenset(params),
        needs_label=label,
        cache=frozenset(cache),
        supports_local=local,
        k_rule=k_rule,
    )


METRICS = (
    _metric(
        "trustworthiness_continuity",
        "tnc",
        params=("k",),
        cache=("knn_ranking_info",),
        local=True,
        k_rule="trustworthiness",
    ),
    _metric(
        "mean_relative_rank_error",
        "mrre",
        params=("k",),
        cache=("knn_ranking_info",),
        local=True,
        k_rule="neighbor",
    ),
    _metric(
        "local_continuity_meta_criteria",
        "lcmc",
        params=("k",),
        cache=("knn_info",),
        local=True,
        k_rule="neighbor",
    ),
    _metric(
        "neighborhood_hit",
        "nh",
        params=("k",),
        inputs=("emb",),
        label=True,
        cache=("knn_emb_info",),
        local=True,
        k_rule="neighbor",
    ),
    _metric(
        "class_aware_trustworthiness_continuity",
        "ca_tnc",
        params=("k",),
        label=True,
        cache=("knn_ranking_info",),
        local=True,
        k_rule="trustworthiness",
    ),
    _metric(
        "label_trustworthiness_and_continuity",
        "l_tnc",
        params=("cvm",),
        label=True,
    ),
    _metric(
        "neighbor_dissimilarity",
        "nd",
        params=("k",),
        cache=("knn_info",),
        k_rule="neighbor",
    ),
    _metric(
        "distance_to_measure", "dtm", params=("sigma",), cache=("distance_matrices",)
    ),
    _metric("kl_divergence", "kl_div", params=("sigma",), cache=("distance_matrices",)),
    _metric("distance_consistency", "dsc", inputs=("emb",), label=True),
    _metric("pearson_r", "pr", cache=("distance_matrices",)),
    _metric("spearman_rho", "srho", cache=("distance_matrices",)),
    _metric(
        "internal_validation_measure",
        "ivm",
        params=("measure",),
        inputs=("emb",),
        label=True,
    ),
    _metric(
        "clustering_and_external_validation_measure",
        "c_evm",
        params=("measure", "clustering", "clustering_args"),
        inputs=("emb",),
        label=True,
    ),
    _metric(
        "steadiness_cohesiveness",
        "snc",
        params=(
            "iteration",
            "walk_num_ratio",
            "alpha",
            "k",
            "clustering_strategy",
            "random_state",
        ),
        local=True,
        k_rule="optional_neighbor",
    ),
    _metric(
        "topographic_product",
        "topo",
        params=("k",),
        cache=("distance_matrices", "knn_info"),
        k_rule="neighbor",
    ),
    _metric(
        "procrustes",
        "proc",
        params=("k",),
        cache=("knn_info",),
        k_rule="neighbor",
    ),
    _metric("stress", "stress", cache=("distance_matrices",)),
    _metric("scale_normalized_stress", "sn_stress", cache=("distance_matrices",)),
    _metric("non_metric_stress", "nm_stress", cache=("distance_matrices",)),
    _metric(
        "class_angular_distortion_index",
        "cadi",
        params=("n_triplets", "random_seed"),
        label=True,
    ),
    _metric("gap_index", "gi", params=("metric",)),
)

METRIC_BY_ID = {metric.id: metric for metric in METRICS}
METRIC_BY_ALIAS = {metric.alias: metric for metric in METRICS}
METRIC_LOOKUP = {**METRIC_BY_ID, **METRIC_BY_ALIAS}
