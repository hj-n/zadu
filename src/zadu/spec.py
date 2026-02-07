from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict


class MeasureId(StrEnum):
    TRUSTWORTHINESS_CONTINUITY = "trustworthiness_continuity"
    MEAN_RELATIVE_RANK_ERROR = "mean_relative_rank_error"
    LOCAL_CONTINUITY_META_CRITERIA = "local_continuity_meta_criteria"
    NEIGHBORHOOD_HIT = "neighborhood_hit"
    CLASS_AWARE_TRUSTWORTHINESS_CONTINUITY = "class_aware_trustworthiness_continuity"
    LABEL_TRUSTWORTHINESS_AND_CONTINUITY = "label_trustworthiness_and_continuity"
    NEIGHBOR_DISSIMILARITY = "neighbor_dissimilarity"
    DISTANCE_TO_MEASURE = "distance_to_measure"
    KL_DIVERGENCE = "kl_divergence"
    DISTANCE_CONSISTENCY = "distance_consistency"
    PEARSON_R = "pearson_r"
    SPEARMAN_RHO = "spearman_rho"
    INTERNAL_VALIDATION_MEASURE = "internal_validation_measure"
    CLUSTERING_AND_EXTERNAL_VALIDATION_MEASURE = (
        "clustering_and_external_validation_measure"
    )
    STEADINESS_COHESIVENESS = "steadiness_cohesiveness"
    TOPOGRAPHIC_PRODUCT = "topographic_product"
    PROCRUSTES = "procrustes"
    STRESS = "stress"
    SCALE_NORMALIZED_STRESS = "scale_normalized_stress"
    NON_METRIC_STRESS = "non_metric_stress"


class Spec(TypedDict, total=False):
    id: str | MeasureId
    params: dict[str, Any]


def make_spec(id: str | MeasureId, **params: Any) -> Spec:
    """Build one ZADU specification item with typed autocomplete-friendly arguments."""

    return {
        "id": id.value if isinstance(id, MeasureId) else id,
        "params": params,
    }
