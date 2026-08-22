from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal Python 3.10-compatible backport used by the public API."""

        def __str__(self) -> str:
            return self.value


from typing import Any, TypedDict


class MEASURE(StrEnum):
    TNC = "trustworthiness_continuity"
    MRRE = "mean_relative_rank_error"
    LCMC = "local_continuity_meta_criteria"
    NH = "neighborhood_hit"
    CA_TNC = "class_aware_trustworthiness_continuity"
    L_TNC = "label_trustworthiness_and_continuity"
    ND = "neighbor_dissimilarity"
    DTM = "distance_to_measure"
    KL_DIV = "kl_divergence"
    DSC = "distance_consistency"
    PR = "pearson_r"
    SRHO = "spearman_rho"
    IVM = "internal_validation_measure"
    C_EVM = "clustering_and_external_validation_measure"
    SNC = "steadiness_cohesiveness"
    TOPO = "topographic_product"
    PROC = "procrustes"
    STRESS = "stress"
    SN_STRESS = "scale_normalized_stress"
    NM_STRESS = "non_metric_stress"
    CADI = "class_angular_distortion_index"
    GI = "gap_index"


# Backward compatibility for previous typed API name.
MeasureId = MEASURE


class Spec(TypedDict, total=False):
    id: str | MEASURE
    params: dict[str, Any]


def make_spec(id: str | MEASURE, **params: Any) -> Spec:
    """Build one ZADU specification item with typed autocomplete-friendly arguments."""

    return {
        "id": id.value if isinstance(id, MEASURE) else id,
        "params": params,
    }
