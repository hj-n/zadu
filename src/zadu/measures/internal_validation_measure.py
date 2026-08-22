from typing import Literal

import numpy.typing as npt
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .utils.validation import as_finite_2d, validate_labels

IVMMeasure = Literal["silhouette", "calinski_harabasz", "davies_bouldin"]


def measure(
    emb: npt.NDArray,
    label: npt.NDArray,
    measure: IVMMeasure | str = "silhouette",
) -> dict:
    """
    Compute internal validation measure of the embedding.
    """

    emb = as_finite_2d(emb, "emb")
    label = validate_labels(label, emb.shape[0], min_classes=2)
    measure_name = measure.lower()
    scorers = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }

    if measure_name not in scorers:
        allowed = ", ".join(sorted(scorers.keys()))
        raise ValueError(
            f"Invalid internal validation measure '{measure}'. Allowed values: {allowed}"
        )

    score = float(scorers[measure_name](emb, label))
    return {measure_name: score}
