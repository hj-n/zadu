from typing import Literal
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import numpy.typing as npt

IVMMeasure = Literal["silhouette", "calinski_harabasz", "davies_bouldin"]


def measure(
    emb: npt.NDArray,
    label: npt.NDArray,
    measure: IVMMeasure | str = "silhouette",
) -> dict:
    """
    Compute internal validation measure of the embedding.
    """

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

    score = scorers[measure_name](emb, label)
    return {measure_name: score}
