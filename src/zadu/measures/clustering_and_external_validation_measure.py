from typing import Literal
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)
import numpy.typing as npt

EVMMeasure = Literal["arand", "ami", "nmi", "vmeasure"]
ClusteringMethod = Literal["kmeans", "dbscan"]


def measure(
    emb: npt.NDArray,
    label: npt.NDArray,
    measure: EVMMeasure | str = "arand",
    clustering: ClusteringMethod | str = "kmeans",
    clustering_args=None,
) -> dict:
    """
    Evaluate DR embedding using clustering and external validation measure.
    """
    measure_name = measure.lower()
    clustering_name = clustering.lower()

    if clustering_args is None:
        clustering_args = {}

    clusterers = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
    }
    if clustering_name not in clusterers:
        allowed = ", ".join(sorted(clusterers.keys()))
        raise ValueError(
            f"Invalid clustering algorithm '{clustering}'. Allowed values: {allowed}"
        )

    clustering_result = clusterers[clustering_name](**clustering_args).fit(emb)

    scorers = {
        "arand": adjusted_rand_score,
        "ami": adjusted_mutual_info_score,
        "nmi": normalized_mutual_info_score,
        "vmeasure": v_measure_score,
    }
    if measure_name not in scorers:
        allowed = ", ".join(sorted(scorers.keys()))
        raise ValueError(
            f"Invalid external validation measure '{measure}'. Allowed values: {allowed}"
        )

    score = scorers[measure_name](label, clustering_result.labels_)
    return {f"{clustering_name}_{measure_name}": score}
