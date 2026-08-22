from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from .utils.validation import as_finite_2d, validate_labels

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
    emb = as_finite_2d(emb, "emb")
    label = validate_labels(label, emb.shape[0], min_classes=2)
    measure_name = measure.lower()
    clustering_name = clustering.lower()

    clustering_args = {} if clustering_args is None else dict(clustering_args)

    clusterers = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
    }
    if clustering_name not in clusterers:
        allowed = ", ".join(sorted(clusterers.keys()))
        raise ValueError(
            f"Invalid clustering algorithm '{clustering}'. Allowed values: {allowed}"
        )

    if clustering_name == "kmeans":
        clustering_args.setdefault("n_clusters", np.unique(label).size)
        clustering_args.setdefault("n_init", "auto")
        clustering_args.setdefault("random_state", 0)

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

    score = float(scorers[measure_name](label, clustering_result.labels_))
    return {f"{clustering_name}_{measure_name}": score}
