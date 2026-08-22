import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils import pairwise_dist as pdist
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    k: int = 20,
    distance_matrices: tuple | None = None,
    knn_info: tuple | None = None,
) -> dict:
    """
    Compute topographic product
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
          int: k: number of nearest neighbors to consider
    OUTPUT:
            topographic product result
    """
    orig, emb = validate_pair(orig, emb)
    points_num = len(emb)

    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices
    orig_distance_matrix = np.asarray(orig_distance_matrix)
    emb_distance_matrix = np.asarray(emb_distance_matrix)

    # k nearest neighbors in original space and embedded space each
    if knn_info is None:
        orig_knn_indices = knn.knn(orig, k)
        emb_knn_indices = knn.knn(emb, k)
    else:
        orig_knn_indices, emb_knn_indices = knn_info
    orig_knn_indices = np.asarray(orig_knn_indices)[:, :k]
    emb_knn_indices = np.asarray(emb_knn_indices)[:, :k]

    rows = np.arange(points_num)[:, None]
    distance_origin_to_emb_knn = orig_distance_matrix[rows, emb_knn_indices]
    distance_origin_to_origin_knn = orig_distance_matrix[rows, orig_knn_indices]
    if np.any(distance_origin_to_origin_knn <= 0):
        raise ValueError(
            "Topographic Product is undefined for zero-distance "
            "original-space neighbors"
        )

    distance_emb_to_emb_knn = emb_distance_matrix[rows, emb_knn_indices]
    distance_emb_to_origin_knn = emb_distance_matrix[rows, orig_knn_indices]
    if np.any(distance_emb_to_origin_knn <= 0):
        raise ValueError(
            "Topographic Product is undefined for zero-distance "
            "embedded-space neighbors"
        )

    q1 = distance_origin_to_emb_knn / distance_origin_to_origin_knn
    q2 = distance_emb_to_emb_knn / distance_emb_to_origin_knn
    ratios = q1 * q2
    if np.any(ratios <= 0) or not np.all(np.isfinite(ratios)):
        raise ValueError("Topographic Product is undefined for coincident points")

    prefix_log_products = np.cumsum(np.log(ratios), axis=1)
    prefix_lengths = np.arange(1, k + 1)
    log_p3 = prefix_log_products / (2 * prefix_lengths)
    topographic_product = np.mean(log_p3)
    return {"topographic_product": float(topographic_product)}
