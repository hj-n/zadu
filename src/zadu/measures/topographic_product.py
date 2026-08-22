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
    N = len(emb)
    sum_of_log_p3 = 0

    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    # k nearest neighbors in original space and embedded space each
    if knn_info is None:
        orig_knn_indices = knn.knn(orig, k)
        emb_knn_indices = knn.knn(emb, k)
    else:
        orig_knn_indices, emb_knn_indices = knn_info

    for j in range(N):
        for ki in range(k):
            q1_product, q2_product = 1, 1
            for neighbor_rank in range(ki + 1):
                distance_origin_to_emb_knn = orig_distance_matrix[j][
                    emb_knn_indices[j][neighbor_rank]
                ]
                distance_origin_to_origin_knn = orig_distance_matrix[j][
                    orig_knn_indices[j][neighbor_rank]
                ]
                if distance_origin_to_origin_knn <= 0:
                    raise ValueError(
                        "Topographic Product is undefined for zero-distance "
                        "original-space neighbors"
                    )
                q1 = distance_origin_to_emb_knn / distance_origin_to_origin_knn
                q1_product *= q1

                distance_emb_to_emb_knn = emb_distance_matrix[j][
                    emb_knn_indices[j][neighbor_rank]
                ]
                distance_emb_to_origin_knn = emb_distance_matrix[j][
                    orig_knn_indices[j][neighbor_rank]
                ]
                if distance_emb_to_origin_knn <= 0:
                    raise ValueError(
                        "Topographic Product is undefined for zero-distance "
                        "embedded-space neighbors"
                    )
                q2 = distance_emb_to_emb_knn / distance_emb_to_origin_knn
                q2_product *= q2

            product = q1_product * q2_product
            if product <= 0 or not np.isfinite(product):
                raise ValueError(
                    "Topographic Product is undefined for coincident points"
                )
            p3 = pow(product, 1 / (2 * (ki + 1)))
            sum_of_log_p3 += np.log(p3)

    topographic_product = sum_of_log_p3 / (N * k)
    return {"topographic_product": float(topographic_product)}
