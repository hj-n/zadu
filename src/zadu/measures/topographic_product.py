from .utils import pairwise_dist as pdist
import numpy as np
from .utils import knn

def measure(orig, emb, k):
    N = len(emb)
    sum_of_log_p3 = 0
    """
	Compute topographic product
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
        int: k: number of nearest neighbors to consider
	OUTPUT:
		topographic product result
	"""

    # k nearest neighbors in original space and embedded space each
    orig_knn_indices = knn.knn(orig, k)
    emb_knn_indices  = knn.knn(emb, k)
    
    for j in range(N):
        for k in range(N-1):
            q1_product, q2_product = 1, 1
            for l in range(k):
                distance_origin_to_emb_knn      = orig_distance_matrix[j][emb_knn_indices[j][l]]
                distance_origin_to_origin_knn   = orig_distance_matrix[j][orig_knn_indices[j][l]]
                q1 = distance_origin_to_emb_knn / distance_origin_to_origin_knn
                q1_product *= q1

                distance_emb_to_emb_knn        = emb_distance_matrix[j][emb_knn_indices[j][l]]
                distance_emb_to_origin_knn     = emb_distance_matrix[j][orig_knn_indices[j][l]]
                q2 = distance_emb_to_emb_knn   / distance_emb_to_origin_knn
                q2_product *= q2

            p3 = pow(q1_product * q2_product, 1 / (2 * k))
            sum_of_log_p3 += np.log(p3)

    return sum_of_log_p3 / (N * (N - 1))
            