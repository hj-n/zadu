import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr

from .utils import pairwise_dist as pdist
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray, emb: npt.NDArray, distance_matrices: tuple | None = None
):
    """
    Compute Pearson's correlation coefficient of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
            tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
    OUTPUT:
            dict: Pearson's correlation coefficient (r)
    """

    orig, emb = validate_pair(orig, emb)
    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    upper = np.triu_indices(orig.shape[0], k=1)
    orig_distances = orig_distance_matrix[upper]
    emb_distances = emb_distance_matrix[upper]
    if np.ptp(orig_distances) == 0 or np.ptp(emb_distances) == 0:
        raise ValueError("Pearson correlation is undefined for constant distances")
    r = pearsonr(orig_distances, emb_distances).statistic
    return {
        "pearson_r": float(r),
    }
