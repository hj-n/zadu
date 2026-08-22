import numpy as np
import numpy.typing as npt

from .utils import pairwise_dist as pdist
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    sigma: float = 0.1,
    distance_matrices: tuple | None = None,
    densities: tuple | None = None,
) -> dict:
    """
    Compute Kullback-Leibler divergence of the embedding
    INPUT:
        ndarray: orig: original data
        ndarray: emb: embedded data
        tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
    OUTPUT:
        dict: Kullback-Leibler divergence (kl)
    """

    orig, emb = validate_pair(orig, emb)
    if densities is not None:
        density_orig, density_emb = densities
    else:
        if distance_matrices is None:
            orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
            emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
        else:
            orig_distance_matrix, emb_distance_matrix = distance_matrices
        density_orig = pdist.distance_matrix_to_density(orig_distance_matrix, sigma)
        density_emb = pdist.distance_matrix_to_density(emb_distance_matrix, sigma)

    kl = np.sum(density_orig * np.log(density_orig / density_emb))
    return {"kl_divergence": float(kl)}
