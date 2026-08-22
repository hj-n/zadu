import numpy as np
import numpy.typing as npt

from zadu.engine.resources import PairStatistics
from zadu.kernels import scale_normalized_stress_from_statistics

from .stress import measure as stressmeasure
from .utils import pairwise_dist as pdist
from .utils.validation import require_nonzero_distances, validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    distance_matrices: tuple | None = None,
    pair_statistics: PairStatistics | None = None,
) -> dict:
    """
    Compute the minimum stress for a given embedding.
    INPUT:
        ndarray: orig: original data
        ndarray: emb: embedded data
        tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
    OUTPUT:
        dict: scale_normalized_stress
    """
    orig, emb = validate_pair(orig, emb)
    if pair_statistics is not None:
        return {
            "scale_normalized_stress": float(
                scale_normalized_stress_from_statistics(pair_statistics)
            )
        }
    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)

    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    require_nonzero_distances(orig_distance_matrix, "Scale-normalized stress")
    require_nonzero_distances(emb_distance_matrix, "Scale-normalized stress")
    alpha = np.sum(np.multiply(orig_distance_matrix, emb_distance_matrix)) / np.sum(
        np.square(emb_distance_matrix)
    )
    sns = stressmeasure(
        orig, alpha * emb, (orig_distance_matrix, alpha * emb_distance_matrix)
    )

    return {"scale_normalized_stress": float(sns["stress"])}
