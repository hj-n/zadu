import numpy as np
import numpy.typing as npt

from zadu.engine.resources import PairStatistics
from zadu.kernels import stress_from_statistics

from .utils import pairwise_dist as pdist
from .utils.validation import require_nonzero_distances, validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    distance_matrices: tuple | None = None,
    pair_statistics: PairStatistics | None = None,
) -> dict:
    """
    Compute stress of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
    tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
    OUTPUT:
            dict: stress
    """
    orig, emb = validate_pair(orig, emb)
    if pair_statistics is not None:
        return {"stress": float(stress_from_statistics(pair_statistics))}
    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)

    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    require_nonzero_distances(orig_distance_matrix, "Stress")
    diff_squared_sum = np.square(orig_distance_matrix - emb_distance_matrix).sum()
    orig_squared_sum = np.square(orig_distance_matrix).sum()

    stress = np.sqrt(diff_squared_sum / orig_squared_sum)

    return {"stress": float(stress)}
