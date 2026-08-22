import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils.validation import validate_pair
from .utils.vectorized import gather_ranks


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    k: int = 20,
    knn_ranking_info: tuple | None = None,
    return_local: bool = False,
) -> tuple | dict:
    """
    Compute Mean Relative Rank Error (MRRE) of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
            int: k: number of nearest neighbors to consider
            tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
            dict: MRRE_false and MRRE_missing
    """
    orig, emb = validate_pair(orig, emb)
    if knn_ranking_info is None:
        orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
        emb_knn_indices, emb_ranking = knn.knn_with_ranking(emb, k)
    else:
        orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

    if return_local:
        mrre_false, local_mrre_false = mrre_computation(
            orig_ranking, emb_ranking, emb_knn_indices, k, return_local
        )
        mrre_missing, local_mrre_missing = mrre_computation(
            emb_ranking, orig_ranking, orig_knn_indices, k, return_local
        )
        return (
            {"mrre_false": mrre_false, "mrre_missing": mrre_missing},
            {
                "local_mrre_false": local_mrre_false,
                "local_mrre_missing": local_mrre_missing,
            },
        )
    else:
        mrre_false = mrre_computation(
            orig_ranking, emb_ranking, emb_knn_indices, k, return_local
        )
        mrre_missing = mrre_computation(
            emb_ranking, orig_ranking, orig_knn_indices, k, return_local
        )

        return {
            "mrre_false": mrre_false,
            "mrre_missing": mrre_missing,
        }


def mrre_computation(
    base_ranking: npt.NDArray,
    target_ranking: npt.NDArray,
    target_knn_indices: npt.NDArray,
    k: int,
    return_local: bool = False,
) -> tuple | dict:
    """
    Core computation of MRRE
    """
    points_num = target_knn_indices.shape[0]
    base_rank_arr = gather_ranks(base_ranking, target_knn_indices)
    target_rank_arr = gather_ranks(target_ranking, target_knn_indices)
    local_distortion_list = np.sum(
        np.abs(base_rank_arr - target_rank_arr) / target_rank_arr, axis=1
    )

    c = sum([abs(points_num - 2 * i + 1) / i for i in range(1, k + 1)])
    local_distortion_list = 1 - local_distortion_list / c

    average_distortion = float(np.mean(local_distortion_list))

    if return_local:
        return average_distortion, local_distortion_list
    else:
        return average_distortion
