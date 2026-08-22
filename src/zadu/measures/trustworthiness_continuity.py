import numpy as np
import numpy.typing as npt

from zadu.engine.resources import RankComparisons

from .utils import knn
from .utils.validation import validate_pair, validate_trustworthiness_k
from .utils.vectorized import gather_ranks, rowwise_membership


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    k: int = 20,
    knn_ranking_info: tuple | None = None,
    return_local: bool = False,
    rank_comparisons: RankComparisons | None = None,
) -> tuple | dict:
    """
    Compute the trustworthiness and continuity of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
            int: k: number of nearest neighbors to consider
            tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
            dict: trustworthiness and continuity
    """
    orig, emb = validate_pair(orig, emb)
    k = validate_trustworthiness_k(orig.shape[0], k)

    if rank_comparisons is not None:
        trust_result = _tnc_from_comparison(
            rank_comparisons.orig_ranks_of_emb[:, :k],
            ~rank_comparisons.emb_in_orig[k],
            k,
            orig.shape[0],
            return_local,
        )
        continuity_result = _tnc_from_comparison(
            rank_comparisons.emb_ranks_of_orig[:, :k],
            ~rank_comparisons.orig_in_emb[k],
            k,
            orig.shape[0],
            return_local,
        )
        if return_local:
            trust, local_trust = trust_result
            cont, local_cont = continuity_result
            return (
                {"trustworthiness": trust, "continuity": cont},
                {
                    "local_trustworthiness": local_trust,
                    "local_continuity": local_cont,
                },
            )
        return {
            "trustworthiness": trust_result,
            "continuity": continuity_result,
        }

    if knn_ranking_info is None:
        orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
        emb_knn_indices, emb_ranking = knn.knn_with_ranking(emb, k)
    else:
        orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

    if return_local:
        trust, local_trust = tnc_computation(
            orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local
        )
        cont, local_cont = tnc_computation(
            emb_knn_indices, emb_ranking, orig_knn_indices, k, return_local
        )
        return (
            {"trustworthiness": trust, "continuity": cont},
            {"local_trustworthiness": local_trust, "local_continuity": local_cont},
        )
    else:
        trust = tnc_computation(
            orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local
        )
        cont = tnc_computation(
            emb_knn_indices, emb_ranking, orig_knn_indices, k, return_local
        )
        return {"trustworthiness": trust, "continuity": cont}


def tnc_computation(
    base_knn_indices: npt.NDArray,
    base_ranking: npt.NDArray,
    target_knn_indices: npt.NDArray,
    k: int,
    return_local: bool = False,
) -> tuple | npt.NDArray:
    """
    Core computation of trustworthiness and continuity
    """
    points_num = base_knn_indices.shape[0]
    missing_mask = ~rowwise_membership(target_knn_indices, base_knn_indices)
    target_ranks = gather_ranks(base_ranking, target_knn_indices)
    local_distortion_list = np.sum((target_ranks - k) * missing_mask, axis=1)
    local_distortion_list = 1 - local_distortion_list * (
        2 / (k * (2 * points_num - 3 * k - 1))
    )

    average_distortion = float(np.mean(local_distortion_list))

    if return_local:
        return average_distortion, local_distortion_list
    else:
        return average_distortion


def _tnc_from_comparison(
    target_ranks: npt.NDArray,
    missing_mask: npt.NDArray,
    k: int,
    points_num: int,
    return_local: bool,
) -> float | tuple[float, npt.NDArray]:
    local = np.sum((target_ranks - k) * missing_mask, axis=1)
    local = 1 - local * (2 / (k * (2 * points_num - 3 * k - 1)))
    average = float(np.mean(local))
    return (average, local) if return_local else average
