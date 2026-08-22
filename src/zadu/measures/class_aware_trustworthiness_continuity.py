import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils.validation import validate_labels, validate_pair, validate_trustworthiness_k


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    label: npt.NDArray,
    k: int = 20,
    knn_ranking_info: tuple | None = None,
    return_local: bool = False,
) -> tuple | dict:
    """
    Compute class-aware trustworthiness and continuity of the embedding
    INPUT:
        ndarray: orig: original data
        ndarray: emb: embedded data
        ndarray: label: label of the original data
        int: k: number of nearest neighbors to consider
        tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
        dict: class-aware trustworthiness (ca_trustworthiness) and class-aware continuity (ca_continuity)
    """

    orig, emb = validate_pair(orig, emb)
    label = validate_labels(label, orig.shape[0], min_classes=2)
    k = validate_trustworthiness_k(orig.shape[0], k)

    if knn_ranking_info is None:
        orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
        emb_knn_indices, emb_ranking = knn.knn_with_ranking(emb, k)
    else:
        orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

    if return_local:
        ca_trust, local_ca_trust = ca_tnc_computation(
            orig_knn_indices,
            orig_ranking,
            emb_knn_indices,
            label,
            k,
            "false",
            return_local,
        )
        ca_cont, local_ca_cont = ca_tnc_computation(
            emb_knn_indices,
            emb_ranking,
            orig_knn_indices,
            label,
            k,
            "missing",
            return_local,
        )
        return (
            {"ca_trustworthiness": ca_trust, "ca_continuity": ca_cont},
            {
                "local_ca_trustworthiness": local_ca_trust,
                "local_ca_continuity": local_ca_cont,
            },
        )
    else:
        ca_trust = ca_tnc_computation(
            orig_knn_indices,
            orig_ranking,
            emb_knn_indices,
            label,
            k,
            "false",
            return_local,
        )
        ca_cont = ca_tnc_computation(
            emb_knn_indices,
            emb_ranking,
            orig_knn_indices,
            label,
            k,
            "missing",
            return_local,
        )

        return {"ca_trustworthiness": ca_trust, "ca_continuity": ca_cont}


def ca_tnc_computation(
    base_knn_indices: npt.NDArray,
    base_ranking: npt.NDArray,
    target_knn_indices: npt.NDArray,
    label: npt.NDArray,
    k: int,
    type_description: str,
    return_local: bool = False,
) -> npt.NDArray | tuple:
    """
    Core computation of class-aware trustworthiness and continuity
    """

    local_distortion_list = []
    points_num = base_knn_indices.shape[0]

    for i in range(points_num):
        missings = np.setdiff1d(target_knn_indices[i], base_knn_indices[i])
        local_distortion = 0.0
        for missing in missings:
            if type_description == "false":
                if label[i] != label[missing]:
                    local_distortion += base_ranking[i, missing] - k
            elif type_description == "missing":
                if label[i] == label[missing]:
                    local_distortion += base_ranking[i, missing] - k
            else:
                raise ValueError("type should be 'false' or 'missing'")

        local_distortion_list.append(local_distortion)

    local_distortion_list = np.array(local_distortion_list)
    local_distortion_list = 1 - local_distortion_list * (
        2 / (k * (2 * points_num - 3 * k - 1))
    )

    average_distortion = float(np.mean(local_distortion_list))

    if return_local:
        return average_distortion, local_distortion_list
    else:
        return average_distortion
