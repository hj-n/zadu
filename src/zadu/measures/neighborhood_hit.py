import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils.validation import as_finite_2d, validate_labels, validate_neighbor_k


def measure(
    emb: npt.NDArray,
    label: npt.NDArray,
    k: int = 20,
    knn_info: tuple | None = None,
    knn_emb_info: tuple | None = None,
    return_local: bool = False,
) -> tuple | dict:
    """
    Compute neighborhood hit of the embedding
    INPUT:
            ndarray: emb: embedded data
            ndarray: label: label of the original data
            int: k: number of nearest neighbors to consider
            tuple: knn_info: precomputed k-nearest neighbors of the original and embedded data (Optional)
    OUTPUT:
            dict: neighborhood hit (nh)
    """
    emb = as_finite_2d(emb, "emb")
    label = validate_labels(label, emb.shape[0])
    k = validate_neighbor_k(emb.shape[0], k)

    if knn_info is not None and knn_emb_info is not None:
        raise ValueError("Provide only one of knn_info or knn_emb_info")

    if knn_info is None:
        knn_info = knn_emb_info

    if knn_info is None:
        emb_knn_indices = knn.knn(emb, k)
    else:
        emb_knn_indices = knn_info[1] if isinstance(knn_info, tuple) else knn_info

    neighbor_labels = label[emb_knn_indices]
    nh_list = np.sum(neighbor_labels == label[:, None], axis=1) / k

    nh = float(np.mean(nh_list))

    if return_local:
        return ({"neighborhood_hit": nh}, {"local_neighborhood_hit": nh_list})
    else:
        return {"neighborhood_hit": nh}
