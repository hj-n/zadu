import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    k: int = 20,
    knn_info: tuple | None = None,
    return_local: bool = False,
) -> tuple | dict:
    """
    Compute the local continuity meta-criteria of the embedding
    INPUT:
                  ndarray: orig: original data
                  ndarray: emb: embedded data
                  int: k: number of nearest neighbors to consider
                  tuple: knn_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
                  dict: local continuity meta-criteria
    """
    orig, emb = validate_pair(orig, emb)
    if knn_info is None:
        orig_knn_indices = knn.knn(orig, k)
        emb_knn_indices = knn.knn(emb, k)
    else:
        orig_knn_indices, emb_knn_indices = knn_info

    point_num = orig.shape[0]
    local_distortion_list = []

    for i in range(point_num):
        local_distortion_list.append(
            np.intersect1d(orig_knn_indices[i], emb_knn_indices[i]).shape[0]
            - ((k * k) / (point_num - 1))
        )

    local_distortion_list = np.array(local_distortion_list)
    local_distortion_list = local_distortion_list / k

    average_distortion = float(np.mean(local_distortion_list))

    if return_local:
        return ({"lcmc": average_distortion}, {"local_lcmc": local_distortion_list})
    else:
        return {"lcmc": average_distortion}
