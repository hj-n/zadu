import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse

from .utils import knn
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    k: int = 20,
    snn_info: tuple | None = None,
    knn_info: tuple | None = None,
) -> dict:
    """
    Compute neighbor dissimilarity (ND) of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
            int: k: number of nearest neighbors to consider
            tuple: knn_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
            dict: neighbor dissimilarity (ND)
    """

    orig, emb = validate_pair(orig, emb)
    if snn_info is None:
        if knn_info is None:
            orig_SNN_graph = knn.snn(orig, k, directed=False)
            emb_SNN_graph = knn.snn(emb, k, directed=False)
        else:
            orig_SNN_graph = knn.snn(orig, k, knn_indices=knn_info[0], directed=False)
            emb_SNN_graph = knn.snn(emb, k, knn_indices=knn_info[1], directed=False)

    else:
        orig_SNN_graph, emb_SNN_graph = snn_info

    D = (orig_SNN_graph - emb_SNN_graph) / k

    if issparse(D):
        # Keep sparse operations sparse-safe: use element-wise square via .power.
        D_plus = D.maximum(0)
        D_minus = (-D).maximum(0)
        dissim_plus = np.sqrt(D_plus.power(2).sum())
        dissim_minus = np.sqrt(D_minus.power(2).sum())
    else:
        D_plus = D[D > 0]
        D_minus = D[D < 0]
        dissim_plus = np.sqrt(np.sum(D_plus**2))
        dissim_minus = np.sqrt(np.sum(D_minus**2))

    nd = float(max(dissim_plus, dissim_minus))

    return {"neighbor_dissimilarity": nd}
