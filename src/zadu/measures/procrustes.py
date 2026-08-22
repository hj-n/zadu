import numpy as np
import numpy.typing as npt

from .utils import knn
from .utils.validation import validate_pair
from .utils.vectorized import DEFAULT_MAX_TEMP_BYTES, iter_row_blocks


def measure(
    orig: npt.NDArray, emb: npt.NDArray, k: int = 20, knn_info: tuple | None = None
) -> dict:
    """
    Compute procrustes statistics
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
          int: k: number of nearest neighbors to consider
    OUTPUT:
            Procrustes score
    """
    orig, emb = validate_pair(orig, emb)

    # k nearest neighbors in original space and embedded space each
    if knn_info is None:
        orig_knn_indices = knn.knn(orig, k)
        emb_knn_indices = knn.knn(emb, k)
    else:
        orig_knn_indices, emb_knn_indices = knn_info

    points_num = orig.shape[0]
    neighbor_count = orig_knn_indices.shape[1]
    work_dtype = np.result_type(orig.dtype, emb.dtype, np.float64)
    orig_dim = orig.shape[1]
    emb_dim = emb.shape[1]
    bytes_per_value = np.dtype(work_dtype).itemsize
    bytes_per_row = bytes_per_value * (
        4 * neighbor_count * (orig_dim + emb_dim) + 4 * orig_dim * emb_dim
    )
    block_budget = min(DEFAULT_MAX_TEMP_BYTES, max(1, bytes_per_row) * 256)
    normalized_scores = np.empty(points_num, dtype=work_dtype)

    for block in iter_row_blocks(
        points_num,
        max(1, bytes_per_row),
        max_block_bytes=block_budget,
    ):
        origin_neighbors = np.asarray(orig[orig_knn_indices[block]], dtype=work_dtype)
        embedded_neighbors = np.asarray(emb[emb_knn_indices[block]], dtype=work_dtype)
        centered_orig = origin_neighbors - np.mean(
            origin_neighbors, axis=1, keepdims=True
        )
        centered_emb = embedded_neighbors - np.mean(
            embedded_neighbors, axis=1, keepdims=True
        )

        cross_covariance = np.einsum(
            "nki,nkj->nij", centered_orig, centered_emb, optimize=True
        )
        u, _, vh = np.linalg.svd(cross_covariance, full_matrices=False)
        rotation = u @ vh
        aligned_emb = np.einsum("nkj,nij->nki", centered_emb, rotation, optimize=True)
        residual = centered_orig - aligned_emb
        numerator = np.sum(residual * residual, axis=(1, 2))
        denominator = np.sum(centered_orig * centered_orig, axis=(1, 2))
        if np.any(denominator <= 0):
            raise ValueError(
                "Procrustes is undefined for an original-space neighborhood "
                "with zero variance"
            )
        normalized_scores[block] = numerator / denominator

    procrustes = np.mean(normalized_scores)

    return {"procrustes": float(procrustes)}
