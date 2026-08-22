import numpy as np
import numpy.typing as npt
from numba import njit

from .utils.validation import validate_labels, validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    label: npt.NDArray,
    n_triplets: int = 0,
    random_seed: int | np.random.Generator | None = None,
) -> dict:
    """
    Computes the Class Angular Distortion Index (CADI) between a dataset X and a projection Y.
    CADI is a cluster-level metric that evaluates how well angles between clusters are preserved.


    Parameters
    ----------
    orig : ndarray of shape (N, d)
        (High-dimensional) dataset.
    emb : ndarray of shape (N, d2)
        (Low-dimensional) projection.
    label : ndarray of shape (N,)
        Maps each vector to a class label.
    n_triplets : int, optional
        Number of inter-cluster triplets to sample. If 0 (default), is automatically set to N * 10
    random_seed : int, np.random.Generator, or None, optional
        Random number generator for reproducibility.

    Returns
    -------
    cadi : dict
        {"class_angular_distortion_index": score}

    References
    ----------
    Gunaratne, K., Kobourov, S., & Miller, J. (2026). Class Angular Distortion
    Index for Dimensionality Reduction. In *Proceedings of the EuroVis 2026 Conference.*
    """
    orig, emb = validate_pair(orig, emb, min_samples=3)
    label = validate_labels(label, orig.shape[0], min_classes=2)

    if isinstance(n_triplets, bool) or not isinstance(n_triplets, (int, np.integer)):
        raise TypeError("n_triplets must be an integer")
    n_triplets = int(n_triplets)
    if n_triplets < 0:
        raise ValueError("n_triplets must be zero or greater")

    if not isinstance(random_seed, np.random.Generator):
        rng = np.random.default_rng(random_seed)
    else:
        rng = random_seed

    if n_triplets == 0:
        n_triplets = orig.shape[0] * 10

    cluster_info = _get_cluster_info(label)

    orig = orig.astype(np.float64)
    emb = emb.astype(np.float64)

    score = _cadi_kernel(
        orig,
        emb,
        cluster_info["offsets"],
        cluster_info["clusterB_idxs"],
        cluster_info["flat"],
        cluster_info["cluster_sizes"],
        n_triplets,
        rng,
    )
    return {"class_angular_distortion_index": float(score)}


def _get_cluster_info(labels):
    classes = np.unique(labels)
    n_clusters = len(classes)

    if len(labels) < 3:
        raise ValueError("Dataset must be composed of at least 3 points")

    if n_clusters < 2:
        raise ValueError(
            "CADI needs at least 2 clusters to sample angles between clusters."
        )

    lengths = np.array([np.sum(labels == c) for c in classes])
    offsets = np.concatenate([[0], lengths[:-1].cumsum()])
    flat = np.concatenate([np.argwhere(labels == c).flatten() for c in classes])

    clusterB_idxs = np.argwhere(lengths >= 2).flatten()
    if not len(clusterB_idxs):
        raise ValueError(
            "At least one cluster must have size >= 2 to sample y,z in "
            "triplet (x,y,z)."
        )

    return {
        "flat": flat,
        "offsets": offsets,
        "cluster_sizes": lengths,
        "clusterB_idxs": clusterB_idxs,
    }


@njit
def _cadi_kernel(
    X,
    Y,
    offsets,
    valid_clusterB_idxs,
    flat,
    cluster_sizes,
    n_triplets: int,
    rng: np.random.Generator,
):
    sum_sqr = 0.0

    for _ in range(n_triplets):

        # sample triplet
        x_idx, y_idx, z_idx = _sample_cadi_triplet(
            offsets,
            valid_clusterB_idxs,
            flat,
            cluster_sizes,
            rng,
        )

        # compute cosines
        cosX = _get_cosine(X, x_idx, y_idx, z_idx)
        cosY = _get_cosine(Y, x_idx, y_idx, z_idx)

        diff = cosX - cosY
        sum_sqr += diff * diff

    return sum_sqr / (4.0 * n_triplets)


@njit
def _sample_cadi_triplet(
    offsets,
    valid_clusterB_idxs,
    flat,
    cluster_sizes,
    rng: np.random.Generator,
):
    n_clusters = offsets.shape[0]
    n_validB = valid_clusterB_idxs.shape[0]

    # Sample cluster B first, then choose A uniformly from every other cluster.
    # This avoids rejection-sampling forever when only one cluster is eligible
    # for B (because it is the only cluster with at least two members).
    cB = valid_clusterB_idxs[rng.integers(0, n_validB)]
    cA = rng.integers(0, n_clusters - 1)
    if cA >= cB:
        cA += 1

    # Sample members
    sizeA = cluster_sizes[cA]
    sizeB = cluster_sizes[cB]

    offA = offsets[cA]
    offB = offsets[cB]

    x_idx = flat[offA + rng.integers(0, sizeA)]

    y_off = rng.integers(0, sizeB)
    z_off = rng.integers(0, sizeB)
    while y_off == z_off:
        z_off = rng.integers(0, sizeB)

    y_idx = flat[offB + y_off]
    z_idx = flat[offB + z_off]

    return x_idx, y_idx, z_idx


@njit
def _get_cosine(X, x_idx, y_idx, z_idx):
    d = X.shape[1]

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for j in range(d):
        v1 = X[y_idx, j] - X[x_idx, j]
        v2 = X[z_idx, j] - X[x_idx, j]

        dot += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    res = dot / np.sqrt(norm1 * norm2)
    return res
