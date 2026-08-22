import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree

from .pairwise_dist import pairwise_distance_matrix
from .validation import validate_neighbor_k


def _validate_k(points: npt.NDArray, k: int) -> None:
    validate_neighbor_k(points.shape[0], k)


def _validate_distance_matrix(distance_matrix: npt.NDArray) -> npt.NDArray:
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square, " f"got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(matrix < 0):
        raise ValueError("distance_matrix must be non-negative")
    return matrix


def _sorted_neighbors_and_ranking(
    distance_matrix: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Sort distances while forcing each sample itself into rank zero."""

    sortable = _validate_distance_matrix(distance_matrix).copy()
    np.fill_diagonal(sortable, -np.inf)
    sorted_indices = np.argsort(sortable, axis=1, kind="stable")
    ranking = np.argsort(sorted_indices, axis=1, kind="stable")
    return sorted_indices[:, 1:], ranking


def knn_from_distance_matrix(distance_matrix: npt.NDArray, k: int) -> npt.NDArray:
    """Return k neighbors from a precomputed distance matrix."""

    matrix = _validate_distance_matrix(distance_matrix)
    validate_neighbor_k(matrix.shape[0], k)
    neighbors, _ = _sorted_neighbors_and_ranking(matrix)
    return neighbors[:, :k]


def knn_with_ranking(
    points: npt.NDArray, k: int, distance_matrix: npt.NDArray | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the k-nearest neighbors of the points along with the
    rankings of other points based on the distance to each point.
    If the distance matrix is not provided, it is computed in O(n^2) time.
    INPUT:
        ndarray: points: list of points
        int: k: number of nearest neighbors to compute
        ndarray: distance_matrix: pairwise distance matrix (Optional)
    OUTPUT:
        ndarray: knn_indices: k-nearest neighbors of each point
        ndarray: ranking: ranking of other points based on the distance to each point
    """

    _validate_k(points, k)

    if distance_matrix is None:
        distance_matrix = pairwise_distance_matrix(points, "euclidean")

    sorted_neighbors, ranking = _sorted_neighbors_and_ranking(distance_matrix)
    knn_indices = sorted_neighbors[:, :k]

    return knn_indices, ranking


def knn(
    points: npt.NDArray, k: int, distance_function: str = "euclidean"
) -> npt.NDArray:
    """
    Compute the k-nearest neighbors of the points
    If the distance function is euclidean, the computation relies on faiss-cpu.
    Otherwise, the computation is done based on scikit-learn KD Tree algorithm
    You can use any distance function supported by scikit-learn KD Tree or specify a callable function
    INPUT:
        ndarray: points: list of points
        int: k: number of nearest neighbors to compute
        str or callable: distance_function: distance function to use
    OUTPUT:
        ndarray: knn_indices: k-nearest neighbors of each point
    """

    _validate_k(points, k)

    if callable(distance_function):
        distance_matrix = pairwise_distance_matrix(points, distance_function)
        return knn_from_distance_matrix(distance_matrix, k)
    if not isinstance(distance_function, str):
        raise TypeError("distance_function must be a metric name or callable")

    # FAISS requires contiguous float32 input.
    points = np.ascontiguousarray(points, dtype=np.float32)

    if distance_function.lower() == "euclidean":
        import faiss

        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points)
        candidates = index.search(points, k + 1)[1]
        knn_indices = np.empty((points.shape[0], k), dtype=candidates.dtype)
        for row_idx, row in enumerate(candidates):
            without_self = row[row != row_idx]
            knn_indices[row_idx] = without_self[:k]
    else:
        tree = KDTree(points, metric=distance_function)
        candidates = tree.query(points, k=k + 1, return_distance=False)
        knn_indices = np.empty((points.shape[0], k), dtype=candidates.dtype)
        for row_idx, row in enumerate(candidates):
            without_self = row[row != row_idx]
            knn_indices[row_idx] = without_self[:k]

    return knn_indices


def snn(
    points: npt.NDArray,
    k: int,
    distance_function: str = "euclidean",
    directed: bool = True,
    knn_indices: tuple | None = None,
) -> npt.NDArray:
    """
    Compute the shared nearest neighbors (SNN) graph of the points
    INPUT:
        ndarray: points: list of points
        int: k: number of nearest neighbors to consider
        str or callable: distance_function: distance function to use
        bool: directed: whether the k-nearest neighbors graph using is directed or not
        tuple: knn_indices: precomputed k-nearest neighbors and rankings of the points (Optional)
    OUTPUT:
        ndarray: snn_graph: shared nearest neighbors (SNN) graph of the points
    """
    if knn_indices is None:
        _validate_k(points, k)
        knn_indices = knn(points, k, distance_function)

    n = knn_indices.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = knn_indices.flatten()
    vals = np.tile(np.arange(k, 0, -1), n)

    knn_graph = csr_matrix((vals, (rows, cols)), shape=(n, n))

    if directed:
        snn_graph = knn_graph @ knn_graph.T
    else:
        sym_graph = ((knn_graph + knn_graph.T) > 0).astype(int)
        snn_graph = sym_graph @ sym_graph

    snn_graph.setdiag(0)
    return snn_graph
