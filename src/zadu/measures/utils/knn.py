from .pairwise_dist import pairwise_distance_matrix
import numpy as np
import numpy.typing as npt
import faiss
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix


def _validate_k(points: npt.NDArray, k: int) -> None:
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    n = points.shape[0]
    if n < 2:
        raise ValueError("At least 2 points are required to compute nearest neighbors")
    if k < 1 or k >= n:
        raise ValueError(f"k must satisfy 1 <= k < n (n={n}), got k={k}")


def knn_with_ranking(
    points: npt.NDArray, k: int, distance_matrix: npt.NDArray | None = None
) -> (npt.NDArray, npt.NDArray):
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

    sorted_indices = np.argsort(distance_matrix, axis=1)
    knn_indices = sorted_indices[:, 1 : k + 1]
    ranking = np.argsort(sorted_indices, axis=1)

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

    # make c-contiguous
    points = np.ascontiguousarray(points, dtype=np.float32)

    if distance_function.lower() == "euclidean":
        index = faiss.IndexFlatL2(points.shape[1])
        index.add(points)
        knn_indices = index.search(points, k + 1)[1][:, 1:]
    else:
        tree = KDTree(points, metric=distance_function)
        knn_indices = tree.query(points, k=k + 1, return_distance=False)[:, 1:]

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
