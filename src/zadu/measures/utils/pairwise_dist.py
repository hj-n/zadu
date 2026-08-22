from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from .validation import as_finite_2d, validate_positive_real


def pairwise_distance_matrix(
    point: npt.NDArray, distance_function: str | Callable = "euclidean"
):
    """
    Compute the pairwise distance matrix of the point list.
    You can use any distance function from scipy.spatial.distance.cdist
    or specify a callable function.

    INPUT:
        ndarray: point: list of points
        str or callable: distance_function: distance function to use
    OUTPUT:
        ndarray: pairwise distance matrix
    """
    points = as_finite_2d(point, "point")
    if callable(distance_function):
        distance_matrix = cdist(points, points, distance_function)
    elif not isinstance(distance_function, str):
        raise TypeError("distance_function must be a metric name or callable")
    elif distance_function.lower() == "snn":
        # TODO
        raise NotImplementedError(
            "snn has not yet been implemented as a distance function"
        )
    else:
        distance_matrix = cdist(points, points, distance_function)

    if not np.all(np.isfinite(distance_matrix)) or np.any(distance_matrix < 0):
        raise ValueError("The distance function must return finite non-negative values")

    return distance_matrix


def distance_matrix_to_density(
    distance_matrix: npt.NDArray, sigma: float
) -> npt.NDArray:
    """
    Compute the density of each point based on the pairwise distance matrix.

    INPUT:
        ndarray: distance_matrix: pairwise distance matrix
        float: sigma: sigma parameter for the Gaussian kernel
    OUTPUT:
        ndarray: density
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("distance_matrix must contain finite non-negative values")
    sigma = validate_positive_real(sigma, "sigma")
    maximum = float(np.max(matrix))
    if maximum <= 0:
        raise ValueError(
            "Density-based measures are undefined when all pairwise distances are zero"
        )

    kernel = np.array(matrix, dtype=float, copy=True)
    kernel /= maximum
    np.square(kernel, out=kernel)
    kernel /= -sigma
    np.exp(kernel, out=kernel)
    density = np.sum(kernel, axis=-1)
    density = density / np.sum(density)
    return density
