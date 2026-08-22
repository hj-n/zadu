"""Shared validation helpers for metric inputs."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import numpy.typing as npt


def as_finite_2d(
    values: npt.ArrayLike,
    name: str,
    *,
    min_samples: int = 2,
) -> npt.NDArray:
    """Return *values* as a finite two-dimensional NumPy array."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    if array.shape[0] < min_samples:
        raise ValueError(
            f"{name} must contain at least {min_samples} samples, "
            f"got {array.shape[0]}"
        )
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    try:
        finite = np.all(np.isfinite(array))
    except TypeError as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_pair(
    orig: npt.ArrayLike,
    emb: npt.ArrayLike,
    *,
    min_samples: int = 2,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Validate a high-dimensional/embedded data pair."""

    orig_array = as_finite_2d(orig, "orig", min_samples=min_samples)
    emb_array = as_finite_2d(emb, "emb", min_samples=min_samples)
    if orig_array.shape[0] != emb_array.shape[0]:
        raise ValueError(
            "orig and emb must have the same number of rows "
            f"(orig={orig_array.shape[0]}, emb={emb_array.shape[0]})"
        )
    return orig_array, emb_array


def validate_labels(
    label: npt.ArrayLike,
    n_samples: int,
    *,
    min_classes: int = 1,
) -> npt.NDArray:
    """Validate one label per sample and a minimum number of classes."""

    labels = np.asarray(label)
    if labels.ndim != 1:
        raise ValueError(f"label must be a 1D array, got shape {labels.shape}")
    if labels.shape[0] != n_samples:
        raise ValueError(
            "label must contain one value per sample "
            f"(labels={labels.shape[0]}, samples={n_samples})"
        )
    if np.issubdtype(labels.dtype, np.number) and not np.all(np.isfinite(labels)):
        raise ValueError("label must contain only finite values")
    try:
        class_count = np.unique(labels).size
    except TypeError as exc:
        raise TypeError("label values must be mutually comparable") from exc
    if class_count < min_classes:
        raise ValueError(
            f"At least {min_classes} distinct class labels are required, "
            f"got {class_count}"
        )
    return labels


def validate_neighbor_k(n_samples: int, k: int, *, name: str = "k") -> int:
    """Validate a conventional nearest-neighbor count."""

    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError(f"{name} must be int, got {type(k).__name__}")
    k_int = int(k)
    if k_int < 1 or k_int >= n_samples:
        raise ValueError(
            f"{name} must satisfy 1 <= {name} < n "
            f"(n={n_samples}), got {name}={k_int}"
        )
    return k_int


def validate_trustworthiness_k(n_samples: int, k: int) -> int:
    """Validate the domain of the standard T&C normalization."""

    k_int = validate_neighbor_k(n_samples, k)
    if k_int >= n_samples / 2:
        raise ValueError(
            "k must be smaller than n / 2 for trustworthiness/continuity "
            f"normalization (n={n_samples}), got k={k_int}"
        )
    return k_int


def validate_positive_real(value: Real, name: str) -> float:
    """Validate a finite, strictly positive scalar parameter."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def require_nonzero_distances(distance_matrix: npt.NDArray, name: str) -> None:
    """Reject a distance matrix whose off-diagonal distances are all zero."""

    if float(np.max(distance_matrix)) <= 0:
        raise ValueError(f"{name} is undefined when all pairwise distances are zero")
