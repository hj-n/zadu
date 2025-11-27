import numpy as np
from itertools import combinations
import numpy.typing as npt


def btw_ch(data: npt.NDArray, labels: npt.NDArray) -> float:
    return btw(data, labels)


def dsc_normalize(data: npt.NDArray, labels: npt.NDArray) -> float:
    """
    Compute the distance consistency in a vectorized manner.
    """
    labels = np.asarray(labels)
    unique_labels, labels = np.unique(labels, return_inverse=True)
    n_classes = len(unique_labels)

    # Compute centroids
    centroids = np.zeros((n_classes, data.shape[1]), dtype=float)
    for c in range(n_classes):
        centroids[c] = data[labels == c].mean(axis=0)

    dists = np.linalg.norm(data[:, None, :] - centroids[None, ...], axis=2)
    pred_labels = np.argmin(dists, axis=1)
    consistent_ratio = np.mean(pred_labels == labels)

    return consistent_ratio


def shift(X: npt.NDArray, label: npt.NDArray) -> float:
    labels = np.asarray(label)
    unique_labels, labels = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)
    n_samples, n_features = X.shape

    global_centroid = X.mean(axis=0)
    dists_to_global = np.linalg.norm(X - global_centroid, axis=1)
    std = dists_to_global.std()
    sums = np.vstack(
        [
            np.bincount(labels, weights=X[:, j], minlength=n_clusters)
            for j in range(n_features)
        ]
    ).T
    counts = np.bincount(labels, minlength=n_clusters)[:, None]
    centroids = sums / counts

    centroid_dists = np.linalg.norm(centroids - global_centroid, axis=1)

    dists_to_centroids = np.linalg.norm(X - centroids[labels], axis=1)
    compactness = np.sum(np.exp(dists_to_centroids / std))

    separability = np.sum(np.exp(centroid_dists / std) * counts.ravel())

    result = (separability * (n_samples - 2)) / compactness

    if not np.isfinite(result):
        raise FloatingPointError("Result became inf or NaN")

    return float(result)


def shift_range(X, label, iter_num):
    orig = shift(X, label)
    orig_result = 1 / (1 + orig ** (-1))
    e_val_sum = 0
    for i in range(iter_num):
        np.random.shuffle(label)
        e_val_sum += shift(X, label)
    e_val = e_val_sum / iter_num
    e_val_result = 1 / (1 + e_val ** (-1))
    if e_val_result == 1:
        return 0
    return (orig_result - e_val_result) / (1 - e_val_result)


def shift_range_class(X: npt.NDArray, label: npt.NDArray, iter_num: int) -> float:
    unique_labels = np.unique(label)
    class_indices = {c: np.where(label == c)[0] for c in unique_labels}

    def process_pair(a, b):
        idx = np.concatenate([class_indices[a], class_indices[b]])
        X_pair = X[idx]
        labels_pair = label[idx]
        labels_pair = (labels_pair == b).astype(np.int32)
        return shift_range(X_pair, labels_pair, iter_num)

    scores = [process_pair(a, b) for a, b in combinations(unique_labels, 2)]

    return float(np.mean(scores))


def btw(X: npt.NDArray, labels: npt.NDArray, iter_num: int = 20) -> float:
    return shift_range_class(X, labels, iter_num)


def centroid(X: npt.NDArray) -> npt.NDArray:
    """
    Compute the centroid of a set of vectors.
    :param X: The set of vectors.
    :return: The centroid.
    """
    return np.mean(X, axis=0)
