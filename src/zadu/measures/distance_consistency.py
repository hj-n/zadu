import numpy as np
import numpy.typing as npt

from .utils.validation import as_finite_2d, validate_labels


def measure(emb: npt.NDArray, label: npt.NDArray) -> dict:
    """
    Compute distance consistency of the embedding
    INPUT:
        ndarray: emb: embedded data
        ndarray: label: label of the original data
    OUTPUT:
        dict: distance consistency (dsc)
    """

    emb = as_finite_2d(emb, "emb")
    label = validate_labels(label, emb.shape[0], min_classes=2)
    classes, encoded = np.unique(label, return_inverse=True)
    centroids = np.vstack(
        [np.mean(emb[encoded == i], axis=0) for i in range(len(classes))]
    )
    distances = np.linalg.norm(emb[:, None, :] - centroids[None, :, :], axis=2)
    predicted = np.argmin(distances, axis=1)
    return {"distance_consistency": float(np.mean(predicted == encoded))}
