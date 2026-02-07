from zadu.measures.utils.knn import knn, knn_with_ranking, snn
import numpy as np


def test_knn():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    indices = knn(X, k=2)

    gt_indices = np.array([[1, 2], [0, 2], [1, 0], [4, 5], [3, 5], [4, 3]])

    assert np.array_equal(indices, gt_indices)


def test_knn_with_ranking():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    indices, rankings = knn_with_ranking(X, k=2)

    gt_indices = np.array([[1, 2], [0, 2], [1, 0], [4, 5], [3, 5], [4, 3]])

    gt_rankings = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 2, 3, 4, 5],
            [2, 1, 0, 3, 4, 5],
            [3, 4, 5, 0, 1, 2],
            [3, 4, 5, 1, 0, 2],
            [3, 4, 5, 2, 1, 0],
        ]
    )

    assert np.array_equal(indices, gt_indices)
    assert np.array_equal(rankings, gt_rankings)


def test_snn():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    snn_graph = snn(X, k=2).toarray()
    snn_gt = np.array(
        [
            [0.0, 1.0, 4.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [4.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 4.0, 2.0, 0.0],
        ]
    )

    assert np.array_equal(snn_graph, snn_gt)
