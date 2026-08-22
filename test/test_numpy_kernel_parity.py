"""Reference-oracle tests for the exact NumPy metric kernels.

The helpers in this file intentionally use straightforward Python loops. They
are independent executable definitions of the pre-0.5.1 formulas, not code that
should be copied into the optimized implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from zadu.measures import (
    class_aware_trustworthiness_continuity as ca_tnc,
)
from zadu.measures import (
    local_continuity_meta_criteria as lcmc,
)
from zadu.measures import (
    mean_relative_rank_error as mrre,
)
from zadu.measures import (
    neighborhood_hit,
    procrustes,
    topographic_product,
)
from zadu.measures import (
    trustworthiness_continuity as tnc,
)
from zadu.measures.utils.knn import knn_from_distance_matrix, knn_with_ranking
from zadu.measures.utils.vectorized import (
    gather_ranks,
    iter_row_blocks,
    rowwise_intersection_count,
    rowwise_membership,
)


def _data_and_resources(
    *, seed: int, n: int, orig_dim: int, emb_dim: int, k: int, dtype
):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, orig_dim)).astype(dtype)
    projection = rng.normal(size=(orig_dim, emb_dim)).astype(dtype)
    emb = (orig @ projection + 0.05 * rng.normal(size=(n, emb_dim))).astype(dtype)
    orig_dist = cdist(orig, orig)
    emb_dist = cdist(emb, emb)
    orig_knn, orig_ranking = knn_with_ranking(orig, k, orig_dist)
    emb_knn, emb_ranking = knn_with_ranking(emb, k, emb_dist)
    return (
        orig,
        emb,
        orig_dist,
        emb_dist,
        orig_knn,
        orig_ranking,
        emb_knn,
        emb_ranking,
    )


def _reference_tnc(base_knn, base_ranking, target_knn, k):
    n = base_knn.shape[0]
    local = []
    for row in range(n):
        missing = np.setdiff1d(target_knn[row], base_knn[row])
        distortion = 0.0
        for index in missing:
            distortion += base_ranking[row, index] - k
        local.append(distortion)
    local = 1 - np.asarray(local) * (2 / (k * (2 * n - 3 * k - 1)))
    return float(np.mean(local)), local


def _reference_ca_tnc(base_knn, base_ranking, target_knn, labels, k, kind):
    n = base_knn.shape[0]
    local = []
    for row in range(n):
        missing = np.setdiff1d(target_knn[row], base_knn[row])
        distortion = 0.0
        for index in missing:
            false_neighbor = kind == "false" and labels[row] != labels[index]
            missing_neighbor = kind == "missing" and labels[row] == labels[index]
            if false_neighbor or missing_neighbor:
                distortion += base_ranking[row, index] - k
        local.append(distortion)
    local = 1 - np.asarray(local) * (2 / (k * (2 * n - 3 * k - 1)))
    return float(np.mean(local)), local


def _reference_mrre(base_ranking, target_ranking, target_knn, k):
    n = target_knn.shape[0]
    local = []
    for row in range(n):
        base_rank = base_ranking[row][target_knn[row]]
        target_rank = target_ranking[row][target_knn[row]]
        local.append(np.sum(np.abs(base_rank - target_rank) / target_rank))
    normalizer = sum(abs(n - 2 * rank + 1) / rank for rank in range(1, k + 1))
    local = 1 - np.asarray(local) / normalizer
    return float(np.mean(local)), local


def _reference_lcmc(orig_knn, emb_knn, n, k):
    local = []
    for row in range(n):
        overlap = np.intersect1d(orig_knn[row], emb_knn[row]).shape[0]
        local.append((overlap - (k * k) / (n - 1)) / k)
    local = np.asarray(local)
    return float(np.mean(local)), local


def _reference_neighborhood_hit(emb_knn, labels, k):
    local = []
    for row in range(len(labels)):
        local.append(np.sum(labels[emb_knn[row]] == labels[row]) / k)
    local = np.asarray(local)
    return float(np.mean(local)), local


def _reference_topographic_product(orig_dist, emb_dist, orig_knn, emb_knn, k):
    total = 0.0
    n = orig_knn.shape[0]
    for row in range(n):
        for prefix_end in range(k):
            q1_product = 1.0
            q2_product = 1.0
            for rank in range(prefix_end + 1):
                orig_denominator = orig_dist[row, orig_knn[row, rank]]
                emb_denominator = emb_dist[row, orig_knn[row, rank]]
                if orig_denominator <= 0:
                    raise ValueError("zero-distance original-space neighbors")
                if emb_denominator <= 0:
                    raise ValueError("zero-distance embedded-space neighbors")
                q1_product *= orig_dist[row, emb_knn[row, rank]] / orig_denominator
                q2_product *= emb_dist[row, emb_knn[row, rank]] / emb_denominator
            product = q1_product * q2_product
            if product <= 0 or not np.isfinite(product):
                raise ValueError("coincident points")
            total += np.log(product ** (1 / (2 * (prefix_end + 1))))
    return total / (n * k)


def _reference_procrustes(orig, emb, orig_knn, emb_knn):
    scores = []
    for row in range(orig.shape[0]):
        orig_neighbors = orig[orig_knn[row]]
        emb_neighbors = emb[emb_knn[row]]
        k = orig_neighbors.shape[0]
        centering = np.eye(k) - np.ones((k, k)) / k
        u, _, vh = np.linalg.svd(
            orig_neighbors.T @ centering @ emb_neighbors, full_matrices=False
        )
        rotation = u @ vh
        residual = centering @ (orig_neighbors - emb_neighbors @ rotation.T)
        numerator = np.linalg.norm(residual, ord="fro") ** 2
        denominator = np.linalg.norm(centering @ orig_neighbors, ord="fro") ** 2
        if denominator <= 0:
            raise ValueError("zero variance")
        scores.append(numerator / denominator)
    return float(np.mean(scores))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 5, 14])
def test_neighborhood_preservation_kernels_match_reference(dtype, k):
    resources = _data_and_resources(
        seed=41, n=31, orig_dim=6, emb_dim=2, k=k, dtype=dtype
    )
    (
        orig,
        emb,
        _,
        _,
        orig_knn,
        orig_ranking,
        emb_knn,
        emb_ranking,
    ) = resources
    labels = np.asarray(["class-a", "class-b", "class-c"])[np.arange(31) % 3]
    ranking_info = (orig_knn, orig_ranking, emb_knn, emb_ranking)

    expected_trust, local_trust = _reference_tnc(orig_knn, orig_ranking, emb_knn, k)
    expected_cont, local_cont = _reference_tnc(emb_knn, emb_ranking, orig_knn, k)
    tnc_score, tnc_local = tnc.measure(
        orig, emb, k, knn_ranking_info=ranking_info, return_local=True
    )
    assert tnc_score == pytest.approx(
        {"trustworthiness": expected_trust, "continuity": expected_cont}
    )
    np.testing.assert_allclose(tnc_local["local_trustworthiness"], local_trust)
    np.testing.assert_allclose(tnc_local["local_continuity"], local_cont)

    expected_false, local_false = _reference_ca_tnc(
        orig_knn, orig_ranking, emb_knn, labels, k, "false"
    )
    expected_missing, local_missing = _reference_ca_tnc(
        emb_knn, emb_ranking, orig_knn, labels, k, "missing"
    )
    ca_score, ca_local = ca_tnc.measure(
        orig,
        emb,
        labels,
        k,
        knn_ranking_info=ranking_info,
        return_local=True,
    )
    assert ca_score == pytest.approx(
        {
            "ca_trustworthiness": expected_false,
            "ca_continuity": expected_missing,
        }
    )
    np.testing.assert_allclose(ca_local["local_ca_trustworthiness"], local_false)
    np.testing.assert_allclose(ca_local["local_ca_continuity"], local_missing)

    expected_false, local_false = _reference_mrre(orig_ranking, emb_ranking, emb_knn, k)
    expected_missing, local_missing = _reference_mrre(
        emb_ranking, orig_ranking, orig_knn, k
    )
    mrre_score, mrre_local = mrre.measure(
        orig, emb, k, knn_ranking_info=ranking_info, return_local=True
    )
    assert mrre_score == pytest.approx(
        {"mrre_false": expected_false, "mrre_missing": expected_missing}
    )
    np.testing.assert_allclose(mrre_local["local_mrre_false"], local_false)
    np.testing.assert_allclose(mrre_local["local_mrre_missing"], local_missing)

    expected_lcmc, local_lcmc = _reference_lcmc(orig_knn, emb_knn, len(orig), k)
    lcmc_score, lcmc_local = lcmc.measure(
        orig, emb, k, knn_info=(orig_knn, emb_knn), return_local=True
    )
    assert lcmc_score["lcmc"] == pytest.approx(expected_lcmc)
    np.testing.assert_allclose(lcmc_local["local_lcmc"], local_lcmc)

    expected_nh, local_nh = _reference_neighborhood_hit(emb_knn, labels, k)
    nh_score, nh_local = neighborhood_hit.measure(
        emb, labels, k, knn_info=emb_knn, return_local=True
    )
    assert nh_score["neighborhood_hit"] == pytest.approx(expected_nh)
    np.testing.assert_allclose(nh_local["local_neighborhood_hit"], local_nh)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 6, 26])
def test_topographic_product_matches_cubic_reference(dtype, k):
    resources = _data_and_resources(
        seed=52, n=27, orig_dim=5, emb_dim=3, k=k, dtype=dtype
    )
    orig, emb, orig_dist, emb_dist, orig_knn, _, emb_knn, _ = resources
    expected = _reference_topographic_product(orig_dist, emb_dist, orig_knn, emb_knn, k)

    actual = topographic_product.measure(
        orig,
        emb,
        k,
        distance_matrices=(orig_dist, emb_dist),
        knn_info=(orig_knn, emb_knn),
    )["topographic_product"]

    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [2, 7, 23])
def test_batched_procrustes_matches_loop_reference(dtype, k):
    resources = _data_and_resources(
        seed=63, n=24, orig_dim=7, emb_dim=3, k=k, dtype=dtype
    )
    orig, emb, _, _, orig_knn, _, emb_knn, _ = resources
    expected = _reference_procrustes(orig, emb, orig_knn, emb_knn)

    actual = procrustes.measure(orig, emb, k, knn_info=(orig_knn, emb_knn))[
        "procrustes"
    ]

    tolerance = 1e-7 if dtype is np.float32 else 1e-12
    assert actual == pytest.approx(expected, abs=tolerance)


def test_direct_and_precomputed_metric_paths_agree():
    resources = _data_and_resources(
        seed=74, n=40, orig_dim=5, emb_dim=2, k=5, dtype=np.float64
    )
    (
        orig,
        emb,
        orig_dist,
        emb_dist,
        orig_knn,
        orig_ranking,
        emb_knn,
        emb_ranking,
    ) = resources
    labels = np.arange(len(orig)) % 4
    ranking_info = (orig_knn, orig_ranking, emb_knn, emb_ranking)
    knn_info = (orig_knn, emb_knn)

    assert tnc.measure(orig, emb, 5) == pytest.approx(
        tnc.measure(orig, emb, 5, ranking_info)
    )
    assert ca_tnc.measure(orig, emb, labels, 5) == pytest.approx(
        ca_tnc.measure(orig, emb, labels, 5, ranking_info)
    )
    assert mrre.measure(orig, emb, 5) == pytest.approx(
        mrre.measure(orig, emb, 5, ranking_info)
    )
    assert lcmc.measure(orig, emb, 5) == pytest.approx(
        lcmc.measure(orig, emb, 5, knn_info)
    )
    assert neighborhood_hit.measure(emb, labels, 5) == pytest.approx(
        neighborhood_hit.measure(emb, labels, 5, emb_knn)
    )
    assert topographic_product.measure(orig, emb, 5) == pytest.approx(
        topographic_product.measure(orig, emb, 5, (orig_dist, emb_dist), knn_info)
    )
    assert procrustes.measure(orig, emb, 5) == pytest.approx(
        procrustes.measure(orig, emb, 5, knn_info)
    )


def test_duplicate_points_preserve_neighborhood_kernel_results():
    orig = np.asarray([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0]])
    emb = np.asarray([[0.0], [0.0], [1.1], [2.2], [2.9], [4.2]])
    labels = np.asarray(["x", "x", "y", "y", "z", "z"])
    k = 2
    orig_dist = cdist(orig, orig)
    emb_dist = cdist(emb, emb)
    orig_knn, orig_ranking = knn_with_ranking(orig, k, orig_dist)
    emb_knn, emb_ranking = knn_with_ranking(emb, k, emb_dist)
    ranking_info = (orig_knn, orig_ranking, emb_knn, emb_ranking)

    expected, expected_local = _reference_tnc(orig_knn, orig_ranking, emb_knn, k)
    score, local = tnc.measure(
        orig, emb, k, knn_ranking_info=ranking_info, return_local=True
    )
    assert score["trustworthiness"] == pytest.approx(expected)
    np.testing.assert_array_equal(local["local_trustworthiness"], expected_local)

    expected_nh, expected_local_nh = _reference_neighborhood_hit(emb_knn, labels, k)
    score, local = neighborhood_hit.measure(
        emb, labels, k, knn_info=emb_knn, return_local=True
    )
    assert score["neighborhood_hit"] == pytest.approx(expected_nh)
    np.testing.assert_array_equal(local["local_neighborhood_hit"], expected_local_nh)


def test_neighborhood_metrics_match_reference_at_large_k_boundary():
    k = 65
    resources = _data_and_resources(
        seed=80, n=131, orig_dim=4, emb_dim=2, k=k, dtype=np.float64
    )
    (
        orig,
        emb,
        _,
        _,
        orig_knn,
        orig_ranking,
        emb_knn,
        _,
    ) = resources
    labels = np.arange(len(orig)) % 5

    expected_tnc, expected_local_tnc = _reference_tnc(
        orig_knn, orig_ranking, emb_knn, k
    )
    actual_tnc, actual_local_tnc = tnc.tnc_computation(
        orig_knn, orig_ranking, emb_knn, k, return_local=True
    )
    assert actual_tnc == pytest.approx(expected_tnc)
    np.testing.assert_array_equal(actual_local_tnc, expected_local_tnc)

    expected_ca, expected_local_ca = _reference_ca_tnc(
        orig_knn, orig_ranking, emb_knn, labels, k, "false"
    )
    actual_ca, actual_local_ca = ca_tnc.ca_tnc_computation(
        orig_knn,
        orig_ranking,
        emb_knn,
        labels,
        k,
        "false",
        return_local=True,
    )
    assert actual_ca == pytest.approx(expected_ca)
    np.testing.assert_array_equal(actual_local_ca, expected_local_ca)

    expected_lcmc, expected_local_lcmc = _reference_lcmc(
        orig_knn, emb_knn, len(orig), k
    )
    actual_lcmc, actual_local_lcmc = lcmc.measure(
        orig,
        emb,
        k,
        knn_info=(orig_knn, emb_knn),
        return_local=True,
    )
    assert actual_lcmc["lcmc"] == pytest.approx(expected_lcmc)
    np.testing.assert_array_equal(actual_local_lcmc["local_lcmc"], expected_local_lcmc)


def test_topographic_product_and_procrustes_keep_degenerate_errors():
    orig = np.asarray([[0.0], [0.0], [1.0]])
    emb = np.asarray([[0.0], [0.5], [1.0]])
    orig_dist = cdist(orig, orig)
    emb_dist = cdist(emb, emb)
    orig_knn = knn_from_distance_matrix(orig_dist, 1)
    emb_knn = knn_from_distance_matrix(emb_dist, 1)

    with pytest.raises(ValueError, match="zero-distance original-space"):
        topographic_product.measure(
            orig,
            emb,
            1,
            distance_matrices=(orig_dist, emb_dist),
            knn_info=(orig_knn, emb_knn),
        )

    constant_orig = np.zeros((4, 2))
    varied_emb = np.arange(8, dtype=float).reshape(4, 2)
    indices = np.tile(np.asarray([1, 2]), (4, 1))
    with pytest.raises(ValueError, match="zero variance"):
        procrustes.measure(
            constant_orig,
            varied_emb,
            2,
            knn_info=(indices, indices),
        )


def test_topographic_product_keeps_other_zero_distance_errors():
    orig = np.asarray([[0.0], [1.0], [10.0]])
    embedded_duplicates = np.asarray([[0.0], [0.0], [10.0]])
    orig_dist = cdist(orig, orig)
    emb_dist = cdist(embedded_duplicates, embedded_duplicates)
    orig_knn = knn_from_distance_matrix(orig_dist, 1)
    emb_knn = knn_from_distance_matrix(emb_dist, 1)

    with pytest.raises(ValueError, match="zero-distance embedded-space"):
        topographic_product.measure(
            orig,
            embedded_duplicates,
            1,
            distance_matrices=(orig_dist, emb_dist),
            knn_info=(orig_knn, emb_knn),
        )

    duplicate_orig = np.asarray([[0.0], [0.0], [1.0]])
    separated_emb = np.asarray([[0.0], [1.0], [2.0]])
    duplicate_dist = cdist(duplicate_orig, duplicate_orig)
    separated_dist = cdist(separated_emb, separated_emb)
    denominator_neighbors = np.asarray([[2], [2], [0]])
    numerator_neighbors = np.asarray([[1], [0], [1]])

    with pytest.raises(ValueError, match="coincident points"):
        topographic_product.measure(
            duplicate_orig,
            separated_emb,
            1,
            distance_matrices=(duplicate_dist, separated_dist),
            knn_info=(denominator_neighbors, numerator_neighbors),
        )


def test_topographic_product_accepts_larger_precomputed_neighbor_tables():
    resources = _data_and_resources(
        seed=84, n=20, orig_dim=4, emb_dim=2, k=6, dtype=np.float64
    )
    orig, emb, orig_dist, emb_dist, orig_knn, _, emb_knn, _ = resources
    expected = _reference_topographic_product(
        orig_dist, emb_dist, orig_knn[:, :3], emb_knn[:, :3], 3
    )

    actual = topographic_product.measure(
        orig,
        emb,
        3,
        distance_matrices=(orig_dist.tolist(), emb_dist.tolist()),
        knn_info=(orig_knn, emb_knn),
    )["topographic_product"]

    assert actual == pytest.approx(expected, abs=1e-12)


def test_vectorized_helpers_honor_small_row_blocks():
    candidates = np.asarray([[1, 2], [3, 4], [5, 6]])
    reference = np.asarray([[0, 2, 7], [3, 8, 9], [6, 5, 4]])

    blocks = list(iter_row_blocks(5, 4, max_block_bytes=8))
    assert [(block.start, block.stop) for block in blocks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    np.testing.assert_array_equal(
        rowwise_membership(candidates, reference, max_block_bytes=2),
        [[False, True], [True, False], [True, True]],
    )
    np.testing.assert_array_equal(
        rowwise_intersection_count(candidates, reference, max_block_bytes=2),
        [1, 1, 2],
    )

    ranking = np.arange(18).reshape(3, 6)
    indices = np.asarray([[5, 0], [2, 4], [1, 3]])
    expected = np.take_along_axis(ranking, indices, axis=1)
    np.testing.assert_array_equal(gather_ranks(ranking, indices), expected)
    np.testing.assert_array_equal(
        gather_ranks(ranking[:, ::-1], indices),
        np.take_along_axis(ranking[:, ::-1], indices, axis=1),
    )


def test_rowwise_membership_uses_memory_bounded_sorted_path_for_large_k():
    reference = np.arange(210).reshape(3, 70)
    candidates = reference[:, 5:][:, ::-1].copy()
    candidates[:, 0] = 10_000 + np.arange(3)
    expected = np.ones(candidates.shape, dtype=bool)
    expected[:, 0] = False

    np.testing.assert_array_equal(
        rowwise_membership(candidates, reference, max_block_bytes=3_000),
        expected,
    )

    max_int = np.iinfo(np.int64).max
    large_reference = np.zeros((2, 65), dtype=np.int64)
    large_candidates = large_reference.copy()
    large_reference[0, 0] = max_int
    large_candidates[0, 0] = max_int
    np.testing.assert_array_equal(
        rowwise_membership(large_candidates, large_reference),
        np.ones_like(large_candidates, dtype=bool),
    )


def test_rowwise_membership_handles_empty_columns():
    assert rowwise_membership(np.empty((3, 0), dtype=int), np.ones((3, 1))).shape == (
        3,
        0,
    )
    np.testing.assert_array_equal(
        rowwise_membership(np.ones((3, 2), dtype=int), np.empty((3, 0))),
        np.zeros((3, 2), dtype=bool),
    )


@pytest.mark.parametrize(
    "arguments,message",
    [
        ((-1, 1), "n_rows"),
        ((1, 0), "bytes_per_row"),
    ],
)
def test_iter_row_blocks_rejects_invalid_sizes(arguments, message):
    with pytest.raises(ValueError, match=message):
        list(iter_row_blocks(*arguments))

    with pytest.raises(ValueError, match="max_block_bytes"):
        list(iter_row_blocks(1, 1, max_block_bytes=0))


def test_vectorized_helpers_reject_incompatible_rows():
    with pytest.raises(ValueError, match="2D"):
        rowwise_membership(np.arange(3), np.arange(3))
    with pytest.raises(ValueError, match="same number of rows"):
        rowwise_membership(np.zeros((2, 1)), np.zeros((3, 1)))


def test_procrustes_processes_more_than_one_internal_batch():
    resources = _data_and_resources(
        seed=85, n=257, orig_dim=3, emb_dim=2, k=2, dtype=np.float64
    )
    orig, emb, _, _, orig_knn, _, emb_knn, _ = resources

    expected = _reference_procrustes(orig, emb, orig_knn, emb_knn)
    actual = procrustes.measure(orig, emb, 2, knn_info=(orig_knn, emb_knn))[
        "procrustes"
    ]

    assert actual == pytest.approx(expected, abs=1e-12)


def test_class_aware_kernel_rejects_unknown_distortion_type():
    ranking = np.tile(np.arange(4), (4, 1))
    neighbors = np.tile(np.asarray([1]), (4, 1))
    with pytest.raises(ValueError, match=r"false.*missing"):
        ca_tnc.ca_tnc_computation(
            neighbors,
            ranking,
            neighbors,
            np.asarray([0, 0, 1, 1]),
            1,
            "unknown",
        )
