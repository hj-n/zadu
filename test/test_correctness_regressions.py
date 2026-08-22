import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from zadu import ZADU
from zadu.measures import (
    class_angular_distortion_index as cadi,
)
from zadu.measures import (
    class_aware_trustworthiness_continuity as ca_tnc,
)
from zadu.measures import (
    clustering_and_external_validation_measure as c_evm,
)
from zadu.measures import (
    distance_consistency,
    distance_to_measure,
    kl_divergence,
    non_metric_stress,
    pearson_r,
    scale_normalized_stress,
    spearman_rho,
    stress,
    topographic_product,
    trustworthiness_continuity,
)
from zadu.measures import (
    label_trustworthiness_and_continuity as label_tnc,
)
from zadu.measures import (
    local_continuity_meta_criteria as lcmc,
)
from zadu.measures import (
    steadiness_cohesiveness as snc,
)
from zadu.measures.utils.knn import knn, knn_with_ranking
from zadu.measures.utils.snc_cpu import SNCCPU
from zadu.measures.utils.validation import (
    as_finite_2d,
    validate_labels,
    validate_neighbor_k,
    validate_pair,
    validate_positive_real,
)


def test_mixed_k_scheduler_matches_direct_metric_calls():
    rng = np.random.default_rng(42)
    orig = rng.normal(size=(60, 4))
    emb = rng.normal(size=(60, 2))
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 10}},
    ]

    scheduled = ZADU(specs, orig).measure(emb)
    expected = lcmc.measure(orig, emb, k=10)

    assert scheduled[1]["lcmc"] == pytest.approx(expected["lcmc"])


def test_cadi_terminates_when_only_one_cluster_can_supply_y_and_z():
    orig = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    result = cadi.measure(
        orig,
        orig.copy(),
        np.array([0, 0, 1]),
        n_triplets=5,
        random_seed=1,
    )

    assert result["class_angular_distortion_index"] == pytest.approx(0.0)


def test_cadi_rejects_negative_triplet_count():
    orig = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="zero or greater"):
        cadi.measure(orig, orig.copy(), np.array([0, 0, 0, 1, 1, 1]), n_triplets=-1)


def test_topographic_product_uses_first_neighbor_at_k_one():
    rng = np.random.default_rng(4)
    orig = rng.normal(size=(30, 3))
    emb = rng.normal(size=(30, 2))

    score = topographic_product.measure(orig, emb, k=1)["topographic_product"]

    assert np.isfinite(score)
    assert abs(score) > 1e-8


def test_knn_never_returns_the_query_point_for_duplicate_rows():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    indices = knn(points, 2)
    ranked_indices, _ = knn_with_ranking(points, 2)

    for row_idx in range(len(points)):
        assert row_idx not in indices[row_idx]
        assert row_idx not in ranked_indices[row_idx]


def test_distance_consistency_is_invariant_to_label_encoding():
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])

    integer = distance_consistency.measure(emb, np.array([0, 0, 1, 1]))
    arbitrary = distance_consistency.measure(emb, np.array([10, 10, 20, 20]))
    strings = distance_consistency.measure(emb, np.array(["a", "a", "b", "b"]))

    assert integer == arbitrary == strings == {"distance_consistency": 1.0}


def test_trustworthiness_rejects_k_outside_normalized_domain():
    rng = np.random.default_rng(2)
    orig = rng.normal(size=(5, 3))
    emb = rng.normal(size=(5, 2))

    with pytest.raises(ValueError, match="smaller than n / 2"):
        trustworthiness_continuity.measure(orig, emb, k=3)
    with pytest.raises(ValueError, match="smaller than n / 2"):
        ZADU([{"id": "tnc", "params": {"k": 3}}], orig)


def test_snc_honors_k_and_does_not_schedule_unused_default_knn():
    rng = np.random.default_rng(8)
    orig = rng.normal(size=(30, 5))
    emb = rng.normal(size=(30, 2))
    params = {
        "iteration": 5,
        "walk_num_ratio": 0.2,
        "clustering_strategy": "kmeans",
        "random_state": 123,
    }

    small_k = snc.measure(orig, emb, k=2, **params)
    large_k = snc.measure(orig, emb, k=8, **params)
    assert small_k != large_k

    small_orig = orig[:10]
    small_emb = emb[:10]
    wrapped = ZADU([{"id": "snc", "params": params}], small_orig).measure(small_emb)
    assert np.isfinite(wrapped[0]["steadiness"])
    assert np.isfinite(wrapped[0]["cohesiveness"])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"iteration": True}, TypeError),
        ({"iteration": 0}, ValueError),
        ({"walk_num_ratio": 0}, ValueError),
        ({"alpha": 0}, ValueError),
        ({"cluster_strategy": 1}, TypeError),
        ({"cluster_strategy": "unknown"}, ValueError),
        ({"cluster_strategy": "0-means"}, ValueError),
        ({"k": 1.5}, TypeError),
        ({"k": 0}, ValueError),
    ],
)
def test_snc_rejects_invalid_configuration(kwargs, error):
    raw = np.arange(30, dtype=float).reshape(10, 3)
    emb = raw[:, :2]

    with pytest.raises(error):
        SNCCPU(raw, emb, **kwargs)


def test_snc_lifecycle_and_precomputed_knn_validation():
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(12, 4))
    emb = rng.normal(size=(12, 2))
    obj = SNCCPU(raw, emb, iteration=1, k=3, random_state=rng)

    with pytest.raises(RuntimeError, match="fit"):
        obj.steadiness()
    with pytest.raises(RuntimeError, match="fit"):
        obj.cohesiveness()
    with pytest.raises(RuntimeError, match="record_vis_info"):
        obj.local_scores()
    with pytest.raises(TypeError, match="tuple"):
        obj.fit(knn_info=[knn(raw, 3), knn(emb, 3)])
    with pytest.raises(ValueError, match="raw knn_info"):
        obj.fit(knn_info=(np.zeros((12, 2), dtype=int), knn(emb, 3)))
    with pytest.raises(ValueError, match="emb knn_info"):
        obj.fit(knn_info=(knn(raw, 3), np.zeros((11, 3), dtype=int)))

    obj.fit(knn_info=(knn(raw, 4), knn(emb, 4)))
    assert np.all(np.diag(obj.raw_snn) == 0)
    assert np.all(np.diag(obj.emb_snn) == 0)


@pytest.mark.parametrize(
    "measure_func,orig,emb",
    [
        (distance_to_measure.measure, np.zeros((8, 3)), np.ones((8, 2))),
        (stress.measure, np.zeros((8, 3)), np.ones((8, 2))),
        (
            scale_normalized_stress.measure,
            np.arange(24, dtype=float).reshape(8, 3),
            np.zeros((8, 2)),
        ),
    ],
)
def test_undefined_degenerate_metrics_raise_value_error(measure_func, orig, emb):
    with pytest.raises(ValueError, match="undefined"):
        measure_func(orig, emb)


def test_label_tnc_rejects_a_single_class():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="distinct class"):
        label_tnc.measure(
            rng.normal(size=(8, 3)),
            rng.normal(size=(8, 2)),
            np.zeros(8, dtype=int),
        )


def test_lcmc_documented_bounds_match_implementation():
    rng = np.random.default_rng(10)
    orig = rng.normal(size=(40, 3))
    k = 5

    score = lcmc.measure(orig, orig.copy(), k=k)["lcmc"]

    assert score == pytest.approx(1 - k / (len(orig) - 1))


def test_external_kmeans_infers_number_of_classes():
    emb = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    result = c_evm.measure(emb, np.array([0, 0, 0, 1, 1, 1]))

    assert result["kmeans_arand"] == pytest.approx(1.0)


def test_distance_correlations_use_unique_off_diagonal_pairs():
    rng = np.random.default_rng(91)
    orig = rng.normal(size=(8, 3))
    emb = rng.normal(size=(8, 2))
    orig_dist = cdist(orig, orig)
    emb_dist = cdist(emb, emb)
    upper = np.triu_indices(len(orig), 1)

    assert pearson_r.measure(orig, emb)["pearson_r"] == pytest.approx(
        pearsonr(orig_dist[upper], emb_dist[upper]).statistic
    )
    assert spearman_rho.measure(orig, emb)["spearman_rho"] == pytest.approx(
        spearmanr(orig_dist[upper], emb_dist[upper]).statistic
    )


def test_geodesic_distance_clamps_roundoff_at_identical_points():
    point = [-2.225888843121074, -1.0795817305425273]
    orig = np.array([point, point, [0.0, 0.0]])

    runner = ZADU([{"id": "stress"}], orig, geodesic=True)

    assert runner.orig_distance_matrix[0, 1] == 0.0


def test_callable_knn_metric_is_supported():
    points = np.array([[0.0], [1.0], [3.0]])
    result = knn(
        points, 1, distance_function=lambda left, right: abs(left[0] - right[0])
    )

    assert result.shape == (3, 1)


def test_remaining_metric_families_return_finite_scores():
    rng = np.random.default_rng(19)
    orig = rng.normal(size=(30, 5))
    emb = rng.normal(size=(30, 2))
    label = np.repeat(["left", "right"], 15)

    class_aware = ca_tnc.measure(orig, emb, label, k=5)
    kl_score = kl_divergence.measure(orig, emb)
    non_metric = non_metric_stress.measure(orig, emb)

    assert all(np.isfinite(value) for value in class_aware.values())
    assert np.isfinite(kl_score["kl_divergence"])
    assert np.isfinite(non_metric["non_metric_stress"])


@pytest.mark.parametrize(
    "values,error",
    [
        ([1.0, 2.0], ValueError),
        (np.ones((1, 2)), ValueError),
        (np.empty((3, 0)), ValueError),
        (np.array([["not-numeric"], ["values"]]), TypeError),
        (np.array([[0.0], [np.nan]]), ValueError),
    ],
)
def test_shared_array_validation_rejects_invalid_inputs(values, error):
    with pytest.raises(error):
        as_finite_2d(values, "values")


def test_shared_label_and_scalar_validation_reject_invalid_inputs():
    with pytest.raises(ValueError, match="1D"):
        validate_labels(np.ones((3, 1)), 3)
    with pytest.raises(ValueError, match="one value per sample"):
        validate_labels(np.ones(2), 3)
    with pytest.raises(ValueError, match="distinct class"):
        validate_labels(np.zeros(3), 3, min_classes=2)
    with pytest.raises(TypeError, match="real number"):
        validate_positive_real("0.1", "sigma")
    with pytest.raises(ValueError, match="greater than zero"):
        validate_positive_real(0, "sigma")
    with pytest.raises(ValueError, match="same number of rows"):
        validate_pair(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(TypeError, match="must be int"):
        validate_neighbor_k(5, 1.5)
