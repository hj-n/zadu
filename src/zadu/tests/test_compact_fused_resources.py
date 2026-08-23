import numpy as np
import pytest

from zadu import ZADU, ExecutionConfig
from zadu.backends import numpy_backend
from zadu.engine.resources import ResourceKind, compact_index_dtype
from zadu.measures import (
    class_aware_trustworthiness_continuity,
    distance_to_measure,
    kl_divergence,
    local_continuity_meta_criteria,
    mean_relative_rank_error,
    neighbor_dissimilarity,
    neighborhood_hit,
    trustworthiness_continuity,
)
from zadu.measures.utils import knn, pairwise_dist


def _sample(seed=0, n=60):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 6))
    emb = rng.normal(size=(n, 2))
    labels = np.arange(n) % 3
    return orig, emb, labels


def _assert_result_close(actual, expected):
    if isinstance(actual, tuple):
        actual_score, actual_local = actual
        expected_score, expected_local = expected
        assert actual_score == pytest.approx(expected_score, abs=1e-14)
        if actual_local is None or expected_local is None:
            assert actual_local is expected_local is None
            return
        for name in actual_local:
            np.testing.assert_allclose(
                actual_local[name], expected_local[name], rtol=0, atol=1e-14
            )
        return
    assert actual == pytest.approx(expected, abs=1e-14)


def test_compact_index_dtype_uses_int32_with_safe_int64_fallback():
    int32_max = np.iinfo(np.int32).max

    assert compact_index_dtype(0) == np.dtype(np.int32)
    assert compact_index_dtype(int32_max) == np.dtype(np.int32)
    assert compact_index_dtype(int32_max + 1) == np.dtype(np.int32)
    assert compact_index_dtype(int32_max + 2) == np.dtype(np.int64)
    with pytest.raises(ValueError, match="zero or greater"):
        compact_index_dtype(-1)


def test_density_resources_fuse_sigma_and_avoid_distance_matrices(monkeypatch):
    orig, emb, _ = _sample()
    distance_calls = 0
    real_cdist = numpy_backend.cdist

    def counted_cdist(*args, **kwargs):
        nonlocal distance_calls
        distance_calls += 1
        return real_cdist(*args, **kwargs)

    monkeypatch.setattr(numpy_backend, "cdist", counted_cdist)
    specs = [
        {"id": "dtm", "params": {"sigma": 0.15}},
        {"id": "kl_div", "params": {"sigma": 0.15}},
        {"id": "dtm", "params": {"sigma": 0.3}},
    ]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    expected = [
        distance_to_measure.measure(orig, emb, sigma=0.15),
        kl_divergence.measure(orig, emb, sigma=0.15),
        distance_to_measure.measure(orig, emb, sigma=0.3),
    ]
    for actual, direct in zip(scores, expected, strict=True):
        _assert_result_close(actual, direct)

    resources = runner.last_run_info["resources"]
    densities = [resource for resource in resources if resource["kind"] == "density"]
    assert len(densities) == 4
    assert {resource["parameter"] for resource in densities} == {0.15, 0.3}
    assert all(
        resource["details"]["algorithm"] == "blockwise_two_pass_gaussian_density"
        for resource in densities
    )
    assert all(
        resource["details"]["fused_sigmas"] == [0.15, 0.3] for resource in densities
    )
    assert not any(resource["kind"] == "distance_matrix" for resource in resources)
    assert distance_calls == 4
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    assert runner.estimated_cache_bytes == 4 * len(orig) * 8
    assert runner._execution_plan.planned_peak_bytes == (
        runner.estimated_cache_bytes + 2 * len(orig) * len(orig) * 8
    )


def test_density_reuses_a_dense_matrix_required_by_other_metrics():
    orig, emb, _ = _sample(seed=2)
    runner = ZADU([{"id": "dtm"}, {"id": "stress"}], orig)

    actual = runner.measure(emb)

    _assert_result_close(actual[0], distance_to_measure.measure(orig, emb))
    assert np.isfinite(actual[1]["stress"])
    densities = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "density"
    ]
    assert densities
    assert all(
        resource["details"]["source"] == "distance_matrix" for resource in densities
    )
    assert any(
        resource["kind"] == "distance_matrix"
        for resource in runner.last_run_info["resources"]
    )


def test_rank_comparisons_fuse_mixed_k_metrics_exactly():
    orig, emb, labels = _sample(seed=3)
    specs = [
        {"id": "tnc", "params": {"k": 3}},
        {"id": "tnc", "params": {"k": 7}},
        {"id": "mrre", "params": {"k": 5}},
        {"id": "ca_tnc", "params": {"k": 3}},
    ]
    runner = ZADU(specs, orig, return_local=True)

    scores, local = runner.measure(emb, labels)
    planned = list(zip(scores, local, strict=True))
    expected = [
        trustworthiness_continuity.measure(orig, emb, k=3, return_local=True),
        trustworthiness_continuity.measure(orig, emb, k=7, return_local=True),
        mean_relative_rank_error.measure(orig, emb, k=5, return_local=True),
        class_aware_trustworthiness_continuity.measure(
            orig, emb, labels, k=3, return_local=True
        ),
    ]
    for actual, direct in zip(planned, expected, strict=True):
        _assert_result_close(actual, direct)

    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["details"]["requested_ks"] == [3, 5, 7]
    assert comparison["details"]["membership_ks"] == [3, 7]
    assert comparison["details"]["fused_metrics"] == [
        "trustworthiness_continuity",
        "trustworthiness_continuity",
        "mean_relative_rank_error",
        "class_aware_trustworthiness_continuity",
    ]
    assert comparison["dtype"] == {
        "indices": "int32",
        "ranks": "int32",
        "membership": "bool",
    }
    assert comparison["details"]["algorithm"] == "blockwise_selected_ranks"
    assert comparison["details"]["block_count"] == 1
    assert comparison["details"]["block_rows"] == len(orig)
    assert comparison["bytes"] == 3 * len(orig) * 7 * 4 + 2 * len(orig) * (3 + 7)
    original_knn = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "stable_knn" and resource["space"] == "orig"
    )
    assert original_knn["bytes"] + comparison["bytes"] == (
        4 * len(orig) * 7 * 4 + 2 * len(orig) * (3 + 7)
    )
    assert comparison["released"] is True
    assert all(
        resource["kind"] != "neighbor_ranking"
        for resource in runner.last_run_info["resources"]
    )
    assert runner.last_run_info["rank_comparison_strategy"] == (
        "blockwise_selected_ranks"
    )


def test_selected_rank_resource_respects_blockwise_memory_plan():
    orig, emb, _ = _sample(seed=5, n=100)
    k = 10
    specs = [{"id": "tnc", "params": {"k": k}}]
    baseline = ZADU(specs, orig)
    bytes_per_row = len(orig) * 24
    budget = baseline.estimated_cache_bytes + 2 * bytes_per_row
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    actual = runner.measure(emb)[0]
    expected = trustworthiness_continuity.measure(orig, emb, k=k)
    assert actual == pytest.approx(expected, abs=1e-14)
    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["details"]["block_rows"] == 2
    assert comparison["details"]["block_count"] == 50
    assert comparison["details"]["working_bytes"] == 2 * bytes_per_row
    assert comparison["details"]["work_budget_bytes"] == 2 * bytes_per_row
    assert runner._execution_plan.planned_peak_bytes == budget

    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            specs,
            orig,
            execution=ExecutionConfig(
                memory_budget=baseline.estimated_cache_bytes + bytes_per_row - 1
            ),
        )


def test_selected_rank_resource_uses_geodesic_original_space():
    orig = np.asarray(
        [
            [-2.4, 0.4],
            [-1.7, -0.2],
            [-0.8, 0.7],
            [-0.1, -0.6],
            [0.5, 0.3],
            [1.1, -0.4],
            [1.8, 0.6],
            [2.5, -0.1],
        ]
    )
    emb = np.column_stack((np.sin(orig[:, 0]), orig[:, 1]))
    k = 2
    runner = ZADU([{"id": "tnc", "params": {"k": k}}], orig, geodesic=True)

    actual = runner.measure(emb)[0]
    orig_distances = runner._provider.pairwise_geodesic_distance_matrix(orig)
    emb_distances = pairwise_dist.pairwise_distance_matrix(emb)
    orig_indices, orig_ranking = knn.knn_with_ranking(
        orig,
        k,
        orig_distances,
    )
    emb_indices, emb_ranking = knn.knn_with_ranking(
        emb,
        k,
        emb_distances,
    )
    expected = trustworthiness_continuity.measure(
        orig,
        emb,
        k=k,
        knn_ranking_info=(
            orig_indices,
            orig_ranking,
            emb_indices,
            emb_ranking,
        ),
    )

    assert actual == pytest.approx(expected, abs=1e-14)
    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["details"]["original_distance_source"] == ("blockwise_geodesic")
    assert comparison["details"]["embedded_distance_source"] == (
        "blockwise_scipy_cdist"
    )


def test_neighbor_statistics_share_max_knn_and_preserve_exact_results():
    orig, emb, labels = _sample(seed=7)
    specs = [
        {"id": "lcmc", "params": {"k": 3}},
        {"id": "lcmc", "params": {"k": 9}},
        {"id": "nd", "params": {"k": 6}},
        {"id": "nh", "params": {"k": 5}},
    ]
    runner = ZADU(specs, orig, return_local=True)

    scores, local = runner.measure(emb, labels)
    planned = list(zip(scores, local, strict=True))
    expected = [
        local_continuity_meta_criteria.measure(orig, emb, k=3, return_local=True),
        local_continuity_meta_criteria.measure(orig, emb, k=9, return_local=True),
        (neighbor_dissimilarity.measure(orig, emb, k=6), None),
        (neighborhood_hit.measure(emb, labels, k=5, return_local=True)),
    ]
    for actual, direct in zip(planned, expected, strict=True):
        _assert_result_close(actual, direct)

    knn_resources = [
        key for key in runner._execution_plan.resources if key.kind is ResourceKind.KNN
    ]
    assert len(knn_resources) == 2
    assert all(key.k == 9 for key in knn_resources)
    statistics = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "neighbor_statistics"
    )
    assert statistics["details"]["lcmc_ks"] == [3, 9]
    assert statistics["details"]["neighbor_dissimilarity_ks"] == [6]
    assert statistics["released"] is True
    assert runner.last_run_info["neighbor_statistics_strategy"] == (
        "fused_neighbor_statistics"
    )


def test_density_peak_memory_guard_runs_before_distance_allocation(monkeypatch):
    orig, _, _ = _sample(n=100)
    specs = [{"id": "dtm"}, {"id": "kl_div"}]
    unbounded = ZADU(specs, orig)
    budget = unbounded.estimated_cache_bytes + len(orig) * 16 - 1

    def unexpected_distance(*args, **kwargs):
        raise AssertionError("distance allocation should not start")

    monkeypatch.setattr(numpy_backend, "cdist", unexpected_distance)
    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            specs,
            orig,
            execution=ExecutionConfig(memory_budget=budget),
        )


def test_density_blocks_respect_budget_and_preserve_geodesic_results():
    rng = np.random.default_rng(13)
    orig = rng.uniform(low=[-np.pi, -np.pi / 2], high=[np.pi, np.pi / 2], size=(40, 2))
    emb = rng.normal(size=(40, 2))
    specs = [{"id": "dtm"}, {"id": "kl_div"}]
    baseline = ZADU(specs, orig, geodesic=True)
    budget = baseline.estimated_cache_bytes + 5 * len(orig) * 16
    runner = ZADU(
        specs,
        orig,
        geodesic=True,
        execution=ExecutionConfig(memory_budget=budget),
    )

    actual = runner.measure(emb)
    distance_matrices = (
        runner._provider.pairwise_geodesic_distance_matrix(orig),
        pairwise_dist.pairwise_distance_matrix(emb),
    )
    expected = [
        distance_to_measure.measure(orig, emb, distance_matrices=distance_matrices),
        kl_divergence.measure(orig, emb, distance_matrices=distance_matrices),
    ]

    for actual_score, expected_score in zip(actual, expected, strict=True):
        _assert_result_close(actual_score, expected_score)
    density_resources = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "density"
    ]
    assert density_resources
    assert all(resource["details"]["block_rows"] == 5 for resource in density_resources)
    assert all(
        resource["details"]["block_count"] == 8 for resource in density_resources
    )
    assert runner._execution_plan.planned_peak_bytes == budget


def test_neighbor_dissimilarity_respects_blockwise_memory_plan():
    orig, emb, _ = _sample(seed=11, n=100)
    k = 10
    unbounded = ZADU([{"id": "nd", "params": {"k": k}}], orig)
    graph_bytes = 4 * len(orig) * k * 16
    product_row_bytes = len(orig) * 48
    budget = unbounded.estimated_cache_bytes + graph_bytes + 2 * product_row_bytes
    runner = ZADU(
        [{"id": "nd", "params": {"k": k}}],
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    score = runner.measure(emb)[0]

    expected = neighbor_dissimilarity.measure(orig, emb, k=k)
    assert score == pytest.approx(expected, abs=1e-14)
    details = runner.last_run_info["resources"][-1]["details"]
    assert details["block_rows"] == 2
    assert details["block_count"] > 1
    assert runner._execution_plan.planned_peak_bytes <= budget
