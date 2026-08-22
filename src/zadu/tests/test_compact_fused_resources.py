import numpy as np
import pytest

from zadu import ZADU, ExecutionConfig
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
from zadu.measures.utils import pairwise_dist


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


def test_density_resources_deduplicate_sigma_and_release_distance_matrices():
    orig, emb, _ = _sample()
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
    distances = [
        resource for resource in resources if resource["kind"] == "distance_matrix"
    ]
    assert len(densities) == 4
    assert {resource["parameter"] for resource in densities} == {0.15, 0.3}
    assert all(resource["released"] for resource in distances)
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    assert runner.estimated_cache_bytes == 4 * len(orig) * 8
    assert runner._execution_plan.planned_peak_bytes == (
        runner.estimated_cache_bytes + 2 * len(orig) * len(orig) * 8
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
    assert comparison["dtype"] == {"ranks": "int32", "membership": "bool"}
    assert comparison["released"] is True
    assert runner.last_run_info["rank_comparison_strategy"] == (
        "fused_gathered_ranks_and_membership"
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
    budget = unbounded._execution_plan.planned_peak_bytes - 1

    def unexpected_distance(*args, **kwargs):
        raise AssertionError("distance allocation should not start")

    monkeypatch.setattr(pairwise_dist, "pairwise_distance_matrix", unexpected_distance)
    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            specs,
            orig,
            execution=ExecutionConfig(memory_budget=budget),
        )


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
