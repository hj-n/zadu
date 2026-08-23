import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from zadu import ZADU, ExecutionConfig
from zadu.backends import NumpyResourceProvider, numpy_backend
from zadu.engine.resources import (
    PairStrategy,
    ResourceKind,
    Space,
    compact_index_dtype,
)
from zadu.measures import non_metric_stress, spearman_rho

ORDERED_SPECS = [{"id": "srho"}, {"id": "nm_stress"}]


def _sample(seed=0, n=80):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 6)), rng.normal(size=(n, 2))


def _dense_reference(orig, emb):
    matrices = cdist(orig, orig), cdist(emb, emb)
    return [
        spearman_rho.measure(orig, emb, matrices),
        non_metric_stress.measure(orig, emb, matrices),
    ]


def _assert_scores_close(actual, expected):
    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score == pytest.approx(expected_score, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ordered_metrics_share_exact_condensed_order_resource(dtype):
    orig, emb = _sample()
    orig = orig.astype(dtype)
    emb = emb.astype(dtype)
    runner = ZADU(ORDERED_SPECS, orig)

    scores = runner.measure(emb)

    pair_plan = runner._execution_plan.pair_plan
    assert pair_plan.strategy is PairStrategy.CONDENSED
    assert pair_plan.metric_ids == ()
    assert pair_plan.ordered_metric_ids == (
        "spearman_rho",
        "non_metric_stress",
    )
    assert [key.kind for key in runner._execution_plan.resources] == [
        ResourceKind.CONDENSED_PAIRS,
        ResourceKind.PAIR_ORDER,
        ResourceKind.ORDERED_PAIR_STATISTICS,
    ]
    assert runner._execution_plan.resources[0].space is Space.EMBEDDED
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    _assert_scores_close(scores, _dense_reference(orig, emb))

    pair_count = len(orig) * (len(orig) - 1) // 2
    index_dtype = compact_index_dtype(len(orig))
    expected_cache = pair_count * (8 + index_dtype.itemsize + 8)
    assert runner.estimated_cache_bytes == expected_cache
    order_resource = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "pair_order"
    )
    assert order_resource["reused"] is True
    assert order_resource["released"] is False
    assert order_resource["dtype"] == {
        "indices": index_dtype.name,
        "sorted_ranks": "float64",
    }
    ordered_resource = runner.last_run_info["resources"][-1]
    assert ordered_resource["consumers"] == [
        "spearman_rho",
        "non_metric_stress",
    ]
    assert ordered_resource["details"]["fused_metrics"] == [
        "spearman_rho",
        "non_metric_stress",
    ]
    assert ordered_resource["released"] is True
    assert runner.last_run_info["pair_strategy"] == "condensed"
    json.dumps(runner.last_run_info)


def test_ordered_pair_statistics_preserve_duplicate_distance_ties():
    orig = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    emb = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )

    scores = ZADU(ORDERED_SPECS, orig).measure(emb)

    _assert_scores_close(scores, _dense_reference(orig, emb))


def test_original_pair_order_is_reused_across_embeddings(monkeypatch):
    orig, emb = _sample()
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append((key.space, key.kind))
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    runner = ZADU(ORDERED_SPECS, orig)
    first = runner.measure(emb)
    second = runner.measure(emb + 0.01)

    assert len(first) == len(second) == 2
    assert calls.count((Space.ORIGINAL, ResourceKind.PAIR_ORDER)) == 1
    assert calls.count((Space.EMBEDDED, ResourceKind.CONDENSED_PAIRS)) == 2
    order_resource = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "pair_order"
    )
    assert order_resource["reused"] is True
    assert order_resource["built_in_run"] is False


def test_ordered_metrics_keep_condensed_sources_with_neighbor_resources():
    orig, emb = _sample(n=60)
    specs = [
        {"id": "lcmc", "params": {"k": 7}},
        *ORDERED_SPECS,
    ]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    assert runner._execution_plan.pair_plan.strategy is PairStrategy.CONDENSED
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    assert all(
        key.kind is not ResourceKind.DISTANCE_MATRIX
        for key in runner._execution_plan.resources
    )
    assert (
        sum(
            key.kind is ResourceKind.CONDENSED_PAIRS
            for key in runner._execution_plan.resources
        )
        == 1
    )
    _assert_scores_close(scores[1:], _dense_reference(orig, emb))


def test_ordered_metrics_preserve_geodesic_dense_results():
    orig = np.asarray([[0.0, 0.0], [0.2, 0.1], [-0.4, 0.3], [0.5, -0.2], [0.7, 0.4]])
    emb = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.3, -0.2], [0.5, 0.4], [-0.1, 0.7]])
    runner = ZADU(ORDERED_SPECS, orig, geodesic=True)

    scores = runner.measure(emb)

    matrices = (
        runner._provider.pairwise_geodesic_distance_matrix(orig),
        cdist(emb, emb),
    )
    expected = [
        spearman_rho.measure(orig, emb, matrices),
        non_metric_stress.measure(orig, emb, matrices),
    ]
    assert runner._execution_plan.pair_plan.strategy is PairStrategy.DENSE
    _assert_scores_close(scores, expected)


def test_ordered_and_reduction_pair_metrics_share_condensed_sources():
    orig, emb = _sample(n=50)
    specs = [{"id": "stress"}, *ORDERED_SPECS, {"id": "pr"}]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    pair_plan = runner._execution_plan.pair_plan
    assert pair_plan.strategy is PairStrategy.CONDENSED
    assert pair_plan.metric_ids == ("stress", "pearson_r")
    assert pair_plan.ordered_metric_ids == (
        "spearman_rho",
        "non_metric_stress",
    )
    assert (
        sum(
            key.kind is ResourceKind.CONDENSED_PAIRS
            for key in runner._execution_plan.resources
        )
        == 2
    )
    assert all(np.isfinite(value) for score in scores for value in score.values())


def test_ordered_pair_memory_guard_fails_before_distance_allocation(monkeypatch):
    orig, _ = _sample(n=100)
    unbounded = ZADU(ORDERED_SPECS, orig)
    budget = unbounded._execution_plan.planned_peak_bytes - 1

    def unexpected_pdist(*args, **kwargs):
        raise AssertionError("distance allocation should not start")

    monkeypatch.setattr(numpy_backend, "scipy_pdist", unexpected_pdist)
    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            ORDERED_SPECS,
            orig,
            execution=ExecutionConfig(memory_budget=budget),
        )


@pytest.mark.parametrize(
    "specs,orig,emb,match",
    [
        (
            [{"id": "srho"}],
            np.zeros((8, 3)),
            np.arange(16, dtype=float).reshape(8, 2),
            "Spearman correlation is undefined",
        ),
        (
            [{"id": "srho"}],
            np.arange(16, dtype=float).reshape(8, 2),
            np.zeros((8, 2)),
            "Spearman correlation is undefined",
        ),
        (
            [{"id": "nm_stress"}],
            np.zeros((8, 3)),
            np.arange(16, dtype=float).reshape(8, 2),
            "Non-metric stress is undefined",
        ),
        (
            [{"id": "nm_stress"}],
            np.arange(16, dtype=float).reshape(8, 2),
            np.zeros((8, 2)),
            "Non-metric stress is undefined",
        ),
    ],
)
def test_ordered_pair_statistics_preserve_degenerate_errors(
    specs,
    orig,
    emb,
    match,
):
    with pytest.raises(ValueError, match=match):
        ZADU(specs, orig).measure(emb)
