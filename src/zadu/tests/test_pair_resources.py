import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from zadu import ZADU, ExecutionConfig
from zadu.backends import NumpyResourceProvider, numpy_backend
from zadu.engine.resources import PairStrategy, ResourceKind, Space
from zadu.kernels import PairAccumulator
from zadu.measures import pearson_r, scale_normalized_stress, stress

PAIR_SPECS = [{"id": "stress"}, {"id": "sn_stress"}, {"id": "pr"}]


def _sample(seed=0, n=80):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 6)), rng.normal(size=(n, 2))


def _dense_reference(orig, emb):
    matrices = cdist(orig, orig), cdist(emb, emb)
    return [
        stress.measure(orig, emb, matrices),
        scale_normalized_stress.measure(orig, emb, matrices),
        pearson_r.measure(orig, emb, matrices),
    ]


def _assert_scores_close(actual, expected):
    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score == pytest.approx(expected_score, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pair_only_metrics_use_one_condensed_statistics_resource(dtype):
    orig, emb = _sample()
    orig = orig.astype(dtype)
    emb = emb.astype(dtype)
    runner = ZADU(PAIR_SPECS, orig)

    scores = runner.measure(emb)

    assert runner._execution_plan.pair_plan.strategy is PairStrategy.CONDENSED
    assert [key.kind for key in runner._execution_plan.resources] == [
        ResourceKind.CONDENSED_PAIRS,
        ResourceKind.CONDENSED_PAIRS,
        ResourceKind.PAIR_STATISTICS,
    ]
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    pair_count = len(orig) * (len(orig) - 1) // 2
    assert runner.estimated_cache_bytes == 2 * pair_count * 8
    _assert_scores_close(scores, _dense_reference(orig, emb))

    pair_resource = runner.last_run_info["resources"][-1]
    assert pair_resource["consumers"] == [
        "stress",
        "scale_normalized_stress",
        "pearson_r",
    ]
    assert pair_resource["details"]["strategy"] == "condensed"
    assert pair_resource["details"]["pair_count"] == pair_count
    assert pair_resource["details"]["fused_metrics"] == [
        "stress",
        "scale_normalized_stress",
        "pearson_r",
    ]
    assert pair_resource["dtype"] == "float64"
    assert pair_resource["released"] is True
    assert runner.last_run_info["pair_strategy"] == "condensed"
    json.dumps(runner.last_run_info)


def test_pair_statistics_preserve_duplicate_zero_distance_pairs():
    orig = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    emb = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [2.0, 0.0]])

    scores = ZADU(PAIR_SPECS, orig).measure(emb)

    _assert_scores_close(scores, _dense_reference(orig, emb))


def test_tight_memory_budget_selects_exact_streaming_blocks(monkeypatch):
    orig, emb = _sample(n=200)
    budget = 64 * 1024
    block_shapes = []
    original_cdist = numpy_backend.cdist

    def recording_cdist(left, right):
        block_shapes.append((len(left), len(right)))
        return original_cdist(left, right)

    monkeypatch.setattr(numpy_backend, "cdist", recording_cdist)
    runner = ZADU(
        PAIR_SPECS,
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    scores = runner.measure(emb)

    pair_plan = runner._execution_plan.pair_plan
    assert pair_plan.strategy is PairStrategy.STREAMING
    assert pair_plan.block_rows == 32
    assert runner._execution_plan.resources == (pair_plan.statistics_key,)
    assert runner.estimated_cache_bytes == 0
    assert runner._execution_plan.planned_peak_bytes <= budget
    _assert_scores_close(scores, _dense_reference(orig, emb))

    details = runner.last_run_info["resources"][0]["details"]
    assert details["strategy"] == "streaming"
    assert details["block_rows"] == 32
    assert details["block_count"] > 1
    assert block_shapes
    assert max(max(shape) for shape in block_shapes) <= 32
    assert runner.last_run_info["planned_peak_bytes"] <= budget
    assert runner.last_run_info["memory_budget_bytes"] == budget


def test_large_pair_only_plan_streams_by_default_without_preallocation():
    orig, _ = _sample(n=6000)

    runner = ZADU([{"id": "stress"}], orig)

    assert runner._execution_plan.pair_plan.strategy is PairStrategy.STREAMING
    assert runner.estimated_cache_bytes == 0
    assert runner._execution_plan.pair_plan.block_rows == 1024


def test_mixed_neighbor_resources_keep_pair_and_knn_work_compact():
    orig, emb = _sample(n=60)
    specs = [
        {"id": "lcmc", "params": {"k": 7}},
        {"id": "stress"},
        {"id": "pr"},
    ]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    assert runner._execution_plan.pair_plan.strategy is PairStrategy.CONDENSED
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    assert {
        (resource["kind"], resource["provider"])
        for resource in runner.last_run_info["resources"]
    } == {
        ("condensed_pairs", "scipy"),
        ("knn", "scipy"),
        ("neighbor_statistics", "scipy"),
        ("pair_statistics", "numpy"),
    }
    assert all(np.isfinite(value) for score in scores for value in score.values())


def test_mixed_cache_bytes_are_considered_before_selecting_condensed_pairs():
    orig, emb = _sample(n=200)
    k = 7
    pair_count = len(orig) * (len(orig) - 1) // 2
    index_bytes = np.dtype(np.int32).itemsize
    non_pair_cache = 2 * len(orig) * k * index_bytes + len(orig) * 8
    budget = 2 * pair_count * 8 + non_pair_cache - 1
    specs = [
        {"id": "lcmc", "params": {"k": k}},
        {"id": "stress"},
        {"id": "pr"},
    ]

    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )
    scores = runner.measure(emb)

    assert runner._execution_plan.pair_plan.strategy is PairStrategy.STREAMING
    assert all(
        key.kind not in {ResourceKind.DISTANCE_MATRIX, ResourceKind.CONDENSED_PAIRS}
        for key in runner._execution_plan.resources
    )
    assert runner._execution_plan.planned_peak_bytes <= budget
    assert all(np.isfinite(value) for score in scores for value in score.values())


def test_original_condensed_pairs_are_reused_across_embeddings(monkeypatch):
    orig, emb = _sample()
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append((key.space, key.kind))
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    runner = ZADU(PAIR_SPECS, orig)
    first = runner.measure(emb)
    second = runner.measure(emb + 0.01)

    assert len(first) == len(second) == 3
    assert calls.count((Space.ORIGINAL, ResourceKind.CONDENSED_PAIRS)) == 1
    assert calls.count((Space.EMBEDDED, ResourceKind.CONDENSED_PAIRS)) == 2
    original_record = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["space"] == "orig"
    )
    embedded_record = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["space"] == "emb"
    )
    assert original_record["reused"] is True
    assert original_record["released"] is False
    assert embedded_record["released"] is True


def test_pair_statistics_preserve_degenerate_and_scaled_results():
    zeros = np.zeros((8, 3))
    separated = np.arange(16, dtype=float).reshape(8, 2)
    with pytest.raises(ValueError, match="Stress is undefined"):
        ZADU([{"id": "stress"}], zeros).measure(separated)
    with pytest.raises(ValueError, match="Scale-normalized stress is undefined"):
        ZADU([{"id": "sn_stress"}], separated).measure(np.zeros((8, 2)))
    with pytest.raises(ValueError, match="Pearson correlation is undefined"):
        ZADU([{"id": "pr"}], separated).measure(separated * 0)

    orig, _ = _sample(n=40)
    scaled = orig[:, :2] * 7.5
    score = ZADU([{"id": "sn_stress"}], orig[:, :2]).measure(scaled)[0]
    assert score["scale_normalized_stress"] == pytest.approx(0.0, abs=1e-7)


def test_pair_peak_guard_fails_before_the_minimum_streaming_block():
    orig, _ = _sample()

    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            PAIR_SPECS,
            orig,
            execution=ExecutionConfig(memory_budget="1B"),
        )


def test_pair_accumulator_rejects_invalid_provider_blocks():
    accumulator = PairAccumulator()
    with pytest.raises(ValueError, match="matching shapes"):
        accumulator.update(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finite and non-negative"):
        accumulator.update(np.asarray([np.nan]), np.ones(1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        accumulator.update(np.ones(1), np.asarray([-1.0]))
    with pytest.raises(RuntimeError, match="at least one"):
        accumulator.finalize(
            strategy=PairStrategy.STREAMING,
            block_rows=1,
            chunk_pairs=None,
        )


def test_geodesic_condensed_and_block_helpers_match_dense_matrix():
    points = np.asarray([[0.0, 0.0], [0.2, 0.1], [-0.4, 0.3], [0.5, -0.2]])
    provider = NumpyResourceProvider()
    dense = provider.pairwise_geodesic_distance_matrix(points)
    upper = np.triu_indices(len(points), k=1)

    condensed = provider.condensed_distances(points, geodesic=True)
    block = provider.geodesic_distance_block(points[:2], points[2:])

    np.testing.assert_allclose(condensed, dense[upper], rtol=0, atol=0)
    np.testing.assert_allclose(block, dense[:2, 2:], rtol=0, atol=0)
