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
    TopographicProductStatistics,
    compact_index_dtype,
)
from zadu.measures import stress, topographic_product
from zadu.measures.utils.knn import knn_from_distance_matrix


def _sample(seed=0, n=80):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 6)), rng.normal(size=(n, 2))


def _dense_reference(orig, emb, k):
    orig_distances = cdist(orig, orig)
    emb_distances = cdist(emb, emb)
    orig_knn = knn_from_distance_matrix(orig_distances, k)
    emb_knn = knn_from_distance_matrix(emb_distances, k)
    return topographic_product.measure(
        orig,
        emb,
        k,
        distance_matrices=(orig_distances, emb_distances),
        knn_info=(orig_knn, emb_knn),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 7, 20])
def test_topographic_product_uses_exact_selected_distances(dtype, k):
    orig, emb = _sample(n=50)
    orig = orig.astype(dtype)
    emb = emb.astype(dtype)
    runner = ZADU([{"id": "topo", "params": {"k": k}}], orig)

    score = runner.measure(emb)[0]

    assert score == pytest.approx(_dense_reference(orig, emb, k), abs=1e-14)
    assert [key.kind for key in runner._execution_plan.resources] == [
        ResourceKind.STABLE_KNN,
        ResourceKind.STABLE_KNN,
        ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS,
    ]
    assert all(
        key.kind is not ResourceKind.DISTANCE_MATRIX
        for key in runner._execution_plan.resources
    )
    assert runner.orig_distance_matrix is None
    assert runner.emb_distance_matrix is None
    assert runner.orig_knn_indices.shape == (len(orig), k)
    assert runner.emb_knn_indices.shape == (len(orig), k)
    expected_cache = 2 * len(orig) * k * compact_index_dtype(len(orig)).itemsize + k * 8
    assert runner.estimated_cache_bytes == expected_cache
    assert runner.last_run_info["topographic_strategy"] == (
        "blockwise_selected_distances"
    )
    statistics = runner.last_run_info["resources"][-1]
    assert statistics["details"]["algorithm"] == ("blockwise_selected_distances")
    assert statistics["released"] is True
    json.dumps(runner.last_run_info)


def test_tight_budget_bounds_stable_knn_distance_blocks(monkeypatch):
    orig, emb = _sample(n=100)
    k = 10
    budget = 32_000
    block_shapes = []
    original_cdist = numpy_backend.cdist

    def recording_cdist(left, right):
        block_shapes.append((len(left), len(right)))
        return original_cdist(left, right)

    monkeypatch.setattr(numpy_backend, "cdist", recording_cdist)
    runner = ZADU(
        [{"id": "topo", "params": {"k": k}}],
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    score = runner.measure(emb)[0]

    assert score == pytest.approx(_dense_reference(orig, emb, k), abs=1e-12)
    assert runner._execution_plan.planned_peak_bytes <= budget
    assert block_shapes
    assert all(right == len(orig) for _, right in block_shapes)
    assert max(left for left, _ in block_shapes) < len(orig)
    stable_details = [
        resource["details"]
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "stable_knn"
    ]
    assert all(details["block_count"] > 1 for details in stable_details)
    assert all(details["working_bytes"] <= budget for details in stable_details)


def test_multiple_k_values_share_one_maximum_prefix_resource():
    orig, emb = _sample(n=70)
    requested = (3, 12, 7)
    specs = [{"id": "topo", "params": {"k": k}} for k in requested]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    for score, k in zip(scores, requested, strict=True):
        assert score == pytest.approx(_dense_reference(orig, emb, k), abs=1e-14)
    plan = runner._execution_plan.topographic_plan
    assert plan.k == 12
    assert plan.requested_ks == (3, 7, 12)
    assert (
        sum(
            key.kind is ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS
            for key in runner._execution_plan.resources
        )
        == 1
    )
    assert all(
        key.k == 12
        for key in runner._execution_plan.resources
        if key.kind
        in {ResourceKind.STABLE_KNN, ResourceKind.TOPOGRAPHIC_PRODUCT_STATISTICS}
    )


def test_original_stable_neighbors_are_reused_across_embeddings(monkeypatch):
    orig, emb = _sample()
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append((key.space, key.kind))
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    runner = ZADU([{"id": "topo", "params": {"k": 10}}], orig)
    runner.measure(emb)
    runner.measure(emb + 0.01)

    assert calls.count((Space.ORIGINAL, ResourceKind.STABLE_KNN)) == 1
    assert calls.count((Space.EMBEDDED, ResourceKind.STABLE_KNN)) == 2
    original_resource = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "stable_knn" and resource["space"] == "orig"
    )
    assert original_resource["reused"] is True


def test_stable_neighbors_preserve_dense_tie_order():
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
        ]
    )
    expected = knn_from_distance_matrix(cdist(points, points), 4)

    actual, block_count, block_rows = NumpyResourceProvider.stable_knn(
        points,
        4,
        working_memory_bytes=len(points) * 32,
        geodesic=False,
    )

    np.testing.assert_array_equal(actual, expected)
    assert block_count == len(points)
    assert block_rows == 1


def test_topographic_product_preserves_geodesic_dense_result():
    orig = np.asarray([[0.0, 0.0], [0.2, 0.1], [-0.4, 0.3], [0.5, -0.2], [0.7, 0.4]])
    emb = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.3, -0.2], [0.5, 0.4], [-0.1, 0.7]])
    provider = NumpyResourceProvider()
    orig_distances = provider.pairwise_geodesic_distance_matrix(orig)
    emb_distances = cdist(emb, emb)
    k = 2
    expected = topographic_product.measure(
        orig,
        emb,
        k,
        distance_matrices=(orig_distances, emb_distances),
        knn_info=(
            knn_from_distance_matrix(orig_distances, k),
            knn_from_distance_matrix(emb_distances, k),
        ),
    )

    runner = ZADU(
        [{"id": "topo", "params": {"k": k}}],
        orig,
        geodesic=True,
    )
    score = runner.measure(emb)[0]

    assert score == pytest.approx(expected, abs=1e-14)


def test_topographic_and_pair_reduction_keep_independent_bounded_plans():
    orig, emb = _sample(n=60)
    specs = [{"id": "topo", "params": {"k": 8}}, {"id": "stress"}]
    runner = ZADU(specs, orig)

    scores = runner.measure(emb)

    assert runner._execution_plan.topographic_plan is not None
    assert runner._execution_plan.pair_plan.strategy is PairStrategy.CONDENSED
    assert scores[0] == pytest.approx(_dense_reference(orig, emb, 8), abs=1e-14)
    matrices = cdist(orig, orig), cdist(emb, emb)
    assert scores[1] == pytest.approx(stress.measure(orig, emb, matrices), abs=1e-14)


def test_topographic_memory_guard_fails_before_knn_allocation(monkeypatch):
    orig, _ = _sample(n=100)
    unbounded = ZADU([{"id": "topo", "params": {"k": 10}}], orig)
    budget = unbounded.estimated_cache_bytes

    def unexpected_cdist(*args, **kwargs):
        raise AssertionError("distance allocation should not start")

    monkeypatch.setattr(numpy_backend, "cdist", unexpected_cdist)
    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            [{"id": "topo", "params": {"k": 10}}],
            orig,
            execution=ExecutionConfig(memory_budget=budget),
        )


def test_topographic_resources_preserve_degenerate_errors():
    orig_duplicates = np.asarray([[0.0], [0.0], [1.0]])
    separated = np.asarray([[0.0], [0.5], [1.0]])
    with pytest.raises(ValueError, match="zero-distance original-space"):
        ZADU([{"id": "topo", "params": {"k": 1}}], orig_duplicates).measure(separated)

    orig = np.asarray([[0.0], [1.0], [10.0]])
    embedded_duplicates = np.asarray([[0.0], [0.0], [10.0]])
    with pytest.raises(ValueError, match="zero-distance embedded-space"):
        ZADU([{"id": "topo", "params": {"k": 1}}], orig).measure(embedded_duplicates)


def test_topographic_metric_rejects_unplanned_statistics_prefixes():
    orig, emb = _sample(n=5)
    statistics = TopographicProductStatistics(
        scores=np.asarray([np.nan]),
        block_count=1,
        block_rows=5,
    )

    with pytest.raises(RuntimeError, match="do not contain"):
        topographic_product.measure(
            orig,
            emb,
            k=1,
            topographic_product_statistics=statistics,
        )
    with pytest.raises(RuntimeError, match="do not contain"):
        topographic_product.measure(
            orig,
            emb,
            k=2,
            topographic_product_statistics=statistics,
        )
