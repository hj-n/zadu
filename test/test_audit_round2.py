"""Numerical and resource-lifetime regressions from the second audit."""

import importlib
import threading
import weakref

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info, threadpool_limits

from zadu import ZADU, EmbeddingExecutionError, ExecutionConfig
from zadu.engine.resources import ResourceCache, ResourceKind, Space
from zadu.measures import class_angular_distortion_index as cadi
from zadu.measures import stress, topographic_product


def _sample():
    rng = np.random.default_rng(42)
    return rng.normal(size=(60, 5)), rng.normal(size=(60, 2)), np.arange(60) % 3


def _topographic_oracle(orig_distances, emb_distances, k):
    # Deliberately scalar, with explicit self-exclusion and stable index ties.
    values = []
    for row in range(len(orig_distances)):
        others = [i for i in range(len(orig_distances)) if i != row]
        left = sorted(others, key=lambda i: (orig_distances[row, i], i))[:k]
        right = sorted(others, key=lambda i: (emb_distances[row, i], i))[:k]
        log_product = 0.0
        for rank, (i, j) in enumerate(zip(left, right, strict=True), 1):
            log_product += np.log(orig_distances[row, j] / orig_distances[row, i])
            log_product += np.log(emb_distances[row, j] / emb_distances[row, i])
            values.append(log_product / (2 * rank))
    return np.mean(values)


@pytest.mark.parametrize("metric", ["euclidean", "cityblock"])
@pytest.mark.parametrize("k", [1, 5])
def test_topographic_precomputed_neighbors_use_the_supplied_geometry(metric, k):
    x, y, _ = _sample()
    distances = cdist(x, x, metric), cdist(y, y, metric)
    expected = _topographic_oracle(*distances, k)
    actual = topographic_product.measure(x, y, k=k, distance_matrices=distances)
    assert actual["topographic_product"] == pytest.approx(expected, abs=1e-14)


def test_topographic_identical_precomputed_geometry_is_zero_without_recomputation(
    monkeypatch,
):
    x, y, _ = _sample()
    distances = cdist(x, x)

    def unexpected(*args, **kwargs):
        pytest.fail("Supplied distances must be reused for neighbor selection")

    monkeypatch.setattr(topographic_product.knn, "knn", unexpected)
    actual = topographic_product.measure(
        x, y, k=5, distance_matrices=(distances, distances)
    )
    assert actual["topographic_product"] == 0.0


@pytest.mark.parametrize("scale", [1e80, 1e-90, 1e300, 1e-300])
@pytest.mark.parametrize("surface", ["direct", "scheduled"])
def test_cadi_preserves_angles_across_finite_coordinate_scales(scale, surface):
    x, y, labels = _sample()
    params = {"n_triplets": 1000, "random_seed": 0}
    expected = cadi.measure(x, y, labels, **params)
    if surface == "direct":
        actual = cadi.measure(x * scale, y * scale, labels, **params)
    else:
        actual = ZADU([{"id": "cadi", "params": params}], x * scale).measure(
            y * scale, labels
        )[0]
    assert expected["class_angular_distortion_index"] > 0.1
    assert actual == pytest.approx(expected, abs=1e-14)


def test_cadi_cosine_handles_overflowing_differences_and_zero_vectors():
    maximum = np.finfo(float).max
    points = np.array([[-maximum, 0], [maximum, 0], [0, maximum]])
    assert cadi._get_cosine(points, 0, 1, 2) == pytest.approx(1 / np.sqrt(2))
    assert cadi._get_cosine(points, 0, 1, 1) == pytest.approx(1.0)
    assert cadi._get_cosine(points, 0, 0, 1) == 0.0


def _pool_state():
    return {info["filepath"]: info["num_threads"] for info in threadpool_info()}


@pytest.mark.parametrize("close_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("finish", ["close", "exhaust", "error"])
def test_interleaved_runners_preserve_callers_native_thread_settings(
    monkeypatch, close_order, finish
):
    # sklearn is a base dependency and loads an observable native runtime on
    # supported wheels. Empty runtimes are covered by the controller guard below.
    importlib.import_module("sklearn")
    x, y, _ = _sample()
    original_measure = stress.measure

    def measure(orig, emb, **kwargs):
        if finish == "error" and emb[0, 0] == 9999:
            raise ValueError("intentional worker failure")
        return original_measure(orig, emb, **kwargs)

    monkeypatch.setattr(stress, "measure", measure)
    later = y.copy()
    later[0, 0] = 9999
    runners = [
        ZADU([{"id": "stress"}], x, execution=ExecutionConfig(embedding_workers=2))
        for _ in range(2)
    ]
    streams = [runner.iter_measure_many([y, later, y]) for runner in runners]
    with threadpool_limits(limits=4):
        expected = _pool_state()
        try:
            for stream in streams:
                next(stream)
                assert _pool_state() == expected
            # A synchronous collection may also run between stream results.
            ZADU(
                [{"id": "stress"}], x, execution=ExecutionConfig(embedding_workers=2)
            ).measure_many([y, y])
            assert _pool_state() == expected
            for index in close_order:
                if finish == "close":
                    streams[index].close()
                elif finish == "exhaust":
                    list(streams[index])
                else:
                    with pytest.raises(EmbeddingExecutionError):
                        list(streams[index])
                assert _pool_state() == expected
        finally:
            for stream in streams:
                stream.close()


@pytest.mark.parametrize(
    "backend,surface",
    [("numpy", "many"), ("numpy", "stream"), ("torch", "many"), ("mlx", "many")],
)
@pytest.mark.parametrize("count", [7, 8])
def test_repeated_batches_release_previous_embedding_arrays(
    monkeypatch, backend, surface, count
):
    if backend != "numpy":
        pytest.importorskip("torch" if backend == "torch" else "mlx.core")
    x, _, labels = _sample()
    rng = np.random.default_rng(12)
    embeddings = [rng.normal(size=(len(x), 2)) for _ in range(count)]
    refs = []
    peaks = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    original_store = ResourceCache._store

    def observe(self, key, built, elapsed):
        original_store(self, key, built, elapsed)
        if key.space is Space.EMBEDDED and key.kind is ResourceKind.KNN:
            with lock:
                refs.append(weakref.ref(built.value))
                peaks.append(sum(ref() is not None for ref in refs))
                completed = len(refs)
            if backend == "numpy" and completed <= count - count % 2:
                # Ensure both workers own arrays before the caller can retire
                # either cache; do not depend on a lucky thread schedule.
                barrier.wait(timeout=10)

    monkeypatch.setattr(ResourceCache, "_store", observe)
    runner = ZADU(
        [{"id": "nh", "params": {"k": 5}}],
        x,
        execution=ExecutionConfig(
            backend=backend,
            device="cpu",
            dtype="float32" if backend == "mlx" else "float64",
            embedding_workers=2,
        ),
    )
    if surface == "many":
        results = runner.measure_many(embeddings, labels)
    else:
        results = [item.result for item in runner.iter_measure_many(embeddings, labels)]
    assert len(results) == count
    assert len(refs) == count
    assert max(peaks) <= 2
    assert sum(ref() is not None for ref in refs) == 1
    expected = ZADU([{"id": "nh", "params": {"k": 5}}], x).measure(
        embeddings[-1], labels
    )
    assert results[-1][0] == pytest.approx(expected[0], abs=1e-14)
    assert runner.emb is embeddings[-1]
    assert runner.emb_knn_indices is not None
    assert runner.last_run_info["native_threads_per_worker"] is None
    if surface == "stream":
        # Resume once so the first yielded cache is retired, then close with
        # another projection pending. The second yielded cache must survive.
        stream = runner.iter_measure_many(embeddings, labels)
        next(stream)
        second = next(stream)
        stream.close()
        assert runner.emb is embeddings[1]
        assert runner.emb_knn_indices is not None
        assert runner.last_run_info["embedding_count"] == 2
        assert not runner.last_run_info["stream_complete"]
        assert second.result == results[1]
        assert sum(ref() is not None for ref in refs) == 1
