import contextlib
import importlib
import json
import threading

import numpy as np
import pytest

from zadu import ZADU, EmbeddingExecutionError, ExecutionConfig
from zadu.engine.resources import ResourceCache, Space
from zadu.measures import stress


def _sample(seed=0, n=72, embedding_count=4):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 10))
    embeddings = [
        orig @ rng.normal(size=(10, 2)) + 0.03 * rng.normal(size=(n, 2))
        for _ in range(embedding_count)
    ]
    labels = np.arange(n) % 4
    return orig, embeddings, labels


def _mixed_specs():
    return [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 9}},
        {"id": "nh", "params": {"k": 7}},
        {"id": "stress"},
    ]


def _assert_nested_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    elif isinstance(expected, (float, np.floating)):
        assert actual == pytest.approx(expected, rel=0, abs=1e-13)
    else:
        assert actual == expected


def test_parallel_measure_many_matches_sequential_local_results_and_order():
    orig, embeddings, labels = _sample(seed=1)
    sequential = ZADU(_mixed_specs(), orig, return_local=True).measure_many(
        embeddings, labels
    )
    runner = ZADU(
        _mixed_specs(),
        orig,
        return_local=True,
        execution=ExecutionConfig(embedding_workers=2),
    )

    parallel = runner.measure_many(embeddings, labels)

    _assert_nested_equal(parallel, sequential)
    assert runner.emb is embeddings[-1]
    assert runner.label is labels
    assert runner.last_run_info["batch_strategy"] == "threaded_shared_original"
    assert runner.last_run_info["requested_workers"] == 2
    assert runner.last_run_info["effective_workers"] == 2
    assert runner.last_run_info["worker_limit_reason"] is None
    assert runner.last_run_info["native_threads_per_worker"] == 1
    assert runner.last_run_info["provider_batching"] is False
    assert runner.last_run_info["native_batch_size"] == 1
    assert [run["embedding_index"] for run in runner.last_run_info["runs"]] == [
        0,
        1,
        2,
        3,
    ]
    json.dumps(runner.last_run_info)

    followup = runner.measure(embeddings[0], labels)
    _assert_nested_equal(followup, sequential[0])


def test_parallel_caches_share_frozen_original_values(monkeypatch):
    orig, embeddings, _ = _sample(seed=2)
    runner = ZADU(
        [{"id": "proc", "params": {"k": 7}}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )
    original_value = next(
        value
        for key, value in runner._resource_cache._values.items()
        if key.space is Space.ORIGINAL
    )
    assert original_value.flags.writeable is True
    forks = []
    original_fork = ResourceCache.fork_original

    def wrapped_fork(self, provider):
        fork = original_fork(self, provider)
        forks.append(fork)
        return fork

    monkeypatch.setattr(ResourceCache, "fork_original", wrapped_fork)

    runner.measure_many(embeddings)

    assert len(forks) == len(embeddings)
    assert len({id(fork.provider) for fork in forks}) == len(embeddings)
    assert original_value.flags.writeable is False
    assert all(
        next(
            value for key, value in fork._values.items() if key.space is Space.ORIGINAL
        )
        is original_value
        for fork in forks
    )
    assert all(
        resource["reused"]
        for run in runner.last_run_info["runs"]
        for resource in run["resources"]
        if resource["space"] == "orig"
    )


def test_memory_budget_caps_collection_workers_before_execution():
    orig, embeddings, labels = _sample(seed=3)
    baseline = ZADU(_mixed_specs(), orig)
    plan = baseline._execution_plan
    budget = plan.original_cache_bytes + 2 * plan.per_embedding_peak_bytes
    runner = ZADU(
        _mixed_specs(),
        orig,
        execution=ExecutionConfig(
            embedding_workers=4,
            memory_budget=budget,
        ),
    )

    runner.measure_many(embeddings, labels)
    info = runner.last_run_info

    assert info["requested_workers"] == 4
    assert info["effective_workers"] == 2
    assert info["worker_limit_reason"] == "memory_budget"
    assert info["shared_original_bytes"] == plan.original_cache_bytes
    assert info["per_embedding_peak_bytes"] == plan.per_embedding_peak_bytes
    assert info["planned_peak_bytes"] == budget
    assert info["planned_peak_bytes"] <= info["memory_budget_bytes"]


def test_worker_sized_batches_bound_concurrent_execution(monkeypatch):
    orig, embeddings, _ = _sample(seed=4)
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )
    original_execute = ZADU._execute_isolated_embedding
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    started = 0
    active = 0
    maximum_active = 0

    def wrapped_execute(self, *args, **kwargs):
        nonlocal started, active, maximum_active
        with lock:
            call_index = started
            started += 1
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call_index < 2:
                barrier.wait(timeout=5)
            return original_execute(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ZADU, "_execute_isolated_embedding", wrapped_execute)

    runner.measure_many(embeddings)

    assert started == len(embeddings)
    assert maximum_active == 2


def test_parallel_execution_limits_native_threads(monkeypatch):
    orig, embeddings, _ = _sample(seed=5, embedding_count=2)
    module = importlib.import_module("zadu.zadu")
    recorded = []

    def record_threadpool_limits(*, limits):
        recorded.append(limits)
        return contextlib.nullcontext()

    monkeypatch.setattr(module, "threadpool_limits", record_threadpool_limits)
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=4),
    )

    runner.measure_many(embeddings)

    assert recorded == [1]
    assert runner.last_run_info["requested_workers"] == 4
    assert runner.last_run_info["effective_workers"] == 2
    assert runner.last_run_info["worker_limit_reason"] == "embedding_count"


def test_parallel_failure_reports_embedding_index_and_preserves_cause(monkeypatch):
    orig, embeddings, _ = _sample(seed=6)
    embeddings[2][0, 0] = 9876.0
    original_measure = stress.measure

    def failing_measure(orig, emb, **kwargs):
        if emb[0, 0] == 9876.0:
            raise ValueError("intentional worker failure")
        return original_measure(orig, emb, **kwargs)

    monkeypatch.setattr(stress, "measure", failing_measure)
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )

    with pytest.raises(EmbeddingExecutionError) as error:
        runner.measure_many(embeddings)

    assert error.value.embedding_index == 2
    assert isinstance(error.value.__cause__, ValueError)
    assert "intentional worker failure" in str(error.value.__cause__)
    assert runner.last_run_info is None


def test_unseeded_snc_falls_back_to_ordered_sequential_execution():
    orig, embeddings, _ = _sample(seed=7, n=48, embedding_count=2)
    runner = ZADU(
        [{"id": "snc", "params": {"iteration": 2, "k": 6}}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )

    runner.measure_many(embeddings)

    assert runner.last_run_info["requested_workers"] == 2
    assert runner.last_run_info["effective_workers"] == 1
    assert runner.last_run_info["worker_limit_reason"] == "unseeded_snc"
    assert runner.last_run_info["batch_strategy"] == "sequential_shared_original"


def test_mutable_random_generator_marks_collection_as_parallel_unsafe():
    orig, _, _ = _sample(seed=8, n=48, embedding_count=2)
    runner = ZADU(
        [
            {
                "id": "cadi",
                "params": {"n_triplets": 10, "random_seed": np.random.default_rng(0)},
            }
        ],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )

    assert runner._parallel_fallback_reason() == "mutable_random_state"


def test_fixed_seed_snc_uses_outer_workers_and_one_inner_worker_exactly():
    orig, embeddings, _ = _sample(seed=9, n=60, embedding_count=3)
    specs = [
        {
            "id": "snc",
            "params": {
                "iteration": 5,
                "walk_num_ratio": 0.2,
                "k": 8,
                "random_state": 42,
                "n_jobs": 3,
            },
        }
    ]
    sequential = ZADU(specs, orig, return_local=True).measure_many(embeddings)
    runner = ZADU(
        specs,
        orig,
        return_local=True,
        execution=ExecutionConfig(embedding_workers=2),
    )

    parallel = runner.measure_many(embeddings)

    _assert_nested_equal(parallel, sequential)
    assert runner.last_run_info["effective_workers"] == 2
    assert runner.last_run_info["snc_strategy"]["requested_workers"] == {0: 3}
    assert runner.last_run_info["snc_strategy"]["effective_workers"] == {0: 1}
    assert all(
        run["snc_strategy"]["effective_workers"] == {0: 1}
        for run in runner.last_run_info["runs"]
    )
