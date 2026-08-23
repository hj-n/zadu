import json
import threading

import numpy as np
import pytest

from zadu import ZADU, EmbeddingExecutionError, EmbeddingResult, ExecutionConfig
from zadu.measures import stress


def _sample(seed=0, n=56, embedding_count=5):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 8))
    embeddings = [
        orig @ rng.normal(size=(8, 2)) + 0.02 * rng.normal(size=(n, 2))
        for _ in range(embedding_count)
    ]
    labels = np.arange(n) % 4
    return orig, embeddings, labels


def _specs():
    return [
        {"id": "tnc", "params": {"k": 5}},
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


def test_stream_is_lazy_and_yields_indexed_measure_many_parity():
    orig, embeddings, labels = _sample()
    expected = ZADU(_specs(), orig).measure_many(embeddings, labels)
    consumed = 0

    def source():
        nonlocal consumed
        for embedding in embeddings:
            consumed += 1
            yield embedding

    runner = ZADU(_specs(), orig)
    stream = runner.iter_measure_many(source(), labels)

    assert consumed == 0
    first = next(stream)
    assert consumed == 1
    assert isinstance(first, EmbeddingResult)
    assert first.index == 0
    assert first.run_info["embedding_index"] == 0
    _assert_nested_equal(first.result, expected[0])
    assert runner.last_run_info is None

    records = [first, *stream]

    assert [record.index for record in records] == list(range(len(embeddings)))
    for record, expected_result in zip(records, expected, strict=True):
        _assert_nested_equal(record.result, expected_result)
    info = runner.last_run_info
    assert info["mode"] == "many_stream"
    assert info["embedding_count"] == len(embeddings)
    assert info["input_consumed_count"] == len(embeddings)
    assert info["stream_complete"] is True
    assert info["runs_retained"] is False
    assert "runs" not in info
    assert info["original_resources_reused"] is True
    json.dumps(info)


def test_parallel_stream_never_consumes_beyond_its_execution_window():
    orig, embeddings, _ = _sample(seed=1, embedding_count=7)
    consumed = 0

    def source():
        nonlocal consumed
        for embedding in embeddings:
            consumed += 1
            yield embedding

    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )
    stream = runner.iter_measure_many(source())

    assert consumed == 0
    records = []
    for record in stream:
        records.append(record)
        assert consumed <= len(records) + 1

    assert [record.index for record in records] == list(range(len(embeddings)))
    assert runner.last_run_info["effective_workers"] == 2
    assert runner.last_run_info["max_in_flight_observed"] == 2
    assert runner.last_run_info["planned_peak_bytes"] == (
        runner._execution_plan.original_cache_bytes
        + 2 * runner._execution_plan.per_embedding_peak_bytes
    )


def test_parallel_stream_bounds_active_execution_and_preserves_order(monkeypatch):
    orig, embeddings, _ = _sample(seed=2, embedding_count=4)
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
        nonlocal active, maximum_active, started
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

    records = list(runner.iter_measure_many(embeddings))

    assert [record.index for record in records] == [0, 1, 2, 3]
    assert started == len(embeddings)
    assert maximum_active == 2


def test_early_close_closes_source_cancels_window_and_adopts_last_yield():
    orig, embeddings, _ = _sample(seed=3, embedding_count=5)
    source_closed = False

    def source():
        nonlocal source_closed
        try:
            yield from embeddings
        finally:
            source_closed = True

    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )
    stream = runner.iter_measure_many(source())
    first = next(stream)

    stream.close()

    assert first.index == 0
    assert source_closed is True
    assert runner.emb is embeddings[0]
    info = runner.last_run_info
    assert info["stream_complete"] is False
    assert info["embedding_count"] == 1
    assert info["input_consumed_count"] == 2
    assert info["max_in_flight_observed"] == 2


def test_stream_failure_keeps_measure_many_error_index_and_cause(monkeypatch):
    orig, embeddings, _ = _sample(seed=4, embedding_count=4)
    embeddings[2][0, 0] = 9876.0
    original_measure = stress.measure

    def failing_measure(orig, emb, **kwargs):
        if emb[0, 0] == 9876.0:
            raise ValueError("intentional stream failure")
        return original_measure(orig, emb, **kwargs)

    monkeypatch.setattr(stress, "measure", failing_measure)
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(embedding_workers=2),
    )

    with pytest.raises(EmbeddingExecutionError) as error:
        list(runner.iter_measure_many(embeddings))

    assert error.value.embedding_index == 2
    assert isinstance(error.value.__cause__, ValueError)
    assert "intentional stream failure" in str(error.value.__cause__)
    assert runner.last_run_info is None


def test_stream_validates_each_embedding_only_when_reached():
    orig, embeddings, _ = _sample(seed=5, embedding_count=2)
    invalid = embeddings[1][:-1]
    runner = ZADU([{"id": "stress"}], orig)
    stream = runner.iter_measure_many(
        embedding for embedding in [embeddings[0], invalid]
    )

    first = next(stream)
    assert first.index == 0
    with pytest.raises(ValueError, match=r"embeddings\[1\].*same number of rows"):
        next(stream)

    assert runner.last_run_info is None


def test_empty_stream_has_bounded_json_diagnostics():
    orig, _, _ = _sample(seed=6)
    runner = ZADU([{"id": "stress"}], orig)

    assert list(runner.iter_measure_many(iter(()))) == []

    info = runner.last_run_info
    assert info["embedding_count"] == 0
    assert info["input_consumed_count"] == 0
    assert info["effective_workers"] == 0
    assert info["max_in_flight_observed"] == 0
    assert info["stream_complete"] is True
    assert info["resource_seconds"] == 0
    assert info["metric_seconds"] == 0
    json.dumps(info)


def test_stream_memory_budget_caps_the_in_flight_width():
    orig, embeddings, _ = _sample(seed=7, embedding_count=5)
    baseline = ZADU([{"id": "stress"}], orig)
    plan = baseline._execution_plan
    budget = plan.original_cache_bytes + 2 * plan.per_embedding_peak_bytes
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            embedding_workers=5,
            memory_budget=budget,
        ),
    )

    records = list(runner.iter_measure_many(embeddings))

    assert len(records) == len(embeddings)
    assert runner.last_run_info["effective_workers"] == 2
    assert runner.last_run_info["worker_limit_reason"] == "memory_budget"
    assert runner.last_run_info["max_in_flight_observed"] == 2
    assert runner.last_run_info["planned_peak_bytes"] <= budget


def test_stream_validates_shared_labels_before_consuming_input():
    orig, embeddings, _ = _sample(seed=8, embedding_count=2)
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield from embeddings

    runner = ZADU([{"id": "nh", "params": {"k": 5}}], orig)

    with pytest.raises(ValueError, match="Label is required"):
        runner.iter_measure_many(source())

    assert consumed is False
