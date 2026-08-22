import json

import numpy as np
import pytest

from zadu import ZADU, ExecutionConfig
from zadu.backends import NumpyResourceProvider
from zadu.engine.resources import ResourceKind, Space


def _sample(seed=0, n=72, embedding_count=3):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 10))
    projections = [rng.normal(size=(10, 2)) for _ in range(embedding_count)]
    embeddings = [
        orig @ projection + 0.03 * rng.normal(size=(n, 2)) for projection in projections
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
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    elif isinstance(expected, (float, np.floating)):
        assert actual == pytest.approx(expected, rel=0, abs=1e-14)
    else:
        assert actual == expected


def test_measure_many_matches_independent_measure_calls_in_input_order():
    orig, embeddings, labels = _sample()
    expected = [
        ZADU(_mixed_specs(), orig).measure(embedding, labels)
        for embedding in embeddings
    ]

    runner = ZADU(_mixed_specs(), orig)
    actual = runner.measure_many((embedding for embedding in embeddings), labels)

    _assert_nested_equal(actual, expected)
    assert runner.emb is embeddings[-1]
    assert runner.label is labels


def test_measure_many_preserves_local_result_shape_and_values():
    orig, embeddings, labels = _sample(seed=2, embedding_count=2)
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "nh", "params": {"k": 7}},
        {"id": "stress"},
    ]
    expected = [
        ZADU(specs, orig, return_local=True).measure(embedding, labels)
        for embedding in embeddings
    ]

    actual = ZADU(specs, orig, return_local=True).measure_many(embeddings, labels)

    _assert_nested_equal(actual, expected)
    assert all(
        len(global_scores) == len(local_scores) == 3
        for global_scores, local_scores in actual
    )
    assert all(local_scores[-1] is None for _, local_scores in actual)


def test_measure_many_reuses_one_maximum_k_original_resource(monkeypatch):
    orig, embeddings, _ = _sample(seed=3)
    specs = [
        {"id": "proc", "params": {"k": 5}},
        {"id": "proc", "params": {"k": 11}},
    ]
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append(key)
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    runner = ZADU(specs, orig)
    results = runner.measure_many(embeddings)

    assert len(results) == len(embeddings)
    original_knn = [
        key
        for key in calls
        if key.space is Space.ORIGINAL and key.kind is ResourceKind.KNN
    ]
    embedded_knn = [
        key
        for key in calls
        if key.space is Space.EMBEDDED and key.kind is ResourceKind.KNN
    ]
    assert [(key.k, key.kind) for key in original_knn] == [(11, ResourceKind.KNN)]
    assert [key.k for key in embedded_knn] == [11] * len(embeddings)


def test_measure_many_reports_json_compatible_batch_diagnostics():
    orig, embeddings, labels = _sample(seed=4)
    runner = ZADU(_mixed_specs(), orig)

    runner.measure_many(np.stack(embeddings), labels)
    info = runner.last_run_info

    assert info["exact"] is True
    assert info["mode"] == "many"
    assert info["batch_strategy"] == "sequential_shared_original"
    assert info["native_batch_size"] == 1
    assert info["embedding_count"] == len(embeddings)
    assert info["original_resources_reused"] is True
    assert info["original_resource_reuse_events"] == (
        info["original_resource_count"] * len(embeddings)
    )
    assert info["total_seconds"] >= sum(run["total_seconds"] for run in info["runs"])
    assert info["resource_seconds"] == pytest.approx(
        sum(run["resource_seconds"] for run in info["runs"])
    )
    assert info["metric_seconds"] == pytest.approx(
        sum(run["metric_seconds"] for run in info["runs"])
    )
    assert [run["embedding_index"] for run in info["runs"]] == [0, 1, 2]
    assert [entry["id"] for entry in info["metrics"]] == [
        "trustworthiness_continuity",
        "local_continuity_meta_criteria",
        "neighborhood_hit",
        "stress",
    ]
    for run in info["runs"]:
        assert all(
            resource["reused"]
            for resource in run["resources"]
            if resource["space"] == "orig"
        )
        assert all(
            resource["built_in_run"]
            for resource in run["resources"]
            if resource["space"] in {"emb", "pair"}
        )
    json.dumps(info)


def test_measure_many_prevalidates_every_embedding_before_execution(monkeypatch):
    orig, embeddings, labels = _sample(seed=5, embedding_count=2)
    runner = ZADU(_mixed_specs(), orig)
    previous = runner.measure(embeddings[0], labels)
    previous_info = runner.last_run_info
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append(key)
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    invalid = embeddings[1][:-1]

    with pytest.raises(ValueError, match="same number of rows"):
        runner.measure_many([embeddings[0], invalid], labels)

    assert previous
    assert runner.last_run_info is previous_info
    assert calls == []


@pytest.mark.parametrize(
    "embeddings,error,match",
    [
        (np.ones((20, 2)), ValueError, "use measure"),
        (3, TypeError, "iterable"),
        ("embedding", TypeError, "iterable"),
    ],
)
def test_measure_many_rejects_ambiguous_or_non_iterable_inputs(
    embeddings, error, match
):
    orig, _, _ = _sample(n=20)

    with pytest.raises(error, match=match):
        ZADU([], orig).measure_many(embeddings)


def test_measure_many_requires_one_shared_label_vector_before_execution(monkeypatch):
    orig, embeddings, _ = _sample(seed=7, embedding_count=2)
    runner = ZADU([{"id": "nh", "params": {"k": 5}}], orig)
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append(key)
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)

    with pytest.raises(ValueError, match="Label is required"):
        runner.measure_many(embeddings)
    with pytest.raises(ValueError, match="1D array"):
        runner.measure_many(embeddings, np.zeros((len(orig), 1)))

    assert calls == []


def test_empty_measure_many_has_stable_diagnostics_and_no_execution():
    orig, _, _ = _sample(seed=8)
    runner = ZADU([{"id": "proc", "params": {"k": 5}}], orig)
    generation = runner._resource_cache.generation

    result = runner.measure_many([])

    assert result == []
    assert runner._resource_cache.generation == generation
    assert runner.last_run_info["embedding_count"] == 0
    assert runner.last_run_info["runs"] == []
    assert runner.last_run_info["original_resources_reused"] is False
    assert runner.last_run_info["resource_seconds"] == 0
    assert runner.last_run_info["metric_seconds"] == 0
    json.dumps(runner.last_run_info)


def test_measure_many_keeps_per_embedding_memory_plan_under_budget():
    orig, embeddings, labels = _sample(seed=9, n=48, embedding_count=4)
    baseline = ZADU(_mixed_specs(), orig)
    budget = baseline._execution_plan.planned_peak_bytes
    runner = ZADU(
        _mixed_specs(),
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    results = runner.measure_many(embeddings, labels)

    assert len(results) == len(embeddings)
    assert runner.last_run_info["planned_peak_bytes"] <= budget
    assert all(
        run["planned_peak_bytes"] <= budget for run in runner.last_run_info["runs"]
    )
