import json

import numpy as np
import pytest

from zadu import ZADU, ExecutionConfig
from zadu.backends import NumpyResourceProvider
from zadu.engine.resources import (
    PairStrategy,
    ResourceKind,
    Space,
    compact_index_dtype,
)
from zadu.measures import (
    local_continuity_meta_criteria,
    neighborhood_hit,
    stress,
    trustworthiness_continuity,
)
from zadu.registry import METRIC_BY_ALIAS


def _sample(seed: int = 0, n: int = 60):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 6))
    emb = rng.normal(size=(n, 2))
    labels = np.arange(n) % 3
    return orig, emb, labels


def _mixed_specs():
    return [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 10}},
        {"id": "nh", "params": {"k": 7}},
        {"id": "stress"},
    ]


def test_execution_config_normalizes_exact_numpy_options():
    config = ExecutionConfig(
        backend="NUMPY",
        device="CPU",
        memory_budget="1.5MiB",
        embedding_workers=np.int64(3),
    )

    assert config.backend == "numpy"
    assert config.device == "cpu"
    assert config.resolved_backend == "numpy"
    assert config.resolved_device == "cpu"
    assert config.resolved_dtype == "float64"
    assert config.memory_budget_bytes == int(1.5 * 1024**2)
    assert config.embedding_workers == 3


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"backend": "other"}, ValueError, "auto.*numpy.*mlx.*torch"),
        ({"backend": 1}, TypeError, "backend"),
        ({"device": "gpu"}, ValueError, "auto.*cpu"),
        ({"device": 1}, TypeError, "device"),
        ({"dtype": 32}, TypeError, "dtype"),
        ({"dtype": "float32"}, ValueError, "float64.*NumPy"),
        ({"backend": "mlx"}, ValueError, "explicit.*float32.*float64"),
        (
            {"backend": "mlx", "device": "gpu", "dtype": "float64"},
            ValueError,
            "GPU.*float32",
        ),
        ({"backend": "torch"}, ValueError, "explicit.*float32.*float64"),
        (
            {"backend": "torch", "device": "gpu", "dtype": "float32"},
            ValueError,
            "auto.*cpu.*mps.*cuda",
        ),
        (
            {"backend": "torch", "device": "mps", "dtype": "float64"},
            ValueError,
            "MPS.*float32",
        ),
        ({"memory_budget": True}, TypeError, "memory_budget"),
        ({"memory_budget": 0}, ValueError, "greater than zero"),
        ({"memory_budget": "many"}, ValueError, "4GiB"),
        ({"embedding_workers": True}, TypeError, "embedding_workers"),
        ({"embedding_workers": 1.5}, TypeError, "embedding_workers"),
        ({"embedding_workers": 0}, ValueError, "embedding_workers"),
    ],
)
def test_execution_config_rejects_unsupported_options(kwargs, error, match):
    with pytest.raises(error, match=match):
        ExecutionConfig(**kwargs)


def test_execution_config_normalizes_explicit_mlx_options():
    config = ExecutionConfig(
        backend="MLX",
        device="GPU",
        dtype="FLOAT32",
        embedding_workers=np.int64(3),
    )

    assert config.backend == "mlx"
    assert config.device == "gpu"
    assert config.dtype == "float32"
    assert config.resolved_backend == "mlx"
    assert config.resolved_device == "gpu"
    assert config.resolved_dtype == "float32"
    assert config.embedding_workers == 3


def test_execution_config_normalizes_explicit_torch_options():
    config = ExecutionConfig(
        backend="TORCH",
        device="CUDA",
        dtype="FLOAT32",
    )

    assert config.backend == "torch"
    assert config.device == "cuda"
    assert config.dtype == "float32"
    assert config.resolved_backend == "torch"
    assert config.resolved_device == "cuda"
    assert config.resolved_dtype == "float32"

    batched = ExecutionConfig(
        backend="torch",
        device="cpu",
        dtype="float64",
        embedding_workers=3,
    )
    assert batched.embedding_workers == 3


def test_legacy_and_config_memory_budgets_are_mutually_exclusive():
    orig, _, _ = _sample()

    with pytest.raises(ValueError, match="Provide only one"):
        ZADU(
            [{"id": "stress"}],
            orig,
            max_memory_bytes=10_000,
            execution=ExecutionConfig(memory_budget="1MiB"),
        )
    with pytest.raises(TypeError, match="ExecutionConfig"):
        ZADU([], orig, execution={})
    with pytest.raises(TypeError, match="integer"):
        ZADU([], orig, max_memory_bytes=1.5)


def test_string_memory_budget_uses_existing_preallocation_guard():
    orig, _, _ = _sample()

    with pytest.raises(MemoryError, match="Estimated ZADU cache size"):
        ZADU(
            [{"id": "stress"}],
            orig,
            execution=ExecutionConfig(memory_budget="1B"),
        )


def test_plan_deduplicates_resources_and_promotes_knn_to_ranking():
    orig, _, _ = _sample()
    first = ZADU(_mixed_specs(), orig)
    second = ZADU(_mixed_specs(), orig)

    assert first._execution_plan.resources == second._execution_plan.resources
    assert first._execution_plan.consumers == second._execution_plan.consumers
    assert [
        (key.kind, key.space, key.k) for key in first._execution_plan.resources
    ] == [
        (ResourceKind.DISTANCE_MATRIX, Space.ORIGINAL, None),
        (ResourceKind.NEIGHBOR_RANKING, Space.ORIGINAL, 10),
        (ResourceKind.DISTANCE_MATRIX, Space.EMBEDDED, None),
        (ResourceKind.NEIGHBOR_RANKING, Space.EMBEDDED, 10),
        (ResourceKind.PAIR_STATISTICS, Space.PAIRED, None),
        (ResourceKind.RANK_COMPARISONS, Space.PAIRED, 5),
        (ResourceKind.NEIGHBOR_STATISTICS, Space.PAIRED, 10),
    ]

    n = len(orig)
    index_bytes = compact_index_dtype(n).itemsize
    expected_bytes = (
        2 * n * n * 8
        + 2 * (n * n * index_bytes + n * 10 * index_bytes)
        + 2 * n * 5 * index_bytes
        + 2 * n * 5
        + n * 8
    )
    assert first.estimated_cache_bytes == expected_bytes
    assert first.ranking_k == 5
    assert first.knn_both_k == 10
    assert first.knn_emb_k == 7
    assert first._execution_plan.pair_plan.strategy is PairStrategy.DENSE


def test_mixed_k_plan_preserves_exact_results_with_duplicate_ties():
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
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    specs = [
        {"id": "tnc", "params": {"k": 2}},
        {"id": "lcmc", "params": {"k": 3}},
        {"id": "nh", "params": {"k": 1}},
        {"id": "stress"},
    ]

    runner = ZADU(specs, orig)
    planned = runner.measure(emb, labels)
    expected = [
        trustworthiness_continuity.measure(
            orig,
            emb,
            k=2,
            knn_ranking_info=(
                runner.orig_knn_indices[:, :2],
                runner.orig_knn_ranking,
                runner.emb_knn_indices[:, :2],
                runner.emb_knn_ranking,
            ),
        ),
        local_continuity_meta_criteria.measure(
            orig,
            emb,
            k=3,
            knn_info=(
                runner.orig_knn_indices[:, :3],
                runner.emb_knn_indices[:, :3],
            ),
        ),
        neighborhood_hit.measure(
            emb,
            labels,
            k=1,
            knn_emb_info=runner.emb_knn_indices[:, :1],
        ),
        stress.measure(
            orig,
            emb,
            distance_matrices=(
                runner.orig_distance_matrix,
                runner.emb_distance_matrix,
            ),
        ),
    ]

    for actual, direct in zip(planned, expected, strict=True):
        assert actual == pytest.approx(direct)


def test_registry_keeps_cache_names_as_typed_compatibility_view():
    assert METRIC_BY_ALIAS["tnc"].cache == {"rank_comparisons"}
    assert METRIC_BY_ALIAS["topo"].cache == {
        "knn_info",
        "topographic_product_statistics",
    }
    assert METRIC_BY_ALIAS["cadi"].cache == set()
    assert all(
        not isinstance(requirement, str)
        for metric in METRIC_BY_ALIAS.values()
        for requirement in metric.resources
    )


def test_last_run_info_is_separate_json_compatible_metadata():
    orig, emb, labels = _sample()
    runner = ZADU(_mixed_specs(), orig)

    scores = runner.measure(emb, labels)
    info = runner.last_run_info

    assert all("backend" not in score for score in scores)
    assert info["exact"] is True
    assert info["backend"] == "numpy"
    assert info["device"] == "cpu"
    assert info["estimated_cache_bytes"] == runner.estimated_cache_bytes
    assert info["total_seconds"] >= info["metric_seconds"] >= 0
    assert info["resource_seconds"] >= 0
    assert [entry["id"] for entry in info["metrics"]] == [
        "trustworthiness_continuity",
        "local_continuity_meta_criteria",
        "neighborhood_hit",
        "stress",
    ]

    orig_ranking = next(
        resource
        for resource in info["resources"]
        if resource["space"] == "orig" and resource["kind"] == "neighbor_ranking"
    )
    emb_ranking = next(
        resource
        for resource in info["resources"]
        if resource["space"] == "emb" and resource["kind"] == "neighbor_ranking"
    )
    assert orig_ranking["reused"] is True
    assert orig_ranking["built_in_run"] is False
    assert orig_ranking["consumers"] == [
        "trustworthiness_continuity",
        "local_continuity_meta_criteria",
    ]
    assert emb_ranking["built_in_run"] is True
    assert emb_ranking["consumer_count"] == 3
    assert emb_ranking["first_consumer"] == 0
    assert emb_ranking["last_consumer"] == 2
    json.dumps(info)


def test_original_resources_are_reused_across_measure_calls(monkeypatch):
    orig, emb, labels = _sample()
    calls = []
    original_build = NumpyResourceProvider.build

    def wrapped_build(self, key, points, **kwargs):
        calls.append((key.space, key.kind))
        return original_build(self, key, points, **kwargs)

    monkeypatch.setattr(NumpyResourceProvider, "build", wrapped_build)
    runner = ZADU(_mixed_specs(), orig)
    first = runner.measure(emb, labels)
    second = runner.measure(emb + 0.01, labels)

    assert len(first) == len(second) == 4
    for kind in (ResourceKind.DISTANCE_MATRIX, ResourceKind.NEIGHBOR_RANKING):
        assert calls.count((Space.ORIGINAL, kind)) == 1
        assert calls.count((Space.EMBEDDED, kind)) == 2


def test_planner_records_mixed_numpy_and_faiss_resource_providers():
    orig, emb, _ = _sample()

    knn_only = ZADU([{"id": "proc", "params": {"k": 5}}], orig)
    knn_only.measure(emb)
    assert {
        resource["provider"] for resource in knn_only.last_run_info["resources"]
    } == {"faiss"}

    selected = ZADU([{"id": "topo", "params": {"k": 5}}], orig)
    selected.measure(emb)
    providers = {
        (resource["kind"], resource["provider"])
        for resource in selected.last_run_info["resources"]
    }
    assert providers == {
        ("stable_knn", "scipy"),
        ("topographic_product_statistics", "numpy"),
    }


def test_metrics_without_resources_produce_an_empty_resource_plan():
    orig, emb, labels = _sample()
    runner = ZADU([{"id": "dsc"}], orig)

    score = runner.measure(emb, labels)

    assert "distance_consistency" in score[0]
    assert runner.estimated_cache_bytes == 0
    assert runner.last_run_info["resources"] == []
