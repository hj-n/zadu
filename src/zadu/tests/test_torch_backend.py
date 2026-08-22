import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from zadu import ZADU, ExecutionConfig
from zadu.backends import torch_backend
from zadu.engine.resources import NeighborRanking, ResourceKey, ResourceKind, Space
from zadu.measures.utils import knn


def _torch_provider(*, device="cpu", dtype="float64"):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is unavailable")
    return torch_backend.TorchResourceProvider(device=device, dtype=dtype)


def _build(
    provider,
    kind,
    points,
    *,
    working_memory_bytes,
    geodesic=False,
    space=Space.ORIGINAL,
    k=None,
    distance_matrix=None,
):
    return provider.build(
        ResourceKey(kind, space, k),
        points,
        distance_matrix=distance_matrix,
        condensed_pairs=None,
        working_memory_bytes=working_memory_bytes,
        geodesic=geodesic,
    )


def test_importing_base_package_does_not_import_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np; from zadu import ZADU; "
            "ZADU([], np.ones((2, 1))); assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_torch_extra_has_actionable_error(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(torch_backend, "import_module", missing_import)

    with pytest.raises(ImportError, match=r"pip install 'zadu\[torch\]'"):
        torch_backend.TorchResourceProvider(device="cpu", dtype="float64")


def test_torch_cpu_float64_pairwise_is_exact_and_zero_copy():
    rng = np.random.default_rng(71)
    points = rng.normal(size=(47, 8))
    provider = _torch_provider()
    bytes_per_row = points.shape[0] * np.dtype(np.float64).itemsize * 4

    built = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=4 * bytes_per_row,
    )

    assert built.implementation == "torch"
    assert built.value.dtype == np.float64
    np.testing.assert_allclose(
        built.value, cdist(points, points), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(built.value, built.value.T)
    np.testing.assert_array_equal(np.diag(built.value), 0)
    assert built.details["device"] == "cpu"
    assert built.details["compute_dtype"] == "float64"
    assert built.details["block_rows"] == 4
    assert built.details["input_zero_copy"] is True
    assert built.details["input_cast_copy"] is False
    assert built.details["provider_fallback"] is False
    assert built.details["timings"]["compile_and_first_execution_seconds"] >= 0
    assert built.details["timings"]["warm_execution_seconds"] > 0
    json.dumps(built.details)

    warm = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=4 * bytes_per_row,
    )
    assert warm.details["input_reused"] is True
    assert warm.details["timings"]["compile_and_first_execution_seconds"] == 0


def test_torch_condensed_pairs_are_block_bounded_and_keep_scipy_order():
    rng = np.random.default_rng(72)
    points = rng.normal(size=(41, 7))
    provider = _torch_provider()
    bytes_per_row = points.shape[0] * np.dtype(np.float64).itemsize * 4

    built = _build(
        provider,
        ResourceKind.CONDENSED_PAIRS,
        points,
        working_memory_bytes=3 * bytes_per_row,
    )

    assert built.details["block_rows"] == 3
    assert built.details["block_count"] == 14
    assert built.details["working_bytes"] == 3 * bytes_per_row
    np.testing.assert_allclose(built.value, pdist(points), rtol=1e-12, atol=1e-12)


def test_torch_mps_float32_pairwise_matches_scipy():
    rng = np.random.default_rng(73)
    points = rng.normal(size=(64, 9))
    provider = _torch_provider(device="mps", dtype="float32")
    working_bytes = points.shape[0] ** 2 * np.dtype(np.float32).itemsize * 4

    built = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=working_bytes,
    )
    expected = cdist(points.astype(np.float32), points.astype(np.float32)).astype(
        np.float32
    )

    assert built.details["device"] == "mps"
    assert built.details["input_zero_copy"] is False
    assert built.details["input_cast_copy"] is True
    np.testing.assert_allclose(built.value, expected, rtol=3e-5, atol=3e-6)


def test_torch_rejects_unavailable_or_unsupported_devices():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
            torch_backend.TorchResourceProvider(device="cuda", dtype="float32")
    with pytest.raises(ValueError, match="MPS requires dtype='float32'"):
        torch_backend.TorchResourceProvider(device="mps", dtype="float64")


def test_torch_float32_rejects_values_that_overflow_during_cast():
    provider = _torch_provider(device="cpu", dtype="float32")
    points = np.array([[0.0, 0.0], [np.finfo(np.float64).max, 1.0]])

    with pytest.raises(OverflowError, match=r"cannot be represented.*float32"):
        _build(
            provider,
            ResourceKind.DISTANCE_MATRIX,
            points,
            working_memory_bytes=1024,
        )


def test_torch_geodesic_resource_falls_back_with_reason():
    points = np.array([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.3]])
    provider = _torch_provider()

    built = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=1024,
        geodesic=True,
    )

    assert built.implementation == "numpy"
    assert built.details["requested_provider"] == "torch"
    assert built.details["provider_fallback"] is True
    assert built.details["fallback_reason"] == "geodesic_not_supported"


@pytest.mark.parametrize(
    ("device", "dtype", "rel", "abs_"),
    [("cpu", "float64", 1e-12, 1e-12), ("mps", "float32", 3e-5, 3e-6)],
)
def test_zadu_torch_routes_pairwise_resources_and_preserves_scores(
    device, dtype, rel, abs_
):
    _torch_provider(device=device, dtype=dtype)
    rng = np.random.default_rng(74)
    orig = rng.normal(size=(96, 8))
    emb = orig @ rng.normal(size=(8, 2))
    specs = [{"id": "stress"}, {"id": "dtm"}]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="torch", device=device, dtype=dtype),
    )

    actual = runner.measure(emb)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score.keys() == expected_score.keys()
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=rel, abs=abs_
            )
    info = runner.last_run_info
    assert info["backend"] == "torch"
    assert info["device"] == device
    assert any(resource["provider"] == "torch" for resource in info["resources"])
    fallbacks = [
        resource
        for resource in info["resources"]
        if resource["details"].get("provider_fallback")
    ]
    assert fallbacks
    assert all(
        resource["details"]["requested_provider"] == "torch" for resource in fallbacks
    )
    json.dumps(info)


def test_torch_plan_accounts_for_pairwise_working_memory():
    _torch_provider()
    rng = np.random.default_rng(75)
    orig = rng.normal(size=(80, 6))
    runner = ZADU(
        [{"id": "stress"}, {"id": "dtm"}],
        orig,
        execution=ExecutionConfig(backend="torch", device="cpu", dtype="float64"),
    )
    plan = runner._execution_plan
    pairwise_keys = [
        key
        for key in plan.resources
        if key.kind in {ResourceKind.DISTANCE_MATRIX, ResourceKind.CONDENSED_PAIRS}
    ]

    assert pairwise_keys
    assert all(key in plan.resource_working_bytes for key in pairwise_keys)
    assert plan.planned_peak_bytes >= (
        plan.estimated_cache_bytes
        + max(plan.resource_working_bytes[key] for key in pairwise_keys)
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_full_ranking_preserves_self_and_stable_duplicate_ties(device):
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    dtype = "float32" if device == "mps" else "float64"
    provider = _torch_provider(device=device, dtype=dtype)
    k = 4
    bytes_per_row = points.shape[0] * (
        4 * np.dtype(dtype).itemsize + 4 * np.dtype(np.int64).itemsize
    )

    built = _build(
        provider,
        ResourceKind.NEIGHBOR_RANKING,
        points,
        k=k,
        working_memory_bytes=3 * bytes_per_row,
    )
    expected_indices, expected_ranking = knn.knn_with_ranking(points, k)

    assert built.implementation == "torch"
    assert isinstance(built.value, NeighborRanking)
    np.testing.assert_array_equal(built.value.indices, expected_indices)
    np.testing.assert_array_equal(built.value.ranking, expected_ranking)
    assert built.value.indices.dtype == np.int32
    assert built.value.ranking.dtype == np.int32
    assert built.details["algorithm"] == "torch_blockwise_stable_full_ranking"
    assert built.details["block_rows"] == 3
    assert built.details["block_count"] == 3
    assert built.details["tie_break"] == "stable_column_index"
    assert built.details["self_exclusion"] == "forced_rank_zero_then_removed"
    assert built.details["distance_source"] == "fused_blockwise_pairwise"
    assert built.details["provider_fallback"] is False
    json.dumps(built.details)


@pytest.mark.parametrize("kind", [ResourceKind.KNN, ResourceKind.STABLE_KNN])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_exact_topk_matches_stable_numpy_tie_order(kind, device):
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    dtype = "float32" if device == "mps" else "float64"
    provider = _torch_provider(device=device, dtype=dtype)
    k = 3
    bytes_per_row = points.shape[0] * (
        4 * np.dtype(dtype).itemsize + 2 * np.dtype(np.int64).itemsize
    )

    built = _build(
        provider,
        kind,
        points,
        k=k,
        working_memory_bytes=bytes_per_row,
    )
    expected = knn.knn_from_distance_matrix(cdist(points, points), k)

    np.testing.assert_array_equal(built.value, expected)
    assert built.value.dtype == np.int32
    assert built.details["algorithm"] == "torch_blockwise_stable_exact_topk"
    assert built.details["top_k_algorithm"] == "stable_full_order_prefix"
    assert built.details["block_rows"] == 1
    assert built.details["block_count"] == points.shape[0]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_ranking_reuses_planned_distance_matrix(device):
    rng = np.random.default_rng(76)
    points = rng.normal(size=(48, 5)).astype(np.float32)
    dtype = "float32" if device == "mps" else "float64"
    provider = _torch_provider(device=device, dtype=dtype)
    pairwise_bytes = points.shape[0] ** 2 * np.dtype(dtype).itemsize * 4
    matrix = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=pairwise_bytes,
    )
    ranking_bytes = points.shape[0] ** 2 * (
        4 * np.dtype(dtype).itemsize + 4 * np.dtype(np.int64).itemsize
    )

    ranked = _build(
        provider,
        ResourceKind.NEIGHBOR_RANKING,
        points,
        k=7,
        distance_matrix=matrix.value,
        working_memory_bytes=ranking_bytes,
    )
    expected_indices, expected_ranking = knn.knn_with_ranking(
        points,
        7,
        distance_matrix=matrix.value,
    )

    np.testing.assert_array_equal(ranked.value.indices, expected_indices)
    np.testing.assert_array_equal(ranked.value.ranking, expected_ranking)
    assert ranked.details["distance_source"] == "shared_distance_matrix"
    assert ranked.details["distance_zero_copy"] is False


@pytest.mark.parametrize(
    ("device", "dtype", "rel", "abs_"),
    [("cpu", "float64", 1e-12, 1e-12), ("mps", "float32", 4e-5, 4e-6)],
)
def test_zadu_torch_routes_neighbor_metrics_and_preserves_scores(
    device, dtype, rel, abs_
):
    _torch_provider(device=device, dtype=dtype)
    rng = np.random.default_rng(77)
    orig = rng.normal(size=(72, 8))
    emb = orig @ rng.normal(size=(8, 2)) + rng.normal(scale=0.05, size=(72, 2))
    labels = np.arange(orig.shape[0]) % 4
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 7}},
        {"id": "nh", "params": {"k": 4}},
        {"id": "proc", "params": {"k": 3}},
        {"id": "topo", "params": {"k": 4}},
    ]
    expected = ZADU(specs, orig).measure(emb, labels)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="torch", device=device, dtype=dtype),
    )

    actual = runner.measure(emb, labels)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score.keys() == expected_score.keys()
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=rel, abs=abs_
            )
    neighbor_resources = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] in {"knn", "stable_knn", "neighbor_ranking"}
    ]
    assert neighbor_resources
    assert all(resource["provider"] == "torch" for resource in neighbor_resources)
    assert all(
        resource["details"]["provider_fallback"] is False
        for resource in neighbor_resources
    )


def test_torch_plan_accounts_for_neighbor_working_memory():
    _torch_provider()
    rng = np.random.default_rng(78)
    orig = rng.normal(size=(80, 6))
    runner = ZADU(
        [
            {"id": "tnc", "params": {"k": 5}},
            {"id": "lcmc", "params": {"k": 7}},
            {"id": "topo", "params": {"k": 4}},
        ],
        orig,
        execution=ExecutionConfig(backend="torch", device="cpu", dtype="float64"),
    )
    plan = runner._execution_plan
    neighbor_keys = [
        key
        for key in plan.resources
        if key.kind
        in {ResourceKind.KNN, ResourceKind.STABLE_KNN, ResourceKind.NEIGHBOR_RANKING}
    ]

    assert neighbor_keys
    assert all(key in plan.resource_working_bytes for key in neighbor_keys)
    assert plan.planned_peak_bytes >= (
        plan.estimated_cache_bytes
        + max(plan.resource_working_bytes[key] for key in neighbor_keys)
    )
