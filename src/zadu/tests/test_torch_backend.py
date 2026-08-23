import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from zadu import ZADU, EmbeddingExecutionError, ExecutionConfig
from zadu.backends import torch_backend
from zadu.backends.base import BatchResourceError
from zadu.backends.numpy_backend import NumpyResourceProvider
from zadu.engine.resources import NeighborRanking, ResourceKey, ResourceKind, Space
from zadu.measures import (
    local_continuity_meta_criteria,
    neighborhood_hit,
    procrustes,
    topographic_product,
    trustworthiness_continuity,
)
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


def _build_batch(
    provider,
    kind,
    points_batch,
    *,
    working_memory_bytes,
    k=None,
    distance_matrices=None,
):
    if distance_matrices is None:
        distance_matrices = [None] * len(points_batch)
    return provider.build_batch(
        ResourceKey(kind, Space.EMBEDDED, k),
        points_batch,
        distance_matrices=distance_matrices,
        condensed_pairs=[None] * len(points_batch),
        working_memory_bytes=working_memory_bytes,
        geodesic=False,
    )


def _neighbor_metric_reference_scores(orig, emb, labels):
    orig_distances = cdist(orig, orig)
    emb_distances = cdist(emb, emb)
    orig_indices, orig_ranking = knn.knn_with_ranking(
        orig,
        7,
        distance_matrix=orig_distances,
    )
    emb_indices, emb_ranking = knn.knn_with_ranking(
        emb,
        7,
        distance_matrix=emb_distances,
    )
    return [
        trustworthiness_continuity.measure(
            orig,
            emb,
            k=5,
            knn_ranking_info=(
                orig_indices[:, :5],
                orig_ranking,
                emb_indices[:, :5],
                emb_ranking,
            ),
        ),
        local_continuity_meta_criteria.measure(
            orig,
            emb,
            k=7,
            knn_info=(orig_indices, emb_indices),
        ),
        neighborhood_hit.measure(
            emb,
            labels,
            k=4,
            knn_emb_info=emb_indices[:, :4],
        ),
        procrustes.measure(
            orig,
            emb,
            k=3,
            knn_info=(orig_indices[:, :3], emb_indices[:, :3]),
        ),
        topographic_product.measure(
            orig,
            emb,
            k=4,
            distance_matrices=(orig_distances, emb_distances),
            knn_info=(orig_indices[:, :4], emb_indices[:, :4]),
        ),
    ]


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


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_selected_ranks_match_numpy_on_ties_and_bounded_blocks(device):
    orig = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 2.0],
            [-2.0, -2.0],
        ],
        dtype=np.float32,
    )
    emb = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ],
        dtype=np.float32,
    )
    template = ZADU(
        [
            {"id": "tnc", "params": {"k": 3}},
            {"id": "mrre", "params": {"k": 4}},
        ],
        orig,
    )._execution_plan.rank_comparison_plan
    assert template is not None
    bytes_per_row = orig.shape[0] * 24 + max(template.membership_ks) ** 2
    fixed_working_bytes = 16 * orig.shape[0] * template.k
    plan = replace(
        template,
        block_rows=2,
        work_budget_bytes=fixed_working_bytes + 2 * bytes_per_row,
        working_bytes=fixed_working_bytes + 2 * bytes_per_row,
        fixed_working_bytes=fixed_working_bytes,
    )
    orig_knn = knn.knn_from_distance_matrix(cdist(orig, orig), plan.k).astype(np.int32)
    expected = (
        NumpyResourceProvider()
        .build_rank_comparisons(
            plan,
            orig,
            emb,
            orig_knn=orig_knn,
            orig_distance_matrix=None,
            emb_distance_matrix=None,
        )
        .value
    )
    dtype = "float32" if device == "mps" else "float64"
    built = _torch_provider(device=device, dtype=dtype).build_rank_comparisons(
        plan,
        orig,
        emb,
        orig_knn=orig_knn,
        orig_distance_matrix=None,
        emb_distance_matrix=None,
    )

    for name in (
        "orig_ranks_of_emb",
        "emb_ranks_of_orig",
        "orig_indices",
        "emb_indices",
    ):
        np.testing.assert_array_equal(
            getattr(built.value, name), getattr(expected, name)
        )
    for requested_k in plan.membership_ks:
        np.testing.assert_array_equal(
            built.value.emb_in_orig[requested_k],
            expected.emb_in_orig[requested_k],
        )
        np.testing.assert_array_equal(
            built.value.orig_in_emb[requested_k],
            expected.orig_in_emb[requested_k],
        )
    assert built.implementation == "torch"
    assert built.details["provider_fallback"] is False
    assert built.details["block_rows"] == 2
    assert built.details["block_count"] == 4
    assert built.details["tie_break"] == "stable_column_index"
    assert built.details["working_bytes"] == (fixed_working_bytes + 2 * bytes_per_row)
    json.dumps(built.details)


def test_torch_selected_ranks_record_geodesic_fallback():
    _torch_provider()

    orig = np.asarray([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [-0.1, -0.3]])
    emb = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    runner = ZADU(
        [{"id": "tnc", "params": {"k": 1}}],
        orig,
        geodesic=True,
        execution=ExecutionConfig(backend="torch", device="cpu", dtype="float64"),
    )

    runner.measure(emb)

    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["provider"] == "numpy"
    assert comparison["details"]["requested_provider"] == "torch"
    assert comparison["details"]["provider_fallback"] is True
    assert comparison["details"]["fallback_reason"] == "geodesic_not_supported"


def test_torch_selected_ranks_support_mrre_without_membership_masks():
    _torch_provider()

    rng = np.random.default_rng(771)
    orig = rng.normal(size=(40, 5))
    emb = orig @ rng.normal(size=(5, 2))
    specs = [{"id": "mrre", "params": {"k": 5}}]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="torch", device="cpu", dtype="float64"),
    )

    actual = runner.measure(emb)

    assert actual == expected
    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["provider"] == "torch"
    assert comparison["details"]["membership_ks"] == []


@pytest.mark.parametrize(
    ("device", "dtype", "rel", "abs_"),
    [("cpu", "float64", 1e-12, 1e-12), ("mps", "float32", 4e-5, 4e-6)],
)
def test_zadu_torch_routes_neighbor_metrics_and_preserves_scores(
    device, dtype, rel, abs_
):
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
    expected = _neighbor_metric_reference_scores(orig, emb, labels)
    _torch_provider(device=device, dtype=dtype)
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
    assert {resource["kind"] for resource in neighbor_resources} == {
        "knn",
        "stable_knn",
    }
    rank_comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert rank_comparison["provider"] == "torch"
    assert rank_comparison["details"]["provider_fallback"] is False
    assert rank_comparison["details"]["algorithm"] == ("torch_blockwise_selected_ranks")


@pytest.mark.parametrize(
    ("device", "dtype", "rel", "abs_"),
    [("cpu", "float64", 1e-12, 1e-12), ("mps", "float32", 4e-5, 4e-6)],
)
def test_zadu_torch_dense_pair_plan_shares_distances_with_selected_ranks(
    device, dtype, rel, abs_
):
    rng = np.random.default_rng(770)
    orig = rng.normal(size=(64, 6))
    emb = orig @ rng.normal(size=(6, 2))
    specs = [
        {"id": "stress"},
        {"id": "tnc", "params": {"k": 5}},
    ]
    expected = ZADU(specs, orig).measure(emb)
    _torch_provider(device=device, dtype=dtype)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="torch", device=device, dtype=dtype),
    )

    actual = runner.measure(emb)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=rel, abs=abs_
            )
    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["provider"] == "torch"
    assert comparison["details"]["provider_fallback"] is False
    assert comparison["details"]["original_distance_source"] == (
        "shared_distance_matrix"
    )
    assert comparison["details"]["embedded_distance_source"] == (
        "shared_distance_matrix"
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
    rank_plan = plan.rank_comparison_plan
    assert rank_plan is not None
    rank_bytes_per_row = orig.shape[0] * 24 + max(rank_plan.membership_ks) ** 2
    assert rank_plan.fixed_working_bytes == 16 * orig.shape[0] * rank_plan.k
    assert rank_plan.working_bytes == (
        rank_plan.fixed_working_bytes + rank_plan.block_rows * rank_bytes_per_row
    )

    bounded_budget = (
        plan.estimated_cache_bytes
        + rank_plan.fixed_working_bytes
        + 2 * rank_bytes_per_row
    )
    bounded = ZADU(
        [
            {"id": "tnc", "params": {"k": 5}},
            {"id": "lcmc", "params": {"k": 7}},
            {"id": "topo", "params": {"k": 4}},
        ],
        orig,
        execution=ExecutionConfig(
            backend="torch",
            device="cpu",
            dtype="float64",
            memory_budget=bounded_budget,
        ),
    )
    assert bounded._execution_plan.rank_comparison_plan.block_rows == 2
    assert bounded._execution_plan.planned_peak_bytes <= bounded_budget


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_torch_native_batch_pairwise_and_ranking_match_independent_results(device):
    rng = np.random.default_rng(79)
    points_batch = [rng.normal(size=(37, 5)).astype(np.float32) for _ in range(3)]
    points_batch[0][1] = points_batch[0][0]
    dtype = "float32" if device == "mps" else "float64"
    provider = _torch_provider(device=device, dtype=dtype)
    pairwise_bytes = points_batch[0].shape[0] ** 2 * np.dtype(dtype).itemsize * 4

    matrices = _build_batch(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points_batch,
        working_memory_bytes=pairwise_bytes,
    )

    assert len(matrices) == len(points_batch)
    tolerance = 3e-5 if dtype == "float32" else 1e-12
    for batch_index, (built, points) in enumerate(
        zip(matrices, points_batch, strict=True)
    ):
        np.testing.assert_allclose(
            built.value,
            cdist(points.astype(dtype), points.astype(dtype)).astype(dtype),
            rtol=tolerance,
            atol=tolerance / 10,
        )
        assert built.details["provider_batching"] is True
        assert built.details["batch_size"] == 3
        assert built.details["batch_index"] == batch_index
        assert built.details["batch_working_bytes"] == (
            3 * built.details["working_bytes"]
        )

    ranking_bytes = points_batch[0].shape[0] ** 2 * (
        4 * np.dtype(dtype).itemsize + 4 * np.dtype(np.int64).itemsize
    )
    rankings = _build_batch(
        provider,
        ResourceKind.NEIGHBOR_RANKING,
        points_batch,
        k=6,
        distance_matrices=[built.value for built in matrices],
        working_memory_bytes=ranking_bytes,
    )
    for built, points, matrix in zip(
        rankings,
        points_batch,
        [item.value for item in matrices],
        strict=True,
    ):
        expected_indices, expected_ranking = knn.knn_with_ranking(
            points,
            6,
            distance_matrix=matrix,
        )
        np.testing.assert_array_equal(built.value.indices, expected_indices)
        np.testing.assert_array_equal(built.value.ranking, expected_ranking)
        assert built.details["distance_source"] == "shared_distance_matrix_batch"
        assert (
            built.details["timings"]["compile_and_first_execution_seconds"]
            + built.details["timings"]["warm_execution_seconds"]
            > 0
        )
        json.dumps(built.details)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_zadu_torch_measure_many_uses_native_batches_and_preserves_scores(device):
    dtype = "float32" if device == "mps" else "float64"
    _torch_provider(device=device, dtype=dtype)
    rng = np.random.default_rng(80)
    orig = rng.normal(size=(64, 7))
    embeddings = [
        orig @ rng.normal(size=(7, 2)) + rng.normal(scale=0.02, size=(64, 2))
        for _ in range(5)
    ]
    specs = [
        {"id": "stress"},
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 7}},
        {"id": "topo", "params": {"k": 4}},
    ]
    sequential_runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="torch", device=device, dtype=dtype),
    )
    expected = sequential_runner.measure_many(embeddings)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="torch",
            device=device,
            dtype=dtype,
            embedding_workers=3,
        ),
    )

    actual = runner.measure_many(embeddings)

    tolerance = 4e-5 if dtype == "float32" else 1e-12
    for actual_run, expected_run in zip(actual, expected, strict=True):
        for actual_score, expected_score in zip(
            actual_run,
            expected_run,
            strict=True,
        ):
            assert actual_score.keys() == expected_score.keys()
            for name in actual_score:
                assert actual_score[name] == pytest.approx(
                    expected_score[name],
                    rel=tolerance,
                    abs=tolerance / 10,
                )
    info = runner.last_run_info
    assert info["batch_strategy"] == "provider_native_batch"
    assert info["provider_batching"] is True
    assert info["requested_workers"] == 3
    assert info["effective_workers"] == 1
    assert info["native_batch_size"] == 3
    assert info["worker_limit_reason"] is None
    assert [run["embedding_index"] for run in info["runs"]] == list(range(5))
    embedded_resources = [
        resource
        for run in info["runs"]
        for resource in run["resources"]
        if resource["space"] == "emb" and resource["provider"] == "torch"
    ]
    assert embedded_resources
    assert all(
        resource["details"].get("provider_batching") is True
        for resource in embedded_resources
    )
    assert {resource["details"]["batch_size"] for resource in embedded_resources} == {
        2,
        3,
    }
    assert info["total_seconds"] >= sum(run["total_seconds"] for run in info["runs"])
    assert runner.emb is embeddings[-1]
    json.dumps(info)


def test_torch_native_batch_size_is_capped_by_memory_budget():
    _torch_provider()
    rng = np.random.default_rng(81)
    orig = rng.normal(size=(56, 6))
    embeddings = [rng.normal(size=(56, 2)) for _ in range(4)]
    config = ExecutionConfig(backend="torch", device="cpu", dtype="float64")
    baseline = ZADU([{"id": "stress"}], orig, execution=config)
    plan = baseline._execution_plan
    batch_input_bytes = embeddings[0].size * np.dtype(np.float64).itemsize
    budget = plan.original_cache_bytes + 2 * (
        plan.per_embedding_peak_bytes + batch_input_bytes
    )
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="torch",
            device="cpu",
            dtype="float64",
            embedding_workers=4,
            memory_budget=budget,
        ),
    )

    runner.measure_many(embeddings)

    info = runner.last_run_info
    assert info["provider_batching"] is True
    assert info["native_batch_size"] == 2
    assert info["worker_limit_reason"] == "memory_budget"
    assert info["planned_peak_bytes"] <= budget
    assert info["per_embedding_peak_bytes"] == (
        plan.per_embedding_peak_bytes + batch_input_bytes
    )


def test_torch_measure_many_shape_mismatch_falls_back_to_ordered_sequential():
    _torch_provider()
    rng = np.random.default_rng(82)
    orig = rng.normal(size=(40, 5))
    embeddings = [rng.normal(size=(40, 2)), rng.normal(size=(40, 3))]
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="torch",
            device="cpu",
            dtype="float64",
            embedding_workers=3,
        ),
    )

    results = runner.measure_many(embeddings)

    assert len(results) == 2
    assert runner.last_run_info["batch_strategy"] == "sequential_shared_original"
    assert runner.last_run_info["provider_batching"] is False
    assert runner.last_run_info["effective_workers"] == 1
    assert runner.last_run_info["worker_limit_reason"] == "embedding_shape_mismatch"


def test_torch_native_batch_overflow_reports_the_exact_embedding_index():
    _torch_provider(device="cpu", dtype="float32")
    rng = np.random.default_rng(83)
    orig = rng.normal(size=(32, 4))
    embeddings = [rng.normal(size=(32, 2)) for _ in range(3)]
    embeddings[1][0, 0] = np.finfo(np.float64).max
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="torch",
            device="cpu",
            dtype="float32",
            embedding_workers=3,
        ),
    )

    with pytest.raises(EmbeddingExecutionError) as error:
        runner.measure_many(embeddings)

    assert error.value.embedding_index == 1
    assert isinstance(error.value.__cause__, BatchResourceError)
    assert "cannot be represented" in str(error.value.__cause__)
