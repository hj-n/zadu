import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from zadu import ZADU, EmbeddingExecutionError, ExecutionConfig
from zadu.backends import mlx_backend
from zadu.backends.base import BatchResourceError
from zadu.backends.numpy_backend import NumpyResourceProvider
from zadu.engine.resources import NeighborRanking, ResourceKey, ResourceKind, Space
from zadu.measures.utils import knn


def _mlx_provider(*, device="gpu", dtype="float32"):
    mx = pytest.importorskip("mlx.core")
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("MLX Metal GPU is unavailable")
    return mlx_backend.MlxResourceProvider(device=device, dtype=dtype)


def _build(
    provider,
    kind,
    points,
    *,
    working_memory_bytes,
    geodesic=False,
    k=None,
    distance_matrix=None,
    space=Space.ORIGINAL,
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


def test_importing_base_package_does_not_import_mlx():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np; from zadu import ZADU; "
            "ZADU([], np.ones((2, 1))); assert 'mlx' not in sys.modules; "
            "assert 'mlx.core' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_mlx_extra_has_actionable_error(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError("No module named 'mlx'", name="mlx")

    monkeypatch.setattr(mlx_backend, "import_module", missing_import)

    with pytest.raises(ImportError, match=r"pip install 'zadu\[mlx\]'"):
        mlx_backend.MlxResourceProvider(device="gpu", dtype="float32")


def test_mlx_float32_distance_matrix_is_symmetric_and_matches_scipy():
    rng = np.random.default_rng(11)
    points = rng.normal(size=(64, 9))
    provider = _mlx_provider()
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

    assert built.implementation == "mlx"
    assert built.value.dtype == np.float32
    np.testing.assert_allclose(built.value, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(built.value, built.value.T)
    np.testing.assert_array_equal(np.diag(built.value), 0)
    assert built.details["provider_fallback"] is False
    assert built.details["device"] == "gpu"
    assert built.details["compute_dtype"] == "float32"
    assert built.details["working_bytes"] <= working_bytes
    assert built.details["timings"]["compile_and_first_execution_seconds"] >= 0
    assert built.details["timings"]["warm_execution_seconds"] > 0
    json.dumps(built.details)

    warm = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=working_bytes,
    )
    assert warm.details["timings"]["compile_and_first_execution_seconds"] == 0


def test_mlx_condensed_pairs_are_block_bounded_and_keep_scipy_order():
    rng = np.random.default_rng(12)
    points = rng.normal(size=(41, 7))
    provider = _mlx_provider()
    bytes_per_row = points.shape[0] * np.dtype(np.float32).itemsize * 4
    working_bytes = 3 * bytes_per_row

    built = _build(
        provider,
        ResourceKind.CONDENSED_PAIRS,
        points,
        working_memory_bytes=working_bytes,
    )

    assert built.details["block_rows"] == 3
    assert built.details["block_count"] == 14
    assert built.details["working_bytes"] == working_bytes
    np.testing.assert_allclose(
        built.value,
        pdist(points.astype(np.float32)),
        rtol=2e-6,
        atol=2e-6,
    )


def test_mlx_cpu_float64_does_not_downgrade_precision():
    rng = np.random.default_rng(13)
    points = rng.normal(size=(24, 5))
    provider = _mlx_provider(device="cpu", dtype="float64")
    working_bytes = points.shape[0] ** 2 * np.dtype(np.float64).itemsize * 4

    built = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=working_bytes,
    )

    assert built.value.dtype == np.float64
    assert built.details["compute_dtype"] == "float64"
    np.testing.assert_allclose(
        built.value, cdist(points, points), rtol=1e-12, atol=1e-12
    )


def test_mlx_cpu_float64_neighbors_work_in_a_fresh_default_gpu_process():
    pytest.importorskip("mlx.core")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy as np; "
            "from zadu.backends.mlx_backend import MlxResourceProvider; "
            "from zadu.engine.resources import ResourceKey, ResourceKind, Space; "
            "p=np.arange(24, dtype=np.float64).reshape(8, 3); "
            "r=MlxResourceProvider(device='cpu', dtype='float64').build("
            "ResourceKey(ResourceKind.STABLE_KNN, Space.ORIGINAL, 3), p, "
            "distance_matrix=None, condensed_pairs=None, "
            "working_memory_bytes=8192, geodesic=False); "
            "assert r.implementation == 'mlx'; assert r.value.shape == (8, 3)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_mlx_float32_rejects_values_that_overflow_during_explicit_cast():
    provider = _mlx_provider()
    points = np.array([[0.0, 0.0], [np.finfo(np.float64).max, 1.0]])

    with pytest.raises(OverflowError, match=r"cannot be represented.*float32"):
        _build(
            provider,
            ResourceKind.DISTANCE_MATRIX,
            points,
            working_memory_bytes=1024,
        )


def test_mlx_geodesic_resource_falls_back_with_reason():
    points = np.array([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.3]])
    provider = _mlx_provider()

    built = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=1024,
        geodesic=True,
    )

    assert built.implementation == "numpy"
    assert built.details["requested_provider"] == "mlx"
    assert built.details["provider_fallback"] is True
    assert built.details["fallback_reason"] == "geodesic_not_supported"


def test_zadu_mlx_routes_pairwise_resources_and_preserves_scores():
    _mlx_provider()
    rng = np.random.default_rng(14)
    orig = rng.normal(size=(96, 8))
    emb = orig @ rng.normal(size=(8, 2))
    specs = [{"id": "stress"}, {"id": "dtm"}]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
        ),
    )

    actual = runner.measure(emb)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score.keys() == expected_score.keys()
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=2e-5, abs=2e-6
            )
    info = runner.last_run_info
    assert info["backend"] == "mlx"
    assert info["device"] == "gpu"
    assert info["dtype"] == "float32"
    assert any(resource["provider"] == "mlx" for resource in info["resources"])
    fallbacks = [
        resource
        for resource in info["resources"]
        if resource["details"].get("provider_fallback")
    ]
    assert fallbacks
    assert all(
        resource["details"]["requested_provider"] == "mlx" for resource in fallbacks
    )
    assert set(info["provider_timings"]) == {
        "input_transfer_seconds",
        "compile_and_first_execution_seconds",
        "warm_execution_seconds",
        "output_transfer_seconds",
    }
    json.dumps(info)


def test_mlx_plan_accounts_for_pairwise_working_memory():
    _mlx_provider()
    rng = np.random.default_rng(15)
    orig = rng.normal(size=(80, 6))
    runner = ZADU(
        [{"id": "stress"}, {"id": "dtm"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
        ),
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


def test_mlx_memory_budget_caps_blocks_and_rejects_less_than_one_row():
    _mlx_provider()
    rng = np.random.default_rng(16)
    orig = rng.normal(size=(80, 6))
    config = ExecutionConfig(backend="mlx", device="gpu", dtype="float32")
    baseline = ZADU([{"id": "stress"}], orig, execution=config)
    bytes_per_row = orig.shape[0] * np.dtype(np.float32).itemsize * 4
    budget = baseline._execution_plan.estimated_cache_bytes + 3 * bytes_per_row
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            memory_budget=budget,
        ),
    )

    runner.measure(orig[:, :2])

    mlx_resources = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["provider"] == "mlx"
    ]
    assert mlx_resources
    assert all(resource["details"]["block_rows"] == 3 for resource in mlx_resources)
    assert runner.last_run_info["planned_peak_bytes"] <= budget

    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            [{"id": "stress"}],
            orig,
            execution=ExecutionConfig(
                backend="mlx",
                device="gpu",
                dtype="float32",
                memory_budget=(
                    baseline._execution_plan.estimated_cache_bytes + bytes_per_row - 1
                ),
            ),
        )


def test_mlx_full_ranking_preserves_self_and_stable_duplicate_ties():
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
    provider = _mlx_provider()
    k = 4
    bytes_per_row = points.shape[0] * (4 * 4 + 4 * 4)
    built = _build(
        provider,
        ResourceKind.NEIGHBOR_RANKING,
        points,
        k=k,
        working_memory_bytes=3 * bytes_per_row,
    )
    expected_indices, expected_ranking = knn.knn_with_ranking(points, k)

    assert built.implementation == "mlx"
    assert isinstance(built.value, NeighborRanking)
    np.testing.assert_array_equal(built.value.indices, expected_indices)
    np.testing.assert_array_equal(built.value.ranking, expected_ranking)
    assert built.value.indices.dtype == np.int32
    assert built.value.ranking.dtype == np.int32
    assert built.details["algorithm"] == "compiled_blockwise_stable_full_ranking"
    assert built.details["block_rows"] == 3
    assert built.details["block_count"] == 3
    assert built.details["tie_break"] == "stable_column_index"
    assert built.details["self_exclusion"] == "forced_rank_zero_then_removed"
    assert built.details["distance_source"] == "fused_blockwise_pairwise"
    assert built.details["provider_fallback"] is False
    json.dumps(built.details)


def test_mlx_cpu_float64_full_ranking_preserves_exact_order():
    rng = np.random.default_rng(170)
    points = rng.normal(size=(32, 7))
    provider = _mlx_provider(device="cpu", dtype="float64")
    bytes_per_row = points.shape[0] * (4 * 8 + 4 * 4)
    built = _build(
        provider,
        ResourceKind.NEIGHBOR_RANKING,
        points,
        k=6,
        working_memory_bytes=bytes_per_row * points.shape[0],
    )
    expected_indices, expected_ranking = knn.knn_with_ranking(points, 6)

    np.testing.assert_array_equal(built.value.indices, expected_indices)
    np.testing.assert_array_equal(built.value.ranking, expected_ranking)
    assert built.details["compute_dtype"] == "float64"
    assert built.details["input_zero_copy"] is False
    assert built.details["input_cast_copy"] is False


@pytest.mark.parametrize("kind", [ResourceKind.KNN, ResourceKind.STABLE_KNN])
def test_mlx_exact_topk_matches_stable_numpy_tie_order(kind):
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
    provider = _mlx_provider()
    k = 3
    bytes_per_row = points.shape[0] * (4 * 4 + 2 * 4)
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
    assert built.details["algorithm"] == "compiled_blockwise_stable_exact_topk"
    assert built.details["top_k_algorithm"] == "stable_full_order_prefix"
    assert built.details["block_rows"] == 1
    assert built.details["block_count"] == points.shape[0]


def test_mlx_ranking_reuses_distance_matrix_through_unified_memory():
    rng = np.random.default_rng(17)
    points = rng.normal(size=(48, 5)).astype(np.float32)
    provider = _mlx_provider()
    pairwise_bytes = points.shape[0] ** 2 * 4 * 4
    matrix = _build(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points,
        working_memory_bytes=pairwise_bytes,
    )
    ranking_bytes = points.shape[0] ** 2 * (4 * 4 + 4 * 4)
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
    assert ranked.details["distance_zero_copy"] is True
    assert ranked.details["input_zero_copy"] is True
    assert ranked.details["input_reused"] is True
    assert ranked.details["output_zero_copy"] is True


def test_mlx_workspace_invalidation_refreshes_reused_mutable_input():
    points = np.asarray(
        [[0.0], [1.0], [3.0], [10.0]],
        dtype=np.float64,
    )
    provider = _mlx_provider()
    bytes_per_row = points.shape[0] * (4 * 4 + 2 * 4)
    first = _build(
        provider,
        ResourceKind.KNN,
        points,
        k=1,
        working_memory_bytes=bytes_per_row * points.shape[0],
    )
    points[1, 0] = 9.0
    provider.invalidate(Space.ORIGINAL)
    second = _build(
        provider,
        ResourceKind.KNN,
        points,
        k=1,
        working_memory_bytes=bytes_per_row * points.shape[0],
    )

    assert not np.array_equal(first.value, second.value)
    np.testing.assert_array_equal(
        second.value,
        knn.knn_from_distance_matrix(cdist(points, points), 1),
    )
    assert second.details["input_reused"] is False
    assert second.details["input_cast_copy"] is True


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", "float64"), ("gpu", "float32")],
)
def test_mlx_selected_ranks_match_numpy_on_ties_and_bounded_blocks(device, dtype):
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
    plan = replace(
        template,
        block_rows=2,
        work_budget_bytes=2 * bytes_per_row,
        working_bytes=2 * bytes_per_row,
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
    built = _mlx_provider(device=device, dtype=dtype).build_rank_comparisons(
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
    assert built.implementation == "mlx"
    assert built.details["provider_fallback"] is False
    assert built.details["block_rows"] == 2
    assert built.details["block_count"] == 4
    assert built.details["tie_break"] == "stable_column_index"
    assert built.details["working_bytes"] == 2 * bytes_per_row
    json.dumps(built.details)


def test_mlx_selected_ranks_record_geodesic_fallback():
    _mlx_provider(device="cpu", dtype="float64")

    orig = np.asarray([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [-0.1, -0.3]])
    emb = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    runner = ZADU(
        [{"id": "tnc", "params": {"k": 1}}],
        orig,
        geodesic=True,
        execution=ExecutionConfig(backend="mlx", device="cpu", dtype="float64"),
    )

    runner.measure(emb)

    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["provider"] == "numpy"
    assert comparison["details"]["requested_provider"] == "mlx"
    assert comparison["details"]["provider_fallback"] is True
    assert comparison["details"]["fallback_reason"] == "geodesic_not_supported"


def test_mlx_selected_ranks_support_mrre_without_membership_masks():
    _mlx_provider()

    rng = np.random.default_rng(181)
    orig = rng.normal(size=(40, 5))
    emb = orig @ rng.normal(size=(5, 2))
    specs = [{"id": "mrre", "params": {"k": 5}}]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(backend="mlx", device="gpu", dtype="float32"),
    )

    actual = runner.measure(emb)

    assert actual[0]["mrre_false"] == pytest.approx(
        expected[0]["mrre_false"], rel=3e-5, abs=3e-6
    )
    assert actual[0]["mrre_missing"] == pytest.approx(
        expected[0]["mrre_missing"], rel=3e-5, abs=3e-6
    )
    comparison = next(
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    )
    assert comparison["provider"] == "mlx"
    assert comparison["details"]["membership_ks"] == []


def test_zadu_mlx_routes_all_neighbor_metrics_and_preserves_scores():
    _mlx_provider()
    rng = np.random.default_rng(18)
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
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
        ),
    )

    actual = runner.measure(emb, labels)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score.keys() == expected_score.keys()
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=3e-5, abs=3e-6
            )
    neighbor_resources = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] in {"knn", "stable_knn", "neighbor_ranking"}
    ]
    assert neighbor_resources
    assert all(resource["provider"] == "mlx" for resource in neighbor_resources)
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
    assert rank_comparison["provider"] == "mlx"
    assert rank_comparison["details"]["provider_fallback"] is False
    assert rank_comparison["details"]["algorithm"] == (
        "compiled_blockwise_selected_ranks"
    )
    json.dumps(runner.last_run_info)


def test_zadu_mlx_mixed_pair_plan_keeps_selected_ranks_blockwise():
    _mlx_provider()
    rng = np.random.default_rng(180)
    orig = rng.normal(size=(64, 6))
    emb = orig @ rng.normal(size=(6, 2))
    specs = [
        {"id": "stress"},
        {"id": "tnc", "params": {"k": 5}},
    ]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
        ),
    )

    actual = runner.measure(emb)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        for name in actual_score:
            assert actual_score[name] == pytest.approx(
                expected_score[name], rel=3e-5, abs=3e-6
            )
    rank_comparisons = [
        resource
        for resource in runner.last_run_info["resources"]
        if resource["kind"] == "rank_comparisons"
    ]
    assert len(rank_comparisons) == 1
    comparison = rank_comparisons[0]
    assert comparison["details"]["original_distance_source"] == (
        "fused_blockwise_pairwise"
    )
    assert comparison["details"]["embedded_distance_source"] == (
        "fused_blockwise_pairwise"
    )
    assert comparison["provider"] == "mlx"
    assert comparison["details"]["provider_fallback"] is False
    assert runner.last_run_info["pair_strategy"] == "condensed"
    assert not any(
        resource["kind"] == "distance_matrix"
        for resource in runner.last_run_info["resources"]
    )


def test_mlx_plan_accounts_for_neighbor_working_memory_and_budget():
    _mlx_provider()
    rng = np.random.default_rng(19)
    orig = rng.normal(size=(80, 6))
    config = ExecutionConfig(backend="mlx", device="gpu", dtype="float32")
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "lcmc", "params": {"k": 7}},
        {"id": "topo", "params": {"k": 4}},
    ]
    baseline = ZADU(specs, orig, execution=config)
    neighbor_keys = [
        key
        for key in baseline._execution_plan.resources
        if key.kind
        in {ResourceKind.KNN, ResourceKind.STABLE_KNN, ResourceKind.NEIGHBOR_RANKING}
    ]

    assert neighbor_keys
    assert all(
        key in baseline._execution_plan.resource_working_bytes for key in neighbor_keys
    )
    assert baseline._execution_plan.planned_peak_bytes >= (
        baseline._execution_plan.estimated_cache_bytes
        + max(
            baseline._execution_plan.resource_working_bytes[key]
            for key in neighbor_keys
        )
    )

    selected_rank_bytes_per_row = orig.shape[0] * 24 + 5**2
    budget = (
        baseline._execution_plan.estimated_cache_bytes + 2 * selected_rank_bytes_per_row
    )
    bounded = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            memory_budget=budget,
        ),
    )
    assert bounded._execution_plan.rank_comparison_plan.block_rows == 2
    bounded.measure(rng.normal(size=(len(orig), 2)))
    comparison = next(
        record
        for record in bounded._resource_cache.records.values()
        if record.key.kind is ResourceKind.RANK_COMPARISONS
    )
    assert comparison.details["block_rows"] == 2
    assert comparison.provider == "mlx"
    assert comparison.details["provider_fallback"] is False
    assert bounded._execution_plan.planned_peak_bytes <= budget

    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            [{"id": "tnc", "params": {"k": 5}}],
            orig,
            execution=ExecutionConfig(
                backend="mlx",
                device="gpu",
                dtype="float32",
                memory_budget=(
                    ZADU(
                        [{"id": "tnc", "params": {"k": 5}}],
                        orig,
                        execution=config,
                    )._execution_plan.estimated_cache_bytes
                    + selected_rank_bytes_per_row
                    - 1
                ),
            ),
        )


def test_mlx_native_batch_pairwise_and_ranking_match_independent_results():
    rng = np.random.default_rng(20)
    points_batch = [rng.normal(size=(37, 5)).astype(np.float32) for _ in range(3)]
    points_batch[0][1] = points_batch[0][0]
    provider = _mlx_provider()
    pairwise_bytes = points_batch[0].shape[0] ** 2 * 4 * 4

    matrices = _build_batch(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points_batch,
        working_memory_bytes=pairwise_bytes,
    )

    assert len(matrices) == len(points_batch)
    for batch_index, (built, points) in enumerate(
        zip(matrices, points_batch, strict=True)
    ):
        np.testing.assert_allclose(
            built.value,
            cdist(points, points).astype(np.float32),
            rtol=2e-6,
            atol=2e-6,
        )
        assert built.details["provider_batching"] is True
        assert built.details["batch_size"] == 3
        assert built.details["batch_index"] == batch_index
        assert built.details["batch_working_bytes"] == (
            3 * built.details["working_bytes"]
        )

    ranking_bytes = points_batch[0].shape[0] ** 2 * (4 * 4 + 4 * 4)
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
        assert built.details["timings"]["warm_execution_seconds"] > 0
        json.dumps(built.details)

    warm = _build_batch(
        provider,
        ResourceKind.DISTANCE_MATRIX,
        points_batch[:2],
        working_memory_bytes=pairwise_bytes,
    )
    assert all(
        built.details["timings"]["compile_and_first_execution_seconds"] == 0
        for built in warm
    )


def test_zadu_mlx_measure_many_uses_native_batches_and_preserves_scores():
    _mlx_provider()
    rng = np.random.default_rng(21)
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
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
        ),
    )
    expected = sequential_runner.measure_many(embeddings)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            embedding_workers=3,
        ),
    )

    actual = runner.measure_many(embeddings)

    for actual_run, expected_run in zip(actual, expected, strict=True):
        for actual_score, expected_score in zip(
            actual_run,
            expected_run,
            strict=True,
        ):
            assert actual_score.keys() == expected_score.keys()
            for name in actual_score:
                assert actual_score[name] == pytest.approx(
                    expected_score[name], rel=3e-5, abs=3e-6
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
        if resource["space"] == "emb" and resource["provider"] == "mlx"
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


def test_mlx_native_batch_size_is_capped_by_memory_budget():
    _mlx_provider()
    rng = np.random.default_rng(22)
    orig = rng.normal(size=(56, 6))
    embeddings = [rng.normal(size=(56, 2)) for _ in range(4)]
    config = ExecutionConfig(backend="mlx", device="gpu", dtype="float32")
    baseline = ZADU([{"id": "stress"}], orig, execution=config)
    plan = baseline._execution_plan
    batch_input_bytes = embeddings[0].size * np.dtype(np.float32).itemsize
    budget = plan.original_cache_bytes + 2 * (
        plan.per_embedding_peak_bytes + batch_input_bytes
    )
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
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

    sequential = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            embedding_workers=4,
            memory_budget=(plan.original_cache_bytes + plan.per_embedding_peak_bytes),
        ),
    )
    sequential.measure_many(embeddings)
    assert sequential.last_run_info["provider_batching"] is False
    assert sequential.last_run_info["native_batch_size"] == 1
    assert sequential.last_run_info["worker_limit_reason"] == "memory_budget"
    assert sequential.last_run_info["planned_peak_bytes"] <= (
        plan.original_cache_bytes + plan.per_embedding_peak_bytes
    )


def test_mlx_measure_many_shape_mismatch_falls_back_to_ordered_sequential():
    _mlx_provider()
    rng = np.random.default_rng(23)
    orig = rng.normal(size=(40, 5))
    embeddings = [rng.normal(size=(40, 2)), rng.normal(size=(40, 3))]
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            embedding_workers=3,
        ),
    )

    results = runner.measure_many(embeddings)

    assert len(results) == 2
    assert runner.last_run_info["batch_strategy"] == "sequential_shared_original"
    assert runner.last_run_info["provider_batching"] is False
    assert runner.last_run_info["effective_workers"] == 1
    assert runner.last_run_info["worker_limit_reason"] == "embedding_shape_mismatch"


def test_mlx_external_pair_order_uses_explicit_numpy_fallback(tmp_path):
    _mlx_provider(device="cpu", dtype="float64")
    rng = np.random.default_rng(911)
    orig = rng.normal(size=(24, 5))
    emb = rng.normal(size=(24, 2))
    specs = [{"id": "srho"}, {"id": "nm_stress"}]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="cpu",
            dtype="float64",
            memory_budget="4KiB",
            pair_order_strategy="external",
            temporary_budget="1MiB",
            temporary_directory=str(tmp_path),
        ),
    )

    actual = runner.measure(emb)

    for actual_score, expected_score in zip(actual, expected, strict=True):
        for key in actual_score:
            assert actual_score[key] == pytest.approx(
                expected_score[key], rel=2e-12, abs=2e-14
            )
    ordered = runner.last_run_info["resources"][0]
    assert ordered["provider"] == "numpy"
    assert ordered["details"]["strategy"] == "external"
    assert ordered["details"]["provider_fallback"] is True
    assert ordered["details"]["requested_provider"] == "mlx"
    assert ordered["details"]["fallback_reason"] == "unsupported_resource"
    assert list(tmp_path.iterdir()) == []


def test_mlx_native_batch_overflow_reports_the_exact_embedding_index():
    _mlx_provider()
    rng = np.random.default_rng(24)
    orig = rng.normal(size=(32, 4))
    embeddings = [rng.normal(size=(32, 2)) for _ in range(3)]
    embeddings[1][0, 0] = np.finfo(np.float64).max
    runner = ZADU(
        [{"id": "stress"}],
        orig,
        execution=ExecutionConfig(
            backend="mlx",
            device="gpu",
            dtype="float32",
            embedding_workers=3,
        ),
    )

    with pytest.raises(EmbeddingExecutionError) as error:
        runner.measure_many(embeddings)

    assert error.value.embedding_index == 1
    assert isinstance(error.value.__cause__, BatchResourceError)
    assert "cannot be represented" in str(error.value.__cause__)
