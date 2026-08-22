import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from zadu import ZADU, ExecutionConfig
from zadu.backends import mlx_backend
from zadu.engine.resources import ResourceKey, ResourceKind, Space


def _mlx_provider(*, device="gpu", dtype="float32"):
    mx = pytest.importorskip("mlx.core")
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("MLX Metal GPU is unavailable")
    return mlx_backend.MlxResourceProvider(device=device, dtype=dtype)


def _build(provider, kind, points, *, working_memory_bytes, geodesic=False):
    return provider.build(
        ResourceKey(kind, Space.ORIGINAL),
        points,
        distance_matrix=None,
        condensed_pairs=None,
        working_memory_bytes=working_memory_bytes,
        geodesic=geodesic,
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
