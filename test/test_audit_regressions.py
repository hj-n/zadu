"""Independent invariance and allocation regressions from the 0.5.3 audit."""

import tracemalloc

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from zadu import ZADU, ExecutionConfig
from zadu.measures import gap_index, scale_normalized_stress


@pytest.mark.parametrize("surface", ["single", "many", "stream"])
@pytest.mark.parametrize("budget", [None, 12000])
def test_original_snapshot_survives_caller_mutation(surface, budget):
    source = np.random.default_rng(0).normal(size=(50, 4))
    original = source.copy()
    specs = [{"id": "stress"}, {"id": "tnc", "params": {"k": 5}}]
    config = ExecutionConfig(memory_budget=budget)
    runner = ZADU(specs, source.view(), execution=config)
    source[:] = np.random.default_rng(1).normal(size=source.shape)
    projection = source[:, :2].copy()
    expected = ZADU(specs, original, execution=config).measure(projection)
    if surface == "single":
        actual = runner.measure(projection)
    elif surface == "many":
        actual = runner.measure_many([projection, projection])[0]
    else:
        (item,) = list(runner.iter_measure_many([projection]))
        actual = item.result
    assert actual == expected
    assert source.flags.writeable
    assert not np.shares_memory(runner.orig, source)
    with pytest.raises(ValueError, match="read-only"):
        runner.orig[0, 0] = 0


@pytest.mark.parametrize("scale", [1e80, 1e-90])
@pytest.mark.parametrize("budget", [None, 8192])
def test_pearson_is_invariant_to_units(scale, budget):
    x = np.random.default_rng(0).normal(size=(40, 4)) * scale
    y = x[:, :2].copy()
    expected = pearsonr(pdist(x), pdist(y)).statistic
    actual = ZADU([{"id": "pr"}], x, max_memory_bytes=budget).measure(y)[0]
    assert actual["pearson_r"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int64])
def test_gap_index_integer_coordinates_preserve_identity(dtype):
    x = np.array([[0, 0], [3, 0], [0, 4], [5, 5], [2, 2]], dtype=dtype)
    y = x.astype(float)
    assert gap_index.measure(x, y)["gap_index"] == pytest.approx(0, abs=1e-14)
    assert ZADU([{"id": "gi"}], x).measure(y)[0]["gap_index"] == pytest.approx(
        0, abs=1e-14
    )


@pytest.mark.parametrize("budget", [None, 8192])
def test_scale_normalized_stress_retains_small_residual(budget):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = 2 * x + 1e-8 * rng.normal(size=x.shape)
    expected = scale_normalized_stress.measure(x, y)["scale_normalized_stress"]
    actual = ZADU([{"id": "sn_stress"}], x, max_memory_bytes=budget).measure(y)[0]
    assert expected > 0
    assert actual["scale_normalized_stress"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "backend,device", [("torch", "cpu"), ("torch", "mps"), ("mlx", "gpu")]
)
@pytest.mark.parametrize("surface", ["single", "many"])
def test_accelerated_translation_preserves_scores(backend, device, surface):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable")
    else:
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("Metal unavailable")
    x = np.random.default_rng(42).normal(size=(60, 2)).astype(np.float32)
    y = x + np.float32(1e4)
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "stress"},
        {"id": "lcmc", "params": {"k": 5}},
    ]
    expected = ZADU(specs, x).measure(y)
    runner = ZADU(
        specs,
        x,
        execution=ExecutionConfig(
            backend=backend, device=device, dtype="float32", embedding_workers=2
        ),
    )
    actual = (
        runner.measure(y) if surface == "single" else runner.measure_many([y, y])[0]
    )
    for measured, reference in zip(actual, expected, strict=True):
        assert measured == pytest.approx(reference, abs=2e-6)


def test_procrustes_scratch_respects_small_budget():
    x = np.random.default_rng(0).normal(size=(100, 300))
    y = x[:, :2].copy()
    specs = [{"id": "proc", "params": {"k": 10}}]
    expected = ZADU(specs, x).measure(y)
    budget = 1024**2
    runner = ZADU(specs, x, max_memory_bytes=budget)
    runner.measure(y)
    tracemalloc.start()
    try:
        actual = runner.measure(y)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    assert peak <= budget
    assert runner.last_run_info["planned_peak_bytes"] <= budget


def test_gap_index_rejects_impossible_budget_before_triangulation(monkeypatch):
    x = np.random.default_rng(0).normal(size=(100, 300))

    def unexpected(*args, **kwargs):
        pytest.fail("Delaunay must not start before the workspace guard")

    monkeypatch.setattr(gap_index, "Delaunay", unexpected)
    with pytest.raises(MemoryError):
        ZADU([{"id": "gi"}], x, max_memory_bytes=1024).measure(x[:, :2])


@pytest.mark.parametrize("surface", ["many", "stream"])
def test_projection_workspace_accounts_for_dimensions_and_collection(surface):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 30))
    projections = [rng.normal(size=(60, d)) for d in (2, 8, 3)]
    specs = [{"id": "proc", "params": {"k": 5}}, {"id": "stress"}]
    expected = [ZADU(specs, x).measure(y) for y in projections]
    runner = ZADU(
        specs, x, execution=ExecutionConfig(memory_budget="512KiB", embedding_workers=4)
    )
    if surface == "many":
        actual = runner.measure_many(projections)
    else:
        actual = [item.result for item in runner.iter_measure_many(iter(projections))]
        assert runner.last_run_info["stream_complete"]
    for measured, reference in zip(actual, expected, strict=True):
        for score, wanted in zip(measured, reference, strict=True):
            assert score == pytest.approx(wanted, abs=1e-12)
    assert runner.last_run_info["metric_working_bytes"][0] > 0
    assert runner.last_run_info["planned_peak_bytes"] <= 512 * 1024


def test_gap_index_bounded_blocks_and_mixed_dag_share_scores():
    x = np.random.default_rng(5).normal(size=(80, 100))
    y = x[:, :2].copy()
    specs = [{"id": "gi"}, {"id": "gi"}, {"id": "stress"}]
    expected = ZADU(specs, x).measure(y)
    runner = ZADU(specs, x, max_memory_bytes=512 * 1024)
    actual = runner.measure(y)
    for score, wanted in zip(actual, expected, strict=True):
        assert score == pytest.approx(wanted, abs=1e-12)
    assert len(runner.last_run_info["metric_working_bytes"]) == 2
    assert runner.last_run_info["planned_peak_bytes"] <= 512 * 1024


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_direct_distance_kernel_handles_near_pairs_and_feature_changes(backend):
    from scipy.spatial.distance import squareform

    from zadu.backends import create_resource_provider
    from zadu.engine.resources import ResourceKey, ResourceKind, Space

    pytest.importorskip("torch" if backend == "torch" else "mlx.core")
    provider = create_resource_provider(
        ExecutionConfig(backend=backend, device="cpu", dtype="float32")
    )
    for dimension in (2, 7):
        points = np.full((12, dimension), 10000, dtype=np.float32)
        points[1:, 0] += np.arange(11, dtype=np.float32) * 0.01
        points[-1] += 10000
        built = provider.build(
            ResourceKey(ResourceKind.DISTANCE_MATRIX, Space.ORIGINAL),
            points,
            distance_matrix=None,
            condensed_pairs=None,
            geodesic=False,
            working_memory_bytes=12 * 4 * 4 * 3,
        )
        expected = squareform(pdist(points))
        np.testing.assert_allclose(built.value, expected, rtol=2e-6, atol=1e-7)
