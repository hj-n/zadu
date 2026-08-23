from pathlib import Path

import numpy as np
import pytest

from zadu import ZADU, ExecutionConfig
from zadu.engine.resources import PairStrategy, ResourceKind
from zadu.kernels import external_order

ORDERED_SPECS = [{"id": "srho"}, {"id": "nm_stress"}]


def _sample(seed=0, n=40):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 6)), rng.normal(size=(n, 2))


def _external_config(
    *,
    memory_budget="8KiB",
    temporary_budget="2MiB",
    strategy="external",
    temporary_directory=None,
):
    return ExecutionConfig(
        memory_budget=memory_budget,
        pair_order_strategy=strategy,
        temporary_budget=temporary_budget,
        temporary_directory=(
            None if temporary_directory is None else str(temporary_directory)
        ),
    )


def _assert_scores_close(actual, expected):
    for actual_score, expected_score in zip(actual, expected, strict=True):
        assert actual_score.keys() == expected_score.keys()
        for key in actual_score:
            assert actual_score[key] == pytest.approx(
                expected_score[key], rel=2e-12, abs=2e-14
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_external_order_matches_in_memory_ordered_metrics(dtype, tmp_path):
    orig, emb = _sample()
    orig = orig.astype(dtype)
    emb = emb.astype(dtype)
    expected = ZADU(ORDERED_SPECS, orig).measure(emb)
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(temporary_directory=tmp_path),
    )

    actual = runner.measure(emb)

    _assert_scores_close(actual, expected)
    plan = runner._execution_plan.pair_plan
    assert plan.strategy is PairStrategy.EXTERNAL
    assert runner._execution_plan.resources == (plan.ordered_statistics_key,)
    assert plan.order_key is None
    assert plan.original_source_key is None
    assert plan.embedded_source_key is None
    assert runner.estimated_cache_bytes == 0
    assert list(tmp_path.iterdir()) == []

    resource = runner.last_run_info["resources"][0]
    details = resource["details"]
    assert resource["kind"] == ResourceKind.ORDERED_PAIR_STATISTICS.value
    assert details["strategy"] == "external"
    assert details["workspace_removed"] is True
    assert details["ordering_reused"] is False
    assert details["temporary_bytes_peak"] <= details["planned_temporary_bytes"]
    assert details["planned_temporary_bytes"] <= details["temporary_budget_bytes"]
    assert runner.last_run_info["pair_strategy"] == "external"
    assert runner.last_run_info["planned_peak_bytes"] <= 8 * 1024


def test_auto_selects_external_only_with_an_explicit_disk_allowance(tmp_path):
    orig, _ = _sample(n=50)
    external = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(
            strategy="auto",
            memory_budget="8KiB",
            temporary_budget="4MiB",
            temporary_directory=tmp_path,
        ),
    )

    assert external._execution_plan.pair_plan.strategy is PairStrategy.EXTERNAL

    with pytest.raises(MemoryError, match=r"exceeds max_memory_bytes|peak working"):
        ZADU(
            ORDERED_SPECS,
            orig,
            execution=ExecutionConfig(memory_budget="8KiB"),
        )
    with pytest.raises(MemoryError, match=r"exceeds max_memory_bytes|peak working"):
        ZADU(
            ORDERED_SPECS,
            orig,
            execution=_external_config(
                strategy="memory",
                memory_budget="8KiB",
                temporary_budget="4MiB",
            ),
        )


def test_external_ties_cross_runs_and_multiple_merge_passes(tmp_path):
    orig = np.asarray(
        [[float(index % 3), float((index // 3) % 2)] for index in range(12)]
    )
    emb = orig[:, ::-1] * 0.5
    expected = ZADU(ORDERED_SPECS, orig).measure(emb)
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(
            memory_budget="64B",
            temporary_budget="1MiB",
            temporary_directory=tmp_path,
        ),
    )

    actual = runner.measure(emb)

    _assert_scores_close(actual, expected)
    details = runner.last_run_info["resources"][0]["details"]
    assert details["run_pairs"] == 1
    assert details["initial_original_run_count"] == 66
    assert details["initial_embedded_run_count"] == 66
    assert details["merge_fan_in"] == 2
    assert details["merge_algorithm"] == "numba_binary"
    assert details["merge_passes"] == 14
    assert details["tie_group_count"] < 66
    assert list(tmp_path.iterdir()) == []


def test_external_order_and_streaming_pair_reductions_share_one_plan(tmp_path):
    orig, emb = _sample(seed=2, n=50)
    specs = [
        {"id": "stress"},
        {"id": "srho"},
        {"id": "pr"},
        {"id": "nm_stress"},
    ]
    expected = ZADU(specs, orig).measure(emb)
    runner = ZADU(
        specs,
        orig,
        execution=_external_config(
            memory_budget="16KiB",
            temporary_budget="2MiB",
            temporary_directory=tmp_path,
        ),
    )

    actual = runner.measure(emb)

    _assert_scores_close(actual, expected)
    assert [key.kind for key in runner._execution_plan.resources] == [
        ResourceKind.PAIR_STATISTICS,
        ResourceKind.ORDERED_PAIR_STATISTICS,
    ]
    assert all(
        resource["details"]["strategy"] == "external"
        for resource in runner.last_run_info["resources"]
    )
    assert list(tmp_path.iterdir()) == []


def test_temporary_budget_caps_parallel_embedding_workspaces(tmp_path):
    orig, emb = _sample(seed=3, n=24)
    baseline = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(
            memory_budget="1MiB",
            temporary_budget="8MiB",
            temporary_directory=tmp_path,
        ),
    )
    per_embedding = baseline._execution_plan.pair_plan.planned_temporary_bytes
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        execution=ExecutionConfig(
            memory_budget="1MiB",
            embedding_workers=4,
            pair_order_strategy="external",
            temporary_budget=2 * per_embedding,
            temporary_directory=str(tmp_path),
        ),
    )

    results = runner.measure_many([emb + offset for offset in (0.0, 0.01, 0.02, 0.03)])

    assert len(results) == 4
    info = runner.last_run_info
    assert info["effective_workers"] == 2
    assert info["worker_limit_reason"] == "temporary_budget"
    assert info["per_embedding_temporary_bytes"] == per_embedding
    assert info["planned_temporary_bytes"] == 2 * per_embedding
    assert info["planned_temporary_bytes"] <= info["temporary_budget_bytes"]
    assert all(
        resource["details"]["workspace_removed"]
        for run in info["runs"]
        for resource in run["resources"]
        if resource["kind"] == "ordered_pair_statistics"
    )
    assert list(tmp_path.iterdir()) == []


def test_external_geodesic_order_matches_dense_geodesic_results(tmp_path):
    orig = np.asarray([[0.0, 0.0], [0.2, 0.1], [-0.4, 0.3], [0.5, -0.2], [0.7, 0.4]])
    emb = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.3, -0.2], [0.5, 0.4], [-0.1, 0.7]])
    expected = ZADU(ORDERED_SPECS, orig, geodesic=True).measure(emb)
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        geodesic=True,
        execution=_external_config(
            memory_budget="1KiB",
            temporary_budget="1MiB",
            temporary_directory=tmp_path,
        ),
    )

    actual = runner.measure(emb)

    _assert_scores_close(actual, expected)
    assert runner._execution_plan.pair_plan.strategy is PairStrategy.EXTERNAL
    assert list(tmp_path.iterdir()) == []


def test_temporary_guard_fails_before_distance_or_workspace_allocation(
    monkeypatch, tmp_path
):
    orig, _ = _sample(n=40)

    def unexpected_cdist(*args, **kwargs):
        raise AssertionError("distance construction should not start")

    monkeypatch.setattr(external_order, "cdist", unexpected_cdist)
    with pytest.raises(MemoryError, match="temporary_budget"):
        ZADU(
            ORDERED_SPECS,
            orig,
            execution=_external_config(
                temporary_budget="1KiB",
                temporary_directory=tmp_path,
            ),
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "orig,emb,match",
    [
        (
            np.zeros((8, 3)),
            np.arange(16, dtype=float).reshape(8, 2),
            "Spearman correlation is undefined",
        ),
        (
            np.arange(16, dtype=float).reshape(8, 2),
            np.zeros((8, 2)),
            "Spearman correlation is undefined",
        ),
    ],
)
def test_external_degenerate_error_cleans_workspace(orig, emb, match, tmp_path):
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(temporary_directory=tmp_path),
    )

    with pytest.raises(ValueError, match=match):
        runner.measure(emb)

    assert list(tmp_path.iterdir()) == []
    assert runner.last_run_info is None


def test_external_interruption_cleans_every_temporary_file(monkeypatch, tmp_path):
    orig, emb = _sample(seed=9, n=16)
    roots: list[Path] = []
    real_temporary_directory = external_order.TemporaryDirectory

    def recording_temporary_directory(*args, **kwargs):
        temporary = real_temporary_directory(*args, **kwargs)
        roots.append(Path(temporary.name))
        return temporary

    def interrupt_merge(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        external_order,
        "TemporaryDirectory",
        recording_temporary_directory,
    )
    monkeypatch.setattr(external_order, "_merge_group", interrupt_merge)
    runner = ZADU(
        ORDERED_SPECS,
        orig,
        execution=_external_config(
            memory_budget="64B",
            temporary_budget="1MiB",
            temporary_directory=tmp_path,
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        runner.measure(emb)

    assert roots
    assert all(not root.exists() for root in roots)
    assert list(tmp_path.iterdir()) == []
    assert runner.last_run_info is None
