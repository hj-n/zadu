"""Exactness gates for the blockwise selected-rank design candidate."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.benchmark_selected_ranks import (
    blockwise_selected_rank_oracle,
    full_ranking_oracle,
)
from zadu import ZADU
from zadu.measures import (
    class_aware_trustworthiness_continuity,
    mean_relative_rank_error,
    trustworthiness_continuity,
)


def _assert_same_comparisons(left, right):
    np.testing.assert_array_equal(left.orig_indices, right.orig_indices)
    np.testing.assert_array_equal(left.emb_indices, right.emb_indices)
    np.testing.assert_array_equal(left.orig_ranks_of_emb, right.orig_ranks_of_emb)
    np.testing.assert_array_equal(left.emb_ranks_of_orig, right.emb_ranks_of_orig)
    assert left.emb_in_orig.keys() == right.emb_in_orig.keys()
    assert left.orig_in_emb.keys() == right.orig_in_emb.keys()
    for k in left.emb_in_orig:
        np.testing.assert_array_equal(left.emb_in_orig[k], right.emb_in_orig[k])
        np.testing.assert_array_equal(left.orig_in_emb[k], right.orig_in_emb[k])


@pytest.mark.parametrize(
    ("n_samples", "k", "block_rows"),
    [(11, 1, 1), (17, 5, 3), (23, 7, 23)],
)
def test_blockwise_selected_ranks_match_full_ranking_random_inputs(
    n_samples, k, block_rows
):
    rng = np.random.default_rng(70 + n_samples)
    orig = rng.normal(size=(n_samples, 6))
    emb = rng.normal(size=(n_samples, 2))
    membership_ks = tuple(sorted({1, k}))

    full = full_ranking_oracle(orig, emb, k, membership_ks=membership_ks)
    selected = blockwise_selected_rank_oracle(
        orig,
        emb,
        k,
        membership_ks=membership_ks,
        max_working_bytes=24 * n_samples * block_rows,
    )

    _assert_same_comparisons(full.comparisons, selected.comparisons)
    assert selected.block_rows == block_rows
    assert selected.block_count == (n_samples + block_rows - 1) // block_rows
    expected_selected_bytes = 4 * n_samples * k * np.dtype(
        np.int32
    ).itemsize + 2 * n_samples * sum(membership_ks)
    assert selected.persistent_bytes == expected_selected_bytes
    assert selected.persistent_bytes < full.persistent_bytes
    assert (
        full.persistent_bytes - selected.persistent_bytes
        == 2 * n_samples * n_samples * np.dtype(np.int32).itemsize
    )


def test_selected_rank_oracle_preserves_self_and_stable_tie_order():
    orig = np.zeros((7, 2))
    emb = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
        ]
    )
    k = 3

    full = full_ranking_oracle(orig, emb, k, membership_ks=(1, 3))
    selected = blockwise_selected_rank_oracle(
        orig,
        emb,
        k,
        membership_ks=(1, 3),
        max_working_bytes=24 * len(orig),
    )

    _assert_same_comparisons(full.comparisons, selected.comparisons)
    np.testing.assert_array_equal(
        selected.comparisons.orig_indices,
        np.asarray(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
            dtype=np.int32,
        ),
    )
    assert all(
        row_index not in neighbors
        for row_index, neighbors in enumerate(selected.comparisons.orig_indices)
    )


@pytest.mark.parametrize("k", [2, 5])
def test_selected_rank_metric_outputs_match_full_ranking(k):
    rng = np.random.default_rng(2026)
    orig = rng.normal(size=(17, 5))
    emb = rng.normal(size=(17, 2))
    labels = np.arange(17) % 3
    full = full_ranking_oracle(orig, emb, 5, membership_ks=(2, 5))
    selected = blockwise_selected_rank_oracle(
        orig,
        emb,
        5,
        membership_ks=(2, 5),
        max_working_bytes=24 * len(orig) * 2,
    )

    for metric, extra_arguments in (
        (trustworthiness_continuity, ()),
        (class_aware_trustworthiness_continuity, (labels,)),
        (mean_relative_rank_error, ()),
    ):
        expected = metric.measure(
            orig,
            emb,
            *extra_arguments,
            k=k,
            rank_comparisons=full.comparisons,
            return_local=True,
        )
        actual = metric.measure(
            orig,
            emb,
            *extra_arguments,
            k=k,
            rank_comparisons=selected.comparisons,
            return_local=True,
        )
        assert actual[0] == pytest.approx(expected[0])
        assert actual[1].keys() == expected[1].keys()
        for name, values in actual[1].items():
            np.testing.assert_array_equal(values, expected[1][name])


def test_selected_rank_oracle_enforces_one_row_memory_floor():
    orig = np.arange(30, dtype=float).reshape(10, 3)
    emb = orig[:, :2]

    with pytest.raises(MemoryError, match="one row"):
        blockwise_selected_rank_oracle(
            orig,
            emb,
            3,
            max_working_bytes=24 * len(orig) - 1,
        )


def test_numpy_production_resource_matches_full_ranking_oracle_with_ties():
    orig = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
            [3.0, 0.0],
        ]
    )
    emb = orig[:, ::-1].copy()
    specs = [
        {"id": "tnc", "params": {"k": 2}},
        {"id": "tnc", "params": {"k": 5}},
        {"id": "mrre", "params": {"k": 4}},
    ]
    runner = ZADU(specs, orig)
    plan = runner._execution_plan.rank_comparison_plan

    built = runner._provider.build_rank_comparisons(
        plan,
        orig,
        emb,
        orig_knn=runner._resource_cache._values[plan.original_knn_key],
        orig_distance_matrix=None,
        emb_distance_matrix=None,
    )
    expected = full_ranking_oracle(orig, emb, 5, membership_ks=(2, 5))

    _assert_same_comparisons(expected.comparisons, built.value)
    assert built.details["algorithm"] == "blockwise_selected_ranks"
    assert built.details["original_distance_source"] == "blockwise_scipy_cdist"
    assert built.details["embedded_distance_source"] == "blockwise_scipy_cdist"


def test_selected_rank_retained_bytes_meet_production_gate():
    n_samples = 2000
    k = 20
    selected_bytes = 4 * n_samples * k * np.dtype(np.int32).itemsize + 2 * n_samples * k
    full_bytes = selected_bytes + 2 * n_samples**2 * np.dtype(np.int32).itemsize

    assert selected_bytes == 720_000
    assert full_bytes / selected_bytes >= 32
