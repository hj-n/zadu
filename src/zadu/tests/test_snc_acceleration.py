import math

import numpy as np
import pytest
from scipy.sparse import issparse

from zadu import ZADU, ExecutionConfig
from zadu.backends import NumpyResourceProvider
from zadu.measures import steadiness_cohesiveness
from zadu.measures.utils import snc_cpu
from zadu.measures.utils.knn import knn
from zadu.measures.utils.snc_cpu import SNCCPU


def _sample(seed=0, n=120):
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, 12))
    emb = orig @ rng.normal(size=(12, 2)) + 0.05 * rng.normal(size=(n, 2))
    return orig, emb


def _knn_info(orig, emb, k):
    return knn(orig, k), knn(emb, k)


class _ScalarDistanceSNCCPU(SNCCPU):
    """Pre-fusion cluster-pair oracle using the same sparse similarities."""

    def _cluster_distance_matrices(self, clusters):
        cluster_count = len(clusters)
        raw_distances = np.empty((cluster_count, cluster_count))
        emb_distances = np.empty((cluster_count, cluster_count))
        for row, cluster_a in enumerate(clusters):
            for column, cluster_b in enumerate(clusters):
                pair_count = cluster_a.size * cluster_b.size
                raw_similarity = float(
                    self.raw_snn[cluster_a][:, cluster_b].sum() / pair_count
                )
                emb_similarity = float(
                    self.emb_snn[cluster_a][:, cluster_b].sum() / pair_count
                )
                raw_distances[row, column] = 1 / (raw_similarity + self.alpha)
                emb_distances[row, column] = 1 / (emb_similarity + self.alpha)
        return raw_distances, emb_distances


def _run_snc(
    cls, orig, emb, info, *, n_jobs=1, record=False, cluster_strategy="dbscan"
):
    obj = cls(
        orig,
        emb,
        iteration=20,
        walk_num_ratio=0.2,
        alpha=0.1,
        k=10,
        cluster_strategy=cluster_strategy,
        random_state=123,
        n_jobs=n_jobs,
    )
    obj.fit(record_vis_info=record, knn_info=info)
    scores = obj.steadiness(), obj.cohesiveness()
    return (scores, obj.local_scores()) if record else scores


def test_precomputed_knn_skips_euclidean_distances_and_keeps_snn_sparse(monkeypatch):
    orig, emb = _sample()
    info = _knn_info(orig, emb, 10)

    def unexpected_distance(*args, **kwargs):
        raise AssertionError("Euclidean distances should not be constructed")

    monkeypatch.setattr(snc_cpu, "pairwise_distance_matrix", unexpected_distance)
    obj = SNCCPU(orig, emb, iteration=2, k=10, random_state=0)
    obj.fit(knn_info=info)

    assert issparse(obj.raw_snn)
    assert issparse(obj.emb_snn)
    assert obj.raw_snn.nnz < orig.shape[0] ** 2
    assert obj.emb_snn.nnz < orig.shape[0] ** 2
    assert np.isfinite(obj.steadiness())
    assert np.isfinite(obj.cohesiveness())


def test_batched_cluster_distances_match_scalar_pair_oracle():
    orig, emb = _sample(seed=2)
    info = _knn_info(orig, emb, 10)

    actual = _run_snc(SNCCPU, orig, emb, info, record=True)
    expected = _run_snc(_ScalarDistanceSNCCPU, orig, emb, info, record=True)

    assert actual[0] == pytest.approx(expected[0], rel=0, abs=1e-14)
    np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[1][1], expected[1][1], rtol=0, atol=1e-14)


@pytest.mark.parametrize("cluster_strategy", ["dbscan", "kmeans"])
def test_fixed_seed_single_and_multi_worker_results_are_identical(cluster_strategy):
    orig, emb = _sample(seed=4)
    info = _knn_info(orig, emb, 10)

    single = _run_snc(
        SNCCPU,
        orig,
        emb,
        info,
        n_jobs=1,
        record=True,
        cluster_strategy=cluster_strategy,
    )
    parallel = _run_snc(
        SNCCPU,
        orig,
        emb,
        info,
        n_jobs=4,
        record=True,
        cluster_strategy=cluster_strategy,
    )

    assert single[0] == parallel[0]
    np.testing.assert_array_equal(single[1][0], parallel[1][0])
    np.testing.assert_array_equal(single[1][1], parallel[1][1])


def test_zadu_plans_default_sqrt_knn_and_injects_it_without_distances(monkeypatch):
    orig, emb = _sample(seed=5, n=100)
    specs = [{"id": "snc", "params": {"iteration": 3, "random_state": 0}}]
    runner = ZADU(specs, orig)

    def unexpected_distance(*args, **kwargs):
        raise AssertionError("planned kNN should bypass SNC Euclidean distances")

    monkeypatch.setattr(snc_cpu, "pairwise_distance_matrix", unexpected_distance)
    score = runner.measure(emb)[0]

    knn_resources = runner._execution_plan.resources
    assert len(knn_resources) == 2
    assert all(key.k == math.isqrt(len(orig)) for key in knn_resources)
    assert all(np.isfinite(value) for value in score.values())
    assert runner.last_run_info["snc_strategy"]["algorithm"] == (
        "sparse_batched_iterations"
    )


def test_memory_budget_reduces_workers_and_rejects_minimum_before_allocation(
    monkeypatch,
):
    orig, emb = _sample(seed=6, n=100)
    k = 10
    ratio = 0.2
    specs = [
        {
            "id": "snc",
            "params": {
                "iteration": 4,
                "walk_num_ratio": ratio,
                "k": k,
                "random_state": 0,
                "n_jobs": 4,
            },
        }
    ]
    base = ZADU(specs, orig)
    graph_bytes = SNCCPU.estimate_graph_bytes(len(orig), k)
    iteration_bytes = SNCCPU.estimate_iteration_bytes(
        len(orig), k, int(len(orig) * ratio)
    )
    budget = base.estimated_cache_bytes + graph_bytes + 2 * iteration_bytes
    bounded = ZADU(
        specs,
        orig,
        execution=ExecutionConfig(memory_budget=budget),
    )

    score = bounded.measure(emb)[0]

    assert all(np.isfinite(value) for value in score.values())
    assert bounded._execution_plan.snc_plan.requested_workers[0] == 4
    assert bounded._execution_plan.snc_plan.effective_workers[0] == 2
    assert bounded._execution_plan.planned_peak_bytes <= budget
    assert bounded.last_run_info["snc_strategy"]["effective_workers"] == {0: 2}

    def unexpected_build(*args, **kwargs):
        raise AssertionError("resource allocation should not start")

    monkeypatch.setattr(NumpyResourceProvider, "build", unexpected_build)
    with pytest.raises(MemoryError, match="peak working memory"):
        ZADU(
            specs,
            orig,
            execution=ExecutionConfig(
                memory_budget=(
                    base.estimated_cache_bytes + graph_bytes + iteration_bytes - 1
                )
            ),
        )


@pytest.mark.parametrize(
    "n_jobs,error", [(True, TypeError), (1.5, TypeError), (0, ValueError)]
)
def test_snc_rejects_invalid_worker_count(n_jobs, error):
    orig, emb = _sample(seed=8, n=30)

    with pytest.raises(error, match="n_jobs"):
        steadiness_cohesiveness.measure(
            orig,
            emb,
            iteration=1,
            k=5,
            random_state=0,
            n_jobs=n_jobs,
        )
    with pytest.raises(error, match="n_jobs"):
        ZADU([{"id": "snc", "params": {"n_jobs": n_jobs}}], orig)
