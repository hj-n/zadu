import numpy as np

from zadu import ZADU


def _sample_data(seed: int = 0, n: int = 80):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 10))
    proj = rng.normal(size=(10, 2))
    emb = raw @ proj + 0.02 * rng.normal(size=(n, 2))
    label = rng.integers(0, 4, size=n)
    return raw, emb, label


def test_zadu_runs_multiple_metrics_with_return_local():
    raw, emb, label = _sample_data()
    spec = [
        {"id": "tnc", "params": {"k": 10}},
        {"id": "mrre", "params": {"k": 10}},
    ]

    scores, local = ZADU(spec, raw, return_local=True).measure(emb, label)

    assert len(scores) == 2
    assert len(local) == 2
    assert "trustworthiness" in scores[0]
    assert "mrre_false" in scores[1]
    assert "local_trustworthiness" in local[0]
    assert "local_mrre_false" in local[1]


def test_zadu_label_metric_variants_run():
    raw, emb, label = _sample_data(seed=1)
    spec = [
        {"id": "l_tnc", "params": {"cvm": "dsc"}},
        {"id": "l_tnc", "params": {"cvm": "ch_btw"}},
    ]

    scores = ZADU(spec, raw).measure(emb, label)

    assert len(scores) == 2
    for score in scores:
        assert "label_trustworthiness" in score
        assert "label_continuity" in score


def test_all_registered_metrics_execute_together():
    raw, emb, label = _sample_data(seed=12, n=60)
    specs = [
        {"id": "tnc", "params": {"k": 5}},
        {"id": "mrre", "params": {"k": 7}},
        {"id": "lcmc", "params": {"k": 9}},
        {"id": "nh", "params": {"k": 6}},
        {"id": "ca_tnc", "params": {"k": 5}},
        {"id": "l_tnc"},
        {"id": "nd", "params": {"k": 5}},
        {"id": "dtm"},
        {"id": "kl_div"},
        {"id": "dsc"},
        {"id": "pr"},
        {"id": "srho"},
        {"id": "ivm"},
        {"id": "c_evm"},
        {
            "id": "snc",
            "params": {
                "iteration": 2,
                "k": 5,
                "clustering_strategy": "kmeans",
                "random_state": 0,
            },
        },
        {"id": "topo", "params": {"k": 5}},
        {"id": "proc", "params": {"k": 5}},
        {"id": "stress"},
        {"id": "sn_stress"},
        {"id": "nm_stress"},
        {"id": "cadi", "params": {"n_triplets": 10, "random_seed": 0}},
        {"id": "gi"},
    ]

    scores = ZADU(specs, raw).measure(emb, label)

    assert len(scores) == 22
    assert all(np.isfinite(value) for score in scores for value in score.values())
