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
