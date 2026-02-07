import numpy as np

from zadu import ZADU


def test_simple_example_spec_executes():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(50, 6))
    emb = raw[:, :2] + 0.01 * rng.normal(size=(50, 2))

    spec = [
        {"id": "tnc", "params": {"k": 8}},
        {"id": "sn_stress", "params": {}},
    ]
    scores = ZADU(spec, raw).measure(emb)

    assert len(scores) == 2
    assert "trustworthiness" in scores[0]
    assert "scale_normalized_stress" in scores[1]
