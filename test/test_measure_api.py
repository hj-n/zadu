import numpy as np

from zadu.measures import (
    class_angular_distortion_index,
    neighborhood_hit,
    trustworthiness_continuity,
)


def test_measure_modules_are_importable_and_callable():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(40, 8))
    emb = raw[:, :2] + 0.01 * rng.normal(size=(40, 2))

    tnc = trustworthiness_continuity.measure(raw, emb, k=5)

    assert "trustworthiness" in tnc
    assert "continuity" in tnc
    assert 0.0 <= tnc["trustworthiness"] <= 1.0
    assert 0.0 <= tnc["continuity"] <= 1.0


def test_neighborhood_hit_returns_unit_for_perfectly_separated_clusters():
    # Two clearly separated clusters; each point's nearest neighbors share labels.
    emb = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
        dtype=float,
    )
    label = np.array([0, 0, 0, 1, 1, 1])

    nh = neighborhood_hit.measure(emb, label, k=2)

    assert "neighborhood_hit" in nh
    assert nh["neighborhood_hit"] == 1.0


def test_cadi_uses_the_standard_snake_case_return_key():
    orig = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1],
            [0.0, 0.2, 0.1],
            [5.0, 5.0, 5.0],
            [5.2, 5.0, 5.1],
            [5.0, 5.2, 5.1],
        ]
    )
    emb = orig[:, :2]
    label = np.array([0, 0, 0, 1, 1, 1])

    cadi = class_angular_distortion_index.measure(
        orig, emb, label, n_triplets=20, random_seed=0
    )

    assert set(cadi) == {"class_angular_distortion_index"}
    assert 0.0 <= cadi["class_angular_distortion_index"] <= 1.0
