import numpy as np
import pytest

from zadu.measures import steadiness_cohesiveness as zadu_snc


snc_external = pytest.importorskip("snc.snc")


def _sample_data(seed: int = 0, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 12))
    proj = rng.normal(size=(12, 2))
    emb = raw @ proj + 0.05 * rng.normal(size=(n, 2))
    return raw, emb


def _legacy_score(raw: np.ndarray, emb: np.ndarray, return_local: bool = False):
    np.random.seed(123)
    obj = snc_external.SNC(
        raw,
        emb,
        iteration=40,
        walk_num_ratio=0.25,
        dist_strategy="snn",
        dist_parameter={"alpha": 0.1},
        dist_function=None,
        cluster_strategy="dbscan",
    )
    obj.fit(record_vis_info=return_local)
    score = {"steadiness": obj.steadiness(), "cohesiveness": obj.cohesiveness()}

    if not return_local:
        return score

    _, _, _, points_info = obj.vis_info()
    local = {
        "local_steadiness": np.array([1 - p["false_val"] for p in points_info]),
        "local_cohesiveness": np.array([1 - p["missing_val"] for p in points_info]),
    }
    return score, local


def _local_score(raw: np.ndarray, emb: np.ndarray, return_local: bool = False):
    np.random.seed(123)
    return zadu_snc.measure(
        raw,
        emb,
        iteration=40,
        walk_num_ratio=0.25,
        alpha=0.1,
        clustering_strategy="dbscan",
        return_local=return_local,
    )


def test_snc_global_score_matches_legacy_package():
    raw, emb = _sample_data(seed=7)
    expected = _legacy_score(raw, emb)
    actual = _local_score(raw, emb)

    assert actual["steadiness"] == pytest.approx(expected["steadiness"], abs=0.03)
    assert actual["cohesiveness"] == pytest.approx(expected["cohesiveness"], abs=0.03)


def test_snc_local_output_contract_and_proximity_to_legacy():
    raw, emb = _sample_data(seed=13)
    expected_score, expected_local = _legacy_score(raw, emb, return_local=True)
    actual_score, actual_local = _local_score(raw, emb, return_local=True)

    assert actual_score["steadiness"] == pytest.approx(expected_score["steadiness"], abs=0.03)
    assert actual_score["cohesiveness"] == pytest.approx(expected_score["cohesiveness"], abs=0.03)

    assert actual_local["local_steadiness"].shape == (raw.shape[0],)
    assert actual_local["local_cohesiveness"].shape == (raw.shape[0],)
    assert np.all(np.isfinite(actual_local["local_steadiness"]))
    assert np.all(np.isfinite(actual_local["local_cohesiveness"]))

    mean_delta_stead = float(
        np.mean(np.abs(actual_local["local_steadiness"] - expected_local["local_steadiness"]))
    )
    mean_delta_cohev = float(
        np.mean(np.abs(actual_local["local_cohesiveness"] - expected_local["local_cohesiveness"]))
    )

    # Local values are stochastic and amplified by normalization, so compare by mean deviation.
    assert mean_delta_stead < 0.08
    assert mean_delta_cohev < 0.2
