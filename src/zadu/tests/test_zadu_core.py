import numpy as np
import pytest

from zadu import MEASURE, ZADU, MeasureId, make_spec
from zadu.measures import (
    clustering_and_external_validation_measure as cevm,
)
from zadu.measures import (
    internal_validation_measure as ivm,
)
from zadu.measures import (
    label_trustworthiness_and_continuity as ltnc,
)
from zadu.measures.utils import knn as knn_mod
from zadu.registry import METRICS


def test_spec_is_not_mutated_by_abbreviation_normalization():
    orig = np.random.RandomState(0).rand(20, 5)
    spec = [{"id": "tnc", "params": {"k": 5}}]

    ZADU(spec, orig)

    assert spec[0]["id"] == "tnc"


def test_invalid_k_in_spec_raises_early():
    orig = np.random.RandomState(0).rand(5, 3)

    with pytest.raises(ValueError, match="1 <= k < n"):
        ZADU([{"id": "tnc", "params": {"k": 5}}], orig)


def test_invalid_string_options_raise_value_error():
    emb = np.random.RandomState(0).rand(30, 2)
    label = np.random.RandomState(1).randint(0, 3, 30)
    raw = np.random.RandomState(2).rand(30, 5)

    with pytest.raises(ValueError, match="Invalid internal validation measure"):
        ivm.measure(emb, label, measure="foo")

    with pytest.raises(ValueError, match="Invalid external validation measure"):
        cevm.measure(emb, label, measure="foo")

    with pytest.raises(ValueError, match="Invalid clustering algorithm"):
        cevm.measure(emb, label, clustering="foo")

    with pytest.raises(ValueError, match="Invalid cvm"):
        ltnc.measure(raw, emb, label, cvm="foo")


def test_knn_precompute_reused_for_knn_info_measures(monkeypatch):
    raw = np.random.RandomState(0).rand(80, 5)
    emb = np.random.RandomState(1).rand(80, 2)
    call_count = {"knn": 0}

    original_knn = knn_mod.knn

    def wrapped_knn(*args, **kwargs):
        call_count["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(knn_mod, "knn", wrapped_knn)

    spec = [
        {"id": "proc", "params": {"k": 10}},
        {"id": "lcmc", "params": {"k": 10}},
    ]
    ZADU(spec, raw).measure(emb)

    # Precomputation should run once for raw and once for emb.
    assert call_count["knn"] == 2


def test_neighbor_dissimilarity_runs_with_sparse_snn_backend():
    raw = np.random.RandomState(0).rand(40, 6)
    emb = np.random.RandomState(1).rand(40, 2)

    score = ZADU([{"id": "nd", "params": {"k": 10}}], raw).measure(emb)[0]

    assert "neighbor_dissimilarity" in score
    assert np.isfinite(score["neighbor_dissimilarity"])


def test_typed_spec_uses_short_measure_enum_names():
    raw = np.random.RandomState(0).rand(40, 6)
    emb = np.random.RandomState(1).rand(40, 2)

    score = ZADU([make_spec(MEASURE.TNC, k=10)], raw).measure(emb)[0]
    assert "trustworthiness" in score

    # Backward compatibility alias should still be supported.
    score_alias = ZADU([make_spec(MeasureId.TNC, k=10)], raw).measure(emb)[0]
    assert "trustworthiness" in score_alias


def test_typed_measure_enum_and_registry_stay_in_sync():
    assert {member.value for member in MEASURE} == {metric.id for metric in METRICS}
    assert len({metric.alias for metric in METRICS}) == len(METRICS)


def test_cache_memory_estimate_can_guard_large_allocations():
    raw = np.random.RandomState(0).rand(60, 5)
    specs = [{"id": "tnc", "params": {"k": 5}}, {"id": "stress"}]

    with pytest.raises(MemoryError, match="Estimated ZADU cache size"):
        ZADU(specs, raw, max_memory_bytes=1)

    runner = ZADU(specs, raw)
    assert runner.estimated_cache_bytes > 0
