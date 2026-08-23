import numpy as np
import pytest
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform

from zadu import MEASURE, ZADU, make_spec
from zadu.measures import gap_index


def _reference_fixture():
    rng = np.random.default_rng(2026)
    orig = rng.normal(size=(40, 5))
    emb = orig[:, :2] + 0.05 * rng.normal(size=(40, 2))
    return orig, emb


def test_gap_index_matches_upstream_reference_implementation():
    orig, emb = _reference_fixture()

    # Golden value from jros/gap-index revision
    # 0a11e4887864fe5d41526d8487eea33685b8f0b4.
    expected = 0.45016282211761127

    result = gap_index.compute(orig, emb)

    assert result.score == pytest.approx(expected, abs=1e-12)
    assert result.triangles.shape[1] == 3
    assert result.deformations.shape == (result.triangles.shape[0],)
    assert np.all((result.deformations >= -1.0) & (result.deformations <= 1.0))
    assert np.sum(result.original_relative_areas) == pytest.approx(1.0)
    assert np.sum(result.embedded_relative_areas) == pytest.approx(1.0)


def test_gap_index_precomputed_distances_match_coordinate_input():
    orig, emb = _reference_fixture()
    distance_matrix = squareform(pdist(orig))

    coordinate_score = gap_index.gap_index(orig, emb)
    precomputed_score = gap_index.gap_index(distance_matrix, emb, metric="precomputed")

    assert precomputed_score == pytest.approx(coordinate_score, abs=1e-12)


def test_vectorized_euclidean_areas_match_scalar_oracle_exactly():
    orig, emb = _reference_fixture()
    triangles = gap_index.Delaunay(emb).simplices

    vectorized = gap_index._compute_areas(orig, triangles, "euclidean")
    scalar = gap_index._scalar_triangle_areas(orig, triangles, distance.euclidean)
    scalar /= np.sum(scalar)

    np.testing.assert_allclose(vectorized, scalar, rtol=0, atol=2e-18)


def test_vectorized_euclidean_areas_preserve_duplicate_point_edges():
    orig, emb = _reference_fixture()
    orig[1] = orig[0]
    orig[3] = orig[2]
    triangles = gap_index.Delaunay(emb).simplices

    vectorized = gap_index._compute_areas(orig, triangles, "euclidean")
    scalar = gap_index._scalar_triangle_areas(orig, triangles, distance.euclidean)
    scalar /= np.sum(scalar)

    np.testing.assert_allclose(vectorized, scalar, rtol=0, atol=0)


def test_vectorized_precomputed_areas_match_scalar_oracle():
    orig, emb = _reference_fixture()
    triangles = gap_index.Delaunay(emb).simplices
    distance_matrix = squareform(pdist(orig))

    vectorized = gap_index._compute_areas(
        distance_matrix,
        triangles,
        "precomputed",
    )
    scalar = np.asarray(
        [
            gap_index._triangle_area_from_sides(
                distance_matrix[a, b],
                distance_matrix[a, c],
                distance_matrix[b, c],
            )
            for a, b, c in triangles
        ]
    )
    scalar /= np.sum(scalar)

    np.testing.assert_allclose(vectorized, scalar, rtol=0, atol=2e-18)


def test_euclidean_area_blocks_preserve_results(monkeypatch):
    orig, emb = _reference_fixture()
    triangles = gap_index.Delaunay(emb).simplices
    expected = gap_index._compute_areas(orig, triangles, "euclidean")
    bytes_for_one_row = 3 * orig.shape[1] * 8 + 6 * 8
    monkeypatch.setattr(gap_index, "_AREA_WORK_BYTES", bytes_for_one_row)

    actual = gap_index._compute_areas(orig, triangles, "euclidean")

    assert gap_index._area_block_rows(orig) == 1
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_precomputed_area_blocks_preserve_results(monkeypatch):
    orig, emb = _reference_fixture()
    triangles = gap_index.Delaunay(emb).simplices
    distance_matrix = squareform(pdist(orig))
    expected = gap_index._compute_areas(distance_matrix, triangles, "precomputed")
    monkeypatch.setattr(gap_index, "_AREA_WORK_BYTES", 64)

    actual = gap_index._compute_areas(distance_matrix, triangles, "precomputed")

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_callable_metric_retains_scalar_compatibility_path():
    orig, emb = _reference_fixture()
    calls = 0

    def counted_cityblock(left, right):
        nonlocal calls
        calls += 1
        return distance.cityblock(left, right)

    result = gap_index.compute(orig, emb, metric=counted_cityblock)

    assert calls == 3 * len(result.triangles)
    assert result.score == pytest.approx(
        gap_index.gap_index(orig, emb, metric="cityblock"),
        abs=1e-12,
    )


def test_vectorized_heron_validation_matches_scalar_contract():
    with pytest.raises(ValueError, match="triangle inequality"):
        gap_index._triangle_areas_from_sides(np.asarray([[1.0, 1.0, 3.0]]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        gap_index._triangle_areas_from_sides(np.asarray([[1.0, np.nan, 1.0]]))


def test_gap_index_is_invariant_to_rigid_transform_and_uniform_scaling():
    rng = np.random.default_rng(7)
    orig = rng.normal(size=(50, 2))
    angle = np.deg2rad(37)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    emb = 7.5 * (orig @ rotation) + np.array([13.0, -4.0])

    assert gap_index.gap_index(orig, emb) == pytest.approx(0.0, abs=1e-12)


def test_gap_index_accepts_scipy_metric_names_and_callables():
    orig, emb = _reference_fixture()

    named = gap_index.gap_index(orig, emb, metric="cityblock")
    callable_score = gap_index.gap_index(orig, emb, metric=distance.cityblock)

    assert named == pytest.approx(callable_score, abs=1e-12)


def test_gap_index_runs_through_short_and_typed_zadu_specs():
    orig, emb = _reference_fixture()

    short_score = ZADU([{"id": "gi", "params": {}}], orig).measure(emb)[0]
    typed_score = ZADU([make_spec(MEASURE.GI)], orig).measure(emb)[0]

    assert short_score == typed_score
    assert short_score["gap_index"] == pytest.approx(0.45016282211761127)


def test_gap_index_rejects_non_2d_and_degenerate_embeddings():
    orig, emb = _reference_fixture()

    with pytest.raises(ValueError, match="2D embedding"):
        gap_index.compute(orig, np.column_stack([emb, np.zeros(len(emb))]))

    collinear = np.column_stack([np.arange(len(orig)), np.zeros(len(orig))])
    with pytest.raises(ValueError, match="non-collinear"):
        gap_index.compute(orig, collinear)


def test_gap_index_validates_precomputed_distance_matrices():
    orig, emb = _reference_fixture()
    distance_matrix = squareform(pdist(orig))
    distance_matrix[0, 1] += 1.0

    with pytest.raises(ValueError, match="symmetric"):
        gap_index.compute(distance_matrix, emb, metric="precomputed")


def test_gap_index_rejects_unknown_metric_names():
    orig, emb = _reference_fixture()

    with pytest.raises(ValueError, match="Unknown scipy distance metric"):
        gap_index.compute(orig, emb, metric="not-a-distance")
