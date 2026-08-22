import matplotlib
import numpy as np
from scipy.spatial import Voronoi

matplotlib.use("Agg")

from zaduvis import checkviz, reliability_map
from zaduvis import zaduvis as zaduvis_module


def _points():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.2, 0.4],
            [0.7, 0.2],
            [0.4, 0.8],
            [0.8, 0.7],
            [0.5, 0.5],
            [0.3, 0.15],
        ]
    )


def test_checkviz_maps_voronoi_regions_through_point_region(monkeypatch):
    points = _points()
    false_values = np.arange(len(points), dtype=float) / len(points)
    missing_values = false_values[::-1]
    calls = []

    def fake_colormap(false_value, missing_value):
        calls.append((false_value, missing_value))
        return "#000000"

    monkeypatch.setattr(zaduvis_module, "checkviz_cmap", fake_colormap)
    ax = checkviz(points, false_values, missing_values)

    vor = Voronoi(points)
    finite_points = [
        point_idx
        for point_idx, region_idx in enumerate(vor.point_region)
        if vor.regions[region_idx] and -1 not in vor.regions[region_idx]
    ]
    expected = [(false_values[i], missing_values[i]) for i in finite_points]
    assert calls == expected
    assert ax is not None


def test_reliability_map_returns_axes_and_deduplicates_mutual_edges():
    points = _points()
    values = np.ones(len(points))

    ax = reliability_map(points, values, values, k=2)
    rendered_edges = []
    for line in ax.lines:
        x_values = line.get_xdata()
        y_values = line.get_ydata()
        rendered_edges.append(
            frozenset(
                (
                    (float(x_values[0]), float(y_values[0])),
                    (float(x_values[1]), float(y_values[1])),
                )
            )
        )

    assert len(set(rendered_edges)) == len(rendered_edges)
    assert ax is not None
