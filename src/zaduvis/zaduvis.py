import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import QhullError, Voronoi, voronoi_plot_2d
from sklearn.neighbors import kneighbors_graph

from .colormap import checkviz_cmap


def checkviz(
    scatter_data,
    false_distortion_list,
    missing_distortion_list,
    ax=None,
    point_c="black",
    point_s=1,
    point_alpha=0.5,
    point_marker="o",
):
    scatter_data, false_distortion_list, missing_distortion_list = _validate_inputs(
        scatter_data, false_distortion_list, missing_distortion_list
    )
    try:
        vor = Voronoi(scatter_data)
    except QhullError as exc:
        raise ValueError("checkviz requires distinct, non-collinear 2D points") from exc
    ## set size
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_width=0)

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if region and -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(
                *zip(*polygon, strict=True),
                checkviz_cmap(
                    false_distortion_list[point_idx],
                    missing_distortion_list[point_idx],
                ),
            )

    ax.scatter(
        scatter_data[:, 0],
        scatter_data[:, 1],
        c=point_c,
        zorder=2,
        s=point_s,
        alpha=point_alpha,
        marker=point_marker,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def reliability_map(
    emb,
    false_distortion_list,
    missing_distortion_list,
    k=7,
    ax=None,
    point_c="black",
    point_s=1,
    point_alpha=0.5,
    point_marker="o",
    linewidth=2.5,
    line_alpha=0.8,
):
    emb, false_distortion_list, missing_distortion_list = _validate_inputs(
        emb, false_distortion_list, missing_distortion_list
    )
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 1 or k >= emb.shape[0]:
        raise ValueError(f"k must satisfy 1 <= k < n (n={emb.shape[0]}), got k={k}")

    ## construct a knn graph
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    knn_graph = kneighbors_graph(emb, k, mode="distance", include_self=False)

    ## visualizae points and knn graph
    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=point_c,
        zorder=2,
        s=point_s,
        alpha=point_alpha,
        marker=point_marker,
    )
    drawn_edges = set()
    for i in range(emb.shape[0]):
        for j in knn_graph[i].indices:
            edge = tuple(sorted((int(i), int(j))))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            color = checkviz_cmap(
                (false_distortion_list[i] + false_distortion_list[j]) / 2,
                (missing_distortion_list[i] + missing_distortion_list[j]) / 2,
            )
            ax.plot(
                [emb[i, 0], emb[j, 0]],
                [emb[i, 1], emb[j, 1]],
                c=color,
                zorder=1,
                linewidth=linewidth,
                alpha=line_alpha,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _validate_inputs(scatter_data, false_distortion_list, missing_distortion_list):
    points = np.asarray(scatter_data, dtype=float)
    false_values = np.asarray(false_distortion_list, dtype=float)
    missing_values = np.asarray(missing_distortion_list, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"scatter data must have shape (n, 2), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("At least three points are required")
    if false_values.shape != (points.shape[0],) or missing_values.shape != (
        points.shape[0],
    ):
        raise ValueError("Distortion arrays must contain one value per point")
    if not all(
        np.all(np.isfinite(values)) for values in (points, false_values, missing_values)
    ):
        raise ValueError("Visualization inputs must contain only finite values")
    return points, false_values, missing_values
