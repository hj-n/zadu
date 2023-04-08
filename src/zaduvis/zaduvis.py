import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from .colormap import checkviz_cmap

from sklearn.neighbors import kneighbors_graph



def checkviz(
	scatter_data, false_distortion_list, missing_distortion_list, 
	ax=None, point_c="black", point_s=1, point_alpha=0.5, point_marker="o",
):
	vor = Voronoi(scatter_data)
	## set size
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 10))

	voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False,  line_width=0)

	for idx, region in enumerate(vor.regions[:-1]):
		if not -1 in region:
			polygon = [vor.vertices[i] for i in region]
			ax.fill(*zip(*polygon),  checkviz_cmap(false_distortion_list[idx], missing_distortion_list[idx]) )

	ax.scatter(scatter_data[:, 0], scatter_data[:, 1], c=point_c, zorder=2, s=point_s, alpha=point_alpha, marker=point_marker)

	ax.set_xticks([])
	ax.set_yticks([])

	plt.show()



def reliability_map(emb, false_distortion_list, missing_distortion_list, k=7, ax=None):
	## construct a knn graph
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 10))

	knn_graph = kneighbors_graph(emb, k, mode="distance", include_self=False)

	## visualizae points and knn graph
	ax.scatter(emb[:, 0], emb[:, 1], c="black", zorder=2, s=1, alpha=0.5, marker="o")
	for i in range(emb.shape[0]):
		for j in knn_graph[i].indices:
			color = checkviz_cmap((false_distortion_list[i] + false_distortion_list[j]) / 2, (missing_distortion_list[i] + missing_distortion_list[j]) / 2)
			ax.plot(
				[emb[i, 0], emb[j, 0]], [emb[i, 1], emb[j, 1]], 
				c=color, zorder=1, linewidth=2.5, alpha=0.8
			)
	

	
