import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from .colormap import checkviz_cmap




def checkviz(
	scatter_data, score_list_A, score_list_B, 
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
			ax.fill(*zip(*polygon),  checkviz_cmap(score_list_A[idx], score_list_B[idx]) )

	ax.scatter(scatter_data[:, 0], scatter_data[:, 1], c=point_c, zorder=2, s=point_s, alpha=point_alpha, marker=point_marker)

	ax.set_xticks([])
	ax.set_yticks([])

	plt.show()



def reliability_map(emb, distortion):
	pass