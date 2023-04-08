import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def cielab_to_rgb_hex(L, a, b):
	# Convert CIELAB to XYZ
	def lab_to_xyz(L, a, b):
		y = (L + 16) / 116
		x = a / 500 + y
		z = y - b / 200

		x = x ** 3 if x ** 3 > 0.008856 else (x - 16 / 116) / 7.787
		y = y ** 3 if y ** 3 > 0.008856 else (y - 16 / 116) / 7.787
		z = z ** 3 if z ** 3 > 0.008856 else (z - 16 / 116) / 7.787

		x = x * 95.047
		y = y * 100
		z = z * 108.883

		return x, y, z

	# Convert XYZ to RGB
	def xyz_to_rgb(x, y, z):
		x /= 100
		y /= 100
		z /= 100

		r = x * 3.2406 + y * -1.5372 + z * -0.4986
		g = x * -0.9689 + y * 1.8758 + z * 0.0415
		b = x * 0.0557 + y * -0.2040 + z * 1.0570

		def convert(color):
			color = np.clip(color, 0, 1)
			color = np.where(color <= 0.0031308, 12.92 * color, 1.055 * np.power(color, 1 / 2.4) - 0.055)
			return np.round(color * 255).astype(int)

		r, g, b = convert(r), convert(g), convert(b)
		return r, g, b

	x, y, z = lab_to_xyz(L, a, b)
	r, g, b = xyz_to_rgb(x, y, z)
	return f"#{r:02x}{g:02x}{b:02x}"

def colormap(dist_false, dist_missing):
	## generate a 2D continous colormap for the missing and false distortion
	"""
	Generate a matplotlib color code that reproduces the above js code
	"""
	cScale = 1.3
	dist_false = 1 - dist_false
	dist_missing = 1 - dist_missing

	powScale = lambda x: x ** 1.5145
	aScale = lambda x: 30 * cScale * x
	bScale = lambda x: 20 * cScale * x

	lab = [powScale(1 - (dist_false + dist_missing) / 2) * 100, aScale(dist_false - dist_missing), bScale(dist_missing - dist_false)]
	## change the cielab color to rgb that can be used by matplotlib
	# print(lab)
	color = cielab_to_rgb_hex(*lab)
	return color

def draw_voronoi_with_score(scatter_data, score_list_A, score_list_B):
	vor = Voronoi(scatter_data)
	## set size
	fig, ax = plt.subplots(figsize=(10, 10))

	voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False,  line_width=0)



	for idx, region in enumerate(vor.regions[:-1]):
		if not -1 in region:
			polygon = [vor.vertices[i] for i in region]
			ax.fill(*zip(*polygon),  colormap(score_list_A[idx], score_list_B[idx]) )

	ax.scatter(scatter_data[:, 0], scatter_data[:, 1], c='black', zorder=2, s=1, alpha=0.5, marker='o')

	plt.show()

# Example usage


def checkviz(emb, distortion_false, distortion_missing, ax=None):
	draw_voronoi_with_score(emb, distortion_false, distortion_missing)

def reliability_map(emb, distortion):
	pass