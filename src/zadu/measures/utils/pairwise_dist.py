from scipy.spatial.distance import cdist
import numpy as np

def pairwise_distance_matrix(point, distance_function="euclidean"):
	"""
	Compute the pairwise distance matrix of the point list
	You can use any distance function from scipy.spatial.distance.cdist or specify a callable function
	INPUT:
		ndarray: point: list of points
		str or callable: distance_function: distance function to use
	OUTPUT:
		ndarry: pairwise distance matrix 
	"""
	if callable(distance_function):
		distance_matrix = cdist(point, point, distance_function)
	elif distance_function == "snn":
		## TODO
		pass
	else:
		distance_matrix = cdist(point, point, distance_function)
	return distance_matrix

def distance_matrix_to_density(distance_matrix, sigma):
	"""
	Compute the density of each point based on the pairwise distance matrix
	INPUT:
		ndarray: distance_matrix: pairwise distance matrix
		float: sigma: sigma parameter for the Gaussian kernel
	OUTPUT:
		ndarry: density
	"""

	normalized_distance_matrix = distance_matrix / np.max(distance_matrix)
	density = np.sum(np.exp(- (normalized_distance_matrix ** 2) / sigma), axis=-1)
	density = density / np.sum(density)
	return density