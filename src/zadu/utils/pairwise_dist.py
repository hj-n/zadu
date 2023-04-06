from scipy.spatial.distance import cdist

def pairwise_distance_matrix(point, distance_function="euclidean"):
	"""
	Compute the pairwise distance matrix of the point list
	INPUT:
		ndarray: point: list of points
		str: distance_function: distance function to use
	OUTPUT:
		ndarry: pairwise distance matrix 
	"""

	if distance_function in set({
		"braycurtis", "canberra", "chebyshev", "cityblock", "correlation", 
		"cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", 
		"kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", 
		"russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"
	}):
		distance_matrix = cdist(point, point, distance_function)
	elif distance_function == "snn":
		## TODO
		pass
	return distance_matrix