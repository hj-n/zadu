import pairwise_dist as pdist
import numpy as np

def knn_with_ranking(points, k, distance_matrix=None):
  """
  Compute the k-nearest neighbors of the points along with the 
  rankings of other points based on the distance to each point
  INPUT:
		ndarray: points: list of points
		int: k: number of nearest neighbors to compute
		ndarray: distance_matrix: pairwise distance matrix (Optional)
  OUTPUT:
		ndarray: knn_indices: k-nearest neighbors of each point 
		ndarray: ranking: ranking of other points based on the distance to each point
  """
  
  if distance_matrix is None:
    distance_matrix = pdist.pairwise_distance_matrix(points, "euclidean")

  knn_indices = np.empty((points.shape[0], k), dtype=np.int32)
  ranking = np.empty((points.shape[0], points.shape[0]), dtype=np.int32)
  
  for i in range(points.shape[0]):
    distance_to_i = distance_matrix[i]
    sorted_indices = np.argsort(distance_to_i)
    knn_indices[i] = sorted_indices[1:k+1]
    ranking[i] = np.argsort(sorted_indices)
  
  return knn_indices, ranking
    
