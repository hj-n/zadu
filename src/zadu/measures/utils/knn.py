from .pairwise_dist import pairwise_distance_matrix
import numpy as np
import faiss
from sklearn.neighbors import KDTree

def knn_with_ranking(points, k, distance_matrix=None):
  """
  Compute the k-nearest neighbors of the points along with the 
  rankings of other points based on the distance to each point.
  If the distance matrix is not provided, it is computed in O(n^2) time.
  INPUT:
		ndarray: points: list of points
		int: k: number of nearest neighbors to compute
		ndarray: distance_matrix: pairwise distance matrix (Optional)
  OUTPUT:
		ndarray: knn_indices: k-nearest neighbors of each point 
		ndarray: ranking: ranking of other points based on the distance to each point
  """
  
  if distance_matrix is None:
    distance_matrix = pairwise_distance_matrix(points, "euclidean")

  knn_indices = np.empty((points.shape[0], k), dtype=np.int32)
  ranking = np.empty((points.shape[0], points.shape[0]), dtype=np.int32)
  
  for i in range(points.shape[0]):
    distance_to_i = distance_matrix[i]
    sorted_indices = np.argsort(distance_to_i)
    knn_indices[i] = sorted_indices[1:k+1]
    ranking[i] = np.argsort(sorted_indices)
  
  return knn_indices, ranking
    

def knn(points, k, distance_function="euclidean"):
  """
  Compute the k-nearest neighbors of the points
  If the distance function is euclidean, the computation relies on faiss-cpu.
  Otherwise, the computation is done based on scikit-learn KD Tree algorithm
  You can use any distance function supported by scikit-learn KD Tree or specify a callable function
  INPUT:
		ndarray: points: list of points
		int: k: number of nearest neighbors to compute
		str or callable: distance_function: distance function to use
  OUTPUT:
		ndarray: knn_indices: k-nearest neighbors of each point 
	"""
	
	## make c-contiguous
  points = np.ascontiguousarray(points, dtype=np.float32)

  if distance_function == "euclidean":
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)
    knn_indices = index.search(points, k+1)[1][:, 1:]
  else:
    tree = KDTree(points, metric=distance_function)
    knn_indices = tree.query(points, k=k+1, return_distance=False)[:, 1:]
	
  return knn_indices

def snn(points, k, distance_function="euclidean", directed=True, knn_indices=None):
  """
	Compute the shared nearest neighbors (SNN) graph of the points
	INPUT:
		ndarray: points: list of points
		int: k: number of nearest neighbors to consider
		str or callable: distance_function: distance function to use
    bool: directed: whether the k-nearest neighbors graph using is directed or not
		tuple: knn_info: precomputed k-nearest neighbors and rankings of the points (Optional)
	OUTPUT:
		ndarray: snn_graph: shared nearest neighbors (SNN) graph of the points
	"""
  if knn_indices is None:
    knn_indices = knn(points, k, distance_function)

  knn_graph = np.zeros((knn_indices.shape[0], knn_indices.shape[0]))
  for i in range(k):
    knn_graph[np.arange(knn_indices.shape[0]), knn_indices[:, i]] = k-i

  if directed:
    snn_graph = knn_graph @ knn_graph.T
  else:
    knn_graph = ((knn_graph + knn_graph.T) > 0).astype(float)
    snn_graph = knn_graph @ knn_graph

  np.fill_diagonal(snn_graph, 0)

  return snn_graph