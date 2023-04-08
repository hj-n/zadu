import numpy as np
from .utils import knn


def run(orig, emb, k=20, knn_info=None):
  """
  Compute the local continuity meta-criteria of the embedding
  INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		int: k: number of nearest neighbors to consider
		tuple: knn_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
  OUTPUT:
		dict: local continuity meta-criteria
  """
  if knn_info is None:
    orig_knn_indices = knn.knn(orig, k)
    emb_knn_indices = knn.knn(emb, k)
  else:
    orig_knn_indices, emb_knn_indices = knn_info

  point_num = orig.shape[0]
  
  value = 0.0
  for i in range(point_num):
    value += np.intersect1d(orig_knn_indices[i], emb_knn_indices[i]).shape[0] - ((k * k) / (point_num - 1))
  
  lcmc = value / (point_num * k)
  return {
    "lcmc": lcmc
	}
     
