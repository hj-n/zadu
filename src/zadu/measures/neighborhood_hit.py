from .utils import knn
import numpy as np

def run(emb, label, k=20, knn_emb_info=None):
  """
	Compute neighborhood hit of the embedding
	INPUT:
		ndarray: emb: embedded data
		ndarray: label: label of the original data
		int: k: number of nearest neighbors to consider
		tuple: knn_info: precomputed k-nearest neighbors of the original and embedded data (Optional)
	OUTPUT:
		dict: neighborhood hit (nh)
	"""
  if knn_emb_info is None:
    emb_knn_indices = knn.knn(emb, k)
  else:
    emb_knn_indices = knn_emb_info
  
  points_num = emb.shape[0]
  nh = 0
  for i in range(points_num):
    emb_knn_index = emb_knn_indices[i]
    emb_knn_index_label = label[emb_knn_index]
    nh += np.sum((emb_knn_index_label == label[i]).astype(int))
  
  nh = nh / (points_num * k)
  
  return {
    "neighborhood_hit": nh
	}
  

  