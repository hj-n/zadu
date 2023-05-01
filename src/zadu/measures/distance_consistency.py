import numpy as np

def measure(emb, label):
  """
	Compute distance consistency of the embedding
	INPUT:
		ndarray: emb: embedded data
		ndarray: label: label of the original data
	OUTPUT:
		dict: distance consistency (dsc)
  """
  
	## compute centroids
  point_num = emb.shape[0]
  label_num = np.unique(label).shape[0]

  centroids = np.zeros((label_num, emb.shape[1]))
  for i in range(label_num):
    centroids[i] = np.mean(emb[label == i], axis=0)
    
	## compute distance consistency
  consistent_num = 0
  for idx in range(point_num):
    current_label = -1
    current_dist = 1e10
    for c_idx in range(len(centroids)):
      dist = np.linalg.norm(emb[idx] - centroids[c_idx])
      if dist < current_dist:
        current_dist = dist
        current_label = c_idx
    if current_label == label[idx]:
      consistent_num += 1
	
  dsc = consistent_num / point_num
  return {
    "distance_consistency": dsc
	}