import numpy as np
from .utils import knn

def run(orig, emb, label, k, knn_ranking_info=None):
	"""
	Compute class-aware trustworthiness and continuity of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		ndarray: label: label of the original data
		int: k: number of nearest neighbors to consider
		tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
	OUTPUT:
		dict: class-aware trustworthiness (ca_trustworthiness) and class-aware continuity (ca_continuity)
	"""

	if knn_ranking_info is None:
		orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
		emb_knn_indices,  emb_ranking  = knn.knn_with_ranking(emb, k)
	else:
		orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info
	
	## class-aware trustworthiness
	ca_trust = ca_tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, label, k, "false")
	## class-aware continuity
	ca_cont  = ca_tnc_computation(emb_knn_indices,  emb_ranking, orig_knn_indices, label, k, "missing")

	return {
		"ca_trustworthiness": ca_trust,
		"ca_continuity": ca_cont
	}

def ca_tnc_computation(base_knn_indices, base_ranking, target_knn_indices, label, k, type):
	"""
	Core computation of class-aware trustworthiness and continuity
	"""
	value = 0.0
	points_num = base_knn_indices.shape[0]

	for i in range(points_num):
		# get nearest neighbors that in the target indices but not in the base indices
		missings = np.setdiff1d(target_knn_indices[i], base_knn_indices[i])

		for missing in missings:
			if type == "false":
				if label[i] != label[missing]:
					value += base_ranking[i, missing] - k
			elif type == "missing":
				if label[i] == label[missing]:
					value += base_ranking[i, missing] - k
			else:
				raise ValueError("type should be 'false' or 'missing'")

	return 1 - 2 / (points_num * k * (2 * points_num - 3 * k - 1)) * value