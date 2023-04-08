import numpy as np 
from .utils import knn

def run(orig, emb, k=20, knn_ranking_info=None, return_local=False):
	"""
	Compute the trustworthiness and continuity of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		int: k: number of nearest neighbors to consider
		tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
	OUTPUT:
		dict: trustworthiness and continuity
	"""
	if knn_ranking_info is None:
		orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
		emb_knn_indices,  emb_ranking  = knn.knn_with_ranking(emb, k)
	else:
		orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

	if return_local:
		trust, local_trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
		cont , local_cont  = tnc_computation(emb_knn_indices,  emb_ranking, orig_knn_indices, k, return_local)
		return ({
			"trustworthiness": trust,
			"continuity": cont
		}, {
			"local_trustworthiness": local_trust,
			"local_continuity": local_cont
		})
	else:
		trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
		cont  = tnc_computation(emb_knn_indices,  emb_ranking, orig_knn_indices, k, return_local)
		return {
			"trustworthiness": trust,
			"continuity": cont
		}

def tnc_computation(base_knn_indices, base_ranking, target_knn_indices, k, return_local=False):
	"""
	Core computation of trustworthiness and continuity
	"""
	local_distortion_list = []
	points_num = base_knn_indices.shape[0]

	for i in range(points_num):
		missings = np.setdiff1d(target_knn_indices[i], base_knn_indices[i])
		local_distortion = 0.0 
		for missing in missings:
			local_distortion += base_ranking[i, missing] - k
		local_distortion_list.append(local_distortion)
	local_distortion_list = np.array(local_distortion_list)
	local_distortion_list = 1 - local_distortion_list * (2 / (k * (2 * points_num - 3 * k - 1)))

	average_distortion = np.mean(local_distortion_list)

	if return_local:
		return average_distortion, local_distortion_list
	else:
		return average_distortion
	

	

	