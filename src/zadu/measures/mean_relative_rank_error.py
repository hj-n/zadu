import numpy as np 
from .utils import knn


def run(orig, emb, k=20, knn_ranking_info=None, return_local=False):
	"""
	Compute Mean Relative Rank Error (MRRE) of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		int: k: number of nearest neighbors to consider
		tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
	OUTPUT:
		dict: MRRE_false and MRRE_missing
	"""
	if knn_ranking_info is None:
		orig_knn_indices, orig_ranking = knn.knn_with_ranking(orig, k)
		emb_knn_indices,  emb_ranking  = knn.knn_with_ranking(emb, k)
	else:
		orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

	## MRRE_false
	mrre_false = mrre_computation(orig_ranking, emb_ranking, emb_knn_indices, k, return_local)
	## MRRE_missing
	mrre_missing = mrre_computation(emb_ranking, orig_ranking, orig_knn_indices, k, return_local)

	return {
		"mrre_false": mrre_false,
		"mrre_missing": mrre_missing,
	}

def mrre_computation(base_ranking,target_ranking, target_knn_indices, k, return_local=False):
	"""
	Core computation of MRRE
	"""
	local_distortion_list = []
	points_num = target_knn_indices.shape[0]
	for i in range(points_num):
		base_rank_arr   = base_ranking[i][target_knn_indices[i]]
		target_rank_arr = target_ranking[i][target_knn_indices[i]]
		local_distortion_list.append(np.sum(np.abs(base_rank_arr - target_rank_arr) / target_rank_arr))
	
	c = sum([abs(points_num - 2 * i + 1) / i for i in range(1, k + 1)])
	local_distortion_list = np.array(local_distortion_list)
	local_distortion_list = 1 - local_distortion_list / c

	average_distortion = np.mean(local_distortion_list)

	if return_local:
		return {
			"average_distortion": average_distortion,
			"local_distortion_list": local_distortion_list
		}
	else:
		return average_distortion
	


	
	
