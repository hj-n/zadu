from snc.snc import SNC
import numpy as np
from .utils import knn



def measure(orig, emb, iteration=150, walk_num_ratio=0.3, alpha=0.1, k=50, clustering_strategy="dbscan", knn_info=None, return_local=False):
	"""
	Compute the Steadiness and Cohesiveness of the embedding
	"""

	if knn_info is None:
		orig_knn_indices = knn.knn(orig, k)
		emb_knn_indices = knn.knn(emb, k)
	else:
		orig_knn_indices, emb_knn_indices = knn_info
	
	orig_snn_graph = knn.snn(orig, k, knn_indices=orig_knn_indices, directed=True)
	emb_snn_graph = knn.snn(emb, k, knn_indices=emb_knn_indices, directed=True)

	snn_knn_matrix = {
		"raw_knn": orig_knn_indices,
		"raw_snn": orig_snn_graph,
		"emb_knn": emb_knn_indices,
		"emb_snn": emb_snn_graph
	}


	snc_obj = SNC(
		orig, emb, 
		iteration=iteration, 
		walk_num_ratio=walk_num_ratio, 
		dist_strategy="inject_snn", 
		dist_parameter={ "alpha": alpha }, 
		dist_function=None, 
		cluster_strategy=clustering_strategy, 
		snn_knn_matrix=snn_knn_matrix
	)

	snc_obj.fit(record_vis_info=return_local)

	steadiness = snc_obj.steadiness()
	cohesiveness = snc_obj.cohesiveness()

	if return_local:
		_, _, _, points_info = snc_obj.vis_info()

		stead_local = [1 - point_info["false_val"] for point_info in points_info]
		cohev_local = [1 - point_info["missing_val"] for point_info in points_info]

	if return_local:
		return { ## TODO
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}, {
			"local_steadiness": stead_local,
			"local_cohesiveness": cohev_local
		}
	else:
		return {
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}
