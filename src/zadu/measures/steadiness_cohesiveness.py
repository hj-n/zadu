from snc.snc import SNC



def run(orig, emb, iteration=150, walk_num_ratio=0.3, alpha=0.1, k="sqrt", clustering_strategy="dbscan", return_local=False):
	"""
	Compute the Steadiness and Cohesiveness of the embedding
	"""

	snc_obj = SNC(orig, emb, iteration, walk_num_ratio, "snn", { "alpha": alpha, "k": k}, None, clustering_strategy)

	snc_obj.fit(record_vis_info=return_local)

	steadiness = snc_obj.steadiness()
	cohesiveness = snc_obj.cohesiveness()

	if return_local:
		return None, { ## TODO
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}
	else:
		return {
			"steadiness": steadiness,
			"cohesiveness": cohesiveness
		}
