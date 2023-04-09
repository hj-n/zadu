from .utils import pairwise_dist as pdist
import numpy as np

def run(orig, emb, distance_matrices=None):
    """
	Compute stress of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
        tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
	OUTPUT:
		dict: stress
	"""
    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
	
    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    diff_squared_sum = np.square(orig_distance_matrix - emb_distance_matrix).sum()
    orig_squared_sum = np.square(orig_distance_matrix).sum()

    stress = np.sqrt(diff_squared_sum / orig_squared_sum)
    
    return {
			"stress": stress
	}