from scipy.stats import pearsonr
from .utils import pairwise_dist as pdist

def pearson_r(orig, emb, distance_matrices=None):
  """
	Compute Pearson's correlation coefficient of the embedding
	INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
	OUTPUT:
		dict: Pearson's correlation coefficient (r)
  """
  
  if distance_matrices is None:
    orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
    emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
  else:
    orig_distance_matrix, emb_distance_matrix = distance_matrices

  r, p = pearsonr(orig_distance_matrix.flatten(), emb_distance_matrix.flatten())
  return {
		"pearson_r": r,
	}
  
