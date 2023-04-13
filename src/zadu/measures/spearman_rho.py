from scipy.stats import spearmanr
from .utils import pairwise_dist as pdist

def measure(orig, emb, distance_matrices=None):
  """
  Compute Spearman's rank correlation coefficient of the embedding
  INPUT:
		ndarray: orig: original data
		ndarray: emb: embedded data
		tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
	OUTPUT:
		dict: Spearman's rank correlation coefficient (rho)
	"""
  
  if distance_matrices is None:
    orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
    emb_distance_matrix = pdist.pairwise_distance_matrix(emb)
  else:
    orig_distance_matrix, emb_distance_matrix = distance_matrices

  rho, p = spearmanr(orig_distance_matrix.flatten(), emb_distance_matrix.flatten())
  return {
		"spearman_rho": rho,
	}