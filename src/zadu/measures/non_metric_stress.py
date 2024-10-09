from .utils import pairwise_dist as pdist 
from sklearn.isotonic import IsotonicRegression
import numpy as np

def measure(orig,emb,distance_matrices=None):
    """
    Compute the non-metric stress of the embedding, described in:
    Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.
    
    Or for a more modern description: 
    Smelser, K., Miller, J., & Kobourov, S. (2024). " Normalized Stress" is Not Normalized: How to Interpret Stress Correctly. arXiv preprint arXiv:2408.07724.
    
    INPUT: 
        ndarray: orig: orignal data
        ndarray: emb: embedded data
        tuple: distance_matrices: precomputed distance matrices of the original and embedded data (Optional)
    OUTPUT:
        dict: non_metric_stress
    """
    if distance_matrices is None:
        orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
        emb_distance_matrix = pdist.pairwise_distance_matrix(emb)

    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

    #Extract upper triangular of both matrices into 1d arrays 
    #Diagonal is always zero, we can offset by one
    N = orig_distance_matrix.shape[0]
    orig_upper_tri = orig_distance_matrix[np.triu_indices(N,k=1)]
    emb_upper_tri = emb_distance_matrix[np.triu_indices(N,k=1)]

    #Find the indices of orignal distance matrix that, when reordered, would sort it
    sorted_indices = np.argsort(orig_upper_tri)

    #Apply order to both arrays
    orig_upper_tri = orig_upper_tri[sorted_indices]
    emb_upper_tri  = emb_upper_tri[sorted_indices]

    #Create the curve which minimizes horizontal distance in the Shepard diagram
    d_hat = IsotonicRegression().fit(orig_upper_tri,emb_upper_tri).predict(orig_upper_tri)

    #Finally, compute the stress of the horizontal distance between the fitted line and embedded distances
    raw_stress           = np.sum(np.square(emb_upper_tri - d_hat ))
    normalization_factor = np.sum(np.square(emb_upper_tri))

    non_metric_stress = np.sqrt(raw_stress / normalization_factor)

    return {"non_metric_stress": non_metric_stress}
	