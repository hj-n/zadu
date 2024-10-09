from .utils import pairwise_dist as pdist
import numpy as np
from .stress import measure as stressmeasure

def measure(orig, emb, distance_matrices=None):
    """
    Compute the minimum stress for a given embedding.
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

    alpha = np.sum(np.multiply(orig_distance_matrix, emb_distance_matrix)) / np.sum(np.square(emb_distance_matrix))
    sns = stressmeasure(orig, alpha * emb, (orig_distance_matrix, alpha * emb_distance_matrix))

    return {
        "scale_normalized_stress": sns["stress"]
    }
