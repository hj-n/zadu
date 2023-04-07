from sklearn.neighbors import kneighbors_graph
import numpy as np



def neighborhood_dissimilarity(orig, emb, k, knn_info=None):
    """
    Compute neighbor dissimilarity (ND) of the embedding
    INPUT:
      ndarray: orig: original data
      ndarray: emb: embedded data
      int: k: number of nearest neighbors to consider
      tuple: knn_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
      dict: neighbor dissimilarity (ND)
    """

    if knn_info is None:
      orig_knn_graph = kneighbors_graph(orig, k, n_jobs=-1).toarray()
      emb_knn_graph = kneighbors_graph(emb, k, n_jobs=-1).toarray()
    else:
      orig_knn_indices, emb_knn_indices = knn_info

      orig_knn_graph = np.zeros((orig_knn_indices.shape[0], orig_knn_indices.shape[0]))
      emb_knn_graph = np.zeros((emb_knn_indices.shape[0], emb_knn_indices.shape[0]))
      
      for i in range(orig_knn_indices.shape[1]):
        orig_knn_graph[np.arange(orig_knn_indices.shape[0]), orig_knn_indices[:, i]] = 1
        emb_knn_graph[np.arange(emb_knn_indices.shape[0]), emb_knn_indices[:, i]] = 1
        
          
      
		


    orig_knn_graph = ((orig_knn_graph + orig_knn_graph.T) > 0).astype(float)
    emb_knn_graph  = ((emb_knn_graph + emb_knn_graph.T) > 0).astype(float)

    orig_SNN = orig_knn_graph @ orig_knn_graph.T
    emb_SNN  = emb_knn_graph @ emb_knn_graph.T

    D = (orig_SNN - emb_SNN) / k
    np.fill_diagonal(D, 0)

    D_plus = D[D > 0]
    D_minus = D[D < 0]

    dissim_plus = np.sqrt(np.sum(D_plus**2))
    dissim_minus = np.sqrt(np.sum(D_minus**2))

    nd = max(dissim_plus, dissim_minus)

    return {
        "neighborhood_dissimilarity": nd
    }