from sklearn.neighbors import kneighbors_graph
import numpy as np



def neighbor_dissimilarity(orig, emb, k, knn_info=None):
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
      point_num = orig_knn_indices.shape[0]

      orig_knn_graph = np.zeros((point_num, point_num))
      emb_knn_graph = np.zeros((point_num, point_num))

      np.add.at(orig_knn_graph, (np.arange(point_num), orig_knn_indices - 1), 1)
      np.add.at(emb_knn_graph, (np.arange(point_num), emb_knn_indices - 1), 1)
      


    orig_knn_graph = ((orig_knn_graph + orig_knn_graph.T) > 0).astype(float)
    emb_knn_graph = ((emb_knn_graph + emb_knn_graph.T) > 0).astype(float)

    orig_similarity = np.matmul(orig_knn_graph, orig_knn_graph.T)
    emb_similarity = np.matmul(emb_knn_graph, emb_knn_graph.T)

    D = (orig_similarity - emb_similarity) / k
    np.fill_diagonal(D, 0)

    nd_plus = np.sqrt(np.sum(D[D > 0]**2))
    nd_minus = np.sqrt(np.sum(D[D < 0]**2))

    nd = max(nd_plus, nd_minus)

    return {
        "neighbor_dissimilarity": nd
    }