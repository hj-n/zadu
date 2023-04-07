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
      G1 = kneighbors_graph(orig, k, n_jobs=-1).toarray()
      G2 = kneighbors_graph(emb, k, n_jobs=-1).toarray()
    else:
      A1, A2 = knn_info

      G1 = np.zeros((A1.shape[0], A1.shape[0]))
      G2 = np.zeros((A2.shape[0], A2.shape[0]))

      np.add.at(G1, (np.arange(A1.shape[0]), A1 - 1), 1)
      np.add.at(G2, (np.arange(A2.shape[0]), A2 - 1), 1)
      


    G1 = ((G1 + G1.T) > 0).astype(float)
    G2 = ((G2 + G2.T) > 0).astype(float)

    S1 = G1 @ G1.T
    S2 = G2 @ G2.T

    D = (S1 - S2) / k
    np.fill_diagonal(D, 0)

    D_plus = D[D > 0]
    D_minus = D[D < 0]

    dissim_plus = np.sqrt(np.sum(D_plus**2))
    dissim_minus = np.sqrt(np.sum(D_minus**2))

    nd = max(dissim_plus, dissim_minus)

    return {
        "neighbor_dissimilarity": nd
    }