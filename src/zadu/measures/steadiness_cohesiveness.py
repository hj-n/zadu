import numpy as np
import numpy.typing as npt

from .utils.snc_cpu import SNCCPU


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    iteration: int = 150,
    walk_num_ratio: float = 0.3,
    alpha: float = 0.1,
    k: int = 50,
    clustering_strategy: str = "dbscan",
    knn_info: tuple | None = None,
    return_local: bool = False,
) -> tuple | dict:
    """
    Compute the Steadiness and Cohesiveness of the embedding
    INPUT:
            ndarray: orig: original data
            ndarray: emb: embedded data
            int: iteration: number of iterations for the SNC algorithm
            float: walk_num_ratio: ratio of the number of random walks to the number of points
            float: alpha: parameter for the SNC algorithm
            int: k: number of nearest neighbors to consider
            str: clustering_strategy: clustering strategy to use (dbscan or kmeans)
            tuple: knn_info: precomputed k-nearest neighbors of the original and embedded data (Optional)
    OUTPUT:
            dict: steadiness and cohesiveness score
    """

    # if knn_info is None:
    #     orig_knn_indices = knn.knn(orig, k)
    #     emb_knn_indices = knn.knn(emb, k)
    # else:
    #     orig_knn_indices, emb_knn_indices = knn_info

    # orig_snn_graph = knn.snn(orig, k, knn_indices=orig_knn_indices, directed=True)
    # emb_snn_graph = knn.snn(emb, k, knn_indices=emb_knn_indices, directed=True)

    # snn_knn_matrix = {
    #     "raw_knn": orig_knn_indices,
    #     "raw_snn": orig_snn_graph,
    #     "emb_knn": emb_knn_indices,
    #     "emb_snn": emb_snn_graph,
    # }

    # Keep parity with the previous zadu behavior backed by `snc` package:
    # zadu passed only alpha to SNC, so SNC internally used k=sqrt(N).
    snc_obj = SNCCPU(
        orig,
        emb,
        iteration=iteration,
        walk_num_ratio=walk_num_ratio,
        alpha=alpha,
        k=None,
        cluster_strategy=clustering_strategy,
    )

    snc_obj.fit(record_vis_info=return_local)

    steadiness = snc_obj.steadiness()
    cohesiveness = snc_obj.cohesiveness()

    if return_local:
        stead_local, cohev_local = snc_obj.local_scores()

    if return_local:
        return {"steadiness": steadiness, "cohesiveness": cohesiveness}, {  # TODO
            "local_steadiness": stead_local,
            "local_cohesiveness": cohev_local,
        }
    else:
        return {"steadiness": steadiness, "cohesiveness": cohesiveness}
