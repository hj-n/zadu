import numpy as np
import numpy.typing as npt

from .utils.snc_cpu import SNCCPU
from .utils.validation import validate_pair


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    iteration: int = 150,
    walk_num_ratio: float = 0.3,
    alpha: float = 0.1,
    k: int | None = None,
    clustering_strategy: str = "dbscan",
    knn_info: tuple | None = None,
    return_local: bool = False,
    random_state: int | np.random.Generator | None = None,
    n_jobs: int = 1,
    working_memory_bytes: int | None = None,
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
            int: random_state: seed for reproducible walks and KMeans clustering
            int: n_jobs: number of exact iteration workers
            int: working_memory_bytes: internal per-measure working-memory budget (Optional)
    OUTPUT:
            dict: steadiness and cohesiveness score
    """

    orig, emb = validate_pair(orig, emb)
    snc_obj = SNCCPU(
        orig,
        emb,
        iteration=iteration,
        walk_num_ratio=walk_num_ratio,
        alpha=alpha,
        k=k,
        cluster_strategy=clustering_strategy,
        random_state=random_state,
        n_jobs=n_jobs,
        working_memory_bytes=working_memory_bytes,
    )

    snc_obj.fit(record_vis_info=return_local, knn_info=knn_info)

    steadiness = snc_obj.steadiness()
    cohesiveness = snc_obj.cohesiveness()

    if return_local:
        stead_local, cohev_local = snc_obj.local_scores()

    if return_local:
        return {"steadiness": steadiness, "cohesiveness": cohesiveness}, {
            "local_steadiness": stead_local,
            "local_cohesiveness": cohev_local,
        }
    else:
        return {"steadiness": steadiness, "cohesiveness": cohesiveness}
