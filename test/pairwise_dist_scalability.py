import sys

sys.path.append('../src/zadu/utils')
import pairwise_dist as pdist 
import numpy as np
import time 
from scipy.spatial.distance import cdist

points = np.random.rand(5000, 20)



start = time.time()
cdist(points, points, 'euclidean')
end = time.time()

print(end - start, "seconds for scipy implementation")


start = time.time()
pdist.euclidean_pairwise_distance_matrix(points)
end = time.time()

print(end - start, "seconds for numba implementation")

