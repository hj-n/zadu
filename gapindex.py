import math
import numpy as np
from scipy.spatial import distance, Delaunay
from scipy.linalg import issymmetric


# Returns the area of a triangle defined by 3 points
def area_triangle(tri, metric):
    if metric == 'precomputed':
        (a,b,c) = tri
    else:
        a = metric(tri[0],tri[1])
        b = metric(tri[0],tri[2])
        c = metric(tri[1],tri[2])

    s = (a+b+c) / 2
    try:
        area = math.sqrt( s * (s-a) * (s-b) * (s-c) )
    except ValueError:
        # If one side is too short, it can cause numerical issues
        area = 0
    return area


def gap_index(X, X_2D, metric=distance.euclidean):

    # Triangulation
    triangles = Delaunay(X_2D).simplices

    areas_pro = np.array([area_triangle([X_2D[i] for i in t], distance.euclidean) for t in triangles])
    if metric == 'precomputed':
        assert issymmetric(X)
        areas_ori = np.array([area_triangle(X[[a,a,b], [b,c,c]], metric) for (a,b,c) in triangles])
    else:
        areas_ori = np.array([area_triangle([X[i] for i in t], metric) for t in triangles])

    # Triangle deformations
    areas_ori /= sum(areas_ori)
    areas_pro /= sum(areas_pro)
    tri_deforms = (areas_pro - areas_ori) / np.maximum(np.maximum(areas_ori,areas_pro), 0.000001)

    # Aggregation
    max_areas = np.maximum(areas_ori,areas_pro)
    score = np.sum(np.absolute(np.array(tri_deforms)) * max_areas) / np.sum(max_areas)

    return score
