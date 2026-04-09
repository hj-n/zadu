import math
import io
from PIL import Image
import numpy as np
from scipy.spatial import distance, Delaunay
from scipy.linalg import issymmetric
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# Returns the area of a triangle defined by 3 points
def area_triangle(tri, metric):
    if metric == 'precomputed':
        (a,b,c) = tri  # tri is assumed to be the three edge lengths already
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


def compute_areas(X, triangles, metric=distance.euclidean):
    if metric == 'precomputed':
        assert issymmetric(X) # X is assumed to be a square distance matrix
        areas = np.array([area_triangle(X[[a,a,b], [b,c,c]], metric) for (a,b,c) in triangles])
    else:
        areas = np.array([area_triangle([X[i] for i in t], metric) for t in triangles])

    return areas / sum(areas)


def gap_index(X, X_2D, metric=distance.euclidean):

    triangles = Delaunay(X_2D).simplices

    areas_ori = compute_areas(X, triangles, metric)
    areas_pro = compute_areas(X_2D, triangles, distance.euclidean) # For X_2D use Euclidean

    tri_deforms = (areas_pro - areas_ori) / np.maximum(np.maximum(areas_ori,areas_pro), 0.000001)

    max_areas = np.maximum(areas_ori,areas_pro)
    score = np.sum(np.absolute(np.array(tri_deforms)) * max_areas) / np.sum(max_areas)

    return score


def gi_visualization(X, X_2D, metric=distance.euclidean, blur=0, filename=None):

    # Compute triangle deformation
    triangles = Delaunay(X_2D).simplices
    areas_ori = compute_areas(X, triangles, metric)
    areas_pro = compute_areas(X_2D, triangles, distance.euclidean)
    tri_deforms = (areas_pro - areas_ori) / np.maximum(np.maximum(areas_ori,areas_pro), 0.000001)

    plt.figure(figsize=(20,20))

    tpc = plt.tripcolor(X_2D[:, 0], X_2D[:, 1], triangles=triangles, facecolors=tri_deforms, cmap='RdYlBu_r', vmin=-1, vmax=1)

    if blur > 0:

        # Plot the triangulation to a buffer, read as a numpy array
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0, bbox_inches='tight')
        buf.seek(0)
        img = np.array(Image.open(buf))
        background = gaussian_filter(img, sigma=(blur,blur,0)) # Blur the array
        plt.close()

        # Same thing for the points
        plt.figure(figsize=(20,20))
        plt.scatter(X_2D[:, 0], X_2D[:, 1], s=10, c='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, pad_inches=0, bbox_inches='tight')
        buf.seek(0)
        foreground = np.array(Image.open(buf))
        plt.close()

        # Overlay the foreground and the blurred background
        alpha = foreground[..., 3:4] / 255.0
        merged_array = alpha*foreground.astype(float) + (1-alpha)*background.astype(float)
        merged_array = merged_array.astype(np.uint8)

        if filename is None:
            plt.imshow(merged_array)
            plt.axis('off')
            plt.show()
        else:
            im = Image.fromarray(merged_array)
            im.save(filename)

    else:
        plt.scatter(X_2D[:, 0], X_2D[:, 1], s=10, c='black')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, pad_inches=0, bbox_inches='tight')
        plt.close()
