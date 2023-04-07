import sys

sys.path.append('../src')
sys.path.append("../legacy")

import provider as prov

from sklearn.datasets import load_iris, load_digits

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits, digits_label = load_digits(return_X_y=True)
iris, iris_label = load_iris(return_X_y=True)

pca = PCA(n_components=2)

digits_pca = pca.fit_transform(digits)
iris_pca = pca.fit_transform(iris)

tsne = TSNE(n_components=2)

digits_tsne = tsne.fit_transform(digits)
iris_tsne = tsne.fit_transform(iris)

from zadu.measures import *

print(trustworthiness_continuity.trustworthiness_continuity(digits, digits_pca, 20))

# m.trustworthiness_continuity.trustworthiness_continuity(digits, digits_pca, 20)