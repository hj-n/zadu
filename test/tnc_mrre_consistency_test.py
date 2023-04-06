import sys

sys.path.append('../src/zadu')
sys.path.append("../legacy")

import provider as prov

from sklearn.datasets import load_iris, load_digits

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits, _ = load_digits(return_X_y=True)
iris, _ = load_iris(return_X_y=True)

pca = PCA(n_components=2)

digits_pca = pca.fit_transform(digits)
iris_pca = pca.fit_transform(iris)

# tsne = TSNE(n_components=2)

# digits_tsne = tsne.fit_transform(digits)
# iris_tsne = tsne.fit_transform(iris)

from measures.trustworthiness_continuity import trustworthiness_continuity
from provider import MDPMetricProvider


print(trustworthiness_continuity(digits, digits_pca, 20))
print(MDPMetricProvider(digits, digits_pca, ["Trustworthiness", "Continuity"], 20).run())

