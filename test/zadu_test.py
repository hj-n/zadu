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

from zadu import zadu

spec_list = [
  {
    "measure": "tnc",
    "params": {
			"k": 25,
      "k2": 25
		}
	}
]

zadu.ZADU(spec_list)