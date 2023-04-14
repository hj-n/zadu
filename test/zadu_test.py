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

digits_tsne = TSNE(n_components=2).fit_transform(digits)

from zadu import zadu


spec_list = [
  {
    "measure": "tnc",
    "params": {
			"k": 25
		}
	},
  {
    "measure": "ca_tnc",
		"params": {
			"k": 30
		}       
	},
  {
    "measure": "dtm"
	},
  {
    "measure": "neighbor_dissimilarity",
    "params": {
			"k": 50
		}
	},
  {
    "measure": "snc",
    "params": { }
	}
]

# specs = {
#   "tnc": { "k": 25 },
#   "ca_tnc": { "k": 30 },
#   "dtm": {},
#   "neighborhood_dissimilarity": { "k": 50 }
# }

zadu_obj = zadu.ZADU(spec_list, digits, return_local=True)

scores, local_list = zadu_obj.measure(digits_pca, digits_label)
print(scores)
scores, local_list = zadu_obj.measure(digits_tsne, digits_label)
print(scores)
