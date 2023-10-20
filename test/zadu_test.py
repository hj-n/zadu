import sys

sys.path.append('../src')
sys.path.append("../legacy")

from zadu import zadu

from sklearn.datasets import load_iris, load_digits

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits, digits_label = load_digits(return_X_y=True)
iris, iris_label = load_iris(return_X_y=True)

pca = PCA(n_components=2)

digits_pca = pca.fit_transform(digits)
iris_pca = pca.fit_transform(iris)

# digits_tsne = TSNE(n_components=2).fit_transform(digits)



spec_list = [
  # {
  #   "id": "tnc",
  #   "params": {
	# 		"k": 25
	# 	}
	# },
  # {
  #   "id": "ca_tnc",
	# 	"params": {
	# 		"k": 30
	# 	}       
	# },
  # {
  #   "id": "dtm"
	# },
  # {
  #   "id": "neighbor_dissimilarity",
  #   "params": {
	# 		"k": 50
	# 	}
	# },
  # {
  #   "id": "snc",
  #   "params": { "k": 60, "iteration": 300}
	# },
  {
    "id": "l_tnc",
    "params": { "cvm": "ch_btw"}
	},
	{
    "id": "l_tnc",
    "params": { "cvm": "dsc"}		
	}
	# {
  #     "id": "stress"
	# }
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
# scores, local_list = zadu_obj.measure(digits_tsne, digits_label)
# print(scores)
