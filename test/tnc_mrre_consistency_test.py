import sys

sys.path.append('../src/zadu')
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

# tsne = TSNE(n_components=2)

# digits_tsne = tsne.fit_transform(digits)
# iris_tsne = tsne.fit_transform(iris)

from provider import MDPMetricProvider
from measures import trustworthiness_continuity, mean_relative_rank_error
# from measures.class_aware_trustworthiness_continuity import class_aware_trustworthiness_continuity
# from measures.mean_relative_rank_error import mean_relative_rank_error
# from measures.local_continuity_meta_criteria import local_continuity_meta_criteria
# from measures.neighborhood_hit import neighborhood_hit
# from measures.distance_consistency import distance_consistency
# from measures.internal_validation_measure import internal_validation_measure
# from measures.kl_divergence import kl_divergence
# from measures.distance_to_measure import distance_to_measure
# from measures.pearson_r import pearson_r
# from measures.spearman_rho import spearman_rho
# from measures.clustering_and_external_validation_measure import clustering_and_external_validation_measure
# from measures.neighborhood_dissimilarity import neighborhood_dissimilarity
# from measures.utils import knn
print(trustworthiness_continuity.run(digits, digits_pca, 20))
print(MDPMetricProvider(digits, digits_pca, ["Trustworthiness", "Continuity"], 20).run())

# print(class_aware_trustworthiness_continuity(digits, digits_pca, digits_label, 20))

print(mean_relative_rank_error.run(digits, digits_pca, 20))
print(MDPMetricProvider(digits, digits_pca, ["MRRE_ZX", "MRRE_XZ"], 20).run())

# print(local_continuity_meta_criteria(digits, digits_pca, 20))
# print(local_continuity_meta_criteria(digits, digits_tsne, 20))
# print(local_continuity_meta_criteria(digits, digits_pca, 50))
# print(local_continuity_meta_criteria(digits, digits_tsne, 50))

# print(neighborhood_hit( digits_pca, digits_label, 20))
# print(neighborhood_hit(digits_tsne, digits_label, 20))


# print(distance_consistency(digits_pca, digits_label))
# print(distance_consistency(digits_tsne, digits_label))

# print(internal_validation_measure(digits_pca, digits_label, "silhouette"))
# print(internal_validation_measure(digits_tsne, digits_label, "silhouette"))

# print(kl_divergence(digits, digits_pca, 0.1))
# print(kl_divergence(digits, digits_tsne, 0.1))

# print(distance_to_measure(digits, digits_pca, 0.1))
# print(distance_to_measure(digits, digits_tsne, 0.1))

# print(pearson_r(digits, digits_pca))
# print(pearson_r(digits, digits_tsne))

# print(spearman_rho(digits, digits_pca))
# print(spearman_rho(digits, digits_tsne))

# print(clustering_and_external_validation_measure(digits_pca, digits_label, "arand", "kmeans", {"n_clusters": 20}))
# print(clustering_and_external_validation_measure(digits_pca, digits_label, "arand", "kmeans", {"n_clusters": 10}))

# orig_knn_indices = knn.knn(digits, 20)
# emb_knn_indices  = knn.knn(digits_pca, 20)

# print(neighborhood_dissimilarity(digits, digits_pca, 20))
# print(neighborhood_dissimilarity(digits, digits_pca, 20, knn_info=(orig_knn_indices, emb_knn_indices)))