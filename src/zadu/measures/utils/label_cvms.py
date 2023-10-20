import numpy as np

def btw_ch(data, labels):
	return btw(data, labels)

def dsc_normalize(data, labels):
	"""
	compute the distance consistency
	"""
	## contert labels to range from 0 to len(np.unique(labels)) - 1
	labels = np.array(labels)
	unique_labels = np.unique(labels)
	newlabels = np.zeros(labels.shape[0])
	for i in range(len(unique_labels)):
		newlabels[labels == unique_labels[i]] = i

	labels = newlabels

	## compute centroids
	centroids = []
	for i in range(len(np.unique(labels))):
		centroids.append(np.mean(data[labels == i], axis = 0))
	
	## compute distance consistency
	consistent_num = 0
	for idx in range(data.shape[0]):
		current_label = -1
		current_dist = 1e10
		for c_idx in range(len(centroids)):
			dist = np.linalg.norm(data[idx] - centroids[c_idx])
			if dist < current_dist:
				current_dist = dist
				current_label = c_idx
		if current_label == labels[idx]:
			consistent_num += 1
	
	return (consistent_num / data.shape[0] - 0.5) * 2





def shift(X, label):
	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std =np.std(np.sqrt(np.sum(np.square(X - centroid(X)), axis=1)))
	
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = centroid(X[label == i, :])

	entire_centroid = centroid(X)

	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		# if np.exp(np.linalg.norm(centroids[i, :] - entire_centroid) / std) == np.inf:
		# 	raise Exception("None")
		# print(np.sqrt(np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1)) / std)
		compactness += np.sum(np.exp(np.sqrt(np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1)) / std, dtype=np.float128))
		separability += ( np.exp(np.linalg.norm(centroids[i, :] - entire_centroid) / std, dtype=np.float128))* X[label == i, :].shape[0] 



	result = (separability *  (n_samples - 2)) / compactness 

	if compactness == np.inf and separability == np.inf:
		raise Exception("None")

	return result

def shift_range(X, label, iter_num):
	orig = shift(X, label)
	orig_result = 1 / (1 + (orig) ** (-1))
	e_val_sum = 0
	for i in range(iter_num):
		np.random.shuffle(label)
		e_val_sum += shift(X, label)
	e_val = e_val_sum / iter_num
	e_val_result = 1 / (1 + (e_val) ** (-1))
	if e_val_result == 1:
		return 0
	return (orig_result - e_val_result) / (1 - e_val_result)

def shift_range_class(X, label, iter_num):
	class_num = len(np.unique(label))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			X_pair      = X[((label == label_a) | (label == label_b))]
			labels_pair = label[((label == label_a) | (label == label_b))]

			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = shift_range(X_pair, labels_pair, iter_num)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)

def btw(X, labels, iter_num=20):
	return shift_range_class(X, labels, iter_num)


def centroid(X):
	"""
	Compute the centroid of a set of vectors.
	:param X: The set of vectors.
	:return: The centroid.
	"""
	return np.mean(X, axis=0)