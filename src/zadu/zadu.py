from .measures import *
from .measures.utils import knn
from .measures.utils import pairwise_dist as pdist
import math
import numpy as np

class ZADU:

	ABBREVIATIONS = {
		"tnc": "trustworthiness_continuity",
		"mrre": "mean_relative_rank_error",
		"lcmc": "local_continuity_meta_criteria",
		"nh": "neighborhood_hit",
		"ca_tnc": "class_aware_trustworthiness_continuity",
		"l_tnc": "label_trustworthiness_and_continuity",
		"nd": "neighbor_dissimilarity",
		"dtm": "distance_to_measure",
		"kl_div": "kl_divergence",
		"dsc": "distance_consistency",
		"pr": "pearson_r",
		"srho":	"spearman_rho",
		"ivm": "internal_validation_measure",
		"c_evm": "clustering_and_external_validation_measure",
		"snc": "steadiness_cohesiveness",
		"topo": "topographic_product",
		"proc": "procrustes",
		"stress": "stress"
	}

	DEFAULT_K = 20
	
	def __init__(self, spec_list, orig, return_local=False, verbose=False, geodesic=False):
		self.spec_list    = spec_list
		self.return_local = return_local
		self.verbose      = verbose

		self.orig  = orig
		self.emb   = None
		self.label = None

		## FLAGS for scheduling
		self.knn_flag   						= False
		self.knn_flag_k 						= -1
		self.knn_ranking_flag 			= False
		self.knn_ranking_flag_k 		= -1
		self.distance_matrices_flag = False

		## variables for holding precomputed results (prerequisite for some measures)
		self.orig_distance_matrix = None
		self.emb_distance_matrix  = None
		self.orig_knn_ranking     = None
		self.emb_knn_ranking      = None
		self.orig_knn_indices		  = None
		self.emb_knn_indices		  = None

		self.__sanity_check_measures_spec()
		self.__interpret_measures_spec()

		if self.distance_matrices_flag:
			if geodesic:
				self.orig_distance_matrix = self.__pairwise_geodesic_distance_matrix(orig)
			else:
				self.orig_distance_matrix = pdist.pairwise_distance_matrix(orig)
		if self.knn_ranking_flag:
			self.orig_knn_indices, self.orig_knn_ranking = knn.knn_with_ranking(orig, self.knn_ranking_flag, distance_matrix=self.orig_distance_matrix)
		elif self.knn_flag and self.knn_flag_k > self.knn_ranking_flag_k:
			self.orig_knn_indices = knn.knn(orig, self.knn_flag_k, distance_function="euclidean")
	

	def measure(self, emb, label=None):
		"""
		Run the functions specified in spec_list
		INPUT:
			emb:  embedded data
		OUTPUT:
			list: list of results
		"""

		self.emb  = emb
		self.label = label

		## compute the distance matrices
		if self.distance_matrices_flag:
			self.emb_distance_matrix  = pdist.pairwise_distance_matrix(emb)
		if self.knn_ranking_flag:
			self.emb_knn_indices,  self.emb_knn_ranking  = knn.knn_with_ranking(emb,  self.knn_ranking_flag, distance_matrix=self.emb_distance_matrix)
		elif self.knn_flag and self.knn_flag_k > self.knn_ranking_flag_k:
			self.emb_knn_indices  = knn.knn(emb,  self.knn_flag_k, distance_function="euclidean")
		
		## compute the measures
		score_results = []
		local_results = []
		for spec in self.spec_list:
			measure_name = spec["id"]
			given_params = spec["params"] if "params" in spec else {}
			real_params  = self.__get_real_params(measure_name)

			## construct the execution parameters to be injected in the function
			exec_params = {}
			for param in given_params.keys():
				exec_params[param] = given_params[param]

			for param in real_params:
				if "orig" == param:
					exec_params["orig"] = self.orig
				elif "emb" == param:
					exec_params["emb"] = self.emb
				elif "label" == param:
					if label is None:
						raise Exception("Label is required for measure {}".format(measure_name))
					exec_params["label"] = label
				elif "distance_matrices" == param:
					exec_params["distance_matrices"] = (self.orig_distance_matrix, self.emb_distance_matrix)
				elif "knn_ranking_info" == param:
					k_val = exec_params["k"] if "k" in exec_params else self.DEFAULT_K
					exec_params["knn_ranking_info"] = (
						self.orig_knn_indices[:, :k_val], 
						self.orig_knn_ranking, 
						self.emb_knn_indices[:, :k_val], 
						self.emb_knn_ranking
					)
				elif "knn_indices" == param:
					k_val = exec_params["k"] if "k" in exec_params else self.DEFAULT_K
					exec_params["knn_indices"] = (
						self.orig_knn_indices[:, :k_val], 
						self.emb_knn_indices[:, :k_val]
					)
				elif "return_local" == param:
					exec_params["return_local"] = self.return_local
			
			## execute the function
			if self.return_local and "return_local" in exec_params:
				score, local = globals()[measure_name].measure(**exec_params)
				score_results.append(score)
				local_results.append(local)
			elif self.return_local and "return_local" not in exec_params:
				score = globals()[measure_name].measure(**exec_params)
				score_results.append(score)
				local_results.append(None)
			else:
				score = globals()[measure_name].measure(**exec_params)
				score_results.append(score)

		if self.return_local:
			return score_results, local_results
		else:
			return score_results
		



	def __sanity_check_measures_spec(self):
		"""
		Perform sanity check on the measures specification list.
		"""
		## check whehter there exists invalid measure name
		for spec in self.spec_list:
			if spec["id"] not in self.ABBREVIATIONS.values():
				if spec["id"] in self.ABBREVIATIONS:
					spec["id"] = self.ABBREVIATIONS[spec["id"]]
				else:
					raise Exception("Invalid measure name: {}".format(spec["id"]))

		## check whether the parameters are valid
		for spec in self.spec_list:
			measure_name = spec["id"]
			given_params = spec["params"] if "params" in spec else {}
			real_params  = self.__get_real_params(measure_name)

			## check whether the given parameters are valid
			for param in given_params:
				if param not in real_params:
					raise Exception(f"Invalid parameter {param} for measure {measure_name}")
	


	def __interpret_measures_spec(self):
		"""
		Interpret the measures spec and specify the preprequisites (knn, distance matrices)
		"""
		for spec in self.spec_list:
			measure_name = spec["id"]
			given_params = spec["params"] if "params" in spec else {}
			real_params  = self.__get_real_params(measure_name)

			if "knn_ranking_info" in real_params:
				self.knn_ranking_flag = True
				if "k" in given_params:
					self.knn_ranking_flag_k = max(self.knn_ranking_flag_k, given_params["k"])
				else:
					self.knn_ranking_flag_k = max(self.knn_ranking_flag_k, self.DEFAULT_K)
			if "knn_info" in real_params:
				self.knn_flag = True
				if "k" in given_params:
					self.knn_flag_k = max(self.knn_flag_k, given_params["k"])
				else:
					self.knn_flag_k = max(self.knn_flag_k, self.DEFAULT_K)
			if "distance_matrices" in real_params:
				self.distance_matrices_flag = True



	
	def __get_real_params(self, measure_name):
		"""
		Get the real parameters of a measure.
		"""
		measure_func = globals()[measure_name].measure
		return measure_func.__code__.co_varnames[:measure_func.__code__.co_argcount]
	


	def __geodesic_distance(self, phi1, lambda1, phi2, lambda2):
		return math.acos(math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(abs(lambda2 - lambda1)))


	def __pairwise_geodesic_distance_matrix(self, orig):
		data_len = len(orig)
		distance_matrix = np.zeros((data_len, data_len))
		for i in range(data_len):
			for j in range(i, data_len):
				if i == j: distance_matrix[i][j] = 0
				else: distance_matrix[i][j] = distance_matrix[j][i] = self.__geodesic_distance(orig[i][1], orig[i][0], orig[j][1], orig[j][0])
		return distance_matrix