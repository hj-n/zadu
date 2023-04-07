from .measures import *

class ZADU:

	ABBREVIATIONS = {
		"tnc": "trustworthiness_continuity",
		"mrre": "mean_relative_ranking_error",
		"nd": "neighborhood_dissimilarity",
		"dtm": "distance_to_measure",
		"lcmc": "local_continuity_meta_criteria",
	}
	
	def __init__(self, spec_list, verbose=False):
		self.spec_list = spec_list
		self.verbose = verbose

		self.__interpret_measures_spec()
	
	def __interpret_measures_spec(self):
		"""
		Interprets the measures specification list.
		"""
		## check whehter there exists invalid measure name
		for spec in self.spec_list:
			if spec["measure"] not in self.ABBREVIATIONS.values():
				if spec["measure"] in self.ABBREVIATIONS:
					spec["measure"] = self.ABBREVIATIONS[spec["measure"]]
				else:
					raise Exception("Invalid measure name: {}".format(spec["measure"]))

		## check whether the parameters are valid
		for spec in self.spec_list:
			measure_name = spec["measure"]
			given_params = spec["params"]

			## get parameters of the function
			measure_func = globals()[measure_name].run
			real_params  = measure_func.__code__.co_varnames[:measure_func.__code__.co_argcount]

			## check whether the given parameters are valid
			for param in given_params:
				if param not in real_params:
					raise Exception(f"Invalid parameter {param} for measure {measure_name}")
			

			