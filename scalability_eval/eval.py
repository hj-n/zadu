import sys

sys.path.append("../src")

from data import reader
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import umap
import time
import os

from bayes_opt import BayesianOptimization

from zadu import zadu
from zadu.measures import *

import pandas as pd


DATASET_LIST = os.listdir("./data/compressed/")
DATASET_LIST.remove(".gitignore")

pbounds = {
	"n_neighbors": (2, 200),
	"min_dist": (0.001, 0.99)
}

spec_list = [
	{ "measure": "tnc", "params": { "k": 25 } },
	{ "measure": "mrre", "params": { "k": 25 } },
	{ "measure": "snc"},
	{  "measure": "dtm" },
	{ "measure": "kl_div" }
]


with_scheduling = []
without_scheduling = []
dataset_size = []
dataset_dim  = []

DATASET_LIST = DATASET_LIST


for dataset in tqdm(DATASET_LIST):
	data, label = reader.read_dataset_by_path(f"./data/compressed/{dataset}/")
	data = StandardScaler().fit_transform(data)

	dataset_size.append(data.shape[0])
	dataset_dim.append(data.shape[1])

	def run_with_scheduling(n_neighbors, min_dist):
		umap_obj = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=min_dist)

		umap_result = umap_obj.fit_transform(data)

		zadu_obj = zadu.ZADU(spec_list, data)
		scores = zadu_obj.measure(umap_result)

		tnc_score  = (scores[0]["trustworthiness"] * scores[0]["continuity"]) / (scores[0]["trustworthiness"] + scores[0]["continuity"])
		mrre_score = (scores[1]["mrre_false"] * scores[1]["mrre_missing"]) / (scores[1]["mrre_false"] + scores[1]["mrre_missing"])
		snc_score  = (scores[2]["steadiness"] * scores[2]["cohesiveness"]) / (scores[2]["steadiness"] + scores[2]["cohesiveness"])
		dtm_score  = 1 - scores[3]["distance_to_measure"]
		kl_score   = 1 - scores[4]["kl_divergence"]

		avg_score = (tnc_score + mrre_score + snc_score + dtm_score + kl_score) / 5

		return avg_score

	def run_without_scheduling(n_neighbors, min_dist):
		umap_obj = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=min_dist)
		umap_result = umap_obj.fit_transform(data)

		tnc  = trustworthiness_continuity.measure(data, umap_result, k=25)
		mrre = mean_relative_rank_error.measure(data, umap_result, k=25)
		snc  = steadiness_cohesiveness.measure(data, umap_result)
		dtm  = distance_to_measure.measure(data, umap_result)
		kl   = kl_divergence.measure(data, umap_result)

		tnc_score = (tnc["trustworthiness"] * tnc["continuity"]) / (tnc["trustworthiness"] + tnc["continuity"])
		mrre_score = (mrre["mrre_false"] * mrre["mrre_missing"]) / (mrre["mrre_false"] + mrre["mrre_missing"])
		snc_score = (snc["steadiness"] * snc["cohesiveness"]) / (snc["steadiness"] + snc["cohesiveness"])
		dtm_score = 1 - dtm["distance_to_measure"]
		kl_score = 1 - kl["kl_divergence"]

		avg_score = (tnc_score + mrre_score + snc_score + dtm_score + kl_score) / 5

		return avg_score

	optimizer = BayesianOptimization(
		f=run_with_scheduling,
		pbounds=pbounds,
		random_state=1
	)

	start = time.time()
	optimizer.maximize(init_points=2, n_iter=8)
	end = time.time()

	with_scheduling.append(end - start)

	optimizer_without = BayesianOptimization(
		f=run_without_scheduling,
		pbounds=pbounds,
		random_state=1
	)

	start = time.time()
	optimizer_without.maximize(init_points=2, n_iter=8)
	end = time.time()

	without_scheduling.append(end - start)


df = pd.DataFrame({
	"dataset": DATASET_LIST,
	"with_scheduling": with_scheduling,
	"without_scheduling": without_scheduling,
	"dataset_size": dataset_size,
	"dataset_dim": dataset_dim
})

df.to_csv("result/results_full_10.csv", index=False)

	






	
