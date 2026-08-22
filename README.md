<p align="center">
  <h2 align="center">ZADU</h2>
	<p align="center"><b>A</b>-to-<b>Z</b> python library for eval<b>U</b>ating <b>D</b>imensionality reduction</p>
</p>

---

ZADU is a Python library that provides distortion measures for evaluating and analyzing dimensionality reduction (DR) embeddings. The library supports a diverse set of local, cluster-level, and global distortion measures, allowing users to assess DR techniques from various structural perspectives. By offering an optimized execution and pointwise local distortions, ZADU enables efficient and in-depth analysis of DR embeddings. 



## Installation

You can install ZADU via `pip`:

```bash
pip install zadu
```

## Quick Start

Run two metrics (`tnc`, `mrre`) on random data:

```python
import numpy as np
from zadu import ZADU

rng = np.random.default_rng(0)
hd = rng.normal(size=(200, 16))
ld = hd[:, :2] + 0.05 * rng.normal(size=(200, 2))

spec = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "mrre", "params": {"k": 20}},
]

scores = ZADU(spec, hd).measure(ld)
print(scores)
```

Input checklist:

- `hd` (`orig`) and `ld` (`emb`) must be finite 2D numeric arrays with the same number of rows.
- For neighbor-based metrics with `k`, use `1 <= k < n`.
- For `tnc` and `ca_tnc`, the standard normalization additionally requires `k < n / 2`.
- Pass `label` to `measure(ld, label)` for label-based metrics (`nh`, `ca_tnc`, `dsc`, `ivm`, `c_evm`, `l_tnc`, `cadi`). Labels may be strings or arbitrary numeric values.
- Metrics that are mathematically undefined for constant distances, a single class, or coincident neighborhoods raise `ValueError` instead of returning `nan` or `inf`.

## Use ZADU with Context7

This repository is prepared for Context7 indexing so AI coding assistants can fetch up-to-date ZADU docs and examples.


### For Users (Cursor / Claude Code)

Install Context7 MCP (recommended: remote HTTP with API key) to your coding assistant.

Refer to [Context7 Installation](https://github.com/upstash/context7#installation) for detailed instructions.


Add the following rule (so users do not have to type `use context7` each time) to agent instructions like CLAUDE.md or AGENTS.md:

```text
Always use Context7 MCP when I need library/API documentation, code generation, setup, or configuration steps. When users want to evaluate dimensionality reduction embeddings or you think that the evalaution is necessary, use Context7 to access ZADU documentation and examples. You can access via "/hj-n/zadu" path.
```


## Supported Distortion Measures

ZADU currently supports a total of 22 distortion measures, including:

- 7 local measures
- 6 cluster-level measures
- 8 global measures
- 1 gap-based regional measure

For a complete list of supported measures, refer to [measures](/src/zadu/measures). The library initially provided 17 measures when it was first introduced by our academic paper. We later added label trustworthiness & continuity, non-metric stress, scale-normalized stress, the class angular distortion index, and the gap index.

## How To Use ZADU

ZADU provides two different interfaces for executing distortion measures.
You can either use the main class that wraps the measures, or directly access and invoke the functions that define each distortion measure.

### Using the Main Class

Use the main class of ZADU to compute distortion measures.
This approach benefits from the optimization, providing faster performance when executing multiple measures.


```python
from zadu import zadu

hd, ld = load_datasets()
spec = [{
    "id"    : "tnc",
    "params": { "k": 20 },
}, {
    "id"    : "snc",
    "params": { "k": 30, "clustering_strategy": "dbscan" }
}]

scores = zadu.ZADU(spec, hd).measure(ld)
print("T&C:", scores[0])
print("S&C:", scores[1])

```

`hd` represents high-dimensional data, `ld` represents low-dimensional data

You can also use a typed helper for better IDE autocomplete:

```python
from zadu import ZADU, MEASURE, make_spec

spec = [
    make_spec(MEASURE.TNC, k=20),
    make_spec(MEASURE.SNC, k=30, clustering_strategy="dbscan"),
]
scores = ZADU(spec, hd).measure(ld)
```

`MEASURE` enum mapping (typed helper):

| MEASURE | ID | Metric Name |
|---|---|---|
| `MEASURE.TNC` | `tnc` | Trustworthiness & Continuity |
| `MEASURE.MRRE` | `mrre` | Mean Relative Rank Error |
| `MEASURE.LCMC` | `lcmc` | Local Continuity Meta-Criteria |
| `MEASURE.NH` | `nh` | Neighborhood Hit |
| `MEASURE.CA_TNC` | `ca_tnc` | Class-Aware Trustworthiness & Continuity |
| `MEASURE.L_TNC` | `l_tnc` | Label Trustworthiness & Continuity |
| `MEASURE.ND` | `nd` | Neighbor Dissimilarity |
| `MEASURE.DTM` | `dtm` | Distance-to-Measure |
| `MEASURE.KL_DIV` | `kl_div` | Kullback-Leibler Divergence |
| `MEASURE.DSC` | `dsc` | Distance Consistency |
| `MEASURE.PR` | `pr` | Pearson's Correlation Coefficient |
| `MEASURE.SRHO` | `srho` | Spearman's Rank Correlation Coefficient |
| `MEASURE.IVM` | `ivm` | Internal Validation Measure |
| `MEASURE.C_EVM` | `c_evm` | Clustering + External Validation Measure |
| `MEASURE.SNC` | `snc` | Steadiness & Cohesiveness |
| `MEASURE.TOPO` | `topo` | Topographic Product |
| `MEASURE.PROC` | `proc` | Procrustes Measure |
| `MEASURE.STRESS` | `stress` | Stress |
| `MEASURE.SN_STRESS` | `sn_stress` | Scale-Normalized Stress |
| `MEASURE.NM_STRESS` | `nm_stress` | Non-Metric Stress |
| `MEASURE.CADI` | `cadi` | Class Angular Distortion Index |
| `MEASURE.GI` | `gi` | Gap Index |

## ZADU Class

The ZADU class provides the main interface for the Zadu library, allowing users to evaluate and analyze dimensionality reduction (DR) embeddings effectively and reliably.

### Class Constructor

The ZADU class constructor has the following signature:

```python
class ZADU(
    spec_list,
    orig,
    return_local: bool = False,
    verbose: bool = False,
    geodesic: bool = False,
    max_memory_bytes: int | None = None,
    execution: ExecutionConfig | None = None,
)

```

### Exact Execution Planning

ZADU plans pair reductions, distance matrices, neighbor tables, and full rankings
as typed exact resources. Compatible requests are computed once: a larger `k`
serves smaller prefixes, a full ranking also serves metrics that only need kNN
indices, and Stress, Scale-Normalized Stress, and Pearson share one exact pass
over unique point pairs. Spearman and Non-Metric Stress share one exact,
tie-aware original-space pair order across repeated embeddings.

Pair-only specifications avoid two persistent `n x n` distance matrices. The
planner uses compact upper-triangle storage when it fits, switches to bounded
block streaming for larger or memory-constrained workloads, and reuses dense
matrices when another requested metric already needs them. Every point pair is
still evaluated; neither path is approximate.

Metrics that require a global pair order cannot use block streaming. For those
metrics, an explicit memory budget that cannot hold the exact condensed/order
plan raises `MemoryError` before distance allocation begins.

Topographic Product keeps exact stable neighbor ordering without persistent
`n x n` matrices. Its neighbor search uses bounded distance-row blocks, the
metric evaluates only the `O(nk)` distances selected by the two neighbor tables,
and multiple requested `k` values share one maximum-`k` prefix calculation.

The optional execution configuration currently exposes the exact NumPy/FAISS
CPU path and a human-readable memory budget:

```python
from zadu import ExecutionConfig, ZADU

runner = ZADU(
    spec,
    hd,
    execution=ExecutionConfig(
        backend="auto",       # "auto" or "numpy"
        device="auto",        # "auto" or "cpu"
        memory_budget="4GiB",
    ),
)
scores = runner.measure(ld)
print(runner.last_run_info)
```

`last_run_info` is separate from metric scores. It records the exact backend,
resource providers, selected pair strategy and block size, estimated cache and
peak working memory, dtype, construction and metric timings, release/reuse, and
each resource's first and last consumer. MLX and PyTorch providers will be added
in later acceleration PRs; unsupported backend or device requests currently
raise an explicit `ValueError`.

### Parameters:

#### `spec` 
&nbsp;&nbsp;&nbsp;&nbsp;
A list of dictionaries that define the distortion measures to execute and their hyperparameters.
Each dictionary must contain the following keys:
  * `"id"`: The identifier of the distortion measure, such as `"tnc"` or `"snc"`.

  * `"params"`: A dictionary containing hyperparameters specific to the chosen distortion measure.

#### List of ID/Parameters for Each Function


***Warning***: While using `dsc`, `ivm`, `c_evm`, `nh`, and `ca_tnc`, please be aware that these measures assume that class labels are *well-separated* in the original high-dimensional space. If the class labels are not well-separated, the measures may produce unreliable results. Use the measure only if you are confident that the class labels are well-separated. Please refer to the related [academic paper](https://www.hyeonjeon.com/assets/pdf/jeon23tvcg.pdf) for more detail. 

> ##### Local Measures
> 
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Trustworthiness & Continuity | tnc | `k=20` | [0.5, 1] | 1 |
> | Mean Relative Rank Errors | mrre | `k=20` | [0, 1] | 1 | 
> | Local Continuity Meta-Criteria | lcmc | `k=20` | [-k/(n-1), 1-k/(n-1)] | 1-k/(n-1) |
> | Neighborhood hit | nh | `k=20` | [0, 1] | 1 |
> | Neighbor Dissimilarity | nd | `k=20` | R+ | 0 |
> | Class-Aware Trustworthiness & Continuity | ca_tnc | `k=20` | [0.5, 1] | 1|
> | Procrustes Measure | proc | `k=20` | R+ | 0 |
> 
> ##### Cluster-level Measures
> 
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Steadiness & Cohesiveness | snc | `iteration=150, walk_num_ratio=0.3, alpha=0.1, k=None, clustering_strategy="dbscan", random_state=None, n_jobs=1` | [0, 1] | 1 |
> | Distance Consistency | dsc | | [0, 1] | 1 |
> | Internal Validation Measures | ivm | `measure="silhouette"` | Depends on IVM | Depends on IVM |
> | Clustering + External Clustering Validation Measures | c_evm | `measure="arand", clustering="kmeans", clustering_args=None` | Depends on EVM | Depends on EVM |
> | Label Trustworthiness & Continuity[^1] | l_tnc | `cvm="dsc"` | [0, 1] | 1 |
> | Class Angular Distortion Index | cadi | `n_triplets=0, random_seed=None` | [0, 1] | 0 |

[^1]: The current implementation does not apply the rescaling step from the [original paper](https://www.hyeonjeon.com/assets/pdf/jeon23tvcg.pdf) on the cvm score when cvm='dsc'.
The original transformation was intended to map the DSC score into the \[0,1\] range, but it is not needed here.


> ##### Global Measures
> 
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Stress | stress | | R+ | 0 |
> | Non-metric stress | nm_stress| | R+ | 0 |
> | Scale-normalized stress | sn_stress | | R+ | 0 |
> | Kullback-Leibler Divergence | kl_div | `sigma=0.1` | R+ | 0 |
> | Distance-to-Measure | dtm | `sigma=0.1` | R+ | 0 |
> | Topographic Product | topo | `k=20` | R | 0 |
> | Pearson’s correlation coefficient | pr | | [-1, 1] | 1
> | Spearman’s rank correlation coefficient | srho | | [-1, 1] | 1 | 

Pearson and Spearman correlations use each unique off-diagonal distance once (the upper triangle of each distance matrix). Stress-family and density-family metrics reject all-zero distance matrices because their normalizations are undefined there.

> ##### Gap-based Regional Measures
>
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Gap Index[^2] | gi | `metric="euclidean"` | [0, 1] | 0 |

The Gap Index operates on empty triangular regions of a 2D projection rather than fitting cleanly into the local, cluster-level, or global categories above. It supports a SciPy distance function or function name, and `metric="precomputed"` when `hd` is a square distance matrix.

[^2]: Introduced by Jaume Ros, Alessio Arleo, and Fernando Paulovich ([paper](https://arxiv.org/abs/2607.28324), [reference implementation](https://codeberg.org/jros/gap-index)); ZADU retains the original [MIT license](/LICENSES/gap-index-MIT.txt) and [provenance notice](/THIRD_PARTY_NOTICES.md).

#### String Option Values

- `ivm` (`internal_validation_measure`): `silhouette`, `calinski_harabasz`, `davies_bouldin`
- `c_evm` (`clustering_and_external_validation_measure`)
  - `measure`: `arand`, `ami`, `nmi`, `vmeasure`
  - `clustering`: `kmeans`, `dbscan`
  - When `clustering="kmeans"`, `n_clusters` defaults to the number of unique labels and `random_state` defaults to `0`; both can be overridden in `clustering_args`.
- `l_tnc` (`label_trustworthiness_and_continuity`): `cvm` = `dsc`, `ch_btw`

If an invalid option string is passed, ZADU raises a `ValueError` with allowed values.

#### Return Key Summary

- `tnc` -> `trustworthiness`, `continuity`
- `mrre` -> `mrre_false`, `mrre_missing`
- `ca_tnc` -> `ca_trustworthiness`, `ca_continuity`
- `l_tnc` -> `label_trustworthiness`, `label_continuity`
- `snc` -> `steadiness`, `cohesiveness`
- `cadi` -> `class_angular_distortion_index`
- `gi` -> `gap_index`
- `ivm` -> key is the selected measure name (e.g., `silhouette`)
- `c_evm` -> key is `{clustering}_{measure}` (e.g., `kmeans_arand`)

For `return_local=True`, local keys are returned in a second list entry per metric where supported.

##### `hd`
&nbsp;&nbsp;&nbsp;&nbsp;
A high-dimensional dataset (numpy array) to register and reuse during the evaluation process.


##### `return_local`
&nbsp;&nbsp;&nbsp;&nbsp;
A boolean flag that, when set to `True`, enables the computation of local pointwise distortions for each data point. The default value is `False`.


### Directly Accessing Functions

You can also directly access and invoke the functions defining each distortion measure for greater flexibility.

```python
from zadu.measures import *

mrre = mean_relative_rank_error.measure(hd, ld, k=20)
pr  = pearson_r.measure(hd, ld)
nh  = neighborhood_hit.measure(ld, label, k=20)
```

## Advanced Features

### Optimizing the Execution

ZADU automatically optimizes the execution of multiple distortion measures. Its explicit metric registry shares exact pair statistics, densities, rankings, and nearest-neighbor indices while retaining the largest requested `k`, so mixed-`k` specifications remain equivalent to direct metric calls. Pair-only Stress, Scale-Normalized Stress, and Pearson runs use condensed or memory-bounded streaming resources; metrics requiring full ranks or global distance ordering still need O(n²) storage. `ZADU(...).estimated_cache_bytes` exposes the persistent-cache estimate, while `last_run_info["planned_peak_bytes"]` includes package-managed working memory. Pass `max_memory_bytes=` or `ExecutionConfig(memory_budget=...)` to select a bounded strategy or fail before an oversized package-managed allocation.

S&C (`snc`) reuses the planner's exact kNN tables, keeps full weighted-SNN graphs sparse, and batches cluster-pair reductions. Set `n_jobs` above 1 to opt into deterministic thread-level iteration evaluation; the default remains 1 because parallel overhead can outweigh the benefit on smaller workloads. With a memory budget, ZADU may reduce the effective worker count. The requested/effective counts and conservative working-set estimate are recorded in `last_run_info["snc_strategy"]`.

For spherical coordinates, pass `geodesic=True` to `ZADU`. In that mode `orig[:, 0]` is longitude, `orig[:, 1]` is latitude, and both must be expressed in radians. Geodesic distance is used only for the registered original space; embedded-space distances remain Euclidean.

### Computing Pointwise Local Distortions

Users can obtain local pointwise distortions by setting the return_local flag. If a specified distortion measure produces local pointwise distortion as intermediate results, it returns a list of pointwise distortions when the flag is raised.

```python
from zadu import zadu

spec = [{
    "id"    : "dtm",
    "params": {}
}, {
    "id"    : "mrre",
    "params": { "k": 30 }
}]

zadu_obj = zadu.ZADU(spec, hd, return_local=True)
global_, local_ = zadu_obj.measure(ld)
print("MRRE local distortions:", local_[1])

```

### Visualizing Local Distortions

With the pointwise local distortions obtained from ZADU, users can visualize the distortions using various distortion visualizations. We provide ZADUVis, a python library that enables the rendering of two disotortion visualizations: [CheckViz](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2010.01835.x) and the [Reliability Map](https://arxiv.org/abs/2107.07859).


![img](https://user-images.githubusercontent.com/38465539/235427171-94dcc220-7cbb-4ee6-94b3-20cc96ffbfa8.png)

```python
from zadu import zadu
from zaduvis import zaduvis
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml


hd = fetch_openml('mnist_784', version=1, cache=True).data.to_numpy()[::7]
ld = TSNE().fit_transform(hd)

## Computing local pointwise distortions
spec = [{
    "id": "tnc",
    "params": {"k": 25}
},{
    "id": "snc",
    "params": {"k": 50}
}]
zadu_obj = zadu.ZADU(spec, hd, return_local=True)
scores, local_list = zadu_obj.measure(ld)

tnc_local = local_list[0]
snc_local = local_list[1]

local_trustworthiness = tnc_local["local_trustworthiness"]
local_continuity = tnc_local["local_continuity"]
local_steadiness = snc_local["local_steadiness"]
local_cohesiveness = snc_local["local_cohesiveness"]

fig, ax = plt.subplots(1, 4, figsize=(50, 12.5))
zaduvis.checkviz(ld, local_trustworthiness, local_continuity, ax=ax[0])
zaduvis.reliability_map(ld, local_trustworthiness, local_continuity, k=10, ax=ax[1])
zaduvis.checkviz(ld, local_steadiness, local_cohesiveness, ax=ax[2])
zaduvis.reliability_map(ld, local_steadiness, local_cohesiveness, k=10, ax=ax[3])


```
The above code snippet demonstrates how to visualize local pointwise distortions using CheckViz and Reliability Map plots, where the results are shown below.

![mnist_1:7](https://github.com/hj-n/zadu/assets/37105201/7c6dc8d7-59c5-48fd-92a5-186e1e44597a)

## Documentation

For more information about the available distortion measures, their use cases, and examples, please refer to our paper (IEEE VIS 2023 Short).

## Citation

> Hyeon Jeon, Aeri Cho, Jinhwa Jang, Soohyun Lee, Jake Hyun, Hyung-Kwon Ko, Jaemin Jo, and Jinwook Seo. Zadu: A python library for evaluating the reliability of dimensionality reduction embeddings. In 2023 IEEE Visualization and Visual Analytics (VIS), pages 196–200, 2023.

```bib
@INPROCEEDINGS{jeon23vis,
  author={Jeon, Hyeon and Cho, Aeri and Jang, Jinhwa and Lee, Soohyun and Hyun, Jake and Ko, Hyung-Kwon and Jo, Jaemin and Seo, Jinwook},
  booktitle={2023 IEEE Visualization and Visual Analytics (VIS)}, 
  title={ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings}, 
  year={2023},
  volume={},
  number={},
  pages={196-200},
  keywords={Dimensionality reduction;Visual analytics;Design methodology;Distortion;Libraries;Time measurement;Distortion measurement;Human-centered computing;Visualization;Visualization design and evaluation methods},
  doi={10.1109/VIS54172.2023.00048}}
```
