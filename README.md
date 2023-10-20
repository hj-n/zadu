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

## Supported Distortion Measures

ZADU currently supports a total of 18 distortion measures, including:

- 7 local measures
- 5 cluster-level measures
- 6 global measures

For a complete list of supported measures, refer to [measures](/src/zadu/measures). The library initially provided 17 measures when it was first introduced by our academic paper, and we added one more measure (label trustworthiness & continuity) to the library.

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

## ZADU Class

The ZADU class provides the main interface for the Zadu library, allowing users to evaluate and analyze dimensionality reduction (DR) embeddings effectively and reliably.

### Class Constructor

The ZADU class constructor has the following signature:

```python
class ZADU(spec: List[Dict[str, Union[str, dict]]], hd: np.ndarray, return_local: bool = False)

```

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
> | Local Continuity Meta-Criteria | lcmc | `k=20` | [0, 1] | 1 |
> | Neighborhood hit | nh | `k=20` | [0, 1] | 1 |
> | Neighbor Dissimilarity | nd | `k=20` | R+ | 0 |
> | Class-Aware Trustworthiness & Continuity | ca_tnc | `k=20` | [0.5, 1] | 1|
> | Procrustes Measure | proc | `k=20` | R+ | 0 |
> 
> ##### Cluster-level Measures
> 
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Steadiness & Cohesiveness | snc | `iteration=150, walk_num_ratio=0.3, alpha=0.1, k=50, clustering_strategy="dbscan"` | [0, 1] | 1 |
> | Distance Consistency | dsc | | [0.5, 1] | 0.5 | 
> | Internal Validation Measures | ivm | `measure="silhouette"` | Depends on IVM | Depends on IVM |
> | Clustering + External Clustering Validation Measures | c_evm | `measure="arand", clustering="kmeans", clustering_args=None` | Depends on EVM | Depends on EVM |
> | Label Trustworthiness & Continuity | l_tnc | `cvm="dsc"` | [0, 1] | 1 |




> ##### Global Measures
> 
> | Measure | ID | Parameters | Range | Optimum |
> |---------|----|------------|-------|---------|
> | Stress | stress | | R+ | 0 |
> | Kullback-Leibler Divergence | kl_div | `sigma=0.1` | R+ | 0 |
> | Distance-to-Measure | dtm | `sigma=0.1` | R+ | 0 |
> | Topographic Product | topo | `k=20` | R | 0 |
> | Pearson’s correlation coefficient | pr | | [-1, 1] | 1
> | Spearman’s rank correlation coefficient | srho | | [-1, 1] | 1



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

ZADU automatically optimizes the execution of multiple distortion measures. It minimizes the computational overhead associated with preprocessing stages such as pairwise distance calculation, pointwise distance ranking determination, and k-nearest neighbor identification.

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
The above code snippet demonstrates how to visualize local pointwise distortions using CheckViz and Reliability Map plots.

![mnist_1:7](https://github.com/hj-n/zadu/assets/37105201/7c6dc8d7-59c5-48fd-92a5-186e1e44597a)

## Documentation

For more information about the available distortion measures, their use cases, and examples, please refer to our paper (IEEE VIS 2023 Short).

## Citation

> Hyeon Jeon, Aeri Cho, Jinhwa Jang, Soohyun Lee, Jake Hyun, Hyung-Kwon Ko, Jaemin Jo, and Jinwook Seo. Zadu: A python library for evaluating the reliability of dimensionality reduction embeddings. In 2023 IEEE Visualization and Visual Analytics (VIS), 2023. to appear.

```bib
@inproceedings{jeon23vis,
  author={Jeon, Hyeon and Cho, Aeri and Jang, Jinhwa and Lee, Soohyun and Hyun, Jake and Ko, Hyung-Kwon and Jo, Jaemin and Seo, Jinwook},
  booktitle={2023 IEEE Visualization and Visual Analytics (VIS)}, 
  title={ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={},
  note={to appear}
}
```

