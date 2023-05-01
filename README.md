# ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings

ZADU is a Python library that provides a comprehensive suite of distortion measures for evaluating and analyzing dimensionality reduction (DR) embeddings. The library supports a diverse set of local, cluster-level, and global distortion measures, allowing users to assess DR techniques from various structural perspectives. By offering an optimized scheduling scheme and pointwise local distortions, ZADU enables efficient and in-depth analysis of DR embeddings.


## Installation

You can install ZADU via `pip`:

```bash
pip install zadu
```

## Supported Distortion Measures

ZADU currently supports a total of 17 distortion measures, including:

- 7 local measures
- 4 cluster-level measures
- 6 global measures

For a complete list of supported measures, refer to [measures](/src/zadu/measures).

## How To Use ZADU

ZADU provides two different interfaces for executing distortion measures.
You can either use the main class that wraps the measures, or directly access and invoke the functions that define each distortion measure.

### Using the Main Class

Use the main class of ZADU to compute distortion measures.
This approach benefits from the scheduling scheme, providing faster performance.


```python
from zadu import zadu

hd, ld = load_datasets()
spec = [{
    "id"    : "tnc",
    "params": { "k": 20 },
}, {
    "id"    : "snc",
    "params": { "k": 30, "clustering": "hdbscan" }
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

#### Parameters:

##### `spec` 
&nbsp;&nbsp;&nbsp;&nbsp;
A list of dictionaries that define the distortion measures to execute and their hyperparameters.
Each dictionary must contain the following keys:
  * `"id"`: The identifier of the distortion measure, such as `"tnc"` or `"snc"`.

  * `"params"`: A dictionary containing hyperparameters specific to the chosen distortion measure.

<details>
<summary style="cursor: pointer; font-weight: bold; color: #0066cc;">List of ID/Parameters for Each Function</summary>

### Local Measures

| Measure | ID | Parameters | Range | Optimum |
|---------|----|------------|-------|---------|
| Trustworthiness & Continuity | tnc | `k=20` | [0.5, 1] | 1 |
| Mean Relative Rank Errors | mrre | `k=20` | [0, 1] | 1 | 
| Local Continuity Meta-Criteria | lcmc | `k=20` | [0, 1] | 1 |
| Neighborhood hit | nh | `k=20` | [0, 1] | 1 |
| Neighbor Dissimilarity | nd | `k=20` | R+ | 0 |
| Class-Aware Trustworthiness & Continuity | ca_tnc | `k=20` | [0.5, 1] | 1|
| Procrustes Measure | proc | `k=20` | R+ | 0 |

### Cluster-level

| Measure | ID | Parameters | Range | Optimum |
|---------|----|------------|-------|---------|
| Steadiness & Cohesiveness | snc | `iteration=150, walk_num_ratio=0.3, alpha=0.1, k=50, clustering_strategy="dbscan"` | [0, 1] | 1 |
| Distance Consistency | dsc | | [0.5, 1] | 0.5 | 
| Internal Validation Measures | ivm | `measure="silhouette"` | Depends on IVM | Depends on IVM |
| Clustering + External Clustering Validation Measures | c_evm | `measure="arand", clustering="kmeans", clustering_args=None` | Depends on EVM | Depends on EVM |

### Global

| Measure | ID | Parameters | Range | Optimum |
|---------|----|------------|-------|---------|
| Stress | stress | | R+ | 0 |
| Kullback-Leibler Divergence | kl_div | `sigma=0.1` | R+ | 0 |
| Distance-to-Measure | dtm | `sigma=0.1` | R+ | 0 |
| Topographic Product | topo | `k=20` | R | 0 |
| Pearson’s correlation coefficient | pr | | [-1, 1] | 1
| Spearman’s rank correlation coefficient | srho | | [-1, 1] | 1


</details>

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

mrre = mean_relative_rank_error.run(hd, ld, k=20)
pr  = pearson_r.run(hd, ld)
nh  = neighborhood_hit.run(ld, label, k=20)
```

## Advanced Features

### Scheduling the Execution

ZADU optimizes the execution of multiple distortion measures through an effective scheduling scheme. It minimizes the computational overhead associated with preprocessing stages such as pairwise distance calculation, pointwise distance ranking determination, and k-nearest neighbor identification.

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

With the pointwise local distortions obtained from ZADU, users can visualize the distortions using various distortion visualizations. For example, CheckViz and the Reliability Map can be implemented using a Python visualization library with zaduvis.

```python
from zadu import zadu
from zaduvis import zaduvis
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

## load datasets and generate an embedding
hd = load_mnist()
ld = TSNE.fit_transform(hd)

## Computing local pointwise distortions
specs = [{"id": "snc", "params": {"k": 50}}]
zadu_obj = zadu.ZADU(spec, hd, return_local=True)
global_, local_ = zadu_obj.measure(ld)
l_s = local_[0]["local_steadiness"]
l_c = local_[0]["local_cohesiveness"]

## Visualizing local distortions
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
zaduvis.checkviz(ld, l_s, l_c, ax=ax[0])
zaduvis.reliability_map(ld, l_s, l_c, ax=ax[1])

```


The above code snippet demonstrates how to visualize local pointwise distortions using CheckViz and Reliability Map plots. `zaduvis.checkviz` generates a CheckViz plot, which shows local Steadiness (x-axis) vs. local Cohesiveness (y-axis) for each point in the embedding. `zaduvis.reliability_map` creates a Reliability Map plot, which colors each point in the embedding according to its local Steadiness and local Cohesiveness values.

## Documentation

For more information about the available distortion measures, their use cases, and examples, please refer to our [paper](zadu.pdf).

##Citation

##License

##Contributing
