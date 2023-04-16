# ZADU: A Python Toolkit for Evaluating the Reliability of Dimensionality Reduction Embeddings

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

For a complete list of supported measures, refer to [the documentation](tree/master/src/zadu/measures).

## How To Use ZADU

ZADU provides two different interfaces for executing distortion measures.
You can either use the main class that wraps the measures, or directly access and invoke the functions that define each distortion measure.

### Using the Main Class

Use the main class of ZADU to compute distortion measures.
This approach benefits from the scheduling scheme, providing faster performance.


```python
from zadu import zadu

spec = {
    "tnc": { "k": 20 },
    "snc": { "k": 30, "clustering": "hdbscan" }
}
scores = zadu.ZADU(spec).run(hd, ld)
print("T&C:", scores[0])
print("S&C:", scores[1])
```

`hd` represents high-dimensional data, `ld` represents low-dimensional data

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

spec = {
    "dtm" : {},
    "mrre": { "k": 30 }
}
zadu_obj = zadu.ZADU(spec, return_local=True)
global_, local_ = zadu_obj.run(hd, ld)
print("MRRE local distortions:", local_["mrre"])
```

### Visualizing Local Distortions

With the pointwise local distortions obtained from ZADU, users can visualize the distortions using various distortion visualizations. For example, CheckViz and the Reliability Map can be implemented using a Python visualization library with zaduvis.

```python
from zadu import zadu
from zaduvis import zaduvis
import matplotlib.pyplot as plt

# Computing local pointwise distortions
specs = [{"measure": "snc", "params": {"k": 50}}]
zadu_obj = zadu.ZADU(spec, return_local=True)
global_, local_ = zadu_obj.run(hd, ld)
l_s = local_["local_steadiness"]
l_c = local_["local_cohesiveness"]

# Visualizing local distortions
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
zaduvis.checkviz(ld, l_s, l_c, ax=ax[0])
zaduvis.reliability_map(ld, l_s, l_c, ax=ax[1])
```


The above code snippet demonstrates how to visualize local pointwise distortions using CheckViz and Reliability Map plots. `zaduvis.checkviz` generates a CheckViz plot, which shows local Steadiness (x-axis) vs. local Cohesiveness (y-axis) for each point in the embedding. `zaduvis.reliability_map` creates a Reliability Map plot, which colors each point in the embedding according to its local distortion scores, providing a spatial representation of the distortions in the DR embeddings.

## Documentation

For more information about the available distortion measures, their use cases, and examples, please refer to our [paper]().

##Citation

##License

##Contributing
