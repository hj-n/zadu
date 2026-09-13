# ZADU

ZADU measures what changes when high-dimensional data is projected into fewer
dimensions. Its 22 measures cover neighborhoods, distances, class structure,
density, and empty regions in scatterplots.

Provide the original data and a projection of the same samples. ZADU returns
one result for each measure you select.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from zadu import ZADU

original, labels = load_iris(return_X_y=True)
projection = PCA(n_components=2).fit_transform(original)
scores = ZADU([{"id": "tnc", "params": {"k": 10}}], original).measure(projection)
print(scores[0])  # trustworthiness and continuity; higher is better
```

[Install ZADU](getting-started/installation.md){ .md-button .md-button--primary }
[Quickstart](getting-started/quickstart.md){ .md-button }

## Find what you need

| Task | Documentation |
| --- | --- |
| Decide what to measure and interpret the result | [Choose measures](guides/choosing-measures.md) |
| Look up parameters, return keys, and papers | [Measure reference](measures/index.md) |
| Compare projections of the same dataset | [Evaluate many projections](guides/many-projections.md) |
| Locate distortion in a scatterplot | [Pointwise scores](guides/local-scores.md) and [Visualization](guides/visualization.md) |
| Control memory use | [Memory and execution](guides/execution.md) |
| Run supported calculations with MLX or PyTorch | [Execution backends](backends.md) |
| Add a measure or report a problem | [Contributing](development/contributing.md) |

## Citation

Cite the [ZADU paper](https://doi.org/10.1109/VIS54172.2023.00048) and the
[software version you used](https://github.com/hj-n/zadu#citation).
The [measure reference](measures/index.md) links each measure's original paper.
