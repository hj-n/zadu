# ZADU

ZADU evaluates how faithfully a dimensionality-reduction embedding preserves
the structure of its original data. It provides **22 local, cluster-level,
global, and gap-based measures** through one consistent Python interface.

```python
import numpy as np
from zadu import ZADU

rng = np.random.default_rng(0)
original = rng.normal(size=(200, 16))
embedding = original[:, :2] + 0.05 * rng.normal(size=(200, 2))

specs = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "mrre", "params": {"k": 20}},
]

scores = ZADU(specs, original).measure(embedding)
print(scores)
```

[Install ZADU](getting-started/installation.md){ .md-button .md-button--primary }
[Follow the quickstart](getting-started/quickstart.md){ .md-button }

## Where to begin

- **New to embedding evaluation?** Read [Choose measures](guides/choosing-measures.md)
  and start with more than one structural perspective.
- **Already know the metric?** Find its ID, parameters, score range, return
  keys, and primary paper in the [measure reference](measures/index.md).
- **Evaluating many projections?** Reuse original-space work with
  [`measure_many()`](guides/many-embeddings.md).
- **Working at larger scale?** Configure bounded exact execution and optional
  accelerator backends in [Memory and exact execution](guides/execution.md).
- **Adding a published metric?** A paper link and an optional reference
  implementation are enough to [propose a measure](development/contributing.md).

## What ZADU provides

### Multiple structural perspectives

Use neighborhood preservation, class-aware validation, distance preservation,
topological, density, and gap-based measures without combining incompatible
score meanings into one opaque number.

### Exact shared execution

When several measures need the same distances, neighbors, ranks, or pair
reductions, ZADU's execution DAG builds the compatible resource once. Memory
budgets select bounded exact strategies or fail before an oversized managed
allocation; they do not silently approximate the published formula.

### Optional acceleration

NumPy/SciPy is the default dependency-light path. MLX and PyTorch are optional,
lazily imported backends for supported exact resources. Backend choices,
fallbacks, memory plans, and timings remain separate from scientific scores in
`last_run_info`.

### Pointwise diagnosis

Measures that expose local contributions can return one score per data point.
The optional ZADUVis package renders these values with CheckViz and Reliability
Map visualizations.

## Citation

If ZADU supports your work, cite the
[ZADU paper](https://doi.org/10.1109/VIS54172.2023.00048). Each measure's
original literature is linked from the [measure reference](measures/index.md).
