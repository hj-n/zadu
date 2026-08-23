# Quickstart

## Evaluate one embedding

Prepare a finite two-dimensional original array and an embedding with the same
number of rows:

```python
import numpy as np
from zadu import ZADU

rng = np.random.default_rng(0)
original = rng.normal(size=(200, 16))
embedding = original[:, :2] + 0.05 * rng.normal(size=(200, 2))

specs = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "stress", "params": {}},
]

runner = ZADU(specs, original)
scores = runner.measure(embedding)

print(scores[0]["trustworthiness"])
print(scores[0]["continuity"])
print(scores[1]["stress"])
```

Results follow specification order. Every result is a dictionary of finite
Python scalar scores.

## Use typed specifications

`MEASURE` and `make_spec()` provide autocomplete-friendly alternatives to raw
dictionaries:

```python
from zadu import MEASURE, ZADU, make_spec

specs = [
    make_spec(MEASURE.TNC, k=20),
    make_spec(MEASURE.STRESS),
]

scores = ZADU(specs, original).measure(embedding)
```

The short aliases such as `"tnc"` and full IDs such as
`"trustworthiness_continuity"` are both accepted.

## Label-based measures

Pass a label vector to `measure()` when a specification contains `nh`,
`ca_tnc`, `dsc`, `ivm`, `c_evm`, `l_tnc`, or `cadi`:

```python
specs = [
    {"id": "nh", "params": {"k": 15}},
    {"id": "dsc", "params": {}},
]

scores = ZADU(specs, original).measure(embedding, label=labels)
```

Labels may be strings or arbitrary numeric values.

## Input checklist

- `original` and `embedding` must be finite numeric 2D arrays with the same
  number of rows.
- For neighbor-based measures, use `1 <= k < n`.
- Standard T&C and class-aware T&C normalization additionally require
  `k < n / 2`.
- Provide labels for every label-based specification.
- Undefined inputs such as constant distances, a single class, or coincident
  neighborhoods raise an actionable `ValueError` instead of returning `nan`
  or `inf`.

Next, use [Choose measures](../guides/choosing-measures.md) to match scientific
questions to metrics, or read [Evaluate embeddings](../guides/evaluating-embeddings.md)
for configuration and diagnostics.
