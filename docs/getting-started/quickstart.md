# Quickstart

## Evaluate a PCA projection

This example uses the Iris dataset bundled with scikit-learn. Each of its 150
rows is one sample with four measurements. PCA produces two coordinates for
each row; no dataset download is needed.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from zadu import ZADU

original, labels = load_iris(return_X_y=True)
projection = PCA(n_components=2).fit_transform(original)

specs = [
    {"id": "tnc", "params": {"k": 10}},
    {"id": "stress"},
]
runner = ZADU(specs, original)
scores = runner.measure(projection)

print(f"Trustworthiness: {scores[0]['trustworthiness']:.3f}")
print(f"Continuity:      {scores[0]['continuity']:.3f}")
print(f"Stress:          {scores[1]['stress']:.3f}")
```

The example prints, rounded to three decimal places:

```text
Trustworthiness: 0.983
Continuity:      0.991
Stress:          0.042
```

Results follow the order of `specs`:

- **Trustworthiness** decreases when points that were far apart become
  neighbors in the projection.
- **Continuity** decreases when original neighbors become separated.
- **Stress** measures the discrepancy between original and projected
  pairwise distances. Smaller is better; 0 means the distances match.

T&C scores range from 0 to 1, with 1 indicating preserved neighborhoods at the
chosen `k`. The `k=10` setting evaluates ten neighbors per sample. These scores
do not establish that every aspect of the dataset is preserved.

## Evaluate class structure

The Iris `labels` array contains one class label per row. Reusing the arrays
above, evaluate how often a projected neighbor has the same label:

```python
specs = [{"id": "nh", "params": {"k": 10}}]
scores = ZADU(specs, original).measure(projection, label=labels)
print(scores[0]["neighborhood_hit"])
```

Neighborhood Hit ranges from 0 to 1. It describes label agreement in the
projection; it does not compare those neighborhoods with the original space.
See [Choose measures](../guides/choosing-measures.md) for that distinction.

## Use named identifiers

These specifications are equivalent to the first example:

```python
from zadu import MEASURE, make_spec

specs = [make_spec(MEASURE.TNC, k=10), make_spec(MEASURE.STRESS)]
scores = ZADU(specs, original).measure(projection)
```

Short aliases such as `"tnc"` and full IDs such as
`"trustworthiness_continuity"` are both accepted.

## Use your own data

- Supply finite numeric arrays of shape `(n, d)` and `(n, p)`. **Row `i` must
  refer to the same sample in both arrays.** ZADU cannot detect a row permutation.
- Most measures accept any positive feature count. Gap Index requires a
  projection with exactly two columns.
- Labels have shape `(n,)` in the same row order. Use numeric labels or strings
  of a consistent, comparable type.
- Choose `1 <= k < n`; T&C and class-aware T&C additionally require `k < n / 2`.
- ZADU does not standardize features or fit projections. Apply any intended
  preprocessing before constructing the runner, and record it with the scores.

Input restrictions differ by measure. For example, Pearson needs nonconstant
pairwise distances, while Neighborhood Hit accepts a single class and returns
1 for it. See the [measure reference](../measures/index.md) for parameters and
[API reference](../reference/zadu.md) for return values and exceptions.
