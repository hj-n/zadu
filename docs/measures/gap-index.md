# Gap Index

Gap Index quantifies how strongly empty triangular regions in a 2D projection
are deformed relative to the corresponding regions in the original space.
Triangle areas are normalized to sum to 1 separately in each space. A score of
0 means those relative areas match; a uniform enlargement alone does not
increase the score. The score ranges from 0 to 1.

The metric was introduced by **Jaume Ros, Alessio Arleo, and Fernando
Paulovich** in
[*Measuring Distortion in the Empty Regions of Dimensionality Reduction
Scatterplots with the Gap Index*](https://arxiv.org/abs/2607.28324).

## Standard ZADU interface

The examples use `original` and `projection` from the
[quickstart](../getting-started/quickstart.md).

```python
from zadu import ZADU

specs = [{"id": "gi", "params": {"metric": "euclidean"}}]
score = ZADU(specs, original).measure(projection)[0]["gap_index"]
```

The projection must have exactly two columns and contain at least three
non-collinear points. Its Delaunay triangulation defines the regions.

## Distance choices

`metric` may be:

- `"euclidean"` for the optimized coordinate path;
- the name of a SciPy distance function such as `"cityblock"`;
- a callable accepting two original-space rows; or
- `"precomputed"` when `original` is a finite, symmetric, non-negative
  `(n, n)` distance matrix with a zero diagonal.

The projection-space triangle edges are always Euclidean, matching the published
formulation.

```python
from scipy.spatial.distance import pdist, squareform
from zadu.measures import gap_index

distances = squareform(pdist(original))
score = gap_index.gap_index(distances, projection, metric="precomputed")
```

## Regional details

Use the direct `compute()` function to obtain the triangulation and its
per-region values:

```python
from zadu.measures import gap_index

result = gap_index.compute(original, projection)

print(result.score)
print(result.triangles)
print(result.deformations)
print(result.original_relative_areas)
print(result.embedded_relative_areas)
```

`triangles` has shape `(m, 3)` and contains sample indices. Each of the other
regional arrays has length `m`. Positive `deformations` indicate a larger
relative triangle area in the projection; negative values indicate a smaller
one. The global score combines their absolute magnitudes with area weights.

`measure()` returns the scalar score dictionary. `compute()` returns these
regional arrays as well; `return_local=True` on the runner does not expose them.

## Provenance

ZADU adapted the authors' MIT-licensed
[reference implementation](https://codeberg.org/jros/gap-index) at revision
`0a11e4887864fe5d41526d8487eea33685b8f0b4`. The port adds input validation,
regional result objects, and block processing for Euclidean and precomputed
areas. Tests compare against a pinned upstream result.

The original algorithm and implementation remain credited to Ros, Arleo, and
Paulovich. See the repository's
[third-party notice](https://github.com/hj-n/zadu/blob/master/THIRD_PARTY_NOTICES.md)
and
[retained MIT license](https://github.com/hj-n/zadu/blob/master/LICENSES/gap-index-MIT.txt).
