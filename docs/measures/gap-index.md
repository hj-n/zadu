# Gap Index

Gap Index quantifies how strongly empty triangular regions in a 2D projection
are deformed relative to the corresponding regions in the original space. A
score of 0 indicates no regional area distortion; the score is bounded by 1.

The metric was introduced by **Jaume Ros, Alessio Arleo, and Fernando
Paulovich** in
[*Measuring Distortion in the Empty Regions of Dimensionality Reduction
Scatterplots with the Gap Index*](https://arxiv.org/abs/2607.28324).

## Standard ZADU interface

```python
from zadu import ZADU

specs = [{"id": "gi", "params": {"metric": "euclidean"}}]
score = ZADU(specs, original).measure(embedding)[0]["gap_index"]
```

The embedding must have exactly two columns and contain at least three
non-collinear points. Its Delaunay triangulation defines the regions.

## Distance choices

`metric` may be:

- `"euclidean"` for the optimized coordinate path;
- the name of a SciPy distance function such as `"cityblock"`;
- a callable accepting two original-space rows; or
- `"precomputed"` when `original` is a finite, symmetric, non-negative
  `(n, n)` distance matrix with a zero diagonal.

The embedded triangle edges are always Euclidean, matching the published
formulation.

```python
from scipy.spatial.distance import pdist, squareform
from zadu.measures import gap_index

distances = squareform(pdist(original))
score = gap_index.gap_index(distances, embedding, metric="precomputed")
```

## Regional details

Use the direct `compute()` function to obtain the triangulation and its
per-region values:

```python
from zadu.measures import gap_index

result = gap_index.compute(original, embedding)

print(result.score)
print(result.triangles)
print(result.deformations)
print(result.original_relative_areas)
print(result.embedded_relative_areas)
```

The standard scheduled API intentionally returns only the finite scalar score.
The direct detailed result is useful for scientific analysis and custom
visualization.

## Provenance

ZADU adapted the authors' MIT-licensed
[reference implementation](https://codeberg.org/jros/gap-index) at revision
`0a11e4887864fe5d41526d8487eea33685b8f0b4`. The port adds ZADU's measure
contract, validation, typed detailed results, bounded vectorization for
Euclidean and precomputed areas, and regression tests pinned to an upstream
golden result.

The original algorithm and implementation remain credited to Ros, Arleo, and
Paulovich. See the repository's
[third-party notice](https://github.com/hj-n/zadu/blob/master/THIRD_PARTY_NOTICES.md)
and
[retained MIT license](https://github.com/hj-n/zadu/blob/master/LICENSES/gap-index-MIT.txt).
