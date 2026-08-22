# Changelog

## 0.5.0

### Added

- Gap Index with upstream-reference tests and full third-party attribution.
- An explicit metric registry for scheduling, parameter validation, and lazy metric imports.
- Shared validation for array shape, finiteness, labels, neighbor counts, and undefined degenerate inputs.
- Reproducible `random_state` support for Steadiness & Cohesiveness.
- Python 3.10–3.14, formatting, lint, coverage, distribution, and license checks in CI.

### Fixed

- Mixed-`k` specifications now retain enough neighbors for every metric.
- CADI no longer loops forever when only one class can supply its paired samples.
- Topographic Product includes the first neighbor for `k=1`.
- Duplicate points can no longer leave the query point in its own k-nearest-neighbor list.
- Distance Consistency supports arbitrary numeric and string class labels.
- Trustworthiness normalizations reject unsupported `k >= n / 2` values.
- SNC now honors `k`, optional precomputed neighbors, and its documented randomness control.
- Pearson and Spearman distance correlations exclude the diagonal.
- KMeans external validation infers the number of target classes by default.
- Undefined constant-distance and single-class cases raise clear errors instead of returning non-finite values.
- Geodesic distance clamps floating-point roundoff before `acos`.
- CheckViz maps Voronoi regions through `point_region`; Reliability Map avoids duplicate mutual edges.

### Changed

- Packaging now uses PEP 621 metadata from `pyproject.toml`.
- Metric results returned through `ZADU` contain standard Python scalars.
- LCMC documentation reports its adjusted theoretical range and optimum.
