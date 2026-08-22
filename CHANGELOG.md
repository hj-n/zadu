# Changelog

## Unreleased

### Added

- Slow-reference parity tests and isolated microbenchmarks for exact NumPy metric
  kernels.
- Typed exact resource contracts, a deterministic execution planner, and a
  NumPy/FAISS resource provider.
- `ExecutionConfig` for backend, device, and memory-budget selection, plus
  `ZADU.last_run_info` for separate execution diagnostics.
- Exact condensed and memory-bounded streaming pair resources, with an isolated
  dense-versus-planned benchmark.
- Shared exact pair ordering and tie-aware rank resources for Spearman and
  Non-Metric Stress, with repeated-embedding reuse and preallocation guards.
- Exact blockwise stable-kNN and selected-distance resources for Topographic
  Product, including shared multi-`k` prefix results.
- Parameterized density, gathered-rank, and fused neighbor-statistics resources,
  with compact exact index/rank storage and explicit resource lifetimes.
- Deterministic optional `n_jobs` execution and memory-budget-aware worker
  planning for Steadiness & Cohesiveness.
- Ordered `ZADU.measure_many()` execution with shared original-space resources,
  per-embedding results, and aggregate JSON-compatible diagnostics.

### Changed

- Vectorized T&C, class-aware T&C, MRRE, LCMC, and Neighborhood Hit kernels.
- Reduced Topographic Product to gathered ratios and cumulative logs, and batched
  local Procrustes alignment with bounded temporary memory.
- Metric registry cache declarations now use typed resource requirements; full
  rankings satisfy compatible kNN requests and larger `k` resources serve
  smaller prefixes.
- Stress, Scale-Normalized Stress, and Pearson now share one stable sufficient-
  statistics pass over unique off-diagonal pairs and release per-run pair
  temporaries after their final consumer.
- Spearman and Non-Metric Stress now share compact condensed distances and one
  reusable original-space pair order instead of requiring two persistent dense
  distance matrices.
- Topographic Product now retains only `O(nk)` neighbor tables and computes its
  selected distances in bounded row blocks instead of caching two `n x n`
  distance matrices.
- Steadiness & Cohesiveness reuses planned kNN tables, keeps its full weighted-SNN
  graphs sparse, batches cluster-pair reductions, and preserves fixed-seed
  single/multi-worker global and local results.
- Repeated embeddings now use one exact maximum-`k` plan and are prevalidated as
  a collection before sequential memory-bounded execution begins.

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
