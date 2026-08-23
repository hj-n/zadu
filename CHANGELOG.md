# Changelog

## Unreleased

### Added

- A no-code metric proposal path for contributors and repository guidance for
  coding agents implementing paper- or reference-backed metrics.
- Repository-wide metric contract tests covering registry, enum, exports,
  signatures, direct-versus-scheduled score parity, and execution-DAG resource
  sharing.
- A development-only exact selected-rank oracle, isolated memory/time benchmark,
  and post-0.5.1 scaling plan for replacing persistent full inverse rankings.
- Bounded `iter_measure_many()` evaluation with lazy indexed results,
  per-result diagnostics, memory-capped concurrency, deterministic ordering,
  indexed failures, and explicit early-close cleanup.
- Opt-in exact external-memory ordering for Spearman and Non-Metric Stress,
  including deterministic sorted runs, bounded-fan-in merge, disk-backed
  tie-average ranks and PAVA, explicit RAM/disk budgets, and exception-safe
  workspace cleanup.

### Changed

- Exact external pair ordering now uses compiled binary merges instead of a
  per-record Python heap merge.
- Scheduled T&C, class-aware T&C, and MRRE now build exact cross-space ranks in
  bounded NumPy row blocks and retain `O(nk)` state instead of two persistent
  `n x n` inverse rankings. Run diagnostics expose block bounds, distance reuse,
  retained dtypes, and optional-provider fallback.
- MLX and PyTorch now execute the paired selected-rank resource natively with
  device-side stable sorting, inverse scatter, rank gather, and membership
  reductions. Geodesic rank resources retain an explicit NumPy fallback, and
  PyTorch planning reserves its fixed target-index transfer workspace.

### Fixed

- Rank-based metrics now honor `geodesic=True` for the registered original
  space even when no other metric requests a dense distance matrix.
- MLX CPU float64 resources now create slices, casts, and workspace views on
  the provider stream instead of inheriting the process-default GPU stream.

## 0.5.1

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
- Opt-in, memory-bounded `embedding_workers` for exact repeated-embedding
  execution and indexed `EmbeddingExecutionError` failures.
- An optional, lazily imported MLX provider for memory-bounded Euclidean
  distance matrices, condensed pairs, stable full/inverse rankings, exact
  neighbor prefixes, and stable-kNN tables on Apple Silicon.
- Provider-native MLX tensor batching for equal-shaped `measure_many()` inputs,
  including memory-capped batch width and indexed batch failures.
- An optional, lazily imported PyTorch provider for exact, memory-planned
  Euclidean distance matrices, condensed pairs, stable full/inverse rankings,
  and exact stable neighbor prefixes on CPU, MPS, and CUDA.
- Provider-native PyTorch batching for equal-shaped repeated embeddings using
  batched `cdist` and stable sorting, with bounded batch width and indexed
  failures.
- A provisional `zadu.backends` entry-point contract for separately packaged
  exact providers, including an optional planner working-memory hook.
- A backend capability guide, scheduled benchmark workflow, reproducible
  release-history benchmark, and 0.5.1 acceleration report.

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
- Parallel repeated-embedding runs share frozen original resources, isolate
  mutable per-embedding caches, limit native inner threads, and automatically
  fall back for unsafe random-state configurations.
- Explicit MLX device and dtype selection now preserves the NumPy default,
  forbids silent precision downgrade, records cold/warm/transfer timings, and
  falls back to NumPy/SciPy/FAISS per unsupported resource.
- MLX distance outputs are reused through unified-memory views by dependent
  rankings; embedded workspaces are invalidated between runs, and neighbor
  diagnostics expose stable tie handling, self exclusion, block bounds, and
  zero-copy boundaries.
- On MLX, `embedding_workers` now requests a native tensor batch width rather
  than Python threads; incompatible shapes or plans fall back sequentially with
  the reason recorded in collection diagnostics.
- On PyTorch, `embedding_workers` likewise requests a native tensor batch width;
  unsupported shapes and memory-constrained plans preserve ordered sequential
  execution with an explicit diagnostic reason.

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
