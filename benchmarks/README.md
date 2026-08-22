# ZADU microbenchmarks

These benchmarks isolate metric kernels from dimensionality reduction and from
distance/kNN resource construction. They compare the current exact implementation
with intentionally slow reference formulas used before the 0.5.1 acceleration
work.

Run the default workload from the repository root:

```bash
python benchmarks/benchmark_numpy_kernels.py
```

Choose the workload and optionally save a machine-readable record:

```bash
python benchmarks/benchmark_numpy_kernels.py \
  --samples 2000 --dimension 50 --neighbors 20 --repeat 5 \
  --json benchmarks/result/numpy-kernels.json
```

The report includes input shape, dtype, median reference/current runtime,
speedup, absolute score delta, process peak RSS, Python/NumPy versions, and
platform information. RSS is process-wide and is not an allocation delta.

The PR targets are 10x for T&C and class-aware T&C, 4x for MRRE, 5x for LCMC,
10x for Neighborhood Hit, 50x for Topographic Product, and 1.5x for Procrustes
on a representative large input. These are review targets, not shared-CI timing
gates; noisy pull-request runners only enforce correctness.

## Pair-resource planner

Compare the legacy two-dense-matrix path with the fused exact pair-resource
planner in isolated child processes:

```bash
python benchmarks/benchmark_pair_resources.py \
  --samples 2000 --dimension 20 --repeat 5
```

Use `--memory-budget 16MiB` with the example workload to exercise block
streaming. The JSON report includes cold resource construction, warm repeated-
embedding time, isolated process peak RSS, planned cache/peak bytes, selected
strategy, speedup, and maximum score delta. Peak RSS remains process-wide rather
than an allocation delta.

## Ordered-pair planner

Compare the legacy dense Spearman/Non-Metric Stress path with the shared exact
condensed/order resources:

```bash
python benchmarks/benchmark_ordered_pair_resources.py \
  --samples 2000 --dimension 20 --repeat 5
```

The isolated JSON report separates cold construction from a warm repeated
embedding, process peak RSS, planned cache/peak bytes, and maximum score delta.

## Topographic Product resources

Compare the legacy dense distance/neighbor cache with exact blockwise stable-kNN
and selected-distance execution:

```bash
python benchmarks/benchmark_topographic_resources.py \
  --samples 3000 --dimension 20 --neighbors 20 --repeat 5
```

The report separates cold and repeated-embedding timing, isolated process peak
RSS, persistent cache and bounded-work estimates, and score delta.

## Compact and fused CPU resources

Compare the pre-fusion exact scheduler with compact indices, shared densities,
gathered-rank fusion, and blockwise neighbor statistics:

```bash
python benchmarks/benchmark_compact_fused_resources.py \
  --samples 2000 --dimension 20 --neighbors 20 --repeat 5
```

The report covers eight representative density, rank, and neighbor metrics. It
includes cold and repeated-embedding time, isolated peak RSS, persistent cache
and planned peak bytes, selected fusion strategies, and maximum score delta.

## Steadiness & Cohesiveness CPU execution

Compare the pre-PR dense SNN and scalar cluster-pair path with the exact sparse,
batched single-worker and deterministic multi-worker paths:

```bash
python benchmarks/benchmark_snc_cpu.py \
  --samples 2000 --neighbors 20 --iterations 50 --workers 4 --repeat 3
```

The three modes run in isolated processes and report execution time, peak RSS,
cache/planned-peak bytes, effective workers, and score deltas. Parallel timing
is reported separately because the single-worker path can be faster when native
library calls are short relative to thread scheduling overhead.

## Repeated embeddings

Compare rebuilding a runner for every embedding with manual runner reuse and the
ordered `measure_many()` API:

```bash
python benchmarks/benchmark_measure_many.py \
  --samples 2000 --dimension 20 --embeddings 8 --repeat 3
```

The three modes run in isolated processes. Construction is included so the
report separates the benefit of sharing original-space pair orders, rankings,
and maximum-`k` neighbor resources from the small API overhead relative to an
already-correct manual `measure()` loop. It also reports exact score deltas,
process peak RSS, planned per-embedding peak bytes, and reuse-event counts.
