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
