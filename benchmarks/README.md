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

## Release-history comparison

Compare the current source against the stable 2023 tag and the release
immediately before the 0.5.1 acceleration work under one interpreter/dependency
environment:

```bash
git worktree add --detach /tmp/zadu-v0.1.1 v0.1.1
git worktree add --detach /tmp/zadu-v0.5.0 v0.5.0

python benchmarks/benchmark_release_history.py \
  --source 2023-v0.1.1=/tmp/zadu-v0.1.1 \
  --source pre-acceleration-v0.5.0=/tmp/zadu-v0.5.0 \
  --source current=. \
  --samples 500 1000 2000 --dimension 20 --k 20 --repeat 5 \
  --json benchmarks/results/0.5.1/history-default.json
```

The worker runs every source in an isolated process and reports constructor,
first-evaluation, warm median, process peak RSS, score deltas, revisions, and
environment metadata. Use `--suite` to select a common workload or
`--embeddings 8` for repeated evaluation. To compare one current explicit
backend with historical defaults, add for example:

```bash
--accelerated-label current --backend mlx --device gpu --dtype float32
```

This is a same-machine source comparison, not a reconstruction of 2023 hardware
or dependencies. Metric fixes can also invalidate equal-work claims; inspect the
reported score delta and the full
[0.5.1 report](../docs/performance/0.5.1-acceleration-report.md).

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
  --samples 2000 --dimension 20 --repeat 5 \
  --memory-budget 64MiB --temporary-budget 512MiB
```

The three isolated modes are the legacy dense reference, the default in-memory
condensed order, and exact external ordering. The JSON report separates cold
construction from a warm repeated embedding, process peak RSS, planned RAM and
temporary bytes, observed temporary peak, slowdown from disk I/O, and maximum
score delta. The external mode deliberately recomputes and cleans its pair
workspace for each embedding instead of retaining unbounded order files.
The maintained Apple M4 `n=2,000` record is stored in
[external-pair-ordering-m4.json](results/post-0.5.1/external-pair-ordering-m4.json).

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
  --samples 2000 --dimension 20 --embeddings 8 --workers 2 --repeat 3
```

The four modes run in isolated processes: independent runners, a manual reused
runner, sequential `measure_many()`, and bounded parallel `measure_many()`.
Construction is included so the
report separates the benefit of sharing original-space pair orders, rankings,
and maximum-`k` neighbor resources from the small API overhead relative to an
already-correct manual `measure()` loop. It also reports exact score deltas,
process peak RSS, planned collection peak bytes, effective workers, and
reuse-event counts.

For long or generated collections, compare the materialized API with sequential
and bounded-parallel iterator execution in isolated processes:

```bash
python benchmarks/benchmark_stream_measure_many.py \
  --samples 2000 --dimension 20 --embeddings 32 --workers 2 --repeat 3
```

The report includes elapsed time, process peak RSS, the planned and observed
in-flight width, whether run diagnostics were retained, and an exact score
checksum delta. Peak RSS is process-wide; use enough generated embeddings for
their retained input/result footprint to rise above interpreter noise.

## Exact selected-rank design gate

Compare the 0.5.1 full inverse-ranking representation with the production NumPy
blockwise selected-rank resource:

```bash
python benchmarks/benchmark_selected_ranks.py \
  --samples 2000 --dimension 20 --neighbors 20 \
  --memory-budget 16777216 --repeat 3
```

The two implementations run in isolated processes. The full-ranking side is an
independent compatibility oracle; the selected side calls the installed NumPy
provider. The report includes exact array digests, metric-score delta, median
construction time, process peak RSS, retained-array bytes, and the selected
path's planned working bytes and block shape. The design thresholds are
documented in the
[post-0.5.1 exact scaling plan](../docs/development/post-0.5.1-exact-scaling-plan.md),
with the initial [Apple M4 result](results/post-0.5.1/selected-ranks-m4-16mib.json)
and the [production resource plus end-to-end comparison](results/post-0.5.1/selected-ranks-production-m4.json)
kept as machine-readable evidence.

To compare the production NumPy path with one native optional provider through
T&C and MRRE, run each framework in a separate process:

```bash
python benchmarks/benchmark_selected_rank_backends.py \
  --backend mlx --device gpu --dtype float32 \
  --samples 2000 --dimension 20 --k 20 --repeat 15

python benchmarks/benchmark_selected_rank_backends.py \
  --backend torch --device cpu --dtype float64 \
  --samples 2000 --dimension 20 --k 20 --repeat 15
```

The report separates construction/first execution, warm total time, selected-
rank resource time, score delta, transfer time, block shape, and fixed provider
workspace, and emits every warm timing sample so device variance stays visible.
For GPU and MPS, repeat with the budgets expected in production; transfer and
scheduling overhead can reverse the result when a tight budget creates many row
blocks.
The maintained Apple M4 measurements are stored in
[native-selected-ranks-m4.json](results/post-0.5.1/native-selected-ranks-m4.json).

## Optional MLX pairwise provider

Install `zadu[mlx]` on Apple Silicon, then compare cold and warm distance
resource construction against the NumPy/SciPy baseline:

```bash
python benchmarks/benchmark_mlx_pairwise.py \
  --samples 2000 --dimension 20 --kind distance-matrix \
  --device gpu --dtype float32 --repeat 3
```

The report separates input/output transfer, first compile plus execution, and
warm execution time. It also records the memory-bounded block plan and maximum
absolute distance delta. On an Apple M4 with MLX 0.32.1, `n=2,000` and 20 input
dimensions, the warm distance-matrix path measured 3.35x faster than SciPy
(`4.77 ms` versus `15.95 ms`). Condensed pairs measured 1.81x faster (`4.01 ms`
versus `7.27 ms`). Both paths had a maximum absolute float32 delta of `1.70e-6`.
Cold MLX times were about 31 ms, so cold and warm results must not be conflated.

## Optional PyTorch pairwise provider

Install `zadu[torch]`, then compare the exact PyTorch `cdist` resource with the
NumPy/SciPy baseline. Apple Silicon uses `--device mps --dtype float32`; CUDA
and CPU use the same benchmark but must be measured on their target hardware.

```bash
python benchmarks/benchmark_torch_pairwise.py \
  --samples 2000 --dimension 20 --kind distance-matrix \
  --device mps --dtype float32 --repeat 3
```

The JSON separates cold and warm end-to-end construction, provider-reported
input/output transfer and execution time, block bounds, and maximum absolute
distance delta. Results are hardware-specific; support in common code is not a
substitute for a benchmark on a real CUDA machine.

Benchmark stable full/inverse ranking or an exact stable neighbor prefix with:

```bash
python benchmarks/benchmark_torch_neighbors.py \
  --samples 2000 --dimension 20 --k 20 --kind ranking \
  --device mps --dtype float32 --repeat 3
```

The report includes index mismatches against the float64 NumPy baseline;
float32 can reorder nearly equal non-tied distances. PyTorch uses stable full
sorting because `torch.topk` does not promise
stable indices for ties; this preserves ZADU's duplicate-distance contract at
the cost of `O(n log n)` work even for a small neighbor prefix.

Compare sequential and provider-native repeated-embedding execution with:

```bash
python benchmarks/benchmark_torch_measure_many.py \
  --samples 2000 --dimension 20 --embeddings 8 --batch-size 4 --k 20 \
  --device mps --dtype float32 --repeat 3
```

The benchmark reports cold and warm collection timings, score deltas, the
effective native width, and the conservative planned peak. Run CPU, MPS, and
CUDA separately: batching can amortize accelerator overhead while providing no
benefit on a CPU workload.

## Optional MLX neighbor resources

Benchmark the stable full/inverse ranking, ordinary exact kNN, or stable-kNN
resource independently:

```bash
python benchmarks/benchmark_mlx_neighbors.py \
  --samples 2000 --dimension 20 --k 20 --kind ranking \
  --device gpu --dtype float32 --repeat 3
```

Then exercise the resource planner and five downstream metrics together:

```bash
python benchmarks/benchmark_mlx_neighbor_metrics.py \
  --samples 2000 --dimension 20 --k 20 \
  --device gpu --dtype float32 --repeat 3
```

Both reports separate construction/compile and warm execution. The resource
report also records exact index mismatches against the float64 NumPy baseline,
block bounds, distance-source reuse, and zero-copy boundaries. The metric report
compares Trustworthiness & Continuity, LCMC, Neighborhood Hit, Procrustes, and
Topographic Product scores and includes the complete MLX run diagnostics.

On an Apple M4 with MLX 0.32.1, `n=2,000`, 20 input dimensions, and `k=20`,
warm full ranking measured 23.75x faster than NumPy (`10.56 ms` versus
`250.88 ms`) and stable-kNN measured 20.30x faster (`7.11 ms` versus
`144.28 ms`). The five-metric warm suite measured 11.38x faster (`33.64 ms`
versus `382.84 ms`) with a maximum absolute score delta of `1.40e-6`.

Standalone ordinary kNN measured `7.64 ms` in MLX versus `4.27 ms` in FAISS, so
the default automatic backend remains NumPy/FAISS. MLX uses a stable full-order
prefix because an `argpartition` boundary does not define duplicate-distance
tie order; exact tie repair was slower than MLX's stable sort in the same
benchmark. Float32 full rankings can also differ from the float64 baseline when
nearly equal non-tied distances cross after rounding, so score and index deltas
are reported rather than hidden.
