# Post-0.5.1 exact scaling plan

> Status: active development record. PR 10-A established the selected-rank
> oracle and measurement gates, PR 10-B integrated it into production NumPy,
> and PR 10-C completed native MLX and PyTorch providers. PR 11 implements
> bounded repeated-embedding iteration. PR 12 implements exact external-memory
> ordering for globally ranked pairs. PR 13 implements exact blockwise density
> resources. PR 14 removes mixed-workload dense coupling and makes default kNN
> exact, stable, float64, and memory-planned. PR 15 targets Gap Index's scalar
> triangle loop.

## Objective

Reduce exact-evaluation memory and elapsed time after the 0.5.1 acceleration
release, especially where quadratic resources still limit practical sample
counts. Preserve metric definitions, stable tie behavior, public results, and
the dependency-light NumPy default.

Approximation remains out of scope. This plan does not introduce approximate
neighbors, sampling, landmarks, reduced iterations, or precision changes that
the user did not request. ZADUVis is also outside this workstream.

The previously considered prepared-original-resource cache is excluded. It
would add invalidation, serialization, and ownership complexity before the
remaining per-embedding quadratic resources are fixed.

## Current bottlenecks

### Resolved bottleneck: full neighbor rankings

Before PR 10-B, T&C, class-aware T&C, and MRRE retained one `n x n` inverse
ranking in each space. The fused metric kernels only read:

- the first `k` neighbor indices in both spaces;
- the original-space ranks of the embedded `k` neighbors;
- the embedded-space ranks of the original `k` neighbors; and
- exact row-wise membership masks for the requested `k` values.

With compact `int32` indices, the two inverse rankings alone retain `8n^2`
bytes. At `n=10,000`, that is about 763 MiB before neighbor prefixes, gathered
ranks, membership masks, distance/sort workspaces, and Python/runtime overhead.

### Repeated embeddings

`measure_many()` returns the entire result collection and its diagnostics. Its
execution width is bounded, but callers evaluating a very large or generated
embedding stream still need a bounded-yield interface.

### Globally ordered pairs

Exact Spearman and Non-Metric Stress require a global order over all
`n(n-1)/2` pairs. The current condensed representation is smaller than two
dense matrices but is still memory-resident, so it eventually becomes the next
exact scaling ceiling.

### Resolved bottleneck: Gaussian densities

DTM and KL retain only one density value per sample, but previously constructed
an `n x n` distance matrix and another full Gaussian-kernel copy. PR 13 computes
the global normalization maximum in one row-block pass and fuses every requested
sigma in a second pass. A dense matrix is consumed only when another metric has
already made it part of the exact plan.

## Semantic contract

Every implementation in this plan must preserve the following exact behavior:

1. A sample is forced to rank zero in its own row and never appears in its kNN
   prefix.
2. Equal distances use stable original-column order, including duplicate and
   equidistant points.
3. Cross-space ranks are integer-identical to values gathered from the current
   full inverse rankings.
4. Multiple requested `k` values share one maximum-`k` resource and obtain exact
   prefixes.
5. Default NumPy remains float64 for distances and uses the smallest safe signed
   integer dtype for retained indices and ranks.
6. A memory budget bounds planned temporary work and fails clearly if even one
   exact row cannot fit. It never silently selects an approximation.

## PR sequence

### PR 10-A — selected-rank oracle and gates (complete)

Add a development-only exact blockwise implementation alongside the existing
full-ranking oracle. It performs stable row sorts, constructs an inverse for one
row block by linear scatter, gathers only the cross-space ranks, and discards
the full inverse block.

Deliverables:

- independent exactness tests for random data, duplicate points, complete ties,
  self exclusion, multiple block sizes, mixed `k`, and local/global metric
  results;
- an isolated-process benchmark reporting median construction time, process
  peak RSS, retained-array bytes, planned work bytes, result digest, and score
  delta;
- these production acceptance gates and the remaining roadmap.

This PR does not alter the planner, providers, installed package, or public API.

Measured locally on Apple M4, Python 3.12.13, NumPy 2.5.2, with
`n=2,000`, 20 original dimensions, 2 embedded dimensions, `k=20`, three
repetitions, and a 16 MiB selected-rank work budget:

| Exact construction | Median | Peak RSS | Retained arrays |
| --- | ---: | ---: | ---: |
| Full inverse rankings | 0.554 s | 341.9 MiB | 32.72 MB |
| Blockwise selected ranks | 0.330 s | 150.7 MiB | 0.72 MB |

The candidate retained 45.44x less array memory, used 44.1% of the full path's
process peak RSS, and took 59.7% of its time. All result-array digests matched
and the maximum metric-score delta was zero. These are same-machine observations,
not portable timing promises.

Acceptance gates for moving the design into production:

- exact array equality and zero metric delta for the oracle suite;
- retained rank state described only by `O(nk)` arrays;
- at least 32x lower retained rank-resource bytes at `n=2,000, k=20`;
- a reported rather than hidden runtime tradeoff at 16 MiB and 64 MiB work
  budgets; and
- no production or public-API change in PR 10-A.

### PR 10-B — NumPy production selected-rank resource (complete)

Make `RANK_COMPARISONS` a direct paired-space resource for its registered
metrics instead of creating two persistent `NEIGHBOR_RANKING` dependencies.
Cache one exact `O(nk)` stable original-space neighbor prefix, obtain it through
stable partial selection, and count only the requested original-space target
ranks. The embedded side uses a stable block sort plus linear inverse scatter.
This avoids re-sorting the original neighbor prefix for every embedding without
bringing back a quadratic cache. Update cache and peak estimates, and expose
the algorithms, block count, block rows, work budget, and retained bytes in run
diagnostics.

Compatibility paths that explicitly call a metric with `knn_ranking_info`
remain unchanged. If another scheduled metric later requires a genuine full
ranking, the planner may still request `NEIGHBOR_RANKING` independently.

Production gates:

- the complete NumPy suite and metric-contract suite remain green;
- scheduled scores and local arrays match the 0.5.1 path on random, duplicate,
  tied, float32-input, float64-input, mixed-metric, and mixed-`k` cases;
- `estimated_cache_bytes`, `planned_peak_bytes`, resource records, and release
  lifetimes reflect actual ownership without double-counting views;
- constrained budgets select multiple blocks and an insufficient one-row budget
  raises a clear `MemoryError`; and
- the isolated benchmark shows no unexplained regression from the reviewed
  oracle.

Production measurement on the same Apple M4 environment and `n=2,000, k=20`
confirmed the standalone resource result: with a 16 MiB work budget, five-run
median construction was 0.329 s instead of 0.543 s, process peak RSS was
149.0 MiB instead of 342.3 MiB, retained arrays were 0.72 MB instead of
32.72 MB, digests matched, and the maximum score delta was zero.

The end-to-end core suite (`T&C`, MRRE, LCMC, and Neighborhood Hit) exposes the
intentional reuse tradeoff. Against v0.5.1 in the same environment, one cold
embedding improved from 0.557 s to 0.360 s and peak RSS fell from 310.3 MB to
234.9 MB. A warm embedding took 0.291 s instead of 0.264 s because selected
original ranks must be counted for each new embedding. Across eight embeddings,
cold total time was effectively equal (2.410 s versus 2.407 s), warm total time
was 9.9% slower (2.329 s versus 2.119 s), and peak RSS remained 24.3% lower.
All scores were identical. The cached exact `O(nk)` original prefix and stable
partial selection reduce this repeated-run cost from the initial direct-paired
prototype's roughly 23% slowdown; PR 10-C can target the remaining rank-count
work on MLX and PyTorch without restoring a quadratic cache.

### PR 10-C — MLX and PyTorch selected-rank providers (current)

Implement the same paired resource natively on optional providers. Keep row
blocks, distance calculation, stable sorting, inverse scatter, rank gather, and
membership reductions on the selected device until the retained `O(nk)` result
crosses the provider boundary.

Package this as another implementation of the existing exact resource contract,
not as a new user-facing metric or optional dependency. Unsupported device,
dtype, stable-sort, or budget combinations must take an explicit recorded
fallback.

Provider gates:

- NumPy float64 remains the semantic oracle;
- MLX is measured on Apple Silicon; PyTorch CPU and MPS are measured on the Mac;
- CUDA support is tested in CI or CUDA hardware and is never claimed from an MPS
  result;
- duplicate-distance tests prove stable ordering independently of random-data
  tolerance checks; and
- diagnostics separate compile/cold time, warm execution, transfer, block size,
  and fallback reason.

The implementation uses the existing paired-resource contract: no public API,
metric, or optional dependency changes are required. Both providers compute two
stable block orders, inverse-scatter ranks, gather only the cross-space targets,
and reduce the requested membership masks before transferring `O(nk)` output.
PyTorch batches the compact outputs into fewer host transfers and reserves a
fixed `int64` target-index workspace in the memory plan. Geodesic execution is
recorded as a NumPy fallback.

Apple M4 measurements with Python 3.12.13, `n=2,000`, `k=20`, fifteen warm
repetitions, and T&C plus MRRE gave the following end-to-end medians:

| Backend | Total budget | NumPy warm | Native warm | Speedup | Max score delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX CPU float64 | 16 MiB | 0.285 s | 0.322 s | 0.89x | 0 |
| MLX GPU float32 | 16 MiB | 0.286 s | 0.338 s | 0.85x | 1.36e-6 |
| MLX GPU float32 | 64 MiB | 0.287 s | 0.072 s | 3.98x | 1.36e-6 |
| Torch CPU float64 | 16 MiB | 0.286 s | 0.098 s | 2.91x | 0 |
| Torch MPS float32 | 16 MiB | 0.285 s | 0.206 s | 1.38x | 1.53e-7 |
| Torch MPS float32 | 64 MiB | 0.287 s | 0.093 s | 3.09x | 1.53e-7 |

At 16 MiB, optional plans needed seven blocks; at 64 MiB, GPU plans needed two.
The GPU sample ranges were wide (`0.059–0.409 s` for MLX at 16 MiB and
`0.052–0.378 s` for MPS), so the medians are observations rather than stable
service guarantees. PyTorch CPU was tightly grouped at `0.098–0.103 s`.
Framework startup and interactive device scheduling also made cold results
unreliable predictors of warm throughput. These results justify keeping NumPy
as `auto` and exposing block/transfer diagnostics rather than selecting a device
from hardware presence alone. CUDA remains unmeasured on this Mac.

### PR 11 — bounded streaming embedding evaluation

Add an iterator-style repeated-evaluation surface that accepts an iterable of
embeddings and yields ordered indexed results while retaining at most the
planned execution window. Reuse the same original-space resources as
`measure_many()` and preserve its failure indexing and deterministic ordering.

Keep `measure_many()` as the convenient materialized API, implemented over the
same execution core where practical. Do not use the stream to weaken exactness
or validation.

Gates:

- generator inputs are consumed lazily rather than materialized;
- sequential and bounded concurrent modes never exceed the documented in-flight
  width;
- yielded values, order, errors, and diagnostics match `measure_many()`; and
- early iterator close releases embedded/provider resources.

Implementation status: `iter_measure_many()` yields
`EmbeddingResult(index, result, run_info)` records. It incrementally aggregates
timings without retaining every run, caps NumPy/thread-safe provider work by the
same collection memory plan, and closes its input plus pending workers on early
termination. Providers whose safe parallel path is native tensor batching run
the iterator sequentially; `measure_many()` retains native batching.

### PR 12 — disk-backed exact pair ordering

Provide a planner-selected exact external-memory path for globally ordered pair
metrics. Generate sorted pair runs within the memory budget, persist them in an
explicit temporary workspace, and perform a deterministic k-way merge with the
same tie/rank semantics as the in-memory condensed path.

This is an internal pair-resource strategy. The public metric API stays the
same; diagnostics report strategy, temporary bytes, run count, merge passes,
timing, and cleanup outcome.

Gates:

- score and rank parity covers random inputs, duplicate distances, and ties
  crossing run boundaries;
- resident planned memory remains bounded as pair count grows;
- normal completion, exceptions, and interruption clean temporary files; and
- the planner uses disk only when requested or when the exact in-memory strategy
  cannot meet the configured budget. It never guesses an unsafe system-wide
  disk allowance.

Implementation status: `PairStrategy.EXTERNAL` creates deterministic
distance/index runs within the RAM plan, performs compiled binary merges, stores
tie-average original ranks in a disk map for Spearman, and evaluates
Non-Metric Stress with a disk-backed weighted PAVA stack. The planner requires
an explicit `temporary_budget`, checks its worst-case file topology before any
distance work, and accepts an application-owned `temporary_directory`.
Workspaces are evaluation-scoped and cleaned after success, errors, and
interruption; unlike the faster condensed path, external original ordering is
therefore recomputed for repeated embeddings.

On the maintained Apple M4 with Python 3.12.13, NumPy 2.5.2, `n=2,000`, 20
original dimensions, Spearman plus Non-Metric Stress, and three repetitions:

| Strategy | Warm median | Process peak RSS | Planned RAM | Persistent pair cache | Temporary peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy dense reference | 1.059 s | 480.4 MiB | — | 61.0 MiB estimate | 0 |
| In-memory condensed | 0.675 s | 361.9 MiB | 160.1 MiB | 38.1 MiB | 0 |
| External, 32 MiB RAM budget | 0.873 s | 338.3 MiB | 32.0 MiB | 0 | 99.1 MiB |

The external path reduced package-planned RAM by about 5x versus condensed,
and compiled binary merges reduced its warm slowdown from 8.18x to 1.29x.
Process RSS fell only about 6.5% because interpreter, SciPy, and Numba/JIT
overhead is not proportional to pair storage; against the dense reference it
fell about 30%. The maximum score delta from condensed was `8.11e-14`. This is
an exact capacity path rather than the default strategy. Cold and warm external
times were similar because its evaluation-scoped workspace is intentionally
recomputed and removed. The raw record is
[`external-pair-ordering-m4.json`](../../benchmarks/results/post-0.5.1/external-pair-ordering-m4.json).

### PR 13 — exact blockwise multi-sigma densities

Build DTM and KL density vectors directly from point blocks. Multiple sigma
values share the same two distance passes, while original-space vectors remain
reusable across embeddings. Dense distance matrices remain a valid DAG source
when another scheduled metric requires them.

On the maintained Apple M4 with Python 3.12.13, NumPy 2.5.2, `n=5,000`, 20
dimensions, sigma values 0.1 and 0.3, a 16 MiB work budget, and three
repetitions:

| Density construction | Median | Process peak RSS | Planned working memory |
| --- | ---: | ---: | ---: |
| Dense distance plus kernel | 0.263 s | 527.1 MiB | 381.5 MiB |
| Two-pass blockwise | 0.296 s | 145.6 MiB | 15.9 MiB |

The blockwise path retained the same 80,000 bytes of final density vectors,
used 23.9x less planned working memory, and took 1.12x the dense runtime. The
maximum density delta was zero. The raw record is
[`blockwise-density-m4.json`](../../benchmarks/results/post-0.5.1/blockwise-density-m4.json).

### PR 14 — memory-aware mixed resources and stable float64 kNN

Select pair representation from the whole resource plan rather than forcing
`DISTANCE_MATRIX` whenever pair reductions coexist with neighbors or ranks.
Unordered reductions may use condensed or streaming pairs; ordered reductions
may use condensed or explicitly budgeted external ordering. A geodesic run
retains the existing dense compatibility path.

The NumPy provider now serves ordinary and topographic kNN through the same
exact row-block algorithm: SciPy computes float64 distances, partial selection
keeps the useful prefix, and boundary ties are repaired by original column
index. Direct metric calls use the equivalent stable dense oracle. This removes
FAISS's implicit float32 input conversion and its mandatory base dependency.

On the maintained Apple M4 with Python 3.12.13, NumPy 2.5.2, `n=2,000`, 20
original dimensions, `k=20`, LCMC plus Stress and Pearson, a 48 MiB plan, and
three repetitions:

| Exact mixed execution | Cold median | Warm median | Process peak RSS | Retained resources |
| --- | ---: | ---: | ---: | ---: |
| Legacy dense matrices and stable full sorts | 0.563 s | 0.290 s | 417.5 MiB | 64.32 MB |
| Condensed pairs and blockwise stable kNN | 0.090 s | 0.039 s | 151.4 MiB | 32.32 MB |

The planned path was 6.24x faster cold and 7.42x faster warm, retained 1.99x
less package resource memory, and held package-planned peak memory to exactly
48 MiB. Process peak RSS was 63.7% lower, and the maximum score delta was
`9.54e-18`. The comparison reproduces the former mixed dense provider path,
including its stable full-matrix neighbor sorts. The raw record is
[`mixed-resources-m4.json`](../../benchmarks/results/post-0.5.1/mixed-resources-m4.json).

### PR 15 — exact Gap Index vectorization

Replace the per-triangle Python distance loop with bounded vectorized Euclidean
and precomputed-distance kernels. Preserve the scalar SciPy/callable fallback,
the published triangulation and Heron formula, source parity, and finite-input
validation. Benchmark random and duplicate-heavy inputs and report exact score
deltas before making this the default fast path.

## Packaging and release policy

- NumPy/SciPy remains the mandatory exact baseline; MLX and PyTorch stay optional
  extras and entry-point providers.
- Selected-rank and external-pair strategies are internal resource choices,
  visible through diagnostics but not new metric IDs.
- Each PR is independently reviewable, contains its own parity tests and
  benchmark command, and leaves a fallback to the preceding exact path until
  the new path is proven.
- Hardware-specific results name the device, dtype, cold/warm boundary, software
  versions, and exactness comparison. Results from this Mac do not stand in for
  CUDA measurements.
- A release report is written only after the production PRs are merged and
  measured together; oracle-only PR 10-A is evidence, not a user-facing speed
  claim.
