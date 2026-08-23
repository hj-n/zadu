# Exact execution backends

ZADU keeps NumPy/SciPy as the dependency-light execution baseline and
loads accelerator frameworks only after an explicit backend request. Every
backend in 0.5.2 evaluates the full exact resource contract: `float32` changes
rounding, not the number of pairs, neighbors, or iterations evaluated.

## Capability table

| Capability | NumPy CPU | MLX CPU | MLX GPU | Torch CPU | Torch MPS | Torch CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| Base install | Yes | Extra | Extra; Apple Silicon | Extra | Extra; Apple | Extra |
| Dtype | float64 | float32/64 | float32 | float32/64 | float32 | float32/64 |
| Dense/condensed Euclidean | Native | Native | Native | Native | Native | Common code |
| Stable full/inverse ranking | Native | Native | Native | Native | Native | Common code |
| Paired selected ranks | Native | Native | Native | Native | Native | Common code |
| Ordinary exact kNN | Stable blockwise partial selection | Stable full-sort prefix | Stable full-sort prefix | Stable full-sort prefix | Stable full-sort prefix | Common code |
| Topographic stable-kNN | SciPy blocks | Native | Native | Native | Native | Common code |
| Derived metric reductions | Native | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback |
| Geodesic resources | Native | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback |
| Native `measure_many` tensors | No; bounded threads | Yes | Yes | Yes | Yes | Common code |
| Maintained hardware parity CI | Yes | Yes | Yes | Yes | Yes | No |

The Torch implementation uses PyTorch's documented batched
[`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html)
and stable
[`torch.argsort`](https://docs.pytorch.org/docs/stable/generated/torch.argsort.html)
operations. It deliberately avoids bare
[`torch.topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)
for exact neighbor ties because PyTorch documents that tied indices are not
stable. [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) is PyTorch's
documented Apple GPU backend; ZADU's maintained job tests its supported float32
resource paths.

“Common code” means the CUDA branch uses the same tested tensor operations and
has explicit availability checks, but 0.5.2 was not benchmarked or parity-tested
on real CUDA hardware. It is supported as a preview, not performance-validated.

`backend="auto"` deliberately selects NumPy/SciPy. Its ordinary kNN path keeps
float64 precision, repairs partial-selection boundary ties by original index,
and bounds distance work by the execution memory plan. Accelerator startup can
dominate small jobs, so choose MLX or PyTorch explicitly after benchmarking the
actual specification and data size.

## Selection and diagnostics

```python
from zadu import ExecutionConfig, ZADU

runner = ZADU(
    specs,
    original,
    execution=ExecutionConfig(
        backend="mlx",       # numpy, mlx, torch, or an installed entry point
        device="gpu",
        dtype="float32",
        memory_budget="4GiB",
        embedding_workers=4,
        pair_order_strategy="auto",
        temporary_budget=None,
    ),
)
scores = runner.measure_many(projections)
print(runner.last_run_info)
```

For a generated or very long input, keep both values and diagnostics bounded:

```python
stream = runner.iter_measure_many(generate_projections())
try:
    for item in stream:
        consume(item.index, item.result, item.run_info)
finally:
    stream.close()
```

The stream is lazy and ordered. NumPy and thread-safe external providers use the
memory-planned in-flight width. MLX and PyTorch currently execute this iterator
sequentially because their repeated-projection acceleration is native tensor
batching; the materialized `measure_many()` API retains that batching path.
`last_run_info` becomes a bounded aggregate after exhaustion or close and does
not retain every per-projection run.

Diagnostics distinguish requested and actual providers, per-resource fallback
reasons, input/output transfers, first execution, warm execution, block sizes,
planned package-managed peak memory, external pair run/merge shape and
temporary peak, and provider-native batch width. Device
framework allocators can retain pools outside that estimate; use isolated peak
RSS or device-profiler measurements when capacity planning.

Exact Spearman and Non-Metric Stress normally use the in-memory condensed pair
order. Set `pair_order_strategy="external"` plus an explicit
`temporary_budget`, or provide that budget with `"auto"` and a restrictive
memory budget, to select bounded external sorting. `temporary_directory` can
target application-owned scratch storage. The planner checks worst-case run,
merge, rank, and PAVA files before distance construction; normal completion,
errors, and interruption remove the per-evaluation workspace. This path is a
NumPy fallback for optional providers and trades I/O and merge time for a
bounded exact RAM footprint. Repeated-projection concurrency is also capped so
the sum of simultaneous planned workspaces stays within `temporary_budget`.

Registered T&C, class-aware T&C, and MRRE specifications request the paired
selected-rank resource. The NumPy provider builds it in exact bounded blocks and
retains `O(nk)` indices, cross-space ranks, and membership masks. MLX and
PyTorch keep block distances, stable sorting, inverse scatter, rank gathers, and
membership reductions on the selected device, then transfer only the compact
retained result. PyTorch plans a fixed target-index conversion/transfer buffer
in addition to row-block work. Geodesic selected ranks remain an explicit NumPy
fallback because neither optional provider implements geodesic distances.

Cold time includes first framework/device initialization. Warm time measures a
reused provider after that initialization. Neither should be substituted for
the other: on the release machine, small MLX jobs were much faster warm but
slower cold, while the crossover appeared by the 2,000-sample representative
suite.

On the maintained Apple M4 at `n=2,000`, `k=20`, the selected-rank T&C/MRRE
suite measured 2.91x faster warm through PyTorch CPU float64 at a 16 MiB total
budget. MLX GPU and PyTorch MPS were sensitive to the block plan and device
scheduling: their 16 MiB medians were 0.85x and 1.38x, while their 64 MiB
two-block medians were 3.98x and 3.09x. MLX CPU float64 measured 0.89x at
16 MiB. The Torch CPU samples were tightly grouped, whereas both GPU APIs had
wide run-to-run ranges on the interactive Mac. Float64 score deltas were zero;
float32 score deltas were at most `1.36e-6`. These are machine-specific
observations, so benchmark the actual device, budget, and specification rather
than treating the capability table as a speed guarantee. Cold framework time
is reported separately and must not be inferred from warm medians.

## Third-party backend entry points

ZADU 0.5.1 introduced the provisional `zadu.backends` entry-point group. An external
package registers one unique lowercase backend name and points it at a factory:

```toml
[project.entry-points."zadu.backends"]
my_backend = "my_zadu_backend:create_provider"
```

The callable receives the normalized `ExecutionConfig` and returns an exact
resource provider implementing the protocol in `zadu.backends.base`. Its
`name` must equal the entry-point name and `exact` must be `True`.

```python
def create_provider(execution):
    return MyExactProvider(
        device=execution.device,
        dtype=execution.resolved_dtype,
    )
```

An accelerator that needs planner-owned scratch memory may implement:

```python
def working_memory_bytes(self, key, n_samples, available_bytes):
    ...
```

Return a positive integer for resources handled by the provider and `None` for
resources that need no provider-specific plan. The value participates in the
preallocation guard and is passed back to `build()`/`build_batch()`.

External providers must preserve stable self exclusion and duplicate-distance
tie behavior, fall back explicitly for unsupported exact resources, keep score
results unchanged, and put execution details only in diagnostics. The entry
point API remains intentionally narrow and provisional in 0.5.x; providers should pin
the ZADU minor series they test against.
