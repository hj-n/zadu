# Exact execution backends

ZADU keeps NumPy/SciPy/FAISS as the dependency-free execution baseline and
loads accelerator frameworks only after an explicit backend request. Every
backend in 0.5.1 evaluates the full exact resource contract: `float32` changes
rounding, not the number of pairs, neighbors, or iterations evaluated.

## Capability table

| Capability | NumPy CPU | MLX CPU | MLX GPU | Torch CPU | Torch MPS | Torch CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| Base install | Yes | Extra | Extra; Apple Silicon | Extra | Extra; Apple | Extra |
| Dtype | float64 | float32/64 | float32 | float32/64 | float32 | float32/64 |
| Dense/condensed Euclidean | Native | Native | Native | Native | Native | Common code |
| Stable full/inverse ranking | Native | Native | Native | Native | Native | Common code |
| Paired selected ranks | Native | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback | NumPy fallback |
| Ordinary exact kNN | FAISS | Stable full-sort prefix | Stable full-sort prefix | Stable full-sort prefix | Stable full-sort prefix | Common code |
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
has explicit availability checks, but 0.5.1 was not benchmarked or parity-tested
on real CUDA hardware. It is supported as a preview, not performance-validated.

`backend="auto"` deliberately selects NumPy/FAISS. FAISS is still the strongest
standalone small-k path on the maintained Apple machine, accelerator startup can
dominate small jobs, and float64 is the compatibility baseline. Choose MLX or
PyTorch explicitly after benchmarking the actual specification and data size.

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
    ),
)
scores = runner.measure_many(embeddings)
print(runner.last_run_info)
```

Diagnostics distinguish requested and actual providers, per-resource fallback
reasons, input/output transfers, first execution, warm execution, block sizes,
planned package-managed peak memory, and provider-native batch width. Device
framework allocators can retain pools outside that estimate; use isolated peak
RSS or device-profiler measurements when capacity planning.

Registered T&C, class-aware T&C, and MRRE specifications request the paired
selected-rank resource. The NumPy provider builds it in exact bounded blocks and
retains `O(nk)` indices, cross-space ranks, and membership masks. MLX and
PyTorch still expose their explicit full-ranking capability, but selected-rank
execution falls back to NumPy until a native paired implementation is available;
the resource record makes that boundary visible.

Cold time includes first framework/device initialization. Warm time measures a
reused provider after that initialization. Neither should be substituted for
the other: on the release machine, small MLX jobs were much faster warm but
slower cold, while the crossover appeared by the 2,000-sample representative
suite.

## Third-party backend entry points

0.5.1 exposes the provisional `zadu.backends` entry-point group. An external
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
point API is intentionally narrow and provisional in 0.5.1; providers should pin
the ZADU minor series they test against.
