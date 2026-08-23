# Memory and exact execution

ZADU plans distances, neighbors, ranks, densities, and pair reductions as typed
resources. Compatible measures share resources, and the largest requested `k`
serves smaller prefixes. The planner never substitutes an approximation for an
exact published measure.

## Configure execution

```python
from zadu import ExecutionConfig, ZADU

execution = ExecutionConfig(
    backend="auto",
    device="auto",
    dtype=None,
    memory_budget="4GiB",
    embedding_workers=1,
    pair_order_strategy="auto",
    temporary_budget=None,
)

runner = ZADU(specs, original, execution=execution)
scores = runner.measure(embedding)
```

`backend="auto"` deliberately resolves to NumPy/SciPy. Optional accelerators
must be selected explicitly. Read the [backend capability table](../backends.md)
before choosing a device or dtype.

## Memory budgets

`memory_budget` accepts a positive byte count or a string such as `"512MiB"`
or `"4GiB"`. The planner uses it to select compact, streaming, or blocked exact
resources. If even one exact row or required retained resource cannot fit, ZADU
raises `MemoryError` before the oversized managed allocation.

Useful diagnostic fields include:

```python
info = runner.last_run_info
print(info["estimated_cache_bytes"])
print(info["planned_peak_bytes"])
print(info["memory_budget_bytes"])
print(info["pair_strategy"])
```

Framework allocators may retain their own pools outside the package-managed
estimate. Use process or device profilers for capacity planning.

## Exact external pair ordering

Spearman and Non-Metric Stress require a global order over all unique pairs.
If the in-memory condensed order does not fit, explicitly permit bounded
temporary storage:

```python
execution = ExecutionConfig(
    memory_budget="512MiB",
    pair_order_strategy="external",
    temporary_budget="20GiB",
    temporary_directory="/application-owned/zadu-scratch",
)
```

ZADU writes sorted runs, performs deterministic bounded-fan-in merges,
computes tie-aware ranks or stress exactly, and removes its workspace on normal
completion or failure. `pair_order_strategy="auto"` can select this route only
when `temporary_budget` was explicitly supplied; ZADU never infers permission
to use arbitrary disk.

## What the DAG shares

- T&C, class-aware T&C, and MRRE share bounded paired selected ranks.
- Stress, scale-normalized stress, and Pearson share one exact unique-pair pass.
- Spearman and Non-Metric Stress share an exact, tie-aware original pair order.
- Multiple density bandwidths share fused bounded distance blocks.
- Topographic Product requests stable neighbor prefixes and only selected
  distances rather than persistent dense matrices.

For provider-level details and measured crossover points, see
[Execution backends](../backends.md). For the internal resource contract, see
[Execution DAG](../development/execution-dag.md).
