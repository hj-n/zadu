# Memory and execution

Set a memory budget when distance or neighbor calculations would otherwise
use too much memory. ZADU can process these calculations in blocks and reuse
them across measures.

## Set a budget

Using the arrays from the [quickstart](../getting-started/quickstart.md):

```python
from zadu import ExecutionConfig, ZADU

specs = [{"id": "tnc", "params": {"k": 10}}, {"id": "stress"}]
runner = ZADU(specs, original, execution=ExecutionConfig(memory_budget="512MiB"))
scores = runner.measure(projection)
print(runner.last_run_info["planned_peak_bytes"])
```

`memory_budget` accepts a positive byte count or a size string such as
`"512MiB"`. If a required planned allocation cannot fit, construction or
evaluation raises `MemoryError`. Reducing the budget may reduce block size or
collection concurrency; it does not enable approximate nearest neighbors.

The budget covers planned shared resources and the working buffers accounted
for by the execution planner. It is **not a process RSS limit**. Inputs,
including the runner's copy of the original array, retained results, Python
objects, framework memory pools, and arbitrary callable allocations are outside
that budget. Gap Index estimates Qhull workspace rather than controlling its
native allocator.

Procrustes and Gap Index finalize their working-memory estimates after the
projection shape is known. Their estimates appear in
`last_run_info["metric_working_bytes"]`, keyed by specification index.

## Use temporary disk storage for pair ordering

Spearman and Non-Metric Stress order all `n * (n - 1) / 2` unique sample pairs.
When that order does not fit in memory, allow ZADU to use temporary files:

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as scratch:
    execution = ExecutionConfig(
        memory_budget="512MiB",
        pair_order_strategy="external",
        temporary_budget="2GiB",
        temporary_directory=scratch,
    )
    runner = ZADU([{"id": "srho"}], original, execution=execution)
    scores = runner.measure(projection)
```

`temporary_budget` bounds planned ZADU scratch files across concurrent
projections. `pair_order_strategy="external"` requires that budget.
With `"auto"`, disk ordering is an option only when a temporary budget is set.
The default does not use this disk-backed strategy.

Files are removed when evaluation completes or unwinds through an exception.
A forced process termination can leave files behind. This strategy trades disk
I/O for lower RAM use and runs through NumPy even with an accelerator selected.

## Read the memory diagnostics

| Field | Meaning |
| --- | --- |
| `estimated_cache_bytes` | Estimated shared-resource cache size |
| `planned_peak_bytes` | Peak planned memory for the call, including collection concurrency |
| `memory_budget_bytes` | Configured RAM budget, or `None` |
| `pair_strategy` | Selected strategy for pair-based measures, if used |
| `metric_working_bytes` | Planned scratch for metrics with a workspace declaration |

Blocking and disk ordering preserve the metric definition. Floating-point
results may differ across dtypes or reduction orders. S&C and CADI retain their
own randomized algorithms; “exact” execution does not mean those measures
enumerate every possible walk or triplet.

See [Execution backends](../backends.md) to select a device, or
[Execution DAG](../development/execution-dag.md) for resource-sharing details.
