# Evaluate embeddings

The `ZADU` class is the recommended interface. It validates specifications,
plans shared exact work, reuses original-space resources, and returns results
in specification order.

## Build a runner

```python
from zadu import ExecutionConfig, ZADU

specs = [
    {"id": "tnc", "params": {"k": 20}},
    {"id": "snc", "params": {"k": 30, "random_state": 0}},
]

runner = ZADU(
    specs,
    original,
    return_local=False,
    execution=ExecutionConfig(memory_budget="4GiB"),
)
scores = runner.measure(embedding)
```

The constructor accepts:

| Argument | Meaning |
| --- | --- |
| `spec_list` | Ordered measure specifications |
| `orig` | Original high-dimensional samples |
| `return_local` | Return pointwise values where a measure supports them |
| `verbose` | Retained compatibility flag |
| `geodesic` | Treat original two-column coordinates as longitude/latitude in radians |
| `max_memory_bytes` | Legacy byte-count memory limit |
| `execution` | Preferred `ExecutionConfig` interface |

Do not set both `max_memory_bytes` and `execution.memory_budget` to conflicting
values.

## Read results and diagnostics separately

`measure()` returns only scientific results. Execution metadata lives in
`last_run_info`:

```python
scores = runner.measure(embedding)

print(scores)
print(runner.last_run_info["backend"])
print(runner.last_run_info["planned_peak_bytes"])
print(runner.last_run_info["resources"])
```

The diagnostic record includes the selected provider, resource fallbacks,
memory estimates, build and metric timings, resource consumers, dtype, and
reuse. Do not mix these fields into metric-score output or serialize them as if
they were scientific results.

## Use geodesic original coordinates

For spherical positions, pass longitude and latitude in radians as the first
two original columns:

```python
runner = ZADU(specs, spherical_coordinates, geodesic=True)
scores = runner.measure(embedding)
```

Geodesic distance applies only to the registered original space. Embedded
coordinates remain Euclidean. Unsupported accelerator resources fall back to
the exact NumPy path and report the reason in diagnostics.

## Direct measure calls

Standalone measure functions remain useful for one-off or research workflows,
but they do not share resources across metrics. See
[Direct measure functions](../reference/direct-measures.md).
