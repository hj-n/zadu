# Evaluate projections

Create a `ZADU` runner with your measures and original data, then pass a
projection to `measure()`. Reuse the runner for projections of the same samples.
The examples below use the arrays from the [quickstart](../getting-started/quickstart.md).

## Configure a runner

```python
from zadu import ExecutionConfig, ZADU

specs = [
    {"id": "tnc", "params": {"k": 10}},
    {"id": "stress"},
]
runner = ZADU(specs, original, execution=ExecutionConfig(memory_budget="512MiB"))
scores = runner.measure(projection)
```

`original` has shape `(n, d)` and `projection` has shape `(n, p)`, with the same
samples in the same row order. The runner stores a read-only copy of `original`;
later changes to your array do not alter that runner. Create a new runner to
evaluate a different original dataset.

The [API reference](../reference/zadu.md) lists all constructor arguments and
return formats. Use `return_local=True` for [pointwise scores](local-scores.md).

## Inspect execution

`scores` contains the measure results. `last_run_info` describes the most recent
completed call:

```python
info = runner.last_run_info
print(info["backend"])
print(info["planned_peak_bytes"])
for resource in info["resources"]:
    print(resource["kind"], resource["provider"], resource["reused"])
```

Use `provider` to see where each calculation ran and `reused` to see whether a
cached result was used. `planned_peak_bytes` estimates the memory covered by
the [execution budget](execution.md); it excludes the original input copy and
is not a process memory measurement.

## Spherical coordinates

For original longitude/latitude coordinates in radians, `geodesic=True` uses
angular distances on a unit sphere when building shared original-space distance
and neighbor resources. Projection distances remain Euclidean.

```python
import numpy as np

# Columns are longitude, latitude; convert degree input to radians.
coordinates = np.deg2rad([[0, 0], [10, 0], [0, 10], [10, 10]])
map_projection = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
runner = ZADU([{"id": "tnc", "params": {"k": 1}}], coordinates, geodesic=True)
scores = runner.measure(map_projection)
```

This option does not transform the coordinates or turn coordinate-based
calculations such as Procrustes alignment, CADI angles, or Gap Index areas into
spherical versions of those measures. Restrict a spherical evaluation to
measures whose use of the shared distances matches your intended definition.
Accelerator providers use NumPy for geodesic resources.

## Direct calls

For a single calculation, you can also call a
[measure function](../reference/direct-measures.md) directly. Direct calls do
not reuse a runner's caches or apply its execution budget.
