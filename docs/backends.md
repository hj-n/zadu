# Execution backends

ZADU uses NumPy/SciPy on the CPU by default. Install an optional backend and
select it explicitly to run supported distance, neighbor, and rank calculations
through MLX or PyTorch. Metric reductions may still run on the CPU.

## Select a backend

Use `original` and `projection` from the [quickstart](getting-started/quickstart.md).
This example requires `python -m pip install "zadu[torch]"`:

```python
from zadu import ExecutionConfig, ZADU

execution = ExecutionConfig(backend="torch", device="cpu", dtype="float64")
runner = ZADU([{"id": "tnc", "params": {"k": 10}}], original, execution=execution)
scores = runner.measure(projection)
```

| Backend | Devices | Dtype | Installation |
| --- | --- | --- | --- |
| `numpy` or `auto` | `cpu` | `float64` | Base package |
| `mlx` | `cpu` | `float32`, `float64` | `zadu[mlx]` on Apple Silicon |
| `mlx` | `gpu` | `float32` | `zadu[mlx]` on Apple Silicon |
| `torch` | `cpu`, `cuda` | `float32`, `float64` | `zadu[torch]` |
| `torch` | `mps` | `float32` | `zadu[torch]` on a supported Mac |

`backend="auto"` selects NumPy, even when an accelerator is installed. MLX and
PyTorch require an explicit dtype. An explicit unavailable device raises an
error. CUDA uses the shared PyTorch implementation, but the repository's
hardware test jobs cover CPU, MLX, and MPS; no real-CUDA parity result is recorded.

## What runs on the selected device

MLX and PyTorch implement Euclidean distances, stable neighbor selection, full
rankings, and selected-rank calculations used by T&C and MRRE. Unsupported
resources, including geodesic distances and external pair ordering, use NumPy.
Selecting a GPU therefore does not move every part of every measure to it.

```python
for resource in runner.last_run_info["resources"]:
    print(resource["kind"], resource["provider"], resource["details"])
```

The top-level `backend` describes the selected provider. Each resource's
`provider` and `details` show where its work ran and any fallback.

## Precision and performance

Backends use stable index ordering for equal distances. `float32` can round
distinct distances to the same value, so it can change neighbor membership as
well as the last digits of a score. Use `float64` where available when small
distance differences matter.

The current distance kernels accumulate coordinate differences to avoid
cancellation from subtracting large squared norms. Their timings can differ
from the historical [0.5.1 performance report](performance/0.5.1-acceleration-report.md).
Use the [timing example](performance/index.md) with your measures, data, and
memory budget; record score differences when comparing backends.

MLX and PyTorch can batch compatible inputs in `measure_many()`. Their
`iter_measure_many()` path is sequential. See
[Evaluate many projections](guides/many-projections.md) for concurrency and
[Memory and execution](guides/execution.md) for budget scope.

To implement another provider, see [Backend extensions](development/backends.md).
