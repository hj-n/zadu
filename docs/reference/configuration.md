# Configuration and specifications

## Execution configuration

`ExecutionConfig` is immutable. It validates settings at construction; a backend
checks device availability when the runner creates its provider.

| Setting | Default | Meaning |
| --- | --- | --- |
| `backend` | `"auto"` | `auto`/`numpy`, `mlx`, `torch`, or an installed backend name |
| `device` | `"auto"` | Device within that backend; select one explicitly for reproducibility |
| `dtype` | `None` | NumPy uses float64; MLX/PyTorch require explicit `"float32"` or `"float64"` |
| `memory_budget` | `None` | Positive bytes or size string; limits planned resources and working buffers |
| `embedding_workers` | `1` | Requested projection concurrency; can be reduced by the plan |
| `pair_order_strategy` | `"auto"` | `auto`, `memory`, or `external` |
| `temporary_directory` | `None` | Parent directory for external ordering files; defaults to the system temporary directory |
| `temporary_budget` | `None` | Positive bytes or size string; required for `external` ordering |

See [backends](../backends.md) for valid device/dtype combinations and
[memory and execution](../guides/execution.md) for what the budgets cover.

::: zadu.ExecutionConfig
    options:
      members:
        - memory_budget_bytes
        - temporary_budget_bytes
        - resolved_backend
        - resolved_device
        - resolved_dtype
      show_root_full_path: false

## Measure identifiers

Use `MEASURE` for named identifiers, or pass a short alias from the
[measure reference](../measures/index.md). Enum values are full module IDs.

::: zadu.MEASURE
    options:
      show_root_full_path: false

## Specification helper

```python
from zadu import MEASURE, make_spec

specs = [make_spec(MEASURE.TNC, k=10), make_spec(MEASURE.STRESS)]
```

::: zadu.make_spec
    options:
      show_root_full_path: false
