# Configuration and specifications

## Execution configuration

`ExecutionConfig` is immutable and validates backend, device, dtype, memory,
temporary-storage, and repeated-embedding settings when it is created.

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

`MEASURE` provides autocomplete-friendly names for every registered measure.
Each enum value is the full module ID; the scheduler also accepts the short
aliases listed in the [measure reference](../measures/index.md).

::: zadu.MEASURE
    options:
      show_root_full_path: false

## Specification helper

::: zadu.make_spec
    options:
      show_root_full_path: false

For example:

```python
from zadu import MEASURE, make_spec

specs = [
    make_spec(MEASURE.TNC, k=20),
    make_spec(MEASURE.STRESS),
]
```
