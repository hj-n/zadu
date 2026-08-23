# Execution DAG

ZADU separates a metric's published reduction from reusable exact resources.
The registry declares requirements; the planner resolves compatible work; a
provider builds resources; and metrics consume injected values through their
ordinary `measure()` functions.

```text
metric specifications
        │
        ▼
registry requirements
        │
        ▼
exact execution plan ─── memory and temporary-storage guards
        │
        ▼
resource provider ────── NumPy / MLX / PyTorch / entry point
        │
        ▼
shared typed cache
        │
        ├── metric A
        ├── metric B
        └── metric C
```

## Resource semantics

A `ResourceRequirement` identifies the resource kind, space, parameters, and
argument name consumed by a metric. Sharing occurs only when those semantics
are compatible. A maximum neighbor prefix can serve smaller `k` requests, but
a dense matrix is not introduced merely because it would be convenient.

The execution plan also records consumers and lifetimes. Resources can be
released after their final consumer, and immutable original-space resources
can be reused across projections.

## Provider boundaries

Providers implement exact construction for resources they support and declare
explicit fallback for the rest. Optional frameworks are loaded lazily. Stable
tie handling, self exclusion, formula parameters, and public scores must remain
identical up to the declared dtype tolerance.

Execution details go to `last_run_info`, not result dictionaries. This keeps
scientific outputs stable when the same metric is evaluated through a different
exact plan.

## Metric contract

The repository-wide contract verifies that every registered measure:

- appears in the enum and package exports;
- accepts its documented public arguments;
- returns finite Python scalar scores;
- agrees between direct and scheduled calls; and
- shares declared DAG resources across duplicate or mixed specifications.

See [Adding a measure](adding-a-measure.md) for the integration workflow and
[Execution backends](../backends.md) for the current provider matrix.
