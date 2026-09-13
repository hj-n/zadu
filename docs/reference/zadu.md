# ZADU API

Construct a runner for one original dataset, then call a measurement method.
The [quickstart](../getting-started/quickstart.md) includes a complete example.

## Constructor arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `spec_list` | required | Ordered dictionaries containing `id` and optional `params` |
| `orig` | required | Finite numeric array `(n, d)`; copied into a read-only snapshot |
| `return_local` | `False` | Return separate per-point scores where supported |
| `verbose` | `False` | Print the name of each measure as it runs |
| `geodesic` | `False` | Use spherical angular distances for shared original-space resources; see [scope](../guides/evaluating-projections.md#spherical-coordinates) |
| `max_memory_bytes` | `None` | Byte-count alternative to `execution.memory_budget`; conflicting limits raise `ValueError` |
| `execution` | default `ExecutionConfig` | Backend, memory, and collection settings |

## Return values

| Call | Default result | With `return_local=True` |
| --- | --- | --- |
| `measure(emb, label=None)` | List of score dictionaries, one per specification | `(global_scores, local_scores)` |
| `measure_many(embeddings, labels=None)` | List of measurement results, one per projection | List of the tuples above |
| `iter_measure_many(embeddings, labels=None)` | Iterator of `EmbeddingResult` objects | Each object's `result` is the tuple above |

The singular call uses **`label`**; collection methods use **`labels`**.
`local_scores` contains a dictionary of arrays for supported measures and
`None` otherwise. Arrays have one entry per sample in input row order.

An `EmbeddingResult` has `index`, `result`, and `run_info` attributes.
`runner.last_run_info` is `None` before a completed call and is reset when a new
evaluation begins. For a stream it is finalized on exhaustion or explicit
closure; [stream diagnostics](../guides/many-projections.md) are aggregate-only.

## Exceptions

- Invalid shapes, values, parameters, or labels raise `ValueError` or `TypeError`.
- A required planned allocation exceeding the budget raises `MemoryError`.
- A calculation failure in a collection is wrapped in `EmbeddingExecutionError`;
  `embedding_index` identifies the input and `__cause__` preserves the error.
  Pre-execution input validation and input-iterator errors are not wrapped.

Keep rows aligned across inputs. Do not overlap calls on a single runner.

## Signatures

::: zadu.ZADU
    options:
      members:
        - measure
        - measure_many
        - iter_measure_many
      show_root_full_path: false
