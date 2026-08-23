# Evaluate many embeddings

Use one `ZADU` instance when comparing multiple embeddings of the same original
data. Immutable original-space resources are constructed once and reused.

## Materialized collections

```python
from zadu import ExecutionConfig, ZADU

runner = ZADU(
    specs,
    original,
    execution=ExecutionConfig(
        embedding_workers=2,
        memory_budget="4GiB",
    ),
)

results = runner.measure_many(
    [pca_embedding, tsne_embedding, umap_embedding],
    labels=labels,
)
```

`measure_many()` preserves input order and returns one ordinary `measure()`
result per embedding. `labels` is one optional vector shared by the collection.

`embedding_workers=1` is the deterministic default. Larger values opt into
bounded threads on thread-safe CPU providers or native tensor batching on
supported MLX and PyTorch workloads. The memory plan may reduce the effective
width or select sequential execution.

Inspect `runner.last_run_info` for:

- requested and effective workers;
- why a requested strategy was limited;
- original-resource reuse;
- aggregate and per-embedding timings;
- provider-native batch width; and
- the planned collection peak.

If one input fails, ZADU raises `EmbeddingExecutionError` with its input index.
A runner is mutable and should not be called concurrently from multiple user
threads.

## Bounded streams

For generated or very long sequences, avoid retaining every input, result, and
diagnostic record:

```python
stream = runner.iter_measure_many(generate_embeddings())
try:
    for item in stream:
        print(item.index, item.result, item.run_info)
finally:
    stream.close()
```

The iterator is lazy, yields in input order, and keeps at most the planned
in-flight window. Exhaustion or explicit closure finalizes a bounded aggregate
in `last_run_info`. Each `EmbeddingResult` carries the detailed diagnostics for
its own embedding.

MLX and PyTorch currently use their native repeated-embedding tensor batching
only for the materialized `measure_many()` interface. The streaming interface
remains ordered and bounded but executes those providers sequentially.
