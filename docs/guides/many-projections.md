# Evaluate many projections

Use one runner to compare projections of the same samples. It reuses the
original-data calculations and returns results in input order.

## Compare a list of projections

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from zadu import ExecutionConfig, ZADU

original, labels = load_iris(return_X_y=True)
pca_projection = PCA(n_components=2).fit_transform(original)
noisy_projection = pca_projection + np.random.default_rng(0).normal(
    scale=0.2, size=pca_projection.shape
)
specs = [{"id": "tnc", "params": {"k": 10}}]
runner = ZADU(specs, original, execution=ExecutionConfig(embedding_workers=2))

results = runner.measure_many([pca_projection, noisy_projection])
for name, scores in zip(["PCA", "PCA + noise"], results):
    print(name, scores[0])
```

For label-based measures, pass `labels=labels`. The same label vector applies
to every projection. With `return_local=True`, each entry in `results` is a
`(global_scores, local_scores)` tuple.

`measure_many()` loads the input iterable into a list and retains all results
and per-projection diagnostics. Use the iterator below when that storage is too
large.

## Process projections as they arrive

Continuing with the runner above:

```python
def generate_projections():
    rng = np.random.default_rng(1)
    for noise in (0.0, 0.1, 0.2):
        yield pca_projection + rng.normal(scale=noise, size=pca_projection.shape)

stream = runner.iter_measure_many(generate_projections())
try:
    for item in stream:
        print(item.index, item.result)
finally:
    stream.close()

print(runner.last_run_info["embedding_count"])
```

Each item contains its input `index`, `result`, and `run_info`. The iterator
reads only its execution window ahead. Exhausting or closing it finalizes
aggregate diagnostics and releases pending work; it retains the cache for the
last yielded projection. If you keep all yielded items yourself, their storage
still grows with the collection.

## Control concurrency

`embedding_workers=1` evaluates projections sequentially. Larger values request
concurrent CPU workers or MLX/PyTorch batching where supported. The memory plan
can reduce that width. Set seeds separately for randomized measures.

MLX and PyTorch batch the list-based interface but run the iterator sequentially.
Streams containing Procrustes or Gap Index also run sequentially because their
memory requirements depend on each projection's dimensions.

Inspect `effective_workers`, `native_batch_size`, and `worker_limit_reason` in
`last_run_info` to see the chosen strategy. ZADU preserves application BLAS and
OpenMP settings; `embedding_workers` does not limit native-library threads.
The diagnostic `native_threads_per_worker` is therefore `None`.

Do not overlap calls on the same runner, including a new call while its stream
is suspended. Use separate runners for independent evaluations.

## Handle errors

A failure while calculating a validated projection raises
`EmbeddingExecutionError`. Its `embedding_index` identifies the input and
`__cause__` contains the underlying exception. Input validation raises
`ValueError` or `TypeError` directly; exceptions from your input iterator also
propagate. These are distinct from execution failures.
