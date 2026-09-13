# Performance

Runtime depends on the measures, sample count, dimensions, neighborhood size,
memory budget, and device. Measure both the first evaluation and subsequent
evaluations if you plan to reuse a runner.

## Measure your workload

Using the arrays from the [quickstart](../getting-started/quickstart.md):

```python
from time import perf_counter
from statistics import median
from zadu import ZADU

specs = [{"id": "tnc", "params": {"k": 10}}, {"id": "stress"}]
start = perf_counter()
runner = ZADU(specs, original)
runner.measure(projection)
first = perf_counter() - start

times = []
for _ in range(5):
    start = perf_counter()
    runner.measure(projection)
    times.append(perf_counter() - start)
print({"construction_and_first_seconds": first, "median_reused_seconds": median(times)})
```

This measures one process. Framework imports and initialization may already
have happened if you ran other code first. Use a fresh process when measuring
startup or process peak RSS. Compare score differences alongside timings when
changing dtype or backend.

## Published measurements

The [0.5.1 report](0.5.1-acceleration-report.md) records measurements from
2026-08-23 at revision `94e71a9`, compared with v0.5.0 and v0.1.1 under one
dependency environment. These are historical results, not measurements of the
current checkout. Later distance kernels and memory plans can change runtime.

The repository also contains [selected-rank measurements](https://github.com/hj-n/zadu/blob/master/benchmarks/results/post-0.5.1/native-selected-ranks-m4.json)
from revision `34730bb4` on an Apple M4. The file records its environment,
budget, score deltas, and timing ranges. See the
[benchmark instructions](https://github.com/hj-n/zadu/blob/master/benchmarks/README.md)
for reproduction commands.
