# Performance

ZADU has no honest library-wide speedup multiplier. Runtime and peak memory
depend on the selected metric resources, sample count, dimensions, neighborhood
size, memory budget, dtype, backend, device, and whether the provider is cold
or warm.

The [0.5.1 acceleration report](0.5.1-acceleration-report.md) compares:

- the current exact default path;
- the pre-acceleration v0.5.0 release;
- the 2023 v0.1.1 implementation; and
- explicit MLX and PyTorch paths on the maintained Apple M4 machine.

It includes absolute timings, memory observations, isolated kernels,
representative mixed workloads, raw-result locations, limitations, and
reproduction commands. Treat those measurements as workload-specific evidence,
not a promise for different hardware.

For current backend capabilities and crossover observations, see
[Execution backends](../backends.md).
