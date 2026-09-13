# Backend extensions


ZADU 0.5.1 introduced the provisional `zadu.backends` entry-point group. An external
package registers one unique lowercase backend name and points it at a factory:

```toml
[project.entry-points."zadu.backends"]
my_backend = "my_zadu_backend:create_provider"
```

The callable receives the normalized `ExecutionConfig` and returns an exact
resource provider implementing the protocol in `zadu.backends.base`. Its
`name` must equal the entry-point name and `exact` must be `True`.

```python
def create_provider(execution):
    return MyExactProvider(
        device=execution.device,
        dtype=execution.resolved_dtype,
    )
```

An accelerator that needs planner-owned scratch memory may implement:

```python
def working_memory_bytes(self, key, n_samples, available_bytes):
    ...
```

Return a positive integer for resources handled by the provider and `None` for
resources that need no provider-specific plan. The value participates in the
preallocation guard and is passed back to `build()`/`build_batch()`.

External providers must preserve stable self exclusion and duplicate-distance
tie behavior, fall back explicitly for unsupported exact resources, keep score
results unchanged, and put execution details only in diagnostics. The entry
point API is provisional in 0.5.x; providers should pin
the ZADU minor series they test against.
