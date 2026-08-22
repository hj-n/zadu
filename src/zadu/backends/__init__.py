"""Built-in and entry-point exact resource providers."""

from importlib.metadata import entry_points

from .numpy_backend import NumpyResourceProvider

_BUILTIN_BACKENDS = {"numpy", "mlx", "torch"}


def create_resource_provider(execution):
    """Create the selected provider without importing unused optional backends."""

    if execution.resolved_backend == "numpy":
        return NumpyResourceProvider()
    if execution.resolved_backend == "mlx":
        from .mlx_backend import MlxResourceProvider

        return MlxResourceProvider(
            device=execution.device,
            dtype=execution.resolved_dtype,
        )
    if execution.resolved_backend == "torch":
        from .torch_backend import TorchResourceProvider

        return TorchResourceProvider(
            device=execution.device,
            dtype=execution.resolved_dtype,
        )
    return _create_entry_point_provider(execution)


def numpy_backend_entrypoint(execution):
    """Entry-point adapter for the built-in NumPy provider."""

    del execution
    return NumpyResourceProvider()


def mlx_backend_entrypoint(execution):
    """Entry-point adapter that preserves lazy MLX imports."""

    from .mlx_backend import MlxResourceProvider

    return MlxResourceProvider(
        device=execution.device,
        dtype=execution.resolved_dtype,
    )


def torch_backend_entrypoint(execution):
    """Entry-point adapter that preserves lazy PyTorch imports."""

    from .torch_backend import TorchResourceProvider

    return TorchResourceProvider(
        device=execution.device,
        dtype=execution.resolved_dtype,
    )


def available_resource_backends() -> tuple[str, ...]:
    """Return built-in and installed external backend entry-point names."""

    names = _BUILTIN_BACKENDS | {entry.name for entry in _backend_entry_points()}
    return tuple(sorted(names))


def _backend_entry_points():
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group="zadu.backends"))
    return tuple(discovered.get("zadu.backends", ()))  # pragma: no cover - py310 API


def _create_entry_point_provider(execution):
    matches = [
        entry
        for entry in _backend_entry_points()
        if entry.name == execution.resolved_backend
    ]
    if not matches:
        available = ", ".join(available_resource_backends())
        raise ValueError(
            f"No installed ZADU backend entry point named "
            f"'{execution.resolved_backend}'. Available backends: {available}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple 'zadu.backends' entry points are named "
            f"'{execution.resolved_backend}'"
        )
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError("A ZADU backend entry point must load a callable factory")
    provider = factory(execution)
    required = (
        "name",
        "device",
        "dtype",
        "exact",
        "fork",
        "invalidate",
        "can_batch",
        "build_batch",
        "build",
        "build_pair_statistics",
        "build_ordered_pair_statistics",
        "build_topographic_product_statistics",
        "build_rank_comparisons",
        "build_neighbor_statistics",
    )
    missing = [name for name in required if not hasattr(provider, name)]
    if missing:
        raise TypeError(
            "Backend provider is missing required attributes: " + ", ".join(missing)
        )
    if provider.name != execution.resolved_backend:
        raise ValueError(
            "Backend provider name must match its requested entry-point name "
            f"('{provider.name}' != '{execution.resolved_backend}')"
        )
    if provider.exact is not True:
        raise ValueError("ZADU 0.5.1 accepts only exact resource providers")
    return provider


__all__ = [
    "NumpyResourceProvider",
    "available_resource_backends",
    "create_resource_provider",
    "mlx_backend_entrypoint",
    "numpy_backend_entrypoint",
    "torch_backend_entrypoint",
]
