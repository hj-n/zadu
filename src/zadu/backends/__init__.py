"""Built-in exact resource providers."""

from .numpy_backend import NumpyResourceProvider


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
    from .torch_backend import TorchResourceProvider

    return TorchResourceProvider(
        device=execution.device,
        dtype=execution.resolved_dtype,
    )


__all__ = ["NumpyResourceProvider", "create_resource_provider"]
