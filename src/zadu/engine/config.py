"""Execution configuration for the exact ZADU engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Integral

_MEMORY_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>B|KIB|MIB|GIB|KB|MB|GB)?\s*$",
    re.IGNORECASE,
)
_MEMORY_MULTIPLIERS = {
    None: 1,
    "B": 1,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
}


def parse_memory_budget(value: int | str | None) -> int | None:
    """Return a positive memory budget in bytes."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("memory_budget must be an integer byte count or size string")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str):
        match = _MEMORY_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                "memory_budget must be a byte count or a size such as '4GiB'"
            )
        number = float(match.group("value"))
        unit = match.group("unit")
        multiplier = _MEMORY_MULTIPLIERS[unit.upper() if unit is not None else None]
        result = int(number * multiplier)
    else:
        raise TypeError("memory_budget must be an integer byte count or size string")
    if result < 1:
        raise ValueError("memory_budget must be greater than zero")
    return result


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Select exact execution capabilities available in ZADU 0.5.1."""

    backend: str = "auto"
    device: str = "auto"
    dtype: str | None = None
    memory_budget: int | str | None = None
    embedding_workers: int = 1
    _memory_budget_bytes: int | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("backend must be a string")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if self.dtype is not None and not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string or None")
        if isinstance(self.embedding_workers, bool) or not isinstance(
            self.embedding_workers, Integral
        ):
            raise TypeError("embedding_workers must be an integer")
        if self.embedding_workers < 1:
            raise ValueError("embedding_workers must be at least 1")
        backend = self.backend.lower()
        device = self.device.lower()
        dtype = self.dtype.lower() if self.dtype is not None else None
        if backend not in {"auto", "numpy", "mlx", "torch"}:
            raise ValueError("backend must be 'auto', 'numpy', 'mlx', or 'torch'")
        if backend in {"auto", "numpy"}:
            if device not in {"auto", "cpu"}:
                raise ValueError("device must be 'auto' or 'cpu' for the NumPy backend")
            if dtype not in {None, "float64"}:
                raise ValueError(
                    "dtype must be None or 'float64' for the NumPy backend"
                )
        elif backend == "mlx":
            if device not in {"auto", "cpu", "gpu"}:
                raise ValueError(
                    "device must be 'auto', 'cpu', or 'gpu' for the MLX backend"
                )
            if dtype not in {"float32", "float64"}:
                raise ValueError(
                    "The MLX backend requires an explicit 'float32' or "
                    "'float64' dtype"
                )
            if device == "gpu" and dtype != "float32":
                raise ValueError("The MLX GPU requires dtype='float32'")
        else:
            if device not in {"auto", "cpu", "mps", "cuda"}:
                raise ValueError(
                    "device must be 'auto', 'cpu', 'mps', or 'cuda' for the "
                    "PyTorch backend"
                )
            if dtype not in {"float32", "float64"}:
                raise ValueError(
                    "The PyTorch backend requires an explicit 'float32' or "
                    "'float64' dtype"
                )
            if device == "mps" and dtype != "float32":
                raise ValueError("PyTorch MPS requires dtype='float32'")
            if self.embedding_workers != 1:
                raise ValueError(
                    "The PyTorch preview requires embedding_workers=1; "
                    "native repeated-embedding batching is planned for PR 7-C"
                )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "embedding_workers", int(self.embedding_workers))
        object.__setattr__(
            self, "_memory_budget_bytes", parse_memory_budget(self.memory_budget)
        )

    @property
    def memory_budget_bytes(self) -> int | None:
        """Normalized memory budget in bytes."""

        return self._memory_budget_bytes

    @property
    def resolved_backend(self) -> str:
        return self.backend if self.backend in {"mlx", "torch"} else "numpy"

    @property
    def resolved_device(self) -> str:
        if self.resolved_backend == "numpy":
            return "cpu"
        if (
            self.resolved_backend == "mlx"
            and self.device == "auto"
            and self.dtype == "float64"
        ):
            return "cpu"
        return self.device

    @property
    def resolved_dtype(self) -> str:
        return self.dtype or "float64"
