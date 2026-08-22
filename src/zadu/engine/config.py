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
    memory_budget: int | str | None = None
    embedding_workers: int = 1
    _memory_budget_bytes: int | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("backend must be a string")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if isinstance(self.embedding_workers, bool) or not isinstance(
            self.embedding_workers, Integral
        ):
            raise TypeError("embedding_workers must be an integer")
        if self.embedding_workers < 1:
            raise ValueError("embedding_workers must be at least 1")
        backend = self.backend.lower()
        device = self.device.lower()
        if backend not in {"auto", "numpy"}:
            raise ValueError("backend must be 'auto' or 'numpy' in ZADU 0.5.1")
        if device not in {"auto", "cpu"}:
            raise ValueError("device must be 'auto' or 'cpu' for the NumPy backend")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)
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
        return "numpy"

    @property
    def resolved_device(self) -> str:
        return "cpu"
