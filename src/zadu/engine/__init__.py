"""Exact resource planning primitives used by :class:`zadu.ZADU`."""

from .config import ExecutionConfig
from .errors import EmbeddingExecutionError

__all__ = ["EmbeddingExecutionError", "ExecutionConfig"]
