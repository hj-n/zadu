"""Execution errors with collection context."""

from __future__ import annotations


class EmbeddingExecutionError(RuntimeError):
    """A validated embedding failed during collection execution."""

    def __init__(self, embedding_index: int) -> None:
        self.embedding_index = embedding_index
        super().__init__(
            f"measure_many failed while evaluating embeddings[{embedding_index}]"
        )
