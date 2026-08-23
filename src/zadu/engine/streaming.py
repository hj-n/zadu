"""Public records yielded by bounded repeated-embedding evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """One ordered exact result from :meth:`zadu.ZADU.iter_measure_many`.

    ``result`` has the same shape and values as the corresponding item returned
    by :meth:`zadu.ZADU.measure_many`. ``run_info`` is that item's normal ZADU
    diagnostic record with ``embedding_index`` added.
    """

    index: int
    result: Any
    run_info: dict[str, Any]
