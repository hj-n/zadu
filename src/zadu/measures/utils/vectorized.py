"""Small memory-bounded primitives shared by exact NumPy metric kernels."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

DEFAULT_MAX_TEMP_BYTES = 64 * 1024**2
_BROADCAST_MEMBERSHIP_MAX_COMPARISONS_PER_ROW = 4096


def iter_row_blocks(
    n_rows: int,
    bytes_per_row: int,
    *,
    max_block_bytes: int = DEFAULT_MAX_TEMP_BYTES,
) -> Iterator[slice]:
    """Yield row slices whose estimated temporary storage fits the budget."""

    if n_rows < 0:
        raise ValueError("n_rows must be zero or greater")
    if bytes_per_row < 1:
        raise ValueError("bytes_per_row must be greater than zero")
    if max_block_bytes < 1:
        raise ValueError("max_block_bytes must be greater than zero")

    rows_per_block = max(1, max_block_bytes // bytes_per_row)
    for start in range(0, n_rows, rows_per_block):
        yield slice(start, min(start + rows_per_block, n_rows))


def _validate_row_pair(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("row-wise inputs must both be 2D arrays")
    if left_array.shape[0] != right_array.shape[0]:
        raise ValueError("row-wise inputs must have the same number of rows")
    return left_array, right_array


def rowwise_membership(
    candidates: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    max_block_bytes: int = DEFAULT_MAX_TEMP_BYTES,
) -> npt.NDArray[np.bool_]:
    """Test whether each candidate occurs in the corresponding reference row."""

    candidate_array, reference_array = _validate_row_pair(candidates, reference)
    result = np.empty(candidate_array.shape, dtype=bool)
    if result.size == 0:
        return result
    if reference_array.shape[1] == 0:
        result.fill(False)
        return result

    comparison_bytes_per_row = max(
        1, candidate_array.shape[1] * reference_array.shape[1]
    )
    can_encode = (
        np.issubdtype(candidate_array.dtype, np.integer)
        and np.issubdtype(reference_array.dtype, np.integer)
        and np.min(candidate_array) >= 0
        and np.min(reference_array) >= 0
    )
    if (
        can_encode
        and comparison_bytes_per_row > _BROADCAST_MEMBERSHIP_MAX_COMPARISONS_PER_ROW
    ):
        return _rowwise_membership_sorted(
            candidate_array,
            reference_array,
            max_block_bytes=max_block_bytes,
        )

    return _rowwise_membership_broadcast(
        candidate_array,
        reference_array,
        max_block_bytes=max_block_bytes,
    )


def _rowwise_membership_broadcast(
    candidates: npt.NDArray,
    reference: npt.NDArray,
    *,
    max_block_bytes: int,
) -> npt.NDArray[np.bool_]:
    """Compare rows directly in chunks."""

    result = np.empty(candidates.shape, dtype=bool)
    comparison_bytes_per_row = max(1, candidates.shape[1] * reference.shape[1])
    for block in iter_row_blocks(
        candidates.shape[0],
        comparison_bytes_per_row,
        max_block_bytes=max_block_bytes,
    ):
        result[block] = np.any(
            candidates[block, :, None] == reference[block, None, :],
            axis=2,
        )
    return result


def _rowwise_membership_sorted(
    candidates: npt.NDArray,
    reference: npt.NDArray,
    *,
    max_block_bytes: int,
) -> npt.NDArray[np.bool_]:
    """Use sorted row encodings when a broadcast would require quadratic work."""

    result = np.empty(candidates.shape, dtype=bool)
    candidate_width = candidates.shape[1]
    reference_width = reference.shape[1]
    bytes_per_row = np.dtype(np.int64).itemsize * (
        3 * candidate_width + 2 * reference_width
    )
    for block in iter_row_blocks(
        candidates.shape[0],
        max(1, bytes_per_row),
        max_block_bytes=max_block_bytes,
    ):
        candidate_block = candidates[block]
        reference_block = reference[block]
        block_rows = candidate_block.shape[0]
        max_index = max(int(np.max(candidate_block)), int(np.max(reference_block)))
        stride = max_index + 1
        if stride > np.iinfo(np.int64).max // block_rows:
            result[block] = _rowwise_membership_broadcast(
                candidate_block,
                reference_block,
                max_block_bytes=max_block_bytes,
            )
            continue

        offsets = np.arange(block_rows, dtype=np.int64)[:, None] * stride
        sorted_reference = reference_block.astype(np.int64, copy=True)
        sorted_reference.sort(axis=1)
        sorted_reference += offsets
        encoded_candidates = candidate_block.astype(np.int64) + offsets

        flat_reference = sorted_reference.reshape(-1)
        flat_candidates = encoded_candidates.reshape(-1)
        positions = np.searchsorted(flat_reference, flat_candidates)
        membership = np.zeros(flat_candidates.shape, dtype=bool)
        valid = positions < flat_reference.size
        membership[valid] = flat_reference[positions[valid]] == flat_candidates[valid]
        result[block] = membership.reshape(candidate_block.shape)
    return result


def rowwise_intersection_count(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    max_block_bytes: int = DEFAULT_MAX_TEMP_BYTES,
) -> npt.NDArray[np.intp]:
    """Count members shared by each pair of rows."""

    membership = rowwise_membership(left, right, max_block_bytes=max_block_bytes)
    return np.sum(membership, axis=1, dtype=np.intp)


def gather_ranks(
    ranking: npt.ArrayLike,
    indices: npt.ArrayLike,
) -> npt.NDArray:
    """Gather full-ranking values at row-aligned neighbor indices."""

    ranking_array, index_array = _validate_row_pair(ranking, indices)
    valid_flat_indices = np.issubdtype(index_array.dtype, np.integer) and (
        index_array.size == 0
        or (np.min(index_array) >= 0 and np.max(index_array) < ranking_array.shape[1])
    )
    if ranking_array.flags.c_contiguous and valid_flat_indices:
        row_offsets = (
            np.arange(ranking_array.shape[0], dtype=np.intp)[:, None]
            * ranking_array.shape[1]
        )
        return ranking_array.reshape(-1)[row_offsets + index_array]
    return np.take_along_axis(ranking_array, index_array, axis=1)
