"""Stable exact reductions over blocks of unique pairwise distances."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from zadu.engine.resources import PairStatistics, PairStrategy


class PairAccumulator:
    """Combine batch moments without retaining individual pair distances."""

    def __init__(
        self,
        *,
        needs_stress: bool = True,
        needs_scale: bool = True,
        needs_pearson: bool = True,
    ) -> None:
        self.needs_stress = needs_stress
        self.needs_scale = needs_scale
        self.needs_pearson = needs_pearson
        self.count = 0
        self.mean_orig = 0.0
        self.mean_emb = 0.0
        self.m2_orig = 0.0
        self.m2_emb = 0.0
        self.co_moment = 0.0
        self.sum_orig_squared = 0.0
        self.sum_emb_squared = 0.0
        self.sum_product = 0.0
        self.sum_squared_difference = 0.0
        self._sum_orig_correction = 0.0
        self._sum_emb_correction = 0.0
        self._sum_product_correction = 0.0
        self._sum_difference_correction = 0.0
        self.min_orig = math.inf
        self.max_orig = -math.inf
        self.min_emb = math.inf
        self.max_emb = -math.inf
        self.block_count = 0

    def update(
        self,
        orig_distances: npt.NDArray,
        emb_distances: npt.NDArray,
    ) -> None:
        orig = np.asarray(orig_distances, dtype=np.float64)
        emb = np.asarray(emb_distances, dtype=np.float64)
        if orig.shape != emb.shape:
            raise ValueError("Pair-distance blocks must have matching shapes")
        batch_count = orig.size
        if batch_count == 0:
            return
        if (
            not np.all(np.isfinite(orig))
            or not np.all(np.isfinite(emb))
            or np.any(orig < 0)
            or np.any(emb < 0)
        ):
            raise ValueError("Pair distances must be finite and non-negative")

        if self.needs_stress or self.needs_scale:
            self._add_sum("orig", float(np.vdot(orig, orig)))
        if self.needs_scale:
            self._add_sum("emb", float(np.vdot(emb, emb)))
            self._add_sum("product", float(np.vdot(orig, emb)))
        if self.needs_stress:
            difference = orig - emb
            self._add_sum("difference", float(np.vdot(difference, difference)))

        new_count = self.count + batch_count
        if self.needs_pearson:
            batch_mean_orig = float(np.mean(orig, dtype=np.float64))
            batch_mean_emb = float(np.mean(emb, dtype=np.float64))
            centered_orig = orig - batch_mean_orig
            centered_emb = emb - batch_mean_emb
            batch_m2_orig = float(np.vdot(centered_orig, centered_orig))
            batch_m2_emb = float(np.vdot(centered_emb, centered_emb))
            batch_co_moment = float(np.vdot(centered_orig, centered_emb))
            if self.count == 0:
                self.mean_orig = batch_mean_orig
                self.mean_emb = batch_mean_emb
                self.m2_orig = batch_m2_orig
                self.m2_emb = batch_m2_emb
                self.co_moment = batch_co_moment
            else:
                combine_weight = self.count * batch_count / new_count
                delta_orig = batch_mean_orig - self.mean_orig
                delta_emb = batch_mean_emb - self.mean_emb
                self.m2_orig += batch_m2_orig + delta_orig**2 * combine_weight
                self.m2_emb += batch_m2_emb + delta_emb**2 * combine_weight
                self.co_moment += (
                    batch_co_moment + delta_orig * delta_emb * combine_weight
                )
                self.mean_orig += delta_orig * batch_count / new_count
                self.mean_emb += delta_emb * batch_count / new_count

        self.count = new_count
        self.min_orig = min(self.min_orig, float(np.min(orig)))
        self.max_orig = max(self.max_orig, float(np.max(orig)))
        self.min_emb = min(self.min_emb, float(np.min(emb)))
        self.max_emb = max(self.max_emb, float(np.max(emb)))
        self.block_count += 1

    def _add_sum(self, name: str, value: float) -> None:
        total_name = {
            "orig": "sum_orig_squared",
            "emb": "sum_emb_squared",
            "product": "sum_product",
            "difference": "sum_squared_difference",
        }[name]
        correction_name = f"_sum_{name}_correction"
        total = getattr(self, total_name)
        adjusted = value - getattr(self, correction_name)
        updated = total + adjusted
        setattr(self, correction_name, (updated - total) - adjusted)
        setattr(self, total_name, updated)

    def finalize(
        self,
        *,
        strategy: PairStrategy,
        block_rows: int | None,
        chunk_pairs: int | None,
    ) -> PairStatistics:
        if self.count < 1:
            raise RuntimeError("Pair statistics require at least one distance pair")
        return PairStatistics(
            count=self.count,
            mean_orig=self.mean_orig,
            mean_emb=self.mean_emb,
            m2_orig=self.m2_orig,
            m2_emb=self.m2_emb,
            co_moment=self.co_moment,
            sum_orig_squared=self.sum_orig_squared,
            sum_emb_squared=self.sum_emb_squared,
            sum_product=self.sum_product,
            sum_squared_difference=self.sum_squared_difference,
            min_orig=self.min_orig,
            max_orig=self.max_orig,
            min_emb=self.min_emb,
            max_emb=self.max_emb,
            strategy=strategy,
            block_count=self.block_count,
            block_rows=block_rows,
            chunk_pairs=chunk_pairs,
        )


def stress_from_statistics(statistics: PairStatistics) -> float:
    if statistics.max_orig <= 0:
        raise ValueError("Stress is undefined when all pairwise distances are zero")
    return math.sqrt(statistics.sum_squared_difference / statistics.sum_orig_squared)


def scale_normalized_stress_from_statistics(statistics: PairStatistics) -> float:
    if statistics.max_orig <= 0 or statistics.max_emb <= 0:
        raise ValueError(
            "Scale-normalized stress is undefined when all pairwise distances "
            "are zero"
        )
    sum_orig_squared = statistics.sum_orig_squared
    sum_emb_squared = statistics.sum_emb_squared
    sum_product = statistics.sum_product
    alpha = sum_product / sum_emb_squared
    numerator = sum_orig_squared - 2 * alpha * sum_product + alpha**2 * sum_emb_squared
    scale = max(sum_orig_squared, abs(2 * alpha * sum_product), 1.0)
    tolerance = 64 * np.finfo(np.float64).eps * scale
    if numerator < -tolerance:
        raise RuntimeError(
            "Scale-normalized stress accumulated a negative squared residual"
        )
    return math.sqrt(max(0.0, numerator) / sum_orig_squared)


def pearson_from_statistics(statistics: PairStatistics) -> float:
    if (
        statistics.min_orig == statistics.max_orig
        or statistics.min_emb == statistics.max_emb
    ):
        raise ValueError("Pearson correlation is undefined for constant distances")
    if statistics.count < 2:
        raise ValueError("`x` and `y` must have length at least 2.")
    denominator = math.sqrt(statistics.m2_orig * statistics.m2_emb)
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for constant distances")
    return float(np.clip(statistics.co_moment / denominator, -1.0, 1.0))
