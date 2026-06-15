"""cartesian undersampling masks the acceleration of accelerated mri

skip phase-encode lines in kspace to shorten the scan
mask keeps a block of central low-freq columns (most of the energy)
plus a sparse set of higher-freq columns to hit target acceleration r
e.g. 4x keeps ~1/4 of the columns

conventions follow fastmri.data.subsample so it matches the baselines
mask selects columns (last/width axis = phase encode) broadcast over rows
center_fractions fraction of columns always kept in the centre
accelerations target r peripheral prob set so realised r matches on average

pure numpy seedable no torch reproducible and testable
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskFunc:
    """base class pairs each acceleration with a center fraction"""

    center_fractions: list[float]
    accelerations: list[int]

    def __post_init__(self) -> None:
        if len(self.center_fractions) != len(self.accelerations):
            raise ValueError(
                "center_fractions and accelerations must have the same length: "
                f"{self.center_fractions} vs {self.accelerations}"
            )

    def _choose(self, rng: np.random.Generator) -> tuple[float, int]:
        i = rng.integers(len(self.accelerations))
        return self.center_fractions[i], self.accelerations[i]

    def __call__(self, num_cols: int, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError


class RandomMaskFunc(MaskFunc):
    """random peripheral sampling + a fully-sampled centre block"""

    def __call__(self, num_cols: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center_fraction, acceleration = self._choose(rng)

        num_low_freqs = int(round(num_cols * center_fraction))
        # peripheral sampling prob chosen so the overall mask hits r
        prob = (num_cols / acceleration - num_low_freqs) / (
            num_cols - num_low_freqs
        )
        prob = float(np.clip(prob, 0.0, 1.0))
        mask = rng.uniform(size=num_cols) < prob

        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True
        return mask.astype(np.float32)


class EquiSpacedMaskFunc(MaskFunc):
    """deterministic evenly-spaced peripheral sampling + central block

    equispaced masks are what real cartesian parallel-imaging looks like
    (grappa/sense) so more clinically faithful random mask is easier to learn from
    """

    def __call__(self, num_cols: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        center_fraction, acceleration = self._choose(rng)

        num_low_freqs = int(round(num_cols * center_fraction))
        mask = np.zeros(num_cols, dtype=bool)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # spacing for the periphery so total sampled count matches r
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * acceleration - num_cols
        )
        offset = int(rng.integers(0, max(1, int(round(adjusted_accel)))))
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        mask[np.round(accel_samples).astype(int)] = True
        return mask.astype(np.float32)


def build_mask_func(cfg: dict) -> MaskFunc:
    """construct a mask function from a config dict (see configs/*.yaml)"""
    kind = cfg.get("type", "random").lower()
    center_fractions = cfg["center_fractions"]
    accelerations = cfg["accelerations"]
    if kind == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    if kind in ("equispaced", "equi", "equal"):
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    raise ValueError(f"Unknown mask type {kind!r} (use 'random' or 'equispaced').")


def expand_mask(col_mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """broadcast a 1d column mask to a full kspace shape (..., h, w)"""
    if col_mask.shape[0] != shape[-1]:
        raise ValueError(
            f"Column mask length {col_mask.shape[0]} != k-space width {shape[-1]}."
        )
    full = np.broadcast_to(col_mask, shape)
    return np.ascontiguousarray(full)


def apply_mask(
    kspace: np.ndarray, mask_func: MaskFunc, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """undersample kspace (..., h, w) returns (masked_kspace full_mask)"""
    col_mask = mask_func(kspace.shape[-1], seed=seed)
    mask = expand_mask(col_mask, kspace.shape)
    return kspace * mask, mask


def realised_acceleration(mask: np.ndarray) -> float:
    """actual r = total columns / sampled columns (for qc logging)"""
    col_mask = np.asarray(mask)
    # collapse to a single column profile
    while col_mask.ndim > 1:
        col_mask = col_mask[0]
    sampled = np.count_nonzero(col_mask)
    return float(col_mask.size / sampled) if sampled else float("inf")
