"""deterministic volume-level dataset splitting pure numpy (no torch)

dependency-free so importable and testable without the deep-learning stack
reused by the fastmri dataset factory
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def split_fastmri_files(
    directory: str | Path,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[Path]]:
    """split a directory of .h5 files into train/val/test by volume

    by whole file never by slice so slices from one scan dont leak across splits
    lets you download only knee_singlecoil_val and carve your own splits for a
    finishable subset demo
    """
    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No .h5 files in {directory}. Extract knee_singlecoil_val there "
            "(see scripts/download_fastmri.md)."
        )
    order = np.random.default_rng(seed).permutation(len(files))
    n = len(files)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    idx = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {k: [files[i] for i in sorted(v)] for k, v in idx.items()}
