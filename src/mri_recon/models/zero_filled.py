"""stage 1 baseline the zero-filled reconstruction

simplest possible recon take the undersampled kspace (zeros at unsampled locations)
and inverse-fft it kspace is incomplete so the result is corrupted by coherent
aliasing deliberately trivial frames the problem and sets the metric floor the unet
(stage 2) and unrolled net (stage 3) have to beat

pure numpy and parameter-free so the baseline needs no training and no gpu
"""
from __future__ import annotations

import numpy as np

from ..fft import ifft2c


def zero_filled_reconstruction(masked_kspace: np.ndarray) -> np.ndarray:
    """magnitude image from undersampled kspace the aliased baseline"""
    return np.abs(ifft2c(masked_kspace)).astype(np.float32)
