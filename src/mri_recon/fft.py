"""centered orthonormal fourier transforms + data consistency operator

mri physics scanner samples k-space (2d fourier transform of image)
reconstruction inverts back to image space
centered orthonormal convention matches fastmri so results comparable

    fft2c(x) = fftshift(fft2(ifftshift(x)))   image -> kspace
    ifft2c(k) = fftshift(ifft2(ifftshift(k)))   kspace -> image

ifftshift/fftshift pair puts zero freq at the array centre where the low
freq mr energy lives makes a keep-central-lines mask meaningful

pure numpy so the core physics is testable without a gpu or torch
torch version in models/layers.py mirrors data_consistency_np exactly
"""
from __future__ import annotations

import numpy as np

_AXES = (-2, -1)  # last two spatial axes leading axes batch


def fft2c(image: np.ndarray) -> np.ndarray:
    """centered 2d fft image -> kspace over last two axes"""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=_AXES), axes=_AXES, norm="ortho"),
        axes=_AXES,
    )


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """centered 2d inverse fft kspace -> image over last two axes"""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=_AXES), axes=_AXES, norm="ortho"),
        axes=_AXES,
    )


def complex_abs(x: np.ndarray) -> np.ndarray:
    """magnitude from a complex array what gets displayed and scored"""
    return np.abs(x)


def data_consistency_np(
    image_pred: np.ndarray,
    measured_kspace: np.ndarray,
    mask: np.ndarray,
    lam: float | None = None,
) -> np.ndarray:
    """re-impose measured kspace on a predicted image the physics step

    wherever the scanner sampled the net cant overwrite the real value
    lam None hard dc trust the measurement exactly noiseless case
    lam finite soft dc blend k = k_pred + v*(k_meas - k_pred) v = lam/(1+lam)
    lam -> inf recovers hard dc
    """
    mask_b = np.asarray(mask).astype(bool)
    k_pred = fft2c(image_pred)
    if lam is None:  # hard dc
        k_out = np.where(mask_b, measured_kspace, k_pred)
    else:  # soft dc
        v = lam / (1.0 + lam)
        blended = k_pred + v * (measured_kspace - k_pred)
        k_out = np.where(mask_b, blended, k_pred)
    return ifft2c(k_out)
