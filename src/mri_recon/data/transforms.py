"""array transforms shared by the synthetic and real data paths

pure numpy (tensor conversion happens in the dataset/collate step) easy to test
two important ideas
center crop fastmri images vary in size crop to a fixed square so batching works
and the metric focuses on anatomy not edges
instance normalization each slice z-scored by its own mean/std before the net then
un-normalized after the fastmri unet convention removes wild per-scan intensity
scaling so the net sees a consistent input distribution
"""
from __future__ import annotations

import numpy as np


def center_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """crop the last two dims of array to shape about the centre"""
    h, w = array.shape[-2:]
    th, tw = shape
    if th > h or tw > w:
        raise ValueError(f"crop {shape} larger than array {(h, w)}")
    top = (h - th) // 2
    left = (w - tw) // 2
    return array[..., top : top + th, left : left + tw]


def normalize_instance(
    array: np.ndarray, eps: float = 1e-11
) -> tuple[np.ndarray, float, float]:
    """z-score by the array own mean/std returns (normalized mean std)"""
    mean = float(array.mean())
    std = float(array.std())
    return (array - mean) / (std + eps), mean, std


def unnormalize_instance(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    """invert normalize_instance"""
    return array * std + mean


def to_magnitude(complex_image: np.ndarray) -> np.ndarray:
    """magnitude (real non-negative) image from a complex array"""
    return np.abs(complex_image).astype(np.float32)


def complex_to_chan(complex_image: np.ndarray) -> np.ndarray:
    """(..., h w) complex -> (..., 2 h w) real with [real imag] channels

    unrolled net works on complex images carried as two real channels this is the
    boundary where that representation is created
    """
    return np.stack([complex_image.real, complex_image.imag], axis=-3).astype(np.float32)


def chan_to_complex(chan_image: np.ndarray) -> np.ndarray:
    """inverse of complex_to_chan (..., 2 h w) real -> complex"""
    return (chan_image[..., 0, :, :] + 1j * chan_image[..., 1, :, :]).astype(np.complex64)
