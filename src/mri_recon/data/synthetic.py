"""synthetic phantom data so the pipeline runs end-to-end on day zero

real target is the fastmri single-coil knee subset (see readme to get it) but that
needs a signed agreement and a download to always have a working version this makes
random ellipse phantoms (shepp-logan-ish) and their fully-sampled kspace with the
same array shapes/dtypes the real loader produces

so the whole pipeline (masking recon qc metrics viz) is exercised and tested before
any clinical data is downloaded synthetic results validate plumbing not science
"""
from __future__ import annotations

import numpy as np

from ..fft import fft2c


def _draw_ellipse(image: np.ndarray, rng: np.random.Generator) -> None:
    """add one random smoothly-varying ellipse to image in place"""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = rng.uniform(0.25, 0.75) * h, rng.uniform(0.25, 0.75) * w
    ry, rx = rng.uniform(0.08, 0.3) * h, rng.uniform(0.08, 0.3) * w
    theta = rng.uniform(0, np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - cx) * ct + (yy - cy) * st
    yr = -(xx - cx) * st + (yy - cy) * ct
    inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
    intensity = rng.uniform(0.2, 1.0)
    image[inside] += intensity


def make_phantom(
    height: int = 320,
    width: int = 320,
    n_ellipses: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    """return a real non-negative magnitude phantom image in [0, 1]"""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float64)
    for _ in range(n_ellipses):
        _draw_ellipse(image, rng)
    image = np.clip(image, 0, None)
    peak = image.max()
    if peak > 0:
        image /= peak
    return image.astype(np.float32)


def make_sample(
    height: int = 320,
    width: int = 320,
    seed: int | None = None,
) -> dict:
    """one synthetic example shaped like a fastmri single-coil slice

    returns a dict
      image  (h w) float32 magnitude ground truth
      kspace (h w) complex64 fully-sampled kspace
    matching the keys FastMRISliceDataset yields so downstream code is identical
    """
    image = make_phantom(height, width, seed=seed)
    kspace = fft2c(image.astype(np.complex64))
    return {"image": image, "kspace": kspace.astype(np.complex64)}


def make_dataset(
    n: int = 16,
    height: int = 256,
    width: int = 256,
    seed: int = 0,
) -> list[dict]:
    """a small in-memory synthetic dataset for smoke tests and demos"""
    return [make_sample(height, width, seed=seed + i) for i in range(n)]
