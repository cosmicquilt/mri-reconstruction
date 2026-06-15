"""reconstruction quality metrics nmse psnr ssim

fastmri-standard metrics computed on magnitude images vs fully-sampled gt
ssim is the headline (tracks perceived structural fidelity) psnr nmse round it out

ssim uses scikit-image when installed (canonical fastmri-leaderboard impl)
else a self-contained numpy version uniform 7x7 window same sample-variance
correction so the whole module (and smoke test) runs on numpy alone
"""
from __future__ import annotations

import numpy as np

try:  # canonical impl when available
    from skimage.metrics import structural_similarity as _sk_ssim

    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover exercised only when skimage absent
    _HAVE_SKIMAGE = False


def _as_stack(x: np.ndarray) -> np.ndarray:
    """promote a single 2d image to a length-1 stack (n h w)"""
    x = np.asarray(x, dtype=np.float64)
    return x[None] if x.ndim == 2 else x


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """normalized mse ||gt - pred||^2 / ||gt||^2 lower is better"""
    gt, pred = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
    denom = np.linalg.norm(gt) ** 2
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(gt - pred) ** 2 / denom)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: float | None = None) -> float:
    """peak signal-to-noise ratio in db higher is better"""
    gt, pred = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
    maxval = float(gt.max()) if maxval is None else float(maxval)
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(maxval) - 10 * np.log10(mse))


def _box_filter_1d(a: np.ndarray, size: int, axis: int) -> np.ndarray:
    """uniform box filter of width size along axis with reflect edges

    integral image (prefix sums) so exact and o(n)
    """
    r = size // 2
    a = np.swapaxes(a, axis, -1)
    pad = [(0, 0)] * a.ndim
    pad[-1] = (r, r)
    ap = np.pad(a, pad, mode="reflect")
    csum = np.cumsum(ap, axis=-1)
    csum = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
    out = (csum[..., size:] - csum[..., :-size]) / size
    return np.swapaxes(out, axis, -1)


def _uniform_filter(img: np.ndarray, size: int) -> np.ndarray:
    return _box_filter_1d(_box_filter_1d(img, size, 0), size, 1)


def _ssim_2d_numpy(gt: np.ndarray, pred: np.ndarray, maxval: float, win: int = 7) -> float:
    """numpy ssim matching scikit-image uniform-window default closely"""
    C1 = (0.01 * maxval) ** 2
    C2 = (0.03 * maxval) ** 2

    ux = _uniform_filter(gt, win)
    uy = _uniform_filter(pred, win)
    uxx = _uniform_filter(gt * gt, win)
    uyy = _uniform_filter(pred * pred, win)
    uxy = _uniform_filter(gt * pred, win)

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    cov_norm = (win * win) / (win * win - 1.0)  # unbiased sample variance as skimage
    vx, vy, vxy = vx * cov_norm, vy * cov_norm, vxy * cov_norm

    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    ssim_map = num / den

    pad = win // 2  # drop the border (skimage valid region)
    return float(ssim_map[pad:-pad, pad:-pad].mean())


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: float | None = None) -> float:
    """mean structural similarity over a (stack of) magnitude image(s)"""
    gt, pred = _as_stack(gt), _as_stack(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    maxval = float(gt.max()) if maxval is None else float(maxval)
    vals = []
    for g, p in zip(gt, pred):
        if _HAVE_SKIMAGE:
            vals.append(_sk_ssim(g, p, data_range=maxval))
        else:
            vals.append(_ssim_2d_numpy(g, p, maxval))
    return float(np.mean(vals))


def all_metrics(gt: np.ndarray, pred: np.ndarray, maxval: float | None = None) -> dict:
    """convenience compute the standard three at once"""
    maxval = float(np.asarray(gt).max()) if maxval is None else float(maxval)
    return {
        "ssim": ssim(gt, pred, maxval=maxval),
        "psnr": psnr(gt, pred, maxval=maxval),
        "nmse": nmse(gt, pred),
    }
