"""visualization comparison panels and the metric-vs-acceleration plot

matplotlib imported lazily (non-interactive agg backend) so the rest of the
pipeline has no hard plotting dep only needed when rendering figures
every results claim in the readme is meant to be paired with one of these
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics as _metrics


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for visualization. Install it: pip install matplotlib"
        ) from exc


def comparison_panel(
    gt: np.ndarray,
    recons: dict[str, np.ndarray],
    out_path: str | Path,
    title: str | None = None,
    error_scale: float = 3.0,
) -> Path:
    """ground truth | each reconstruction | matched error maps one slice

    recons maps a method name (e.g. zero-filled unrolled) to its magnitude recon
    top row images bottom row |recon - gt| error maps on a shared scale each panel
    annotated with its ssim
    """
    plt = _plt()
    names = list(recons)
    ncols = 1 + len(names)
    maxval = float(gt.max())
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))

    axes[0, 0].imshow(gt, cmap="gray", vmin=0, vmax=maxval)
    axes[0, 0].set_title("ground truth")
    axes[1, 0].axis("off")

    err_vmax = maxval / error_scale
    for j, name in enumerate(names, start=1):
        recon = recons[name]
        ssim = _metrics.ssim(gt, recon, maxval=maxval)
        axes[0, j].imshow(recon, cmap="gray", vmin=0, vmax=maxval)
        axes[0, j].set_title(f"{name}\nSSIM={ssim:.3f}")
        axes[1, j].imshow(np.abs(recon - gt), cmap="magma", vmin=0, vmax=err_vmax)
        axes[1, j].set_title("error")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metric_vs_acceleration(
    results: dict[str, dict[int, dict]],
    metric: str,
    out_path: str | Path,
) -> Path:
    """line plot of one metric vs acceleration one line per method

    results[method][acceleration] = {ssim:... psnr:... nmse:...}
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, by_accel in results.items():
        accels = sorted(by_accel)
        ys = [by_accel[a][metric] for a in accels]
        ax.plot(accels, ys, marker="o", label=method)
    ax.set_xlabel("acceleration factor (R)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
