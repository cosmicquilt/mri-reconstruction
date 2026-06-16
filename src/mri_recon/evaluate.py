"""evaluation run a method over a val set report ssim/psnr/nmse

reconstruct_batch is the single source of truth for given a method and a batch
produce a magnitude reconstruction both train and eval call it so the forward
pass cant silently diverge
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from . import metrics as _metrics
from . import qc as _qc
from .models.layers import complex_abs
from .utils import center_crop_tensor

logger = logging.getLogger("mri_recon.evaluate")

_ZF = ("zero_filled", "zerofilled", "zf")
_UNROLLED = ("unrolled", "dccnn", "varnet")


def reconstruct_batch(kind: str, model, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """return (pred_magnitude target) both (b 1 h w) for one batch"""
    kind = kind.lower()
    target = batch["target"].to(device)
    crop = target.shape[-2:]

    if kind in _ZF:
        pred = center_crop_tensor(batch["zf_image"].to(device), crop)
    elif kind == "unet":
        zf = batch["zf_image"].to(device)
        b = zf.shape[0]
        mean = zf.reshape(b, -1).mean(1).view(b, 1, 1, 1)
        std = zf.reshape(b, -1).std(1).view(b, 1, 1, 1) + 1e-11
        out = model((zf - mean) / std) * std + mean  # instance norm round-trip
        pred = center_crop_tensor(out, crop)
    elif kind in _UNROLLED:
        masked_kspace = batch["masked_kspace"].to(device)
        mask = batch["mask"].to(device)
        out = model(masked_kspace, mask)  # (b 2 h w)
        pred = center_crop_tensor(complex_abs(out), crop)  # (b 1 h w)
    else:
        raise ValueError(f"Unknown method {kind!r}.")
    return pred, target


@torch.no_grad()
def evaluate_loader(
    kind: str,
    model,
    loader,
    device,
    report: _qc.QCReport | None = None,
) -> dict:
    """mean ssim/psnr/nmse over a loader with per-sample qc range checks"""
    if model is not None:
        model.eval()
    report = report or _qc.QCReport()
    acc = {"ssim": [], "psnr": [], "nmse": []}

    for batch in loader:
        pred, target = reconstruct_batch(kind, model, batch, device)
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        for p, t in zip(pred_np, target_np):
            p2, t2 = p[0], t[0]  # drop channel dim
            report.checked += 1
            if not report.record(_qc.check_finite(p2, "recon")):
                continue
            maxval = float(t2.max())
            m = _metrics.all_metrics(t2, p2, maxval=maxval)
            _qc.qc_metrics(m, report)
            for key in acc:
                acc[key].append(m[key])

    return {key: float(np.mean(vals)) if vals else float("nan") for key, vals in acc.items()}


def load_model(kind: str, model_cfg: dict, checkpoint: str | None, device):
    """build a model and optionally load weights returns none for zero-filled"""
    if kind.lower() in _ZF:
        return None
    from .models import build_model

    model = build_model(kind, **model_cfg.get(kind, {})).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        logger.info("loaded checkpoint %s", checkpoint)
    return model


def run_evaluation(cfg: dict, device) -> dict:
    """evaluate every (method acceleration) in the config return a results tree

    results[method][acceleration] = {ssim psnr nmse}
    """
    from torch.utils.data import DataLoader

    from .data.fastmri_dataset import build_dataset, collate_samples
    from .masking import build_mask_func

    methods = cfg["eval"]["methods"]
    accelerations = cfg["eval"]["accelerations"]
    center_fractions = cfg["eval"].get("center_fractions", [0.08] * len(accelerations))
    checkpoints = cfg["eval"].get("checkpoints", {})
    results: dict[str, dict[int, dict]] = {}

    for method in methods:
        results[method] = {}
        model = load_model(method, cfg.get("model", {}), checkpoints.get(method), device)
        for accel, cf in zip(accelerations, center_fractions):
            mask_cfg = {
                "type": cfg["mask"].get("type", "random"),
                "center_fractions": [cf],
                "accelerations": [accel],
            }
            mask_func = build_mask_func(mask_cfg)
            val_set = build_dataset(cfg["data"], mask_func, split="val")
            loader = DataLoader(
                val_set, batch_size=cfg["eval"].get("batch_size", 1), collate_fn=collate_samples
            )
            report = _qc.QCReport()
            scores = evaluate_loader(method, model, loader, device, report)
            results[method][accel] = scores
            logger.info(
                "%-12s R=%dx  SSIM=%.4f  PSNR=%.2f  NMSE=%.4f | %s",
                method, accel, scores["ssim"], scores["psnr"], scores["nmse"], report.summary(),
            )
    return results


def results_to_markdown(results: dict) -> str:
    """render the results tree as a markdown table for the readme"""
    accels = sorted({a for by in results.values() for a in by})
    header = "| Method | " + " | ".join(f"R={a}x SSIM" for a in accels) + " |"
    sep = "|" + "---|" * (len(accels) + 1)
    lines = [header, sep]
    for method, by_accel in results.items():
        cells = [f"{by_accel[a]['ssim']:.3f}" if a in by_accel else "-" for a in accels]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
