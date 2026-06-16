"""training loop for the unet (stage 2) and unrolled (stage 3) models

config-driven seed-fixed same reconstruct_batch as eval produces the
prediction here so what gets trained is exactly what gets scored
best checkpoint (by val ssim) + full epoch history written to the output dir
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.fastmri_dataset import build_dataset, collate_samples
from .evaluate import evaluate_loader, reconstruct_batch
from .masking import build_mask_func
from .models import build_model
from .models.layers import SSIMLoss
from .utils import configure_backend, count_params, get_device, set_seed

logger = logging.getLogger("mri_recon.train")


def train(cfg: dict) -> dict:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    configure_backend(device)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_func = build_mask_func(cfg["mask"])
    train_set = build_dataset(cfg["data"], mask_func, split="train")
    val_set = build_dataset(cfg["data"], mask_func, split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        drop_last=False,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["eval"].get("batch_size", 1), collate_fn=collate_samples
    )

    kind = cfg["model"]["name"]
    model = build_model(kind, **cfg["model"].get(kind, {})).to(device)
    logger.info("model=%s trainable_params=%d device=%s", kind, count_params(model), device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    loss_name = cfg["train"].get("loss", "l1").lower()
    ssim_loss = SSIMLoss().to(device) if loss_name == "ssim" else None

    # mixed precision big speedup on ampere/hopper gpus enabled only for the
    # real-valued unet the unrolled model complex fft is not autocast-safe and
    # stays fp32 regardless
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    if use_amp and kind != "unet":
        logger.warning("AMP requested but model=%s uses complex FFT; keeping fp32.", kind)
        use_amp = False
    amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "bf16")) == "bf16" else torch.float16
    scaler_enabled = use_amp and amp_dtype == torch.float16
    try:  # torch >= 2.3 api
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if use_amp:
        logger.info("mixed precision enabled (%s)", "bf16" if amp_dtype == torch.bfloat16 else "fp16")

    history: list[dict] = []
    best_ssim = -math.inf
    epochs = cfg["train"]["epochs"]

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with ctx:
                pred, target = reconstruct_batch(kind, model, batch, device)
                if loss_name == "ssim":
                    maxval = batch["max_value"].to(device).float()
                    loss = ssim_loss(pred, target, maxval)
                else:
                    loss = F.l1_loss(pred, target)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += loss.item() * pred.size(0)
            seen += pred.size(0)

        train_loss = running / max(seen, 1)
        val = evaluate_loader(kind, model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        logger.info(
            "epoch %d/%d  train_%s=%.4f  val_ssim=%.4f  val_psnr=%.2f  val_nmse=%.4f",
            epoch + 1, epochs, loss_name, train_loss, val["ssim"], val["psnr"], val["nmse"],
        )

        if val["ssim"] > best_ssim:
            best_ssim = val["ssim"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val},
                out_dir / "best.pt",
            )

    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("done. best val SSIM=%.4f  ->  %s", best_ssim, out_dir / "best.pt")
    return {"best_ssim": best_ssim, "checkpoint": str(out_dir / "best.pt"), "history": history}
