"""single config-driven entry point for the whole pipeline

    python -m mri_recon.cli smoke
    python -m mri_recon.cli train --config configs/unet_4x.yaml
    python -m mri_recon.cli eval  --config configs/eval.yaml --set data.source=synthetic
    python -m mri_recon.cli figures --config configs/eval.yaml

torch work imported inside each handler so --help and smoke run on numpy alone
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .utils import apply_overrides, load_config, set_seed, setup_logging

logger = logging.getLogger("mri_recon.cli")


def _cmd_smoke(args: argparse.Namespace) -> int:
    """fast numpy-only sanity check of the core physics (no torch)"""
    from .data.synthetic import make_phantom
    from .fft import data_consistency_np, fft2c, ifft2c
    from .masking import RandomMaskFunc, apply_mask, realised_acceleration
    from .metrics import all_metrics
    from .models.zero_filled import zero_filled_reconstruction

    set_seed(0)
    img = make_phantom(192, 192, seed=1)
    kspace = fft2c(img.astype(np.complex64))
    assert np.abs(ifft2c(kspace) - img).max() < 1e-4, "FFT round-trip failed"

    masked, mask = apply_mask(kspace, RandomMaskFunc([0.08], [4]), seed=3)
    zf = zero_filled_reconstruction(masked)
    m = all_metrics(img, zf)
    dc = data_consistency_np((img + 0.2 * np.random.randn(*img.shape)).astype(np.complex64), masked, mask)
    consistent = np.abs(fft2c(dc) * mask - masked).max()

    print(f"R={realised_acceleration(mask):.2f}x  zero-filled SSIM={m['ssim']:.3f} "
          f"PSNR={m['psnr']:.2f} NMSE={m['nmse']:.4f}  DC-consistency={consistent:.2e}")
    print("smoke OK (run scripts/smoke_test.py for the full assertion suite)")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    cfg = load_config(args.config, args.set)
    result = train(cfg)
    print(f"best val SSIM = {result['best_ssim']:.4f}  ->  {result['checkpoint']}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import results_to_markdown, run_evaluation
    from .utils import get_device

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("device", "auto"))
    results = run_evaluation(cfg, device)

    out_dir = Path(cfg.get("output", {}).get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = results_to_markdown(results)
    (out_dir / "results_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'results_table.md'}")
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader

    from .data.fastmri_dataset import build_dataset
    from .evaluate import load_model, reconstruct_batch, run_evaluation
    from .masking import build_mask_func
    from .utils import get_device
    from .viz import comparison_panel, metric_vs_acceleration

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg.get("output", {}).get("dir", "results"))
    fig_dir = out_dir / "figures"

    # metric-vs-acceleration plot from a fresh eval
    results = run_evaluation(cfg, device)
    metric_vs_acceleration(results, cfg["eval"].get("plot_metric", "ssim"), fig_dir / "ssim_vs_acceleration.png")

    # comparison panel on the first val sample at the headline acceleration
    accel = cfg["eval"]["accelerations"][0]
    cf = cfg["eval"].get("center_fractions", [0.08])[0]
    mask_func = build_mask_func({"type": cfg["mask"].get("type", "random"),
                                 "center_fractions": [cf], "accelerations": [accel]})
    val_set = build_dataset(cfg["data"], mask_func, split="val")
    batch = next(iter(DataLoader(val_set, batch_size=1)))

    import torch

    recons = {}
    target = None
    with torch.no_grad():
        for method in cfg["eval"]["methods"]:
            model = load_model(method, cfg.get("model", {}),
                               cfg["eval"].get("checkpoints", {}).get(method), device)
            if model is not None:
                model.eval()
            pred, target = reconstruct_batch(method, model, batch, device)
            recons[method] = pred[0, 0].detach().cpu().numpy()
    gt = target[0, 0].detach().cpu().numpy()
    comparison_panel(gt, recons, fig_dir / "hero_panel.png", title=f"Reconstruction at R={accel}x")
    print(f"wrote figures to {fig_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mri_recon", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="NumPy-only physics sanity check")
    p_smoke.set_defaults(func=_cmd_smoke)

    for name, func, helptext in [
        ("train", _cmd_train, "train a model from a config"),
        ("eval", _cmd_eval, "evaluate methods x accelerations -> metrics table"),
        ("figures", _cmd_figures, "render comparison panel + metric plot"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", required=True, help="path to a YAML config")
        p.add_argument("--set", nargs="*", action="append", default=[], metavar="key=value",
                       help="dotted config override(s); repeatable AND space-separated both work, "
                            "e.g. --set train.epochs=5 or --set a=1 --set b=2")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    # --set uses nargs=* + action=append so it arrives as a list of lists
    # (one sublist per flag occurrence) flatten so both --set a=1 --set b=2 and
    # --set a=1 b=2 end up as a single [a=1, b=2] list of overrides
    if hasattr(args, "set"):
        flat: list[str] = []
        for group in args.set or []:
            flat.extend(group if isinstance(group, list) else [group])
        args.set = flat
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
