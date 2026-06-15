"""zero-filled (stage 1) baseline metrics on synthetic data numpy only

produces a real metrics table without torch or any download so there is always a
concrete number to show and a floor for the learned models to beat numbers are on
synthetic phantoms labelled as such re-run on real fastmri via

    python -m mri_recon.cli eval --config configs/eval.yaml   # data.source=fastmri

usage
    python scripts/baseline_table.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mri_recon.data.synthetic import make_sample  # noqa: E402
from mri_recon.masking import RandomMaskFunc, apply_mask, realised_acceleration  # noqa: E402
from mri_recon.metrics import all_metrics  # noqa: E402
from mri_recon.models.zero_filled import zero_filled_reconstruction  # noqa: E402


def main() -> int:
    accel_settings = [(4, 0.08), (8, 0.04)]
    n_samples = 32
    rows = []

    for accel, center_fraction in accel_settings:
        mask_func = RandomMaskFunc([center_fraction], [accel])
        scores = {"ssim": [], "psnr": [], "nmse": [], "R": []}
        for i in range(n_samples):
            sample = make_sample(256, 256, seed=1000 + i)
            masked, mask = apply_mask(sample["kspace"], mask_func, seed=1000 + i)
            zf = zero_filled_reconstruction(masked)
            m = all_metrics(sample["image"], zf)
            for k in ("ssim", "psnr", "nmse"):
                scores[k].append(m[k])
            scores["R"].append(realised_acceleration(mask))
        rows.append(
            {
                "target_R": accel,
                "realised_R": float(np.mean(scores["R"])),
                "ssim": float(np.mean(scores["ssim"])),
                "psnr": float(np.mean(scores["psnr"])),
                "nmse": float(np.mean(scores["nmse"])),
            }
        )

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_synthetic.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["target_R", "realised_R", "ssim", "psnr", "nmse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Zero-filled baseline on {n_samples} synthetic slices (256x256):\n")
    print("| Method | R (target) | SSIM | PSNR (dB) | NMSE |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(
            f"| zero-filled | {r['target_R']}x | {r['ssim']:.3f} | "
            f"{r['psnr']:.2f} | {r['nmse']:.4f} |"
        )
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
