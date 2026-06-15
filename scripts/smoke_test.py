"""end-to-end smoke test of the reconstruction physics numpy only no torch

run me to prove the plumbing works before any data is downloaded or any model is
trained

    python scripts/smoke_test.py

exercises the fft conventions undersampling masks the zero-filled baseline the
metrics the data-consistency operator (the centerpiece) and the qc checks asserting
the properties that must hold exits non-zero on any failure
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# make src/ importable without installing the package
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mri_recon import metrics, qc  # noqa: E402
from mri_recon.data.synthetic import make_sample  # noqa: E402
from mri_recon.fft import data_consistency_np, fft2c, ifft2c  # noqa: E402
from mri_recon.masking import (  # noqa: E402
    RandomMaskFunc,
    apply_mask,
    realised_acceleration,
)
from mri_recon.models.zero_filled import zero_filled_reconstruction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    rng = np.random.default_rng(0)
    report = qc.QCReport()

    section("1. Synthetic sample + FFT round-trip")
    sample = make_sample(height=256, width=256, seed=1)
    image, kspace = sample["image"], sample["kspace"]
    roundtrip = ifft2c(fft2c(image.astype(np.complex64)))
    rt_err = float(np.abs(roundtrip - image).max())
    print(f"image {image.shape} {image.dtype}, kspace {kspace.shape} {kspace.dtype}")
    print(f"max |ifft2c(fft2c(x)) - x| = {rt_err:.2e}")
    assert rt_err < 1e-4, "FFT round-trip is not identity, check shift conventions"

    section("2. Undersampling mask (target 4x)")
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
    masked_kspace, mask = apply_mask(kspace, mask_func, seed=7)
    R = realised_acceleration(mask)
    print(f"realised acceleration R = {R:.2f}x (target 4x)")
    assert 3.0 <= R <= 5.5, "realised acceleration far from target"

    section("3. Zero-filled baseline + metrics")
    zf = zero_filled_reconstruction(masked_kspace)
    m = metrics.all_metrics(image, zf)
    print(f"zero-filled: SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f} dB  NMSE={m['nmse']:.4f}")
    # identity sanity a perfect reconstruction maxes out every metric
    perfect = metrics.all_metrics(image, image)
    print(f"identity   : SSIM={perfect['ssim']:.4f}  PSNR={perfect['psnr']}  NMSE={perfect['nmse']:.2e}")
    assert perfect["ssim"] > 0.999, "SSIM(x, x) should be ~1"
    assert perfect["nmse"] < 1e-9, "NMSE(x, x) should be ~0"
    assert 0.0 <= m["ssim"] < perfect["ssim"], "aliased recon must be worse than perfect"
    assert m["nmse"] > 0, "aliased recon must have non-zero error"

    section("4. Data-consistency operator (the centerpiece)")
    # measured kspace = true kspace at sampled locations only
    measured = fft2c(image.astype(np.complex64)) * mask
    # a corrupted starting estimate (what a half-trained denoiser might emit)
    corrupt = (image + 0.3 * rng.standard_normal(image.shape)).astype(np.complex64)
    dc_img = data_consistency_np(corrupt, measured, mask, lam=None)  # hard dc

    # defining property after hard dc sampled kspace equals the measurement
    resampled = fft2c(dc_img) * mask
    dc_consistency_err = float(np.abs(resampled - measured).max())
    print(f"max |F(DC(x))*mask - measured| = {dc_consistency_err:.2e}  (should be ~0)")
    assert dc_consistency_err < 1e-4, "hard DC did not reproduce the measured samples"

    # and dc must move the estimate closer to truth (it injects true samples)
    err_before = float(np.linalg.norm(np.abs(corrupt) - image))
    err_after = float(np.linalg.norm(np.abs(dc_img) - image))
    print(f"||corrupt - gt|| = {err_before:.3f}  ->  ||DC(corrupt) - gt|| = {err_after:.3f}")
    assert err_after < err_before, "DC should reduce the error toward ground truth"

    # soft dc (noisy-measurement mode) must also run and stay finite
    dc_soft = data_consistency_np(corrupt, measured, mask, lam=10.0)
    assert np.all(np.isfinite(np.abs(dc_soft))), "soft DC produced non-finite output"
    print("soft DC (lam=10) ran and stayed finite.")

    section("5. QC checks")
    for res in (
        qc.check_finite(kspace, "kspace"),
        qc.check_kspace_centered(kspace),
        qc.check_not_empty(image),
        qc.check_shape(image, 2, "image"),
    ):
        report.checked += 1
        report.record(res)
    qc.qc_metrics(m, report)
    print(report.summary())
    assert report.n_failed == 0, "unexpected QC failure on a clean synthetic sample"

    section("RESULT")
    print("ALL SMOKE-TEST CHECKS PASSED [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
