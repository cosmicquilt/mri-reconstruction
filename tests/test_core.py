"""unit tests for the pure-numpy physics core (no torch needed)

    pytest -q

lock down the properties the whole project depends on the fft convention the
acceleration of the masks the metric definitions the data-consistency operator and
the qc checks
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mri_recon import metrics, qc  # noqa: E402
from mri_recon.data.synthetic import make_sample  # noqa: E402
from mri_recon.fft import data_consistency_np, fft2c, ifft2c  # noqa: E402
from mri_recon.masking import (  # noqa: E402
    EquiSpacedMaskFunc,
    RandomMaskFunc,
    apply_mask,
    realised_acceleration,
)
from mri_recon.models.zero_filled import zero_filled_reconstruction  # noqa: E402


@pytest.fixture
def sample():
    return make_sample(height=128, width=128, seed=1)


def test_fft_roundtrip(sample):
    img = sample["image"].astype(np.complex64)
    assert np.abs(ifft2c(fft2c(img)) - img).max() < 1e-4


@pytest.mark.parametrize("R", [4, 8])
def test_random_mask_acceleration(sample, R):
    mf = RandomMaskFunc(center_fractions=[0.08 if R == 4 else 0.04], accelerations=[R])
    _, mask = apply_mask(sample["kspace"], mf, seed=0)
    assert abs(realised_acceleration(mask) - R) <= 1.5


def test_equispaced_mask_runs(sample):
    mf = EquiSpacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    masked, mask = apply_mask(sample["kspace"], mf, seed=0)
    assert masked.shape == sample["kspace"].shape
    assert 2.5 <= realised_acceleration(mask) <= 5.5


def test_metrics_identity(sample):
    img = sample["image"]
    assert metrics.ssim(img, img) == pytest.approx(1.0, abs=1e-3)
    assert metrics.nmse(img, img) == pytest.approx(0.0, abs=1e-9)
    assert np.isinf(metrics.psnr(img, img))


def test_metrics_degrade_monotonically(sample):
    img = sample["image"]
    rng = np.random.default_rng(0)
    light = img + 0.02 * rng.standard_normal(img.shape)
    heavy = img + 0.20 * rng.standard_normal(img.shape)
    assert metrics.ssim(img, light) > metrics.ssim(img, heavy)
    assert metrics.psnr(img, light) > metrics.psnr(img, heavy)
    assert metrics.nmse(img, light) < metrics.nmse(img, heavy)


def test_data_consistency_reproduces_measurements(sample):
    img = sample["image"]
    mf = RandomMaskFunc([0.08], [4])
    _, mask = apply_mask(sample["kspace"], mf, seed=2)
    measured = fft2c(img.astype(np.complex64)) * mask

    corrupt = (img + 0.3 * np.random.default_rng(0).standard_normal(img.shape)).astype(np.complex64)
    dc = data_consistency_np(corrupt, measured, mask, lam=None)

    # defining property of hard dc sampled kspace equals the measurement
    assert np.abs(fft2c(dc) * mask - measured).max() < 1e-4
    # and dc moves the estimate closer to ground truth
    assert np.linalg.norm(np.abs(dc) - img) < np.linalg.norm(np.abs(corrupt) - img)


def test_zero_filled_beats_nothing_but_not_perfect(sample):
    mf = RandomMaskFunc([0.08], [4])
    masked, _ = apply_mask(sample["kspace"], mf, seed=3)
    zf = zero_filled_reconstruction(masked)
    m = metrics.all_metrics(sample["image"], zf)
    assert 0.0 < m["ssim"] < 1.0
    assert m["nmse"] > 0.0


def test_qc_flags_corruption(sample):
    bad = sample["image"].copy()
    bad[0, 0] = np.nan
    assert not qc.check_finite(bad, "image").passed
    # a non-centered kspace (forgot fftshift) should be flagged
    uncentered = np.fft.fft2(sample["image"])  # dc at the corner not the centre
    assert not qc.check_kspace_centered(uncentered).passed
    assert qc.check_kspace_centered(sample["kspace"]).passed
