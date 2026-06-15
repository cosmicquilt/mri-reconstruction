"""quality control a first-class stage not an afterthought

in a clinical pipeline silently passing a corrupt slice or out-of-range metric is
worse than crashing this runs cheap explicit checks on inputs (kspace/images) and
outputs (metrics) records every pass/fail with a reason and lets the pipeline drop
a sample or abort robustness and traceability over silent best-effort

pure numpy so qc runs anywhere including the no-torch smoke test
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("mri_recon.qc")


@dataclass
class QCResult:
    """outcome of a single check"""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class QCReport:
    """accumulates qc results across a run and summarises drops"""

    results: list[QCResult] = field(default_factory=list)
    dropped: int = 0
    checked: int = 0

    def record(self, result: QCResult, *, drop_on_fail: bool = True) -> bool:
        """log a result returns true if the sample should be kept"""
        self.results.append(result)
        if result.passed:
            logger.debug("QC PASS [%s] %s", result.name, result.detail)
            return True
        logger.warning("QC FAIL [%s] %s", result.name, result.detail)
        if drop_on_fail:
            self.dropped += 1
            return False
        return True

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        return (
            f"QC: {passed}/{len(self.results)} checks passed, "
            f"{self.dropped} sample(s) dropped of {self.checked} examined."
        )


def check_finite(array: np.ndarray, name: str) -> QCResult:
    """reject nan/inf usual symptom of a corrupt file or a bad normalization"""
    ok = bool(np.all(np.isfinite(array)))
    detail = "all finite" if ok else "contains NaN or Inf"
    return QCResult(f"{name}.finite", ok, detail)


def check_shape(array: np.ndarray, expected_ndim: int, name: str) -> QCResult:
    ok = array.ndim == expected_ndim and min(array.shape[-2:]) > 0
    return QCResult(
        f"{name}.shape", ok, f"shape={array.shape} (expected ndim={expected_ndim})"
    )


def check_not_empty(image: np.ndarray, name: str = "image", rel_tol: float = 1e-6) -> QCResult:
    """flag all-zero / flat slices (empty acquisitions failed crops)"""
    mag = np.abs(image)
    dynamic = float(mag.max() - mag.min())
    ok = dynamic > rel_tol * (float(mag.max()) + 1e-12)
    return QCResult(f"{name}.not_empty", ok, f"dynamic_range={dynamic:.3e}")


def check_kspace_centered(kspace: np.ndarray, name: str = "kspace") -> QCResult:
    """sanity check that dc (zero-freq) energy is at the array centre

    with the centered fft convention the brightest kspace point sits near the
    middle if its at a corner someone forgot an fftshift a classic silent bug
    """
    mag = np.abs(kspace)
    while mag.ndim > 2:
        mag = mag[0]
    peak = np.unravel_index(int(np.argmax(mag)), mag.shape)
    center = (mag.shape[0] // 2, mag.shape[1] // 2)
    tol = (max(2, mag.shape[0] // 8), max(2, mag.shape[1] // 8))
    ok = abs(peak[0] - center[0]) <= tol[0] and abs(peak[1] - center[1]) <= tol[1]
    return QCResult(f"{name}.centered", ok, f"peak={peak} center={center}")


def check_metric_range(value: float, name: str, lo: float, hi: float) -> QCResult:
    """confirm a computed metric lands in a physically plausible range"""
    ok = bool(np.isfinite(value) and lo <= value <= hi)
    return QCResult(f"metric.{name}", ok, f"{name}={value:.4f} expected in [{lo}, {hi}]")


# plausible ranges for a successful magnitude reconstruction wide on purpose
# catch gross failures (negative ssim nmse > 1 meaning worse than zeros)
# not subtle quality differences
METRIC_BOUNDS = {
    "ssim": (0.0, 1.0),
    "psnr": (5.0, 80.0),
    "nmse": (0.0, 1.0),
}


def qc_metrics(metrics: dict, report: QCReport, *, strict: bool = False) -> bool:
    """range-check a metrics dict returns true if all in range"""
    keep = True
    for key, value in metrics.items():
        if key not in METRIC_BOUNDS:
            continue
        lo, hi = METRIC_BOUNDS[key]
        result = check_metric_range(value, key, lo, hi)
        keep &= report.record(result, drop_on_fail=strict)
    return keep
