"""radiomic feature stability under reconstruction

the bridge from reconstruction to quantitative imaging. instead of scoring a
reconstruction only by ssim/psnr, measure how it perturbs the downstream radiomic
features a biomarker pipeline would actually compute.

method (the part that has to be right):
- the roi is segmented ONCE on the fully-sampled ground truth and reused unchanged on
  every reconstruction. segmenting each recon separately would confound segmentation
  error with reconstruction error, so it is not done
- features come from pyradiomics with ibsi-style settings (z-score normalize x100,
  fixed bin width, original + wavelet) when it is installed, else a self-contained
  numpy fallback (first-order + a mask-aware glcm) so the study runs anywhere
- agreement between a recon feature and its ground-truth value is summarized per
  feature with lin's ccc and icc(2,1), then aggregated by feature class
- combat-lite capstone shows how much of the feature shift is a correctable systematic
  location/scale effect vs irreducible per-slice noise

method names support an @label suffix (e.g. unet@l1, unet@ssim) so the same
architecture trained with different losses can be compared in one run.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("mri_recon.radiomics")


# --- region of interest, defined on the ground truth only ----------------------

def gt_tissue_roi(gt: np.ndarray, frac: float = 0.10, min_pixels: int = 64) -> np.ndarray:
    """whole-knee tissue mask from the GROUND TRUTH magnitude

    threshold at a fraction of the gt max, keep the largest connected component, fill
    holes. reused on every recon so feature change reflects reconstruction not
    re-segmentation. a cartilage/bone model trained on the gt would be a drop-in
    upgrade, the decoupling principle is the same
    """
    from scipy import ndimage

    thr = gt > (frac * float(gt.max()) + 1e-12)
    labelled, n = ndimage.label(thr)
    if n == 0:
        return thr
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background label
    largest = labelled == int(sizes.argmax())
    roi = ndimage.binary_fill_holes(largest)
    return roi if roi.sum() >= min_pixels else thr


# --- feature extraction: pyradiomics if present, numpy fallback otherwise -------

def _have_pyradiomics() -> bool:
    try:
        import radiomics  # noqa: F401
        return True
    except Exception:
        return False


HAVE_PYRADIOMICS = _have_pyradiomics()


def extract_features(img: np.ndarray, mask: np.ndarray, bin_width: float = 25.0) -> dict:
    """radiomic features from one image inside one mask"""
    if HAVE_PYRADIOMICS:
        return _pyradiomics_features(img, mask, bin_width)
    return _fallback_features(img, mask, bin_width)


def _pyradiomics_features(img: np.ndarray, mask: np.ndarray, bin_width: float) -> dict:
    import SimpleITK as sitk
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    settings = {
        "binWidth": bin_width,
        "normalize": True,        # ibsi: standardize relative mr intensities
        "normalizeScale": 100,    # avoid fractional binning after z-score
        "force2D": True,
        "force2Ddimension": 0,
        "label": 1,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")  # the texture features most sensitive to recon
    # add a singleton z so simpleitk sees a volume, force2D does the rest
    si = sitk.GetImageFromArray(img.astype(np.float32)[None])
    sm = sitk.GetImageFromArray(mask.astype(np.uint8)[None])
    res = extractor.execute(si, sm)
    return {
        k: float(v) for k, v in res.items()
        if not k.startswith("diagnostics") and np.isfinite(_as_float(v))
    }


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fallback_features(img: np.ndarray, mask: np.ndarray, bin_width: float) -> dict:
    """self-contained first-order + mask-aware glcm, ibsi z-score x100 discretization

    approximate vs pyradiomics (no wavelet/log filters) but mask-exact and runs with
    no extra install, so the stability methodology is demonstrable anywhere
    """
    v = img.astype(np.float64)
    inside = v[mask]
    mu, sd = inside.mean(), inside.std() + 1e-12
    z = (inside - mu) / sd * 100.0
    feats = {
        "original_firstorder_Mean": float(z.mean()),
        "original_firstorder_Variance": float(z.var()),
        "original_firstorder_Skewness": float((((z - z.mean()) / (z.std() + 1e-12)) ** 3).mean()),
        "original_firstorder_Kurtosis": float((((z - z.mean()) / (z.std() + 1e-12)) ** 4).mean()),
        "original_firstorder_Energy": float(np.sum(z ** 2)),
        "original_firstorder_Entropy": float(_entropy(z, bin_width)),
        "original_firstorder_10Percentile": float(np.percentile(z, 10)),
        "original_firstorder_90Percentile": float(np.percentile(z, 90)),
        "original_firstorder_InterquartileRange": float(np.percentile(z, 75) - np.percentile(z, 25)),
        "original_firstorder_RootMeanSquared": float(np.sqrt(np.mean(z ** 2))),
    }
    feats.update(_masked_glcm(img, mask, bin_width))
    return feats


def _entropy(z: np.ndarray, bin_width: float) -> float:
    disc = np.floor((z - z.min()) / bin_width).astype(int)
    counts = np.bincount(disc - disc.min())
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _masked_glcm(img: np.ndarray, mask: np.ndarray, bin_width: float) -> dict:
    """symmetric, normalized glcm counting only neighbor pairs both inside the mask"""
    v = img.astype(np.float64)
    mu, sd = v[mask].mean(), v[mask].std() + 1e-12
    z = (v - mu) / sd * 100.0
    base = z[mask].min()
    disc = np.floor((z - base) / bin_width).astype(int)
    levels = int(disc[mask].max()) + 1
    if levels < 2:
        return {}
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = disc.shape
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        ai, am = disc[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        bi, bm = disc[y0 + dy:y1 + dy, x0 + dx:x1 + dx], mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        both = am & bm
        ii, jj = ai[both].clip(0, levels - 1), bi[both].clip(0, levels - 1)
        np.add.at(glcm, (ii, jj), 1.0)
        np.add.at(glcm, (jj, ii), 1.0)  # symmetric
    total = glcm.sum()
    if total == 0:
        return {}
    glcm /= total
    i_idx, j_idx = np.mgrid[0:levels, 0:levels]
    mu_i, mu_j = (i_idx * glcm).sum(), (j_idx * glcm).sum()
    sig_i = np.sqrt(((i_idx - mu_i) ** 2 * glcm).sum()) + 1e-12
    sig_j = np.sqrt(((j_idx - mu_j) ** 2 * glcm).sum()) + 1e-12
    return {
        "original_glcm_Contrast": float(((i_idx - j_idx) ** 2 * glcm).sum()),
        "original_glcm_Homogeneity": float((glcm / (1.0 + np.abs(i_idx - j_idx))).sum()),
        "original_glcm_Energy": float((glcm ** 2).sum()),
        "original_glcm_Correlation": float(((i_idx - mu_i) * (j_idx - mu_j) * glcm).sum() / (sig_i * sig_j)),
    }


def feature_class(name: str) -> str:
    """coarse radiomic feature family from a pyradiomics-style name"""
    n = name.lower()
    if "wavelet" in n:
        return "wavelet"
    if "log-sigma" in n or "_log_" in n:
        return "log"
    for fam in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
        if fam in n:
            return fam
    return "other"


# --- agreement statistics ------------------------------------------------------

def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """lin's concordance correlation coefficient, agreement to the identity line"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(), y.var()
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 1.0


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """icc(2,1): two-way random effects, absolute agreement, single measurement

    each slice is a target, the ground truth and the recon are the two raters
    """
    data = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1)
    n, k = data.shape
    if n < 2:
        return float("nan")
    grand = data.mean()
    row_means, col_means = data.mean(1), data.mean(0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1) + 1e-12)
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")


# --- reconstruction over the validation set ------------------------------------

def reconstruct_val(cfg: dict, checkpoints: dict, accel: int, cf: float, device, limit=None):
    """return (gt_images, {method: [recon_images]}) at one acceleration

    method keys may carry an @label suffix, the part before @ selects the architecture
    """
    import torch
    from torch.utils.data import DataLoader

    from .data.fastmri_dataset import build_dataset, collate_samples
    from .evaluate import load_model, reconstruct_batch
    from .masking import build_mask_func

    mask_func = build_mask_func(
        {"type": cfg["mask"].get("type", "random"), "center_fractions": [cf], "accelerations": [accel]}
    )
    val_set = build_dataset(cfg["data"], mask_func, split="val")
    loader = DataLoader(val_set, batch_size=1, collate_fn=collate_samples)

    model_cfg = cfg.get("model", {})
    models = {}
    for name, ckpt in checkpoints.items():
        kind = name.split("@")[0]
        models[name] = (kind, load_model(kind, model_cfg, ckpt, device))

    gts: list[np.ndarray] = []
    recons: dict[str, list[np.ndarray]] = {name: [] for name in checkpoints}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break
            gt = None
            for name, (kind, model) in models.items():
                if model is not None:
                    model.eval()
                pred, target = reconstruct_batch(kind, model, batch, device)
                recons[name].append(pred[0, 0].detach().cpu().numpy())
                gt = target[0, 0].detach().cpu().numpy()
            gts.append(gt)
    return gts, recons


# --- the study -----------------------------------------------------------------

def run_stability_study(cfg, checkpoints, device, accel=8, cf=0.04, limit=120, bin_width=25.0):
    """reconstruct, extract features under a gt-defined roi, score agreement per feature"""
    gts, recons = reconstruct_val(cfg, checkpoints, accel, cf, device, limit)
    methods = list(recons.keys())
    logger.info("extracting radiomic features over %d slices, pyradiomics=%s", len(gts), HAVE_PYRADIOMICS)

    gt_feats: list[dict] = []
    method_feats: dict[str, list[dict]] = {m: [] for m in methods}
    for j, gt in enumerate(gts):
        roi = gt_tissue_roi(gt)
        if roi.sum() < 64:
            continue
        gt_feats.append(extract_features(gt, roi, bin_width))
        for m in methods:
            method_feats[m].append(extract_features(recons[m][j], roi, bin_width))

    if not gt_feats:
        raise RuntimeError("no usable rois, every slice mask was below the pixel floor")

    # features common to every extraction
    keys = sorted(set.intersection(*[set(f) for f in gt_feats]))
    gt_arr = {k: np.array([f[k] for f in gt_feats], float) for k in keys}

    rows = []
    for m in methods:
        for k in keys:
            rec = np.array([f.get(k, np.nan) for f in method_feats[m]], float)
            ok = np.isfinite(rec) & np.isfinite(gt_arr[k])
            if ok.sum() < 3:
                continue
            rows.append({
                "method": m, "feature": k, "fclass": feature_class(k),
                "ccc": lin_ccc(gt_arr[k][ok], rec[ok]),
                "icc": icc_2_1(gt_arr[k][ok], rec[ok]),
            })
    return rows, gt_arr, method_feats, keys


def combat_capstone(gt_arr, method_feats, keys, method):
    """location/scale align each recon feature to the gt distribution, recompute ccc

    combat-lite (no empirical-bayes shrinkage). the gain is the systematic, correctable
    part of the shift, the residual is irreducible per-slice variation. full neuroCombat
    / OPNested ComBat add eb shrinkage and covariate preservation
    """
    before, after = [], []
    for k in keys:
        rec = np.array([f.get(k, np.nan) for f in method_feats[method]], float)
        g = gt_arr[k]
        ok = np.isfinite(rec) & np.isfinite(g)
        if ok.sum() < 3:
            continue
        r, gg = rec[ok], g[ok]
        harmonized = (r - r.mean()) / (r.std() + 1e-12) * (gg.std() + 1e-12) + gg.mean()
        before.append(lin_ccc(gg, r))
        after.append(lin_ccc(gg, harmonized))
    return float(np.median(before)), float(np.median(after))


def summarize(rows: list[dict]) -> dict:
    """percent of features with ccc>0.85 and median ccc, per method overall and by class"""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        buckets[(r["method"], r["fclass"])].append(r["ccc"])
        buckets[(r["method"], "ALL")].append(r["ccc"])
    out: dict[str, dict] = {}
    for (method, fclass), cccs in buckets.items():
        arr = np.array(cccs, float)
        out.setdefault(method, {})[fclass] = {
            "n": int(arr.size),
            "pct_ccc_gt_0.85": round(100.0 * float(np.mean(arr > 0.85)), 1),
            "median_ccc": round(float(np.median(arr)), 3),
        }
    return out


def run_and_report(cfg: dict, device, out_dir: str | Path) -> dict:
    """cli entry: run the study, print a readable summary, write csv + json"""
    rcfg = cfg.get("radiomics", {})
    accel = int(rcfg.get("acceleration", 8))
    cf = float(rcfg.get("center_fraction", 0.04))
    limit = rcfg.get("limit", 120)
    bin_width = float(rcfg.get("bin_width", 25))

    methods = cfg["eval"]["methods"]
    ckpt_cfg = cfg["eval"].get("checkpoints", {}) or {}
    checkpoints = {m: ckpt_cfg.get(m) for m in methods if m.split("@")[0] == "zero_filled" or ckpt_cfg.get(m)}
    if not checkpoints:
        raise ValueError("no methods with checkpoints, pass eval.checkpoints.<method>=... via --set")

    rows, gt_arr, method_feats, keys = run_stability_study(
        cfg, checkpoints, device, accel=accel, cf=cf, limit=limit, bin_width=bin_width
    )
    summary = summarize(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "radiomics_per_feature.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "feature", "fclass", "ccc", "icc"])
        writer.writeheader()
        writer.writerows(rows)

    # combat capstone for the learned reconstructors
    capstone = {}
    for m in checkpoints:
        if m.split("@")[0] == "zero_filled":
            continue
        b, a = combat_capstone(gt_arr, method_feats, keys, m)
        capstone[m] = {"median_ccc_before": round(b, 3), "median_ccc_after_combat": round(a, 3)}

    report = {"accel": accel, "n_features": len(keys), "pyradiomics": HAVE_PYRADIOMICS,
              "summary": summary, "combat": capstone}
    (out_dir / "radiomics_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_report(report)
    print(f"\nwrote {out_dir/'radiomics_per_feature.csv'} and {out_dir/'radiomics_summary.json'}")
    return report


def _print_report(report: dict) -> None:
    print(f"\n=== radiomic feature stability vs ground truth, R={report['accel']}x "
          f"({report['n_features']} features, pyradiomics={report['pyradiomics']}) ===")
    print("  higher ccc = more reproducible feature\n")
    print(f"{'method':16s}{'class':12s}{'n':>4s}{'%ccc>0.85':>11s}{'median ccc':>12s}")
    for method, classes in report["summary"].items():
        for fclass in ["ALL"] + sorted(k for k in classes if k != "ALL"):
            s = classes[fclass]
            print(f"{method:16s}{fclass:12s}{s['n']:>4d}{s['pct_ccc_gt_0.85']:>11.1f}{s['median_ccc']:>12.3f}")
        print()
    if report["combat"]:
        print("combat-lite capstone (median ccc, recon vs gt):")
        for method, c in report["combat"].items():
            print(f"  {method:16s} before={c['median_ccc_before']:.3f}  "
                  f"after={c['median_ccc_after_combat']:.3f}  "
                  f"(gain = correctable systematic shift)")
