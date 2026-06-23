"""reproduce the headline 3-seed radiomic-stability result from the committed per-seed CCC tables

each `radiomics_per_feature.csv` (written by `mri_recon.cli radiomics` on one training seed) holds
the per-feature lin ccc of each reconstruction vs the ground truth. this seed-averages each
feature's ccc, then runs the paired wilcoxon signed-rank test (unrolled vs u-net) over the
seed-averaged features, so the "+0.037 mean ccc, 19/25 favoring the unrolled, p ~ 1e-4" headline is
reproducible from committed data alone, with no checkpoints or fastmri access. writes headline.json.

    python scripts/aggregate_radiomics_seeds.py                 # reads docs/radiomics/seed*/
    python scripts/aggregate_radiomics_seeds.py a.csv b.csv c.csv
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RADIO = ROOT / "docs" / "radiomics"
PAIR = ("unrolled", "unet")  # the two learned reconstructors compared in the headline


def load_ccc(path: str) -> dict:
    """{method: {feature: ccc}} from one radiomics_per_feature.csv"""
    out: dict = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["method"].split("@")[0]][row["feature"]] = float(row["ccc"])
    return out


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob(str(RADIO / "seed*" / "radiomics_per_feature.csv")))
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 per-seed csvs; found {len(paths)} (looked in {RADIO}/seed*/)")
    per_seed = [load_ccc(p) for p in paths]
    feats = sorted(set.intersection(*[set(s[PAIR[0]]) & set(s[PAIR[1]]) for s in per_seed]))
    if not feats:
        raise SystemExit("no features shared by both methods across all seeds")

    # seed-average each feature's ccc for the two reconstructors
    unrolled = np.array([np.mean([s[PAIR[0]][f] for s in per_seed]) for f in feats])
    unet = np.array([np.mean([s[PAIR[1]][f] for s in per_seed]) for f in feats])
    n_favor = int((unrolled > unet).sum())

    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(unrolled, unet).pvalue)
    except Exception:  # scipy absent (e.g. local) -> the p-value comes from a colab run
        p = float("nan")

    out = {
        "n_seeds": len(paths),
        "n_features": len(feats),
        "median_ccc_unrolled": round(float(np.median(unrolled)), 4),
        "median_ccc_unet": round(float(np.median(unet)), 4),
        "mean_ccc_diff_unrolled_minus_unet": round(float((unrolled - unet).mean()), 4),
        "n_features_favoring_unrolled": n_favor,
        "wilcoxon_p": p,
        "seeds": [Path(pp).parent.name for pp in paths],
        "per_feature_seed_avg": {f: {"unrolled": round(float(u), 4), "unet": round(float(e), 4)}
                                 for f, u, e in zip(feats, unrolled, unet)},
    }
    RADIO.mkdir(parents=True, exist_ok=True)
    (RADIO / "headline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(paths)} seeds, {len(feats)} features")
    print(f"median ccc  unrolled {out['median_ccc_unrolled']}  u-net {out['median_ccc_unet']}  "
          f"(+{out['mean_ccc_diff_unrolled_minus_unet']} mean, favoring unrolled in {n_favor}/{len(feats)})")
    print(f"paired wilcoxon p = {p:.2e}")
    print(f"wrote {RADIO / 'headline.json'}")


if __name__ == "__main__":
    main()
