"""render the texture-decomposition figure for the radiomic-stability finding

snapshot of the per-feature ccc decomposition on the 3-seed 8x run, so the figure regenerates
without the fastmri data. each glcm feature's accuracy loss is split into a scale penalty
S = (v + 1/v - 2)/2 (spread mismatch) and a location penalty L = u^2/2 (mean offset), from
ccc_components (1/c_b = 1 + S + L). the point: the texture accuracy loss is location-dominated
(a coherent smoothing mean-shift), with hard spread compression confined to contrast and
correlation. run:

    python scripts/make_radiomics_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

# seed-median per-texture-feature decomposition at 8x for the unrolled recon (the headline model),
# ordered worst-to-best accuracy. v = std(gt)/std(recon) scale shift, u = mean offset
# (feature, accuracy, v, u)
_UNROLLED = [
    ("Contrast", 0.205, 5.084, 2.112),
    ("Correlation", 0.219, 4.467, -2.108),
    ("Dissimilarity", 0.312, 2.446, 1.883),
    ("Homogeneity", 0.399, 1.180, -1.726),
    ("JointEntropy", 0.407, 1.261, 1.690),
    ("Energy", 0.442, 0.657, -1.511),
    ("MaximumProbability", 0.564, 0.836, -1.222),
    ("ClusterShade", 0.962, 0.809, -0.182),
    ("ClusterProminence", 0.974, 0.828, -0.148),
    ("Autocorrelation", 0.987, 0.893, -0.119),
]


def _penalties(v: float, u: float) -> tuple[float, float]:
    """scale and location penalties, 1/c_b = 1 + S + L"""
    return (v + 1.0 / v - 2.0) / 2.0, u**2 / 2.0


def texture_decomposition_figure() -> None:
    feats = [r[0] for r in _UNROLLED]
    S = np.array([_penalties(r[2], r[3])[0] for r in _UNROLLED])
    L = np.array([_penalties(r[2], r[3])[1] for r in _UNROLLED])
    y = np.arange(len(feats))[::-1]  # worst accuracy at the top

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(y + 0.19, L, 0.38, color="#0072B2", label="location penalty  u²/2  (mean offset)")
    ax.barh(y - 0.19, S, 0.38, color="#D55E00", label="scale penalty  (v+1/v-2)/2  (spread mismatch)")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_ylim(-0.6, len(feats) - 0.4)
    ax.set_xlabel("contribution to lin's accuracy loss   (1/c_b = 1 + S + L)")
    ax.set_title("the texture bias is location-dominated, not a scale compression\n"
                 "per-feature accuracy-loss split, unrolled recon at 8x", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", ls=":", color="#dddddd")
    ax.set_axisbelow(True)
    ax.annotate("only contrast + correlation\ncompress the spread (v 5-6x)",
                xy=(S[0], y[0] - 0.19), xytext=(2.45, len(feats) - 3.1),
                fontsize=7.5, color="#9a3a00",
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "radiomics_texture_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    texture_decomposition_figure()
