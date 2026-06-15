# portfolio project spec: undersampled mri reconstruction (fastmri)

**goal:** reconstruct high-fidelity mri images from undersampled k-space using a
physics-informed unrolled network with a data-consistency layer, packaged as a clean,
reproducible pipeline with qc, metrics, and visualization.

**why this one:** it bridges the pinn sparse-data reconstruction work directly into
medical imaging (the strongest narrative), it's the jd's exact word ("reconstruction"),
and a single project hits five jd responsibilities at once: reconstruction, image
analysis, scalable pipeline, image qc, and visualization. scoped to finish in ~5 days.

## dataset & setup

- **dataset:** fastmri (nyu/meta). register and sign the data-use agreement at
  fastmri.med.nyu.edu, free for research, de-identified. start with the **single-coil
  knee** subset (simplest entry point).
- **use a subset.** don't train on the full dataset, a few hundred to a couple thousand
  slices is plenty to demonstrate the method. this keeps training tractable on a single
  gpu (colab/kaggle free tier is fine).
- **tooling:** python + pytorch. lean on the `facebookresearch/fastMRI` repo, it
  provides data loaders, undersampling masks (`subsample.py`), baseline models (u-net,
  varnet), and the standard evaluation metrics. adapt, don't reimplement.

## build path (incremental, always keep a working version)

**the rule:** each stage produces a submittable result, so time pressure never leaves
you with nothing.

- **stage 1, zero-filled baseline (day 1).** apply an undersampling mask to
  fully-sampled k-space (start at 4x acceleration, keeping ~8% central low-frequency
  lines), inverse-fft the result, and view the aliased reconstruction. this frames the
  problem and sets the metric floor. trivial but essential.
- **stage 2, u-net reconstruction (days 1-2).** train a u-net to map the zero-filled
  (aliased) image -> ground-truth image. this is the fastmri u-net baseline, the repo
  has it. now there's a working deep-learning reconstruction and a real metric
  improvement over stage 1.
- **stage 3, unrolled / physics-informed network (days 2-4).** this is the centerpiece.
  build an unrolled cascade that alternates a learned cnn denoiser with a
  **data-consistency layer** that re-enforces the measured k-space at each iteration (a
  dc-cnn / variational-network style architecture, the repo's e2e-varnet is the
  reference if you move to multi-coil). the data-consistency layer is the
  physics-informed piece, it guarantees the output stays faithful to the scanner's
  actual measurements rather than hallucinating structure.

> if time runs short, stage 2 is a complete, submittable project on its own. stage 3 is
> the differentiator, start simple (1-2 unrolled iterations) and add cascades if it's
> training well.

## pipeline structure (demonstrates the "scalable pipeline / qc / data consistency" jd bullets)

build it as a pipeline, not a single notebook:

1. **data loading**, fastmri loaders, configurable subset.
2. **undersampling**, apply masks at configurable acceleration (4x and 8x).
3. **reconstruction**, the model (selectable: zero-filled / u-net / unrolled).
4. **qc checks**, validate slice shapes and k-space normalization, flag empty/corrupt
   slices, sanity-check that metrics fall in expected ranges. log what's dropped and why.
5. **evaluation**, metrics on a held-out validation set, reported per acceleration factor.
6. **visualization**, output figures (below).

make it config-driven (a yaml or argparse config), fix random seeds for reproducibility,
and write a clear readme. a lightweight dockerfile is a nice reproducibility nod, skip
the heavier mlops apparatus.

## metrics

use the fastmri standards, computed against ground truth on a validation set:

- **ssim** (structural similarity, the headline metric)
- **psnr** (peak signal-to-noise ratio)
- **nmse** (normalized mean squared error)

report a small table: each metric for zero-filled vs u-net vs unrolled, at 4x and 8x.
the improvement story (zero-filled -> u-net -> unrolled-with-data-consistency) is the
results narrative.

## visualization

- side-by-side panels: **ground truth | zero-filled | model reconstruction | error
  map**, for a few representative slices.
- metric plot: ssim (and/or psnr) vs acceleration factor across the three methods.
- one clean "hero figure" for the readme and resume/portfolio link.

## scope guards (so it actually finishes)

- single-coil knee, data subset, existing repo code. resist scope creep.
- **drop entirely:** segmentation, radiomics, radiogenomic outcome modeling, clinical
  deployment, all out of scope for this project.
- **finish-over-features:** a clean, complete stage-2/stage-3 result beats an ambitious
  half-built pipeline.
- be able to **defend every design choice for 30 minutes**, depth on this one method
  beats breadth.

## deliverables

- **github repo:** pipeline code (loading, masking, model, training, eval, qc, viz),
  config files, readme, fixed seeds, optional dockerfile.
- **readme / 1-page writeup:** the problem, the approach, the results table + hero
  figure, and the explicit bridge from the pinn work.
- **portfolio/resume link** to the repo.
