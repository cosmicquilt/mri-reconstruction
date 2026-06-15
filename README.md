# physics-informed mri reconstruction

**why this matters for quantitative imaging.** a biomarker is only as trustworthy
as the image it's measured from. accelerated mri scans skip measurements to save
time, leaving the raw data incomplete, and a reconstruction that *invents*
plausible-looking structure to fill the gaps poisons every downstream quantitative
feature. this project reconstructs high-fidelity images from **undersampled
k-space** with an unrolled network whose **data-consistency layer** keeps the output
provably faithful to what the scanner actually measured rather than hallucinating.
that faithfulness is the whole point for quantitative work.

**the bridge from my research.** my pinn work recovered reliable signals from
sparse, noisy measurements by building the governing physics into the model instead
of relying on data alone. accelerated-mri reconstruction is the same problem: the
data-consistency operator here *is* the physics constraint, it enforces the mr
forward model (image <-> k-space via the fourier transform) at every iteration.
same idea, medical-imaging setting.

> framing note: this is a *complementary* reconstruction approach. it's an
> unrolled/data-consistency method (schlemper-style dc-cnn), distinct from gan-based
> super-resolution, not a claim to reproduce any one group's exact pipeline.

## the three stages (each one is a complete, submittable result)

| stage | method | idea | beats |
|-------|--------|------|-------|
| 1 | **zero-filled** | inverse-fft the undersampled k-space. aliased, trivial. | the floor |
| 2 | **u-net** | learn to de-alias in the image domain. strong baseline. | stage 1 |
| 3 | **unrolled dc-cnn** | alternate a learned cnn denoiser with a data-consistency layer that re-imposes the measured k-space. **the centerpiece.** | stage 2 |

the unrolled network is "physics-informed" because of step 3's data-consistency
layer: wherever the scanner actually sampled k-space, the network is **not allowed**
to overwrite it. the cnn proposes, the physics disposes.

## quickstart

```bash
# 0. core physics sanity check, numpy only, no torch, no download, ~2s
python scripts/smoke_test.py

# 1. install (python 3.10/3.11 recommended, torch has no 3.14 wheels yet)
pip install -r requirements.txt && pip install -e .

# 2. full pipeline on synthetic data, no download, runs on cpu in minutes
python -m mri_recon.cli train --config configs/synthetic.yaml
python -m mri_recon.cli eval  --config configs/eval.yaml
python -m mri_recon.cli figures --config configs/eval.yaml

# 3. the real thing: download ONLY knee_singlecoil_val (~15 GB), extract it,
#    and split it by volume (see scripts/download_fastmri.md), then
python -m mri_recon.cli train --config configs/fastmri_knee_split.yaml          # u-net @ 4x
python -m mri_recon.cli train --config configs/fastmri_knee_split.yaml \
  --set model.name=unrolled --set train.loss=ssim --set train.batch_size=1 \
  --set mask.accelerations=[8] --set mask.center_fractions=[0.04] \
  --set output.dir=results/knee_split_unrolled_8x                                # unrolled @ 8x
```

windows: `.\run.ps1 train --config configs\synthetic.yaml`.
docker: `docker build -t mri-recon . && docker run --rm mri-recon smoke`.
colab: open [`notebooks/colab_mri_reconstruction.ipynb`](notebooks/colab_mri_reconstruction.ipynb).
it auto-detects the accelerator, runs synthetic end-to-end, then fastmri. **use a gpu
runtime (t4/l4/a100/h100), not tpu** (the complex fft isn't xla-friendly). on ampere+
gpus, mixed precision (`train.amp=true`, `amp_dtype=bf16`) speeds the u-net up.

## the pipeline (this is a pipeline, not a notebook)

```
data loading -> undersampling -> reconstruction -> qc -> evaluation -> visualization
   (synthetic     (configurable    (zero-filled /   (input &   (ssim/psnr/   (panels +
    or fastmri)    R = 4x, 8x)      u-net /         output      nmse per      metric
                                    unrolled)        checks)     acceleration) plots)
```

* **config-driven**, every run is `(yaml + cli overrides + fixed seed)`, so results
  are reproducible. see `configs/`.
* **quality control is first-class** (`src/mri_recon/qc.py`): finite/shape/empty
  checks, a k-space-centering check that catches the classic forgotten-`fftshift`
  bug, and metric range checks. every drop is logged with a reason.
* **containerized**, a `Dockerfile` for a pinned cpu environment, one-command runs
  via `run.ps1` / `Makefile`.

## results

**zero-filled baseline, real numbers, synthetic data** (32 slices, 256x256,
reproduce with `python scripts/baseline_table.py`):

| method | R (target) | SSIM | PSNR (dB) | NMSE |
|---|---|---|---|---|
| zero-filled | 4x | 0.702 | 30.60 | 0.0114 |
| zero-filled | 8x | 0.628 | 27.09 | 0.0255 |

**headline table, fill after training on fastmri** (`mri_recon.cli eval` writes this
automatically to `results/eval/results_table.md`):

| method | R=4x SSIM | R=8x SSIM |
|---|---|---|
| zero-filled | _baseline_ | _baseline_ |
| u-net | _train me_ | _train me_ |
| unrolled (dc-cnn) | _train me_ | _train me_ |

the story is the monotone climb (zero-filled -> u-net -> unrolled-with-data-consistency)
and the gap *widening* at the harder 8x, which is exactly where enforcing the
measured physics pays off most.

**hero figure** (`results/eval/figures/`, generated by `mri_recon.cli figures`):
ground truth | zero-filled | reconstruction | error map, plus ssim-vs-acceleration.

## design decisions (be ready to defend each)

* **centered orthonormal fft** matching fastmri, so dc energy sits at the array
  centre and "keep the central lines" is meaningful. unit-tested for round-trip
  identity.
* **hard vs soft data consistency.** hard dc trusts the measurement exactly
  (noiseless), soft dc blends `k = (k_pred + lam*k_meas)/(1+lam)` with a learnable
  lam (schlemper) for noisy data. parameterise `v = lam/(1+lam) = sigmoid(theta)`
  and learn theta, starting near hard dc. `lam -> inf` recovers hard dc.
* **residual cnn denoiser**, predicts a correction not the image, which trains more
  stably across a deep cascade.
* **train the unrolled net on ssim loss**, optimise the headline metric directly
  (as e2e-varnet does), the u-net baseline uses l1.
* **instance normalization** per slice for the u-net, removes per-scan intensity
  scaling so the network sees a stable input distribution.
* **the numpy reference *is* the test.** `mri_recon.fft.data_consistency_np` and the
  torch `DataConsistency` layer implement the same arithmetic, the numpy version is
  asserted correct in `scripts/smoke_test.py` and `tests/`, which is what gives
  confidence in the differentiable version.

## repo layout

```
src/mri_recon/
├── fft.py              # centered fft + numpy data-consistency reference
├── masking.py          # cartesian undersampling masks (random / equispaced)
├── metrics.py          # ssim / psnr / nmse (skimage or numpy fallback)
├── qc.py               # quality-control checks + report
├── data/
│   ├── synthetic.py    # phantom generator (no download)
│   ├── fastmri_dataset.py  # real .h5 loader + shared transform
│   ├── splits.py       # volume-level train/val/test split
│   └── transforms.py
├── models/
│   ├── zero_filled.py  # stage 1 (numpy)
│   ├── unet.py         # stage 2
│   ├── unrolled.py     # stage 3, the cascade (centerpiece)
│   └── layers.py       # torch fft, DataConsistency, ConvBlock, SSIMLoss
├── train.py  evaluate.py  viz.py  utils.py  cli.py
scripts/   smoke_test.py · baseline_table.py · download_fastmri.md
configs/   default · synthetic · unet_4x · unrolled_8x · eval · fastmri_knee_split
tests/     test_core.py
```

## status

pipeline scaffolded and **verified end-to-end on synthetic data** (physics core
unit-tested, exit-0). next: download the fastmri subset and run stages 2-3 to fill
the headline table and render the hero figure. see
[`mri-reconstruction-project-spec.md`](mri-reconstruction-project-spec.md) for the
full spec.
