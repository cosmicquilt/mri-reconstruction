# getting the fastmri single-coil knee data

the pipeline runs immediately on **synthetic phantoms** (no download). to produce the
real results, get the fastmri data. it's free for research but requires a signed
agreement, so it can't be scripted end-to-end.

## 1. register and accept the data-use agreement

go to <https://fastmri.med.nyu.edu/>, request access, and accept the agreement. nyu
emails time-limited download links (aws). the data is de-identified.

## 2. download the single-coil knee subset

you only need the **single-coil knee** data, skip multicoil, brain, prostate, and
breast entirely (terabytes you'd never touch).

**recommended shortcut:** download **only `knee_singlecoil_val` (~15 GB)** and carve
your own train/val/test split from it (split is by volume, no leakage). that avoids
the 72 GB train download and is plenty for a subset demo + the 4x/8x comparison. just
say plainly in the readme it's a subset, not a leaderboard run.

```bash
# paste your signed curl command from the nyu email (urls expire in 90 days):
curl -C - "https://fastmri-dataset.s3.amazonaws.com/.../knee_singlecoil_val.tar.xz?..." \
  --output knee_singlecoil_val.tar.xz
tar -xf knee_singlecoil_val.tar.xz -C data/fastmri/    # -> data/fastmri/singlecoil_val/
```

> the signed urls are time-limited access tokens, never commit them to a repo.

## 3. lay it out like this

```
data/fastmri/
├── singlecoil_train/
│   ├── file1000001.h5
│   └── ...
└── singlecoil_val/
    ├── file1000123.h5
    └── ...
```

each `.h5` holds a fully-sampled `kspace` volume of shape `(num_slices, H, W)`
(complex) plus a `reconstruction_esc` magnitude target. this project applies its own
undersampling mask to `kspace`, so the fully-sampled data is the ground truth.

## 4. point the config at it

**val-only split (recommended):** `configs/fastmri_knee_split.yaml` carves
train/val/test out of the single directory:

```yaml
data:
  source: fastmri
  split_dir: data/fastmri/singlecoil_val   # extracted knee_singlecoil_val/
  split_fractions: [0.7, 0.15, 0.15]        # by volume, no slice leakage
```

```bash
python -m mri_recon.cli train --config configs/fastmri_knee_split.yaml          # u-net @ 4x
python -m mri_recon.cli train --config configs/fastmri_knee_split.yaml \
  --set model.name=unrolled --set train.loss=ssim --set train.batch_size=1 \
  --set mask.accelerations=[8] --set mask.center_fractions=[0.04] \
  --set output.dir=results/knee_split_unrolled_8x                                # unrolled @ 8x
python -m mri_recon.cli eval --config configs/fastmri_knee_split.yaml \
  --set eval.methods=[zero_filled,unet] --set eval.checkpoints.unet=results/knee_split_unet_4x/best.pt
```

**separate train/val dirs (if you downloaded both):** use `configs/unet_4x.yaml` /
`configs/unrolled_8x.yaml` with `data.root` + `train_dir`/`val_dir` instead of
`split_dir`.

> clinical/imaging data must never be committed to git, `data/` is in `.gitignore`.
> keep the download local.
