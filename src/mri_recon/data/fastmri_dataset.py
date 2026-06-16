"""pytorch datasets for the reconstruction pipeline

two datasets one shared transform
FastMRISliceDataset reads real single-coil knee .h5 (each a fully-sampled kspace volume)
SyntheticSliceDataset wraps the numpy phantom gen so the same code runs with no download

both yield the same sample dict from make_sample_tensors
    masked_kspace (h w) complex64    undersampled measurement
    mask          (1 h w) float32    1 where sampled (full kspace size)
    target        (1 ch cw) float32  fully-sampled magnitude centre-cropped
    zf_image      (1 ch cw) float32  zero-filled magnitude (unet input / baseline)
    max_value     float             target.max() the ssim data range

batching note target/zf_image are a fixed crop always batch
full-size masked_kspace/mask vary across real volumes so train unrolled on real
data with batch_size 1 (synthetic is fixed-size batches at any size)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..fft import ifft2c
from ..masking import MaskFunc, expand_mask
from .splits import split_fastmri_files
from .transforms import center_crop
from .synthetic import make_phantom


def make_sample_tensors(
    kspace_full: np.ndarray,
    mask_func: MaskFunc,
    crop: tuple[int, int],
    seed: int | None = None,
) -> dict:
    """turn one fully-sampled complex kspace slice into a training sample dict"""
    col_mask = mask_func(kspace_full.shape[-1], seed=seed)
    mask = expand_mask(col_mask, kspace_full.shape)
    masked_kspace = kspace_full * mask

    # image-domain quantities centre-cropped to a fixed size for loss/metrics
    crop = (min(crop[0], kspace_full.shape[-2]), min(crop[1], kspace_full.shape[-1]))
    target = np.abs(center_crop(ifft2c(kspace_full), crop)).astype(np.float32)
    zf = np.abs(center_crop(ifft2c(masked_kspace), crop)).astype(np.float32)

    return {
        "masked_kspace": torch.from_numpy(masked_kspace.astype(np.complex64)),
        "mask": torch.from_numpy(mask.astype(np.float32))[None],
        "target": torch.from_numpy(target)[None],
        "zf_image": torch.from_numpy(zf)[None],
        "max_value": float(target.max()),
    }


class SyntheticSliceDataset(Dataset):
    """fixed-size synthetic phantoms for smoke tests ci and offline demos"""

    def __init__(
        self,
        mask_func: MaskFunc,
        length: int = 64,
        height: int = 256,
        width: int = 256,
        crop: tuple[int, int] = (256, 256),
        seed: int = 0,
    ):
        from ..fft import fft2c  # local import keeps module load cheap

        self._fft2c = fft2c
        self.mask_func = mask_func
        self.length = length
        self.hw = (height, width)
        self.crop = crop
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        image = make_phantom(*self.hw, seed=self.seed + idx)
        kspace_full = self._fft2c(image.astype(np.complex64))
        # deterministic per-sample mask so validation is reproducible
        return make_sample_tensors(kspace_full, self.mask_func, self.crop, seed=self.seed + idx)


class FastMRISliceDataset(Dataset):
    """real fastmri single-coil knee slices read from .h5 files

    mask_func the undersampling mask function
    root directory of .h5 (e.g. singlecoil_val) mutually exclusive with files
    files explicit list of .h5 paths for carving a split out of one dir (split_fastmri_files)
    crop image-domain crop for target/zf (fastmri knee esc target is 320x320)
    center_slice_frac keep only this central fraction of slices (outer slices mostly noise)
    sample_limit cap total slices (use a subset per the spec)
    """

    def __init__(
        self,
        mask_func: MaskFunc,
        root: str | Path | None = None,
        files: list[str | Path] | None = None,
        crop: tuple[int, int] = (320, 320),
        center_slice_frac: float = 0.8,
        sample_limit: int | None = None,
        seed: int = 0,
    ):
        import h5py  # local import only needed for the real-data path

        self._h5py = h5py
        self.mask_func = mask_func
        self.crop = crop
        self.seed = seed

        if files is not None:
            file_list = [Path(f) for f in files]
        elif root is not None:
            file_list = sorted(Path(root).glob("*.h5"))
        else:
            raise ValueError("Provide either `root` (a directory) or `files` (a list).")
        if not file_list:
            raise FileNotFoundError(
                "No .h5 files found. Download the fastMRI single-coil knee subset and "
                "point `data.root` (or `data.split_dir`) at it (see README)."
            )

        self.index: list[tuple[Path, int]] = []
        for path in file_list:
            with h5py.File(path, "r") as f:
                if "kspace" not in f:
                    continue
                num_slices = f["kspace"].shape[0]
            lo = int(num_slices * (1 - center_slice_frac) / 2)
            hi = num_slices - lo
            self.index.extend((path, s) for s in range(lo, hi))

        if sample_limit is not None:
            rng = np.random.default_rng(seed)
            keep = rng.permutation(len(self.index))[:sample_limit]
            self.index = [self.index[i] for i in sorted(keep)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        path, slice_idx = self.index[idx]
        with self._h5py.File(path, "r") as f:
            kspace_full = np.asarray(f["kspace"][slice_idx]).astype(np.complex64)
        return make_sample_tensors(kspace_full, self.mask_func, self.crop, seed=self.seed + idx)


def build_dataset(data_cfg: dict, mask_func: MaskFunc, split: str = "train") -> Dataset:
    """factory synthetic or fastmri selected by data.source in the config"""
    source = data_cfg.get("source", "synthetic").lower()
    crop = tuple(data_cfg.get("crop", (320, 320)))
    if source == "synthetic":
        sizes = {"train": data_cfg.get("train_size", 64), "val": data_cfg.get("val_size", 16)}
        seeds = {"train": 0, "val": 100000}
        hw = tuple(data_cfg.get("synthetic_shape", (256, 256)))
        return SyntheticSliceDataset(
            mask_func,
            length=sizes.get(split, 16),
            height=hw[0],
            width=hw[1],
            crop=tuple(data_cfg.get("crop", (256, 256))),
            seed=seeds.get(split, 0),
        )
    if source == "fastmri":
        common = dict(
            crop=crop,
            center_slice_frac=data_cfg.get("center_slice_frac", 0.8),
            sample_limit=data_cfg.get(f"{split}_limit"),
            seed=0 if split == "train" else 100000,
        )
        if data_cfg.get("split_dir"):
            # carve train/val/test out of one directory (e.g. knee_singlecoil_val)
            splits = split_fastmri_files(
                data_cfg["split_dir"],
                tuple(data_cfg.get("split_fractions", (0.7, 0.15, 0.15))),
                seed=data_cfg.get("split_seed", 0),
            )
            return FastMRISliceDataset(mask_func, files=splits[split], **common)
        root = Path(data_cfg["root"]) / data_cfg.get(f"{split}_dir", f"singlecoil_{split}")
        return FastMRISliceDataset(mask_func, root=root, **common)
    raise ValueError(f"Unknown data.source {source!r} (use 'synthetic' or 'fastmri').")
