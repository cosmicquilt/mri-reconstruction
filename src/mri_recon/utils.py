"""reproducibility and config plumbing (numpy-safe torch imported lazily)"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """seed every rng so a run is reproducible end to end"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not installed (numpy-only paths)
        pass


def get_device(prefer: str = "auto"):
    """resolve a torch device prefer in {auto cuda cpu tpu}

    auto picks cuda if present else cpu
    tpu/xla is opt-in only (pass tpu) reconstruction uses complex ffts not
    reliably supported on xla so a gpu is recommended for stages 2-3
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("tpu", "xla"):
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "TPU/XLA requested but torch_xla is unavailable, or the complex-FFT "
                "model is not XLA-compatible. Use a GPU runtime for reconstruction."
            ) from exc
    if torch.cuda.is_available() and prefer in ("auto", "cuda", "gpu"):
        return torch.device("cuda")
    return torch.device("cpu")


def configure_backend(device, allow_tf32: bool = True) -> None:
    """enable safe gpu speedups tf32 matmuls (ampere+) and cudnn autotuning

    tf32 big throughput boost on a100/h100/l4 at negligible accuracy cost
    cudnn.benchmark picks fastest conv algos for the fixed input sizes
    no-op on cpu/tpu
    """
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True


def _coerce(value: str) -> Any:
    """best-effort string -> python value for cli overrides

    handles scalars and [a,b,c] list literals (no spaces) so a list field can
    be overridden inline e.g. --set eval.methods=[zero_filled,unet]
    """
    v = value.strip()
    if len(v) >= 2 and v[0] == "[" and v[-1] == "]":
        inner = v[1:-1].strip()
        return [_coerce(x.strip()) for x in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """apply a.b.c=value dotted overrides (from the cli) onto a config dict"""
    for override in overrides or []:
        key, _, raw = override.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                # missing or present-but-none (an empty key: in yaml) make a dict
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """load a yaml config and apply optional dotted cli overrides"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return apply_overrides(cfg, overrides)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def center_crop_tensor(x, shape: tuple[int, int]):
    """centre-crop the last two dims of a torch tensor to shape (clamped)"""
    h, w = x.shape[-2:]
    th, tw = min(shape[0], h), min(shape[1], w)
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top : top + th, left : left + tw]


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
