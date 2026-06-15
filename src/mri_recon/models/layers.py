"""torch building blocks torch fft the data-consistency layer ssim loss

fft helpers and DataConsistency here mirror the numpy reference in mri_recon.fft
exactly same centered/orthonormal convention same hard/soft dc arithmetic the numpy
version is unit-tested in scripts/smoke_test.py this is the differentiable tensor
version the network trains through

complex images carried as 2-channel real tensors (b 2 h w) = [real imag]
kspace as a native complex tensor (b h w) masks real {0 1} tensors (b 1 h w)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIM = (-2, -1)


# --------------------------------------------------------------------------- #
# centered orthonormal fft (matches mri_recon.fft)
# --------------------------------------------------------------------------- #
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """image -> kspace complex in / complex out over the last two dims"""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=_DIM), dim=_DIM, norm="ortho"),
        dim=_DIM,
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """kspace -> image complex in / complex out over the last two dims"""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=_DIM), dim=_DIM, norm="ortho"),
        dim=_DIM,
    )


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    """(b 2 h w) real -> (b h w) complex"""
    return torch.complex(x[:, 0], x[:, 1])


def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    """(b h w) complex -> (b 2 h w) real"""
    return torch.stack([x.real, x.imag], dim=1)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """(b 2 h w) real channels -> (b 1 h w) magnitude"""
    return torch.sqrt((x**2).sum(dim=1, keepdim=True).clamp_min(1e-12))


# --------------------------------------------------------------------------- #
# data consistency the physics-informed layer
# --------------------------------------------------------------------------- #
class DataConsistency(nn.Module):
    """re-impose the measured kspace samples on the current image estimate

    hard dc (learnable=false) at every sampled location output kspace is set
    exactly to the measurement the net cant overwrite real data

    soft dc (learnable=true) blends predicted and measured kspace at sampled
    locations as k = (k_pred + lam * k_meas) / (1 + lam) with a learnable lam >= 0
    (schlemper et al dc-cnn) right when measurements are noisy lam -> inf recovers
    hard dc parameterise v = lam / (1 + lam) = sigmoid(theta) in (0 1) learn theta
    """

    def __init__(self, learnable: bool = True, init_v: float = 0.95):
        super().__init__()
        self.learnable = learnable
        if learnable:
            init_v = float(min(max(init_v, 1e-3), 1 - 1e-3))
            theta0 = torch.logit(torch.tensor(init_v))  # near-hard by default
            self.theta = nn.Parameter(theta0)

    def forward(
        self,
        image_chan: torch.Tensor,  # (b 2 h w)
        measured_kspace: torch.Tensor,  # (b h w) complex
        mask: torch.Tensor,  # (b 1 h w) real {0,1}
    ) -> torch.Tensor:
        img = chan_to_complex(image_chan)  # (b h w) complex
        k_pred = fft2c(img)
        m = mask[:, 0] > 0  # (b h w) bool
        if self.learnable:
            v = torch.sigmoid(self.theta)
            blended = k_pred + v * (measured_kspace - k_pred)
            k_out = torch.where(m, blended, k_pred)
        else:
            k_out = torch.where(m, measured_kspace, k_pred)
        out = ifft2c(k_out)
        return complex_to_chan(out)


# --------------------------------------------------------------------------- #
# residual cnn denoiser (one regularization step of the unrolled cascade)
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """a small residual cnn that denoises a 2-channel complex image

    the learned regularizer half of an unrolled iteration proposes a cleaner image
    which the following DataConsistency layer snaps back onto the measured data
    residual (predicts a correction not the image from scratch) trains more stably
    in a deep cascade
    """

    def __init__(self, chans: int = 32, n_convs: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(2, chans, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_convs - 2):
            layers += [nn.Conv2d(chans, chans, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(chans, 2, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)  # residual correction


# --------------------------------------------------------------------------- #
# ssim loss (train directly on the headline metric)
# --------------------------------------------------------------------------- #
class SSIMLoss(nn.Module):
    """1 - ssim the fastmri training loss operates on magnitude (b 1 h w)"""

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        self.cov_norm = win_size**2 / (win_size**2 - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        data_range = data_range.view(-1, 1, 1, 1)
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(x, self.w)
        uy = F.conv2d(y, self.w)
        uxx = F.conv2d(x * x, self.w)
        uyy = F.conv2d(y * y, self.w)
        uxy = F.conv2d(x * y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        a1 = 2 * ux * uy + c1
        a2 = 2 * vxy + c2
        b1 = ux * ux + uy * uy + c1
        b2 = vx + vy + c2
        ssim = (a1 * a2) / (b1 * b2)
        return 1 - ssim.mean()
