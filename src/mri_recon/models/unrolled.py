"""stage 3 the unrolled physics-informed reconstruction network (the centerpiece)

deep-unrolling / variational-network idea and the direct bridge from pinn-style
sparse-data reconstruction to mri instead of one black-box pass unroll an iterative
reconstruction into a fixed number of cascades each cascade does two things

  1. learned regularization a small residual cnn (ConvBlock) cleans up the current
     image estimate
  2. data consistency a DataConsistency layer re-imposes the scanner measured kspace
     so the cleaned image cant drift from what was acquired

step 2 is the physics lets us trust the output as a measurement not a plausible
hallucination the whole point for quantitative biomarkers start with num_cascades=1
to get it training then add cascades more cascades = more denoise-then-enforce iters

refs schlemper et al dc-cnn (single-coil) sriram et al e2e-varnet (multi-coil) this
implements the single-coil dc-cnn form
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvBlock, DataConsistency, complex_abs, complex_to_chan, ifft2c


class CascadeBlock(nn.Module):
    """one unrolled iteration residual cnn denoiser then data consistency"""

    def __init__(self, chans: int, n_convs: int, learnable_dc: bool, init_v: float):
        super().__init__()
        self.denoiser = ConvBlock(chans=chans, n_convs=n_convs)
        self.dc = DataConsistency(learnable=learnable_dc, init_v=init_v)

    def forward(
        self, x: torch.Tensor, measured_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.denoiser(x)
        x = self.dc(x, measured_kspace, mask)
        return x


class UnrolledRecon(nn.Module):
    """a cascade of (denoiser -> data-consistency) blocks

    num_cascades number of unrolled iterations (depth)
    chans n_convs width and depth of each cascade cnn denoiser
    shared_weights if true reuse one cascade weights every iteration (a recurrent
        unroll far fewer parameters good when data is scarce)
    learnable_dc soft learnable data consistency vs hard (exact) dc
    init_v initial dc blend in (0 1) ~0.95 starts near hard dc
    """

    def __init__(
        self,
        num_cascades: int = 5,
        chans: int = 32,
        n_convs: int = 5,
        shared_weights: bool = False,
        learnable_dc: bool = True,
        init_v: float = 0.95,
    ):
        super().__init__()
        self.num_cascades = num_cascades
        if shared_weights:
            block = CascadeBlock(chans, n_convs, learnable_dc, init_v)
            self.cascades = nn.ModuleList([block for _ in range(num_cascades)])
        else:
            self.cascades = nn.ModuleList(
                CascadeBlock(chans, n_convs, learnable_dc, init_v)
                for _ in range(num_cascades)
            )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """masked_kspace (b h w) complex mask (b 1 h w) returns (b 2 h w)"""
        x = complex_to_chan(ifft2c(masked_kspace))  # zero-filled init as 2-channel
        for cascade in self.cascades:
            x = cascade(x, masked_kspace, mask)
        return x

    @staticmethod
    def magnitude(chan_image: torch.Tensor) -> torch.Tensor:
        """(b 2 h w) -> (b 1 h w) magnitude for loss and metrics"""
        return complex_abs(chan_image)
