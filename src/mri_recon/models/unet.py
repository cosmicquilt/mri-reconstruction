"""stage 2 the fastmri unet baseline

image-domain denoiser takes the (normalized) zero-filled magnitude image (the
aliased stage-1 recon) and learns to map it to the fully-sampled ground truth knows
nothing about kspace or the forward physics just learns what aliased knee images
should look like when fixed a strong honest baseline the physics-informed unrolled
net (stage 3) has to beat

faithful to fastmri.models.Unet symmetric encoder/decoder instance norm leakyrelu
average-pool downsampling transpose-conv upsampling with skip connections
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, 2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnetModel(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.down_sample_layers = nn.ModuleList([_ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(_ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = _ConvBlock(ch, ch * 2, drop_prob)

        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(_TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(_ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        self.up_transpose_conv.append(_TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(_ConvBlock(ch * 2, ch, drop_prob), nn.Conv2d(ch, out_chans, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack = []
        output = x
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            # pad if encoder/decoder feature maps are off by one (odd input sizes)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if any(padding):
                output = F.pad(output, padding, mode="reflect")
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output
