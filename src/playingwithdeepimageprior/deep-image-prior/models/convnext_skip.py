"""DIP ConvNeXt-Skip Architecture.

Combines the classic DIP U-Net / Skip hierarchy with modern ConvNeXt blocks:
- 7x7 Depthwise Convolution (large receptive field for global texture & edges)
- Inverted Bottleneck (1x1 pointwise expansion C -> 2*C)
- GELU non-linear activation
- 1x1 Pointwise projection (2*C -> C)
- Local Residual addition (x + F(x)) at every encoder and decoder stage
- Long skip connections across scales (Concat)
"""

import torch
import torch.nn as nn
from .common import Concat, conv, bn, act


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block: 7x7 depthwise -> BatchNorm -> 1x1 expand -> GELU -> 1x1 project + Residual."""
    def __init__(self, dim, kernel_size=7, expansion=2, pad='reflection'):
        super().__init__()
        to_pad = kernel_size // 2
        pad_layer = nn.ReflectionPad2d(to_pad) if pad == 'reflection' else nn.ZeroPad2d(to_pad)
        self.dw = nn.Sequential(
            pad_layer,
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, bias=False)
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


def convnext_skip(
    num_input_channels=32,
    num_output_channels=3,
    num_channels_down=[128, 128, 128, 128, 128],
    num_channels_up=[128, 128, 128, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    kernel_size=7,
    expansion=2,
    need_sigmoid=True,
    need_bias=True,
    pad='reflection',
    upsample_mode='bilinear',
    downsample_mode='stride',
    act_fun='LeakyReLU',
):
    """Builds a U-Net / Skip architecture powered by ConvNeXt inverted residual blocks."""
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels

    for i in range(n_scales):
        deeper = nn.Sequential()
        skip_branch = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_branch, deeper))
        else:
            model_tmp.add(deeper)

        # Skip pathway: 1x1 conv to reduce skip dimension + BN + act
        if num_channels_skip[i] != 0:
            skip_branch.add(conv(input_depth, num_channels_skip[i], 1, bias=need_bias, pad=pad))
            skip_branch.add(bn(num_channels_skip[i]))
            skip_branch.add(act(act_fun))

        # Downsampling transition (stride 2)
        deeper.add(conv(input_depth, num_channels_down[i], 3, stride=2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # ConvNeXt Block with 7x7 depthwise + inverted bottleneck + residual shortcut
        deeper.add(ConvNeXtBlock(num_channels_down[i], kernel_size=kernel_size, expansion=expansion, pad=pad))

        deeper_main = nn.Sequential()

        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        # Upsampling
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        # Decoder fusion: combine skip + upsampled features
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        # Decoder ConvNeXt Block
        model_tmp.add(ConvNeXtBlock(num_channels_up[i], kernel_size=kernel_size, expansion=expansion, pad=pad))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    # Final 1x1 projection to RGB
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

