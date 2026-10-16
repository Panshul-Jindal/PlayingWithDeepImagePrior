"""DIP Skip Architecture with Depthwise-Separable Convolutions.

This is a strict 1-to-1 drop-in replacement for models/skip.py:
- Preserves the exact recursive encoder-decoder hierarchy, skip pathways, and Concat blocks.
- Preserves identical feature map shapes (H x W x C) at every layer.
- Preserves identical normalization (bn) and activation (act) placements.
- Preserves strided-convolution downsampling (stride=2) and upsampling modes.
- The ONLY modification is replacing standard convolutions (kernel_size > 1)
  with depthwise (spatial filtering per channel) + pointwise (1x1 channel mixing).
"""

import torch
import torch.nn as nn
from .common import *


def depthwise_conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """Replaces standard convolution with depthwise-separable convolution when kernel_size > 1."""
    if kernel_size == 1:
        return conv(in_f, out_f, kernel_size=1, stride=stride, bias=bias, pad=pad, downsample_mode=downsample_mode)

    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=in_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    # Depthwise convolution (spatial filtering per channel, with downsampling stride)
    dw = nn.Conv2d(in_f, in_f, kernel_size, stride=stride, padding=to_pad, groups=in_f, bias=False)
    # Pointwise convolution (1x1 cross-channel linear combination)
    pw = nn.Conv2d(in_f, out_f, 1, stride=1, padding=0, bias=bias)

    layers = filter(lambda x: x is not None, [padder, dw, downsampler, pw])
    return nn.Sequential(*layers)


def dpwise_skip(
    num_input_channels=2,
    num_output_channels=3,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=1,
    need_sigmoid=True,
    need_bias=True,
    pad='zero',
    upsample_mode='nearest',
    downsample_mode='stride',
    act_fun='LeakyReLU',
    need1x1_up=True,
):
    """Exact replica of models.skip.skip, replacing standard convolutions with depthwise-separable convs."""
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_branch = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_branch, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        # Skip branch
        if num_channels_skip[i] != 0:
            skip_branch.add(depthwise_conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_branch.add(bn(num_channels_skip[i]))
            skip_branch.add(act(act_fun))

        # Deeper branch: Downsampling conv (stride=2)
        deeper.add(depthwise_conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # Deeper branch: Second conv (stride=1)
        deeper.add(depthwise_conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        # Upsampling
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        # Fusion conv
        model_tmp.add(depthwise_conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(depthwise_conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(depthwise_conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
