"""Skip Architecture with Attention Gates between downsampling and upsampling layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate weighting encoder skip features conditioned on decoder gating signal."""
    def __init__(self, in_skip, in_gate, inter_channels=32):
        super().__init__()
        self.wx = nn.Conv2d(in_skip, inter_channels, 1, bias=False)
        self.wg = nn.Conv2d(in_gate, inter_channels, 1, bias=False)
        self.psi = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.wx(x) + self.wg(g))
        return x * alpha


def conv_block(in_c, out_c, k=3, pad="reflection"):
    padding = (k - 1) // 2
    pad_layer = nn.ReflectionPad2d(padding) if pad == "reflection" else nn.Identity()
    pad_len = 0 if pad == "reflection" else padding
    return nn.Sequential(
        pad_layer,
        nn.Conv2d(in_c, out_c, k, padding=pad_len),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SkipAttention(nn.Module):
    """DIP Skip encoder-decoder with Attention Gates between down and up branches."""
    def __init__(
        self,
        num_input_channels=32,
        num_output_channels=3,
        channels_down=(128, 128, 128, 128, 128),
        channels_up=(128, 128, 128, 128, 128),
        channels_skip=(4, 4, 4, 4, 4),
        pad="reflection",
        upsample_mode="bilinear",
    ):
        super().__init__()
        self.n_scales = len(channels_down)
        self.upsample_mode = upsample_mode

        self.downs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_c = num_input_channels
        for i in range(self.n_scales):
            self.downs.append(nn.Sequential(
                conv_block(prev_c, channels_down[i], 3, pad=pad),
                conv_block(channels_down[i], channels_down[i], 3, pad=pad),
            ))
            self.skips.append(conv_block(prev_c, channels_skip[i], 1, pad=pad))
            prev_c = channels_down[i]

        self.deepest = conv_block(channels_down[-1], channels_down[-1], 3, pad=pad)

        for i in reversed(range(self.n_scales)):
            deeper_c = channels_down[-1] if i == self.n_scales - 1 else channels_up[i + 1]
            self.attn_gates.insert(0, AttentionGate(channels_skip[i], deeper_c, inter_channels=max(channels_skip[i], 16)))
            self.ups.insert(0, nn.Sequential(
                conv_block(channels_skip[i] + deeper_c, channels_up[i], 3, pad=pad),
                conv_block(channels_up[i], channels_up[i], 1, pad=pad),
            ))

        self.final = nn.Sequential(
            nn.Conv2d(channels_up[0], num_output_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_feats = []
        curr = x
        for i in range(self.n_scales):
            skip_feats.append(self.skips[i](curr))
            curr = self.downs[i](curr)
            if i < self.n_scales - 1:
                curr = F.avg_pool2d(curr, 2)

        curr = self.deepest(curr)

        for i in reversed(range(self.n_scales)):
            if i < self.n_scales - 1:
                curr = F.interpolate(curr, size=skip_feats[i].shape[-2:], mode=self.upsample_mode, align_corners=False)
            attended_skip = self.attn_gates[i](skip_feats[i], curr)
            curr = torch.cat([attended_skip, curr], dim=1)
            curr = self.ups[i](curr)

        return self.final(curr)


def skip_attention(input_depth=32, output_channels=3, **kwargs):
    return SkipAttention(num_input_channels=input_depth, num_output_channels=output_channels, **kwargs)

