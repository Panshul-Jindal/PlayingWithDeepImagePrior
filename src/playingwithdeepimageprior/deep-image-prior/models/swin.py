import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformerBlock


class _SwinStage(nn.Module):
	def __init__(self, channels, depth, heads, window_size):
		super().__init__()
		self.blocks = nn.ModuleList([
			SwinTransformerBlock(
				channels,
				heads,
				[window_size, window_size],
				[0, 0] if index % 2 == 0 else [window_size // 2, window_size // 2],
			)
			for index in range(depth)
		])

	def forward(self, x):
		x = x.permute(0, 2, 3, 1)
		for block in self.blocks:
			x = block(x)
		return x.permute(0, 3, 1, 2)


class _PatchMerging(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.norm = nn.LayerNorm(channels * 4)
		self.reduction = nn.Linear(channels * 4, channels * 2, bias=False)

	def forward(self, x):
		x = x.permute(0, 2, 3, 1)
		height, width = x.shape[1:3]
		x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
		x = torch.cat([
			x[:, 0::2, 0::2], x[:, 1::2, 0::2],
			x[:, 0::2, 1::2], x[:, 1::2, 1::2],
		], dim=-1)
		x = self.reduction(self.norm(x))
		return x.permute(0, 3, 1, 2)


class _DecoderStage(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels, depth, heads, window_size):
		super().__init__()
		self.fuse = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
		self.stage = _SwinStage(out_channels, depth, heads, window_size)

	def forward(self, x, skip):
		x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		return self.stage(self.fuse(torch.cat([x, skip], dim=1)))


class SwinImagePrior(nn.Module):
	def __init__(self, input_depth, output_channels=3, patch_size=4, embed_dim=32,
				 depth=2, num_heads=4, window_size=8, need_sigmoid=True, **kwargs):
		super().__init__()
		channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
		heads = [max(1, min(num_heads, channel // 32)) for channel in channels]

		self.patch_embed = nn.Conv2d(input_depth, channels[0], patch_size, patch_size)
		self.encoder = nn.ModuleList([
			_SwinStage(channel, depth, heads[index], window_size)
			for index, channel in enumerate(channels)
		])
		self.merge = nn.ModuleList([_PatchMerging(channel) for channel in channels[:-1]])
		self.decoder = nn.ModuleList([
			_DecoderStage(channels[3], channels[2], channels[2], depth, heads[2], window_size),
			_DecoderStage(channels[2], channels[1], channels[1], depth, heads[1], window_size),
			_DecoderStage(channels[1], channels[0], channels[0], depth, heads[0], window_size),
		])
		self.output = nn.Conv2d(channels[0], output_channels, 3, padding=1)
		self.need_sigmoid = need_sigmoid

	def forward(self, x):
		input_size = x.shape[-2:]
		x = self.patch_embed(x)
		skips = []
		for index, stage in enumerate(self.encoder):
			x = stage(x)
			if index < len(self.merge):
				skips.append(x)
				x = self.merge[index](x)

		for decoder, skip in zip(self.decoder, reversed(skips)):
			x = decoder(x, skip)

		x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
		x = self.output(x)
		return torch.sigmoid(x) if self.need_sigmoid else x


def swin(input_depth, output_channels, patch_size=4, embed_dim=32, depth=2,
		 num_heads=4, window_size=8, need_sigmoid=True, **kwargs):
	return SwinImagePrior(
		input_depth,
		output_channels,
		patch_size,
		embed_dim,
		depth,
		num_heads,
		window_size,
		need_sigmoid,
		**kwargs,
	)
