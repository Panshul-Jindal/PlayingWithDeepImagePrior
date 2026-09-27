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
	def __init__(self, channels, out_channels):
		super().__init__()
		self.norm = nn.LayerNorm(channels * 4)
		self.reduction = nn.Linear(channels * 4, out_channels, bias=False)

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


class _SwinSkipStage(nn.Module):
	def __init__(self, input_channels, skip_channels, deeper_channels, output_channels,
				 depth, heads, window_size):
		super().__init__()
		self.skip = nn.Sequential(
			nn.Conv2d(input_channels, skip_channels, 1),
			nn.BatchNorm2d(skip_channels),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.fuse = nn.Conv2d(skip_channels + deeper_channels, output_channels, 1)
		self.stage = _SwinStage(output_channels, depth, heads, window_size)

	def forward(self, x, deeper):
		skip = self.skip(x)
		deeper = F.interpolate(deeper, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		return self.stage(self.fuse(torch.cat([skip, deeper], dim=1)))


class SwinSkipImagePrior(nn.Module):
	"""Swin encoder-decoder with a recursive skip branch at every scale."""

	def __init__(self, input_depth, output_channels=3, patch_size=4, embed_dim=32,
				 depth=2, num_heads=4, window_size=8, num_scales=4,
				 num_channels_skip=None, need_sigmoid=True, **kwargs):
		super().__init__()
		if num_scales < 2:
			raise ValueError("num_scales must be at least 2")

		self.channels = [embed_dim * 2 ** index for index in range(num_scales)]
		if num_channels_skip is None:
			num_channels_skip = self.channels
		if len(num_channels_skip) != num_scales:
			raise ValueError("num_channels_skip must match num_scales")

		heads = [max(1, min(num_heads, channel // 32)) for channel in self.channels]
		self.patch_embed = nn.Conv2d(input_depth, self.channels[0], patch_size, patch_size)
		self.encoder = nn.ModuleList([
			_SwinStage(channel, depth, heads[index], window_size)
			for index, channel in enumerate(self.channels)
		])
		self.merge = nn.ModuleList([
			_PatchMerging(self.channels[index], self.channels[index + 1])
			for index in range(num_scales - 1)
		])
		self.decoder = nn.ModuleList([
			_SwinSkipStage(
				self.channels[index],
				num_channels_skip[index],
				self.channels[index + 1] if index < num_scales - 1 else self.channels[index],
				self.channels[index],
				depth,
				heads[index],
				window_size,
			)
			for index in range(num_scales)
		])
		self.output = nn.Conv2d(self.channels[0], output_channels, 1)
		self.need_sigmoid = need_sigmoid

	def _forward_scale(self, index, x):
		encoded = self.encoder[index](x)
		if index == len(self.encoder) - 1:
			deeper = encoded
		else:
			deeper = self._forward_scale(index + 1, self.merge[index](encoded))
		return self.decoder[index](x, deeper)

	def forward(self, x):
		input_size = x.shape[-2:]
		x = self.patch_embed(x)
		x = self._forward_scale(0, x)
		x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
		x = self.output(x)
		return torch.sigmoid(x) if self.need_sigmoid else x


def swin_skip(input_depth, output_channels, patch_size=4, embed_dim=32, depth=2,
			  num_heads=4, window_size=8, num_scales=4, num_channels_skip=None,
			  need_sigmoid=True, **kwargs):
	return SwinSkipImagePrior(
		input_depth,
		output_channels,
		patch_size,
		embed_dim,
		depth,
		num_heads,
		window_size,
		num_scales,
		num_channels_skip,
		need_sigmoid,
		**kwargs,
	)