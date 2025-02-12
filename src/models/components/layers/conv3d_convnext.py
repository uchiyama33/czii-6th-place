import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class LayerNorm2d(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def downsample_conv(in_channels: int, out_channels: int, stride_xy: int = 2, stride_z: int = 1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride=(stride_z, stride_xy, stride_xy), padding=1),
        LayerNorm(out_channels, data_format="channels_first"),
    )


class ResidualConv3DConvNeXt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding="same",
            groups=out_channels,
        )
        self.norm1 = LayerNorm(out_channels, data_format="channels_first")

        # Depthwise convolution (spatial)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels * 4,
            kernel_size=1,
            stride=1,
            padding="valid",
        )
        self.act2 = nn.GELU()

        # Final pointwise convolution
        self.conv3 = nn.Conv3d(
            out_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="valid",
        )

        if in_channels != out_channels:
            self.downsamp = downsample_conv(in_channels, out_channels, stride_xy=1)
        else:
            self.downsamp = nn.Identity()

    def forward(self, x: torch.Tensor):
        shortcut = x

        # First 1x1 Conv -> LayerNorm -> GELU
        x = self.conv1(x)
        x = self.norm1(x)

        # Depthwise Conv -> LayerNorm -> GELU
        x = self.conv2(x)
        x = self.act2(x)

        # Last 1x1 Conv -> LayerNorm
        x = self.conv3(x)

        # Add residual connection
        x += self.downsamp(shortcut)

        return x
