import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch.nn import LayerNorm

import numpy as np
from functools import partial

from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers.factories import Conv
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.timm_2_5d_unet_hengck23 import Head

# from src.models.components.layers.conv3d_convnext import LayerNorm


class InterpolateConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=None,
        groups=1,
        bias=True,
        dilation=1,
    ):
        super().__init__()

        self.interp = partial(F.interpolate, scale_factor=2, mode="linear")

        ks = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        padding = ks // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.interp(x)
        x = self.conv(x)
        return x


class InterpolateConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=None,
        groups=1,
        bias=True,
        dilation=1,
    ):
        super().__init__()

        self.interp = partial(F.interpolate, scale_factor=2, mode="bilinear")

        ks = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        padding = ks // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.interp(x)
        x = self.conv(x)
        return x


class InterpolateConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        output_padding=None,
        groups=1,
        bias=True,
        dilation=1,
    ):
        super().__init__()

        self.interp = partial(F.interpolate, scale_factor=2, mode="trilinear")

        ks = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        padding = ks // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.interp(x)
        x = self.conv(x)
        return x


@Conv.factory_function("convtrans")
def convtrans_factory(dim: int) -> nn.Module:
    """
    Transposed convolutional layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the transposed convolutional layer

    Returnmode:
        ConvTranspose[dim]d
    """
    types = (InterpolateConv1d, InterpolateConv2d, InterpolateConv3d)
    return types[dim - 1]


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans,
        channels,
        strides=[2, 2, 2, 2],
        output_classmap=True,
        output_seg=True,
        output_det=True,
        norm_head="bn",
        act_head="relu",
        num_res_units=2,
        norm="instance",
        num_head_layers=0,
    ):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.output_classmap = output_classmap
        self.output_seg = output_seg
        self.output_det = output_det

        assert len(channels) == len(strides) + 1

        if norm == "group":
            norm = ("group", {"num_groups": 32})

        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=channels[0],
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

        if output_classmap:
            self.head_classmap = Head(
                channels[0], num_classes, num_layers=num_head_layers, norm=norm_head, act=act_head
            )
        if output_seg:
            self.head_seg = Head(
                channels[0], num_classes + 1, num_layers=num_head_layers, norm=norm_head, act=act_head
            )
        if output_det:
            self.head_det = Head(channels[0], 1, num_layers=num_head_layers, norm=norm_head, act=act_head)

    def forward(self, x):
        x = self.unet(x)

        outputs = {}

        if self.output_classmap:
            outputs["classmap"] = self.head_classmap(x)
        if self.output_seg:
            outputs["seg"] = self.head_seg(x)
        if self.output_det:
            outputs["det"] = self.head_det(x)

        return outputs
