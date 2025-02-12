import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from functools import partial

from monai.networks.nets import UNet, SwinUNETR
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.timm_2_5d_unet_hengck23 import Head


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans,
        img_size,
        feature_size=12,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        out_channels=16,
        norm="instance",
        use_checkpoint=False,
        output_classmap=True,
        output_seg=False,
        output_det=False,
        norm_head="bn",
        act_head="relu",
        num_head_layers=0,
    ):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.output_classmap = output_classmap
        self.output_seg = output_seg
        self.output_det = output_det

        self.unet = SwinUNETR(
            img_size=img_size,
            in_channels=in_chans,
            depths=depths,
            num_heads=num_heads,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=3,
            use_checkpoint=use_checkpoint,
            norm_name=norm,
        )

        if output_classmap:
            self.head_classmap = Head(
                out_channels, num_classes, num_layers=num_head_layers, norm=norm_head, act=act_head
            )
        if output_seg:
            self.head_seg = Head(
                out_channels, num_classes + 1, num_layers=num_head_layers, norm=norm_head, act=act_head
            )
        if output_det:
            self.head_det = Head(out_channels, 1, num_layers=num_head_layers, norm=norm_head, act=act_head)

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
