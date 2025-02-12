import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch.nn import LayerNorm

import numpy as np
from functools import partial

from monai.networks.nets import SegResNet
from monai.networks.layers.factories import Conv
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans,
        init_filters=8,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm="group",
    ):
        super(Net, self).__init__()
        self.num_classes = num_classes

        if norm == "group":
            norm = ("GROUP", {"num_groups": 8})

        self.net = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_chans,
            out_channels=num_classes,
            act="RELU",
            norm=norm,
            use_conv_final=True,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode="nontrainable",
        )

    def forward(self, x):
        x = self.net(x)

        outputs = {}
        outputs["classmap"] = x

        return outputs
