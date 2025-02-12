import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from functools import partial

import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.layers.conv3d_convnext import ResidualConv3DConvNeXt, LayerNorm

# https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder/input?scriptVersionId=207638234&select=model2.py
# v3: stageごとに3d convを追加


def get_norm(norm):
    if norm == "bn":
        return nn.BatchNorm3d
    elif norm == "ln":
        return partial(LayerNorm, data_format="channels_first")
    elif norm == "gn":
        return lambda ch: nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
    elif norm == "in":
        return nn.InstanceNorm3d
    else:
        assert False, f"Unsupported norm type: {norm}"


def get_act(act):
    if act == "relu":
        return nn.ReLU
    elif act == "gelu":
        return nn.GELU
    else:
        assert False, f"Unsupported act type: {act}"


# ------------------------------------------------
# 3d decoder
class MyDecoderBlock3d(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
        norm="ln",
        act="gelu",
    ):
        super().__init__()
        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channel),
            act_layer(),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channel),
            act_layer(),
        )
        self.attention2 = nn.Identity()

    def forward(
        self,
        x,
        skip=None,
        depth_scaling=2,
        spatial_scaling=2,
    ):
        x = F.interpolate(x, scale_factor=(depth_scaling, spatial_scaling, spatial_scaling), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder3d(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
        depth_scaling,
        spatial_scaling,
        norm="ln",
        act="gelu",
        checkpoint=False,
    ):
        super().__init__()
        self.center = nn.Identity()
        self.depth_scaling = depth_scaling
        self.spatial_scaling = spatial_scaling
        self.checkpoint = checkpoint

        i_channel = [
            in_channel,
        ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o, norm=norm, act=act) for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            if self.checkpoint:
                d = checkpoint.checkpoint(block, d, s, self.depth_scaling[i], self.spatial_scaling[i])
            else:
                d = block(d, s, self.depth_scaling[i], self.spatial_scaling[i])
            decode.append(d)
        last = d
        return last, decode


class ConvDepth(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, depth_stride=1, padding=1):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size=(kernel_size, 1, kernel_size),
                stride=(depth_stride, 1, 1),
                padding=(padding, 0, padding),
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=(depth_stride, 1, 1),
                padding=(padding, padding, 0),
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x = x1 + x2
        return x


def conv_in_depth(x, B, depth_scaling):
    bd, c, h, w = x.shape
    x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
    x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
    x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    return x, x1


def get_efficinentnet_blocks(info):
    blocks = []
    for stage_info in info:
        block = int(stage_info["module"].split(".")[1])
        blocks.append(block)
    return blocks


def Head(in_channel, out_channel, num_layers=2, norm="ln", act="gelu"):
    norm_layer = get_norm(norm)
    act_layer = get_act(act)

    layers = []
    for _ in range(num_layers):
        layers += [
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1),
            norm_layer(in_channel),
            act_layer(),
        ]
    layers += [nn.Conv3d(in_channel, out_channel, kernel_size=1)]
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(
        self,
        model_name,
        in_chans,
        num_classes,
        pretrained=False,
        grad_checkpointing=False,
        decoder_dim=[256, 128, 64, 32, 16],
        output_classmap=True,
        output_seg=True,
        output_det=True,
        norm_decoder="bn",
        act_decoder="relu",
        num_middle_conv=0,
        num_head_layers=0,
    ):
        super(Net, self).__init__()
        self.model_name = model_name
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.output_classmap = output_classmap
        self.output_seg = output_seg
        self.output_det = output_det

        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            features_only=True,
        )
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing()

        encoder_dim = self.encoder.feature_info.channels()

        if "convnext" in model_name:
            encoder_dim.insert(1, encoder_dim[0])
            self.depth_scaling = [2, 2, 2, 2, 1]
            self.spatial_scaling = [2, 2, 2, 1, 4]
        elif "efficientnet" in model_name:
            self.depth_scaling = [2, 2, 2, 2, 1]
            self.spatial_scaling = [2, 2, 2, 2, 2]
            self.feature_blocks = get_efficinentnet_blocks(self.encoder.feature_info.info)
        elif "resnet" in model_name:
            self.depth_scaling = [2, 2, 2, 2, 1]
            self.spatial_scaling = [2, 2, 2, 2, 2]
        else:
            raise ValueError(f"No encode function for model {model_name}")

        assert len(encoder_dim) == len(
            decoder_dim
        ), f"len(encoder_dim)={len(encoder_dim)} != len(decoder_dim)={len(decoder_dim)}"

        self.middle_conv_pool = nn.ModuleList(
            [
                ConvDepth(
                    encoder_dim[i],
                    encoder_dim[i],
                    kernel_size=3,
                    depth_stride=self.depth_scaling[i],
                    padding=1,
                )
                for i in range(len(self.depth_scaling))
            ]
        )
        self.middle_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvDepth(encoder_dim[i], encoder_dim[i], kernel_size=3, padding=1)
                        for _ in range(num_middle_conv)
                    ]
                )
                for i in range(len(self.depth_scaling))
            ]
        )

        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            depth_scaling=self.depth_scaling[::-1],
            spatial_scaling=self.spatial_scaling,
            norm=norm_decoder,
            act=act_decoder,
            checkpoint=grad_checkpointing,
        )
        if output_classmap:
            self.head_classmap = Head(
                decoder_dim[-1], num_classes, num_layers=num_head_layers, norm=norm_decoder, act=act_decoder
            )
        if output_seg:
            self.head_seg = Head(
                decoder_dim[-1],
                num_classes + 1,
                num_layers=num_head_layers,
                norm=norm_decoder,
                act=act_decoder,
            )
        if output_det:
            self.head_det = Head(
                decoder_dim[-1], 1, num_layers=num_head_layers, norm=norm_decoder, act=act_decoder
            )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * D, C, H, W)

        encode = self.encode(x, B)
        last, decode = self.decoder(feature=encode[-1], skip=encode[:-1][::-1] + [None])

        outputs = {}

        if self.output_classmap:
            outputs["classmap"] = self.head_classmap(last)
        if self.output_seg:
            outputs["seg"] = self.head_seg(last)
        if self.output_det:
            outputs["det"] = self.head_det(last)

        return outputs

    def encode(self, x, B):
        if "efficientnet" in self.model_name:
            return self.encode_efficientnet(x, B)
        elif "resnet" in self.model_name:
            return self.encode_resnet(x, B)
        elif "convnext" in self.model_name:
            return self.encode_convnext(x, B)
        else:
            raise ValueError(f"No encode function for model {self.model_name}")

    def conv_in_depth(self, x, B, stage):
        depth_scaling = self.depth_scaling[stage]
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = self.middle_conv_pool[stage](x1)
        for conv in self.middle_conv[stage]:
            x1 = conv(x1)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    def encode_efficientnet(self, x, B):
        encode = []

        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)

        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.feature_blocks:
                x, x1 = self.conv_in_depth(x, B, len(encode))
                encode.append(x1)

        return encode

    def encode_resnet(self, x, B):
        encode = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x, x1 = self.conv_in_depth(x, B, 0)
        encode.append(x1)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.encoder.layer1(x)
        x, x1 = self.conv_in_depth(x, B, 1)
        encode.append(x1)

        x = self.encoder.layer2(x)
        x, x1 = self.conv_in_depth(x, B, 2)
        encode.append(x1)

        x = self.encoder.layer3(x)
        x, x1 = self.conv_in_depth(x, B, 3)
        encode.append(x1)

        x = self.encoder.layer4(x)
        x, x1 = self.conv_in_depth(x, B, 4)
        encode.append(x1)

        return encode

    def encode_convnext(self, x, B):
        encode = []

        x = self.encoder.stem_0(x)
        x = self.encoder.stem_1(x)
        x, x1 = self.conv_in_depth(x, B, 0)
        encode.append(x1)

        x = self.encoder.stages_0(x)
        x, x1 = self.conv_in_depth(x, B, 1)
        encode.append(x1)

        x = self.encoder.stages_1(x)
        x, x1 = self.conv_in_depth(x, B, 2)
        encode.append(x1)

        x = self.encoder.stages_2(x)
        x, x1 = self.conv_in_depth(x, B, 3)
        encode.append(x1)

        x = self.encoder.stages_3(x)
        x, x1 = self.conv_in_depth(x, B, 4)
        encode.append(x1)

        return encode
