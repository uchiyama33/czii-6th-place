import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.conv3d_convnext import ResidualConv3DConvNeXt, downsample_conv, LayerNorm2d


class UNet3DDecoder(nn.Module):
    def __init__(self, encoder_channels, scale_factors):
        super(UNet3DDecoder, self).__init__()

        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(encoder_channels) - 1, 0, -1):
            scale = scale_factors[i - 1]
            self.upconvs.append(
                nn.ConvTranspose3d(
                    encoder_channels[i],
                    encoder_channels[i - 1],
                    kernel_size=(2, scale, scale),
                    stride=(2, scale, scale),
                )
            )
            self.convs.append(ResidualConv3DConvNeXt(encoder_channels[i - 1] * 2, encoder_channels[i - 1]))

    def forward(self, encoder_features):
        if encoder_features[0].dim() == 4:
            encoder_features = [f.unsqueeze(0) for f in encoder_features]

        x = encoder_features[-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[-(i + 2)]], dim=1)
            x = self.convs[i](x)

        return x


class Net(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        channels_3d=[8, 16, 32, 64, 128, 256],
        num_3d_layer=1,
        pretrained=True,
        in_chans=1,
        stem_chans=30,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.extra_stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chans, 3, 2, 1),
            nn.BatchNorm2d(stem_chans),
        )
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=stem_chans,
            features_only=True,
        )
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing()
        self.output_fmt = getattr(self.encoder, "output_fmt", "NCHW")
        self.num_features = [stem_chans] + self.encoder.feature_info.channels()
        assert len(channels_3d) == len(self.num_features)

        # Determine the scale factors for the final upsampling layer
        reductions = [1] + [info["reduction"] for info in self.encoder.feature_info.info]
        self.scale_factors = [reductions[i + 1] // reductions[i] for i in range(len(reductions) - 1)]

        self.conv_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.num_features[i], channels_3d[i], 1, stride=1),
                    nn.BatchNorm2d(channels_3d[i]),
                    nn.ReLU(),
                )
                for i in range(len(channels_3d))
            ]
        )
        self.conv_3d_middle = nn.ModuleList(
            [
                nn.Sequential(
                    (
                        nn.Sequential(
                            *[
                                downsample_conv(channels_3d[i], channels_3d[i], stride_xy=1, stride_z=2)
                                for _ in range(i)
                            ]
                        )
                        if i != 0
                        else nn.Identity()
                    ),
                    *[
                        ResidualConv3DConvNeXt(
                            channels_3d[i],
                            channels_3d[i],
                        )
                        for j in range(num_3d_layer)
                    ],
                )
                for i in range(len(channels_3d))
            ]
        )
        self.unet_decoder = UNet3DDecoder(channels_3d, self.scale_factors)

        self.head_seg = nn.Conv3d(channels_3d[0], num_classes, kernel_size=1)
        self.head_map = nn.Conv3d(channels_3d[0], 1, kernel_size=1)

    def forward_image_feats(self, x):
        b, c, d, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * d, c, h, w)
        with torch.no_grad():
            x = F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        x_stem = self.extra_stem(x)
        feats = self.encoder(x_stem)
        feats = [x_stem] + feats

        if self.output_fmt == "NHWC":
            for i in range(len(feats)):
                feats[i] = feats[i].permute(0, 3, 1, 2).contiguous()

        for i in range(len(feats)):
            feats[i] = self.conv_proj[i](feats[i])

        feats = [f.view(b, d, *f.shape[1:]) for f in feats]
        feats = [f.transpose(1, 2) for f in feats]  # B, C, D, H, W

        return feats

    def forward(self, x):
        feats = self.forward_image_feats(x)
        feats = [conv(f) for f, conv in zip(feats, self.conv_3d_middle)]

        x = self.unet_decoder(feats)
        seg = self.head_seg(x)
        map = self.head_map(x)

        return seg, map
