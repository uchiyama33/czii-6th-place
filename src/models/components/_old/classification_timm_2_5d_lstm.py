import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.conv3d_convnext import ResidualConv3DConvNeXt, downsample_conv, LayerNorm2d


class Net(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        pretrained=True,
        in_chans=1,
        grad_checkpointing=False,
        num_lstm_layers=1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing()
        self.output_fmt = getattr(self.encoder, "output_fmt", "NCHW")
        self.num_features = self.encoder.feature_info.channels()[-1]

        self.lstm = nn.LSTM(
            self.num_features,
            self.num_features,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(self.num_features * 2, num_classes)

    def forward_image_feats(self, x):
        b, c, d, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * d, c, h, w)
        feats = self.encoder(x)[-1]

        if self.output_fmt == "NHWC":
            feats = feats.permute(0, 3, 1, 2).contiguous()

        # GAP
        feats = F.adaptive_avg_pool2d(feats, (1, 1))

        # Reshape
        feats = feats.view(b, d, -1)

        return feats

    def forward(self, x):
        feats = self.forward_image_feats(x)

        x = self.lstm(feats)[0]
        x = x.mean(dim=1)
        x = self.head(x)

        return x
