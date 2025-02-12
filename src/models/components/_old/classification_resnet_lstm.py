import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# reference: https://github.dev/macsifan/ResNet20
def conv3x3(in_channels, out_channels, stride=1):
    """
    return 3x3 Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Initialize basic ResidualBlock with forward propogation
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Initialize  ResNet with forward propogation
    """

    def __init__(self, in_chans, block, num_layers, channels):
        super(ResNet, self).__init__()
        self.in_channels = channels[0]
        self.conv = conv3x3(in_chans, channels[0])
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        strides = [1, 2, 2, 2]
        self.layers = nn.ModuleList()
        for channel, layer, stride in zip(channels, num_layers, strides):
            self.layers.append(self.make_layer(block, channel, layer, stride))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return out


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans=1,
        num_layers=[3, 3, 3],
        channels=[16, 32, 64],
        num_lstm_layers=1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = ResNet(in_chans, ResidualBlock, num_layers, channels)
        self.num_features = channels[-1]

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
        feats = self.encoder(x)

        # Reshape
        feats = feats.view(b, d, -1)

        return feats

    def forward(self, x):
        feats = self.forward_image_feats(x)

        x = self.lstm(feats)[0]
        x = x.mean(dim=1)
        x = self.head(x)

        return x
