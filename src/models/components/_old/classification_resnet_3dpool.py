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


def pool_in_depth(x, B, depth_scaling, max_or_avg="avg", kernel_size: int = None):
    if max_or_avg == "max":
        pool_fn = F.max_pool3d
    elif max_or_avg == "avg":
        pool_fn = F.avg_pool3d
    else:
        assert False, f"Unsupported pool type: {max_or_avg}"

    if depth_scaling == 1:
        return x

    if kernel_size is None:
        kernel_size = (depth_scaling, 1, 1)
        padding = 0
    else:
        kernel_size = (kernel_size, 1, 1)
        padding = ((kernel_size[0] - 1) // 2, 0, 0)
    bd, c, h, w = x.shape
    x = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
    x = pool_fn(x, kernel_size=kernel_size, stride=(depth_scaling, 1, 1), padding=padding)
    x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    return x


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

    def __init__(self, in_chans, block, num_layers, channels, depth_pool_sizes):
        super(ResNet, self).__init__()
        self.in_channels = channels[0]
        self.conv = conv3x3(in_chans, channels[0])
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        strides = [1, 2, 2, 2]
        self.layers = nn.ModuleList()
        for channel, layer, stride in zip(channels, num_layers, strides):
            self.layers.append(self.make_layer(block, channel, layer, stride))
        self.depth_pool_sizes = depth_pool_sizes

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

    def forward(self, x, batch_size):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = pool_in_depth(out, batch_size, self.depth_pool_sizes[0], max_or_avg="avg")

        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = pool_in_depth(out, batch_size, self.depth_pool_sizes[i + 1], max_or_avg="avg")

        return out


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans=1,
        num_layers=[3, 3, 3],
        channels=[16, 32, 64],
        depth_pool_sizes=[1, 1, 2, 2],
        num_lstm_layers=1,
    ):
        super().__init__()
        assert len(num_layers) == len(channels)
        assert len(channels) + 1 == len(depth_pool_sizes)

        self.num_classes = num_classes

        self.encoder = ResNet(in_chans, ResidualBlock, num_layers, channels, depth_pool_sizes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Linear(channels[-1], num_classes)

    def forward_image_feats(self, x):
        b, c, d, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * d, c, h, w)
        feats = self.encoder(x, b)

        # Reshape
        bd, c, h, w = feats.shape
        feats = feats.reshape(b, -1, c, h, w).permute(0, 2, 1, 3, 4)
        feats = self.avg_pool(feats)
        feats = torch.flatten(feats, 1)

        return feats

    def forward(self, x):
        x = self.forward_image_feats(x)
        x = self.head(x)

        return x
