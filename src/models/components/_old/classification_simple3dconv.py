import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        in_chans=1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv3d(in_chans, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)

        return x
