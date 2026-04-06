"""VGG11 encoder"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with skip connections for U-Net."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Block 1 - output: [B, 64, 112, 112]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 - output: [B, 128, 56, 56]
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 - output: [B, 256, 28, 28]
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 - output: [B, 512, 14, 14]
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5 - output: [B, 512, 7, 7]
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, 224, 224]
            return_features: if True, return skip maps for U-Net decoder

        Returns:
            - if return_features=False: bottleneck tensor [B, 512, 7, 7]
            - if return_features=True: (bottleneck, feature_dict)
        """
        s1 = self.block1(x)        # [B, 64,  112, 112]
        x  = self.pool1(s1)

        s2 = self.block2(x)        # [B, 128, 56,  56]
        x  = self.pool2(s2)

        s3 = self.block3(x)        # [B, 256, 28,  28]
        x  = self.pool3(s3)

        s4 = self.block4(x)        # [B, 512, 14,  14]
        x  = self.pool4(s4)

        s5 = self.block5(x)        # [B, 512, 7,   7]
        x  = self.pool5(s5)        # [B, 512, 7,   7] bottleneck

        if return_features:
            features = {
                "s1": s1,  # skip1
                "s2": s2,  # skip2
                "s3": s3,  # skip3
                "s4": s4,  # skip4
                "s5": s5,  # skip5
            }
            return x, features

        return x
class VGG11EncoderNoBN(nn.Module):
    """VGG11 encoder WITHOUT BatchNorm for comparison."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, return_features=False):
        s1 = self.block1(x);  x = self.pool1(s1)
        s2 = self.block2(x);  x = self.pool2(s2)
        s3 = self.block3(x);  x = self.pool3(s3)
        s4 = self.block4(x);  x = self.pool4(s4)
        s5 = self.block5(x);  x = self.pool5(s5)

        if return_features:
            return x, {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
        return x