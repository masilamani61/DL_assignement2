"""Localization model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class SpatialAttention(nn.Module):
    """Spatial attention to focus on relevant regions."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class VGG11Localizer(nn.Module):
    """VGG11-based bounding box regressor with attention."""

    def __init__(self, in_channels: int = 3, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Spatial attention
        self.attention = SpatialAttention(512)
        self.training=False
        # Better regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = self.attention(features)
        bbox = self.regressor(features)     # [B, 4] normalized 0-1
        
        # Always convert to pixel space for output
        
        return bbox
    def load_encoder_weights(self, classifier_checkpoint: str, device: str = "cpu"):
        checkpoint = torch.load(classifier_checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder weights from {classifier_checkpoint}")