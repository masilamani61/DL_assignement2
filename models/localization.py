"""Localization model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based bounding box regressor.
    
    Encoder outputs [B, 512, 7, 7] bottleneck.
    Regression head predicts [xc, yc, w, h] normalized to [0, 1].
    """

    def __init__(self, in_channels: int = 3, freeze_encoder: bool = False):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Optionally freeze encoder weights
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),       # [B, 512, 7, 7]
            nn.Flatten(),                        # [B, 25088]

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 4),                 # [B, 4]
            nn.Sigmoid(),                        # normalize output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            Bounding box [B, 4] in (xc, yc, w, h) format, values in [0, 1]
        """
        features = self.encoder(x)          # [B, 512, 7, 7]
        bbox = self.regressor(features)     # [B, 4]
        return bbox

    def load_encoder_weights(self, classifier_checkpoint: str, device: str = "cpu"):
        """Load encoder weights from a trained classifier checkpoint."""
        checkpoint = torch.load(classifier_checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract only encoder weights
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder weights from {classifier_checkpoint}")