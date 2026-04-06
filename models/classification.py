"""Classification model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder,VGG11EncoderNoBN
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + Classification Head."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head
        # After encoder: [B, 512, 7, 7] → AdaptivePool → [B, 512, 1, 1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),       # ensure fixed size
            nn.Flatten(),                        # [B, 512*7*7] = [B, 25088]

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),        # [B, 37]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Returns:
            Classification logits [B, num_classes]
        """
        features = self.encoder(x)       # [B, 512, 7, 7]
        logits = self.classifier(features)  # [B, 37]
        return logits
class VGG11ClassifierNoBN(nn.Module):
    """Classifier WITHOUT BatchNorm for comparison."""

    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5):
        super().__init__()
        self.encoder = VGG11EncoderNoBN(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))