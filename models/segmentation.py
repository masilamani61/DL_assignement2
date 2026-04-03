"""Segmentation model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class DecoderBlock(nn.Module):
    """Single decoder block: TransposedConv upsample + skip concat + conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Transposed conv to upsample spatial dims x2
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        # After concat with skip: channels = out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # concat along channel dim
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation using VGG11 encoder.

    Encoder skip connections:
        s1: [B, 64,  224, 224]
        s2: [B, 128, 112, 112]
        s3: [B, 256,  56,  56]
        s4: [B, 512,  28,  28]
        s5: [B, 512,  14,  14]
    Bottleneck: [B, 512, 7, 7]
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Bottleneck conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks (bottom to top)
        # Each: (in_channels, skip_channels, out_channels)
        self.dec5 = DecoderBlock(1024, 512, 512)  # bottleneck + s5
        self.dec4 = DecoderBlock(512,  512, 256)  # dec5 + s4
        self.dec3 = DecoderBlock(256,  256, 128)  # dec4 + s3
        self.dec2 = DecoderBlock(128,  128, 64)   # dec3 + s2
        self.dec1 = DecoderBlock(64,   64,  32)   # dec2 + s1

        # Final 1x1 conv to get class logits
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # Bottleneck
        x = self.bottleneck(bottleneck)     # [B, 1024, 7, 7]

        # Decoder with skip connections
        x = self.dec5(x, feats["s5"])       # [B, 512, 14, 14]
        x = self.dec4(x, feats["s4"])       # [B, 256, 28, 28]
        x = self.dec3(x, feats["s3"])       # [B, 128, 56, 56]
        x = self.dec2(x, feats["s2"])       # [B, 64,  112, 112]
        x = self.dec1(x, feats["s1"])       # [B, 32,  224, 224]

        # Final classification
        out = self.final_conv(x)            # [B, num_classes, 224, 224]
        return out

    def load_encoder_weights(self, checkpoint_path: str, device: str = "cpu"):
        """Load encoder weights from a trained classifier checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder weights from {checkpoint_path}")