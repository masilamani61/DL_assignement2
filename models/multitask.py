"""Unified multi-task model"""

import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.
    
    Integrates Task 1, 2, 3 into single forward pass.
    Weights loaded from individually trained checkpoints.
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()

        # Shared VGG11 encoder backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Shared bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # ── Task 1: Classification Head ──
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Task 2: Localization Head ──
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4),
            nn.Sigmoid(),
        )

        # ── Task 3: Segmentation Decoder ──
        self.dec5 = DecoderBlock(1024, 512, 512)
        self.dec4 = DecoderBlock(512,  512, 256)
        self.dec3 = DecoderBlock(256,  256, 128)
        self.dec2 = DecoderBlock(128,  128, 64)
        self.dec1 = DecoderBlock(64,   64,  32)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        # Auto load weights if checkpoints exist
        self._auto_load_weights()

    def _auto_load_weights(self):
        """Auto load weights from checkpoints folder if available."""
        checkpoint_paths = [
            "checkpoints",
            "../checkpoints",
            "../../checkpoints",
        ]

        # Find checkpoints folder
        ckpt_dir = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                ckpt_dir = path
                break

        if ckpt_dir is None:
            print("No checkpoints folder found - using random weights")
            return

        classifier_ckpt   = os.path.join(ckpt_dir, "classifier.pth")
        localizer_ckpt    = os.path.join(ckpt_dir, "localizer.pth")
        segmentation_ckpt = os.path.join(ckpt_dir, "unet.pth")

        self.load_pretrained_weights(
            classifier_ckpt   = classifier_ckpt   if os.path.exists(classifier_ckpt)   else None,
            localizer_ckpt    = localizer_ckpt    if os.path.exists(localizer_ckpt)    else None,
            segmentation_ckpt = segmentation_ckpt if os.path.exists(segmentation_ckpt) else None,
            device            = "cpu"
        )

    def forward(self, x: torch.Tensor):
        """Single forward pass for all 3 tasks.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            dict with keys:
            - 'classification': [B, num_breeds] logits
            - 'localization':   [B, 4] bbox coordinates
            - 'segmentation':   [B, seg_classes, H, W] pixel logits
        """
        # Shared encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # Shared bottleneck
        b = self.bottleneck(bottleneck)

        # Task 1: Breed classification
        cls_out = self.cls_head(b)              # [B, 37]

        # Task 2: Bounding box regression
        loc_out = self.loc_head(b)              # [B, 4]

        # Task 3: Pixel-wise segmentation
        s = self.dec5(b, feats["s5"])
        s = self.dec4(s, feats["s4"])
        s = self.dec3(s, feats["s3"])
        s = self.dec2(s, feats["s2"])
        s = self.dec1(s, feats["s1"])
        seg_out = self.seg_final(s)             # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }

    def load_pretrained_weights(self,
                                 classifier_ckpt: str = None,
                                 localizer_ckpt: str = None,
                                 segmentation_ckpt: str = None,
                                 device: str = "cpu"):
        """Load pretrained weights from individual task checkpoints.

        Args:
            classifier_ckpt:   path to classifier.pth  (Task 1)
            localizer_ckpt:    path to localizer.pth   (Task 2)
            segmentation_ckpt: path to unet.pth        (Task 3)
            device:            device to load weights on
        """

        # ── Load encoder from classifier (Task 1) ──
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            ckpt = torch.load(classifier_ckpt, map_location=device)
            sd   = ckpt.get("state_dict", ckpt)
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in sd.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(encoder_state)
            print(f"Loaded encoder from classifier: {classifier_ckpt}")

        # ── Load encoder from localizer (Task 2) if classifier not available ──
        elif localizer_ckpt and os.path.exists(localizer_ckpt):
            ckpt = torch.load(localizer_ckpt, map_location=device)
            sd   = ckpt.get("state_dict", ckpt)
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in sd.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(encoder_state)
            print(f"Loaded encoder from localizer: {localizer_ckpt}")

        # ── Load segmentation decoder (Task 3) ──
        if segmentation_ckpt and os.path.exists(segmentation_ckpt):
            ckpt = torch.load(segmentation_ckpt, map_location=device)
            sd   = ckpt.get("state_dict", ckpt)

            # Load decoder blocks
            for name, module in [
                ("dec5", self.dec5), ("dec4", self.dec4),
                ("dec3", self.dec3), ("dec2", self.dec2),
                ("dec1", self.dec1)
            ]:
                dec_state = {
                    k.replace(f"{name}.", ""): v
                    for k, v in sd.items()
                    if k.startswith(f"{name}.")
                }
                if dec_state:
                    module.load_state_dict(dec_state)

            # Load final conv
            seg_final_state = {
                k.replace("final_conv.", ""): v
                for k, v in sd.items()
                if k.startswith("final_conv.")
            }
            if seg_final_state:
                self.seg_final.load_state_dict(seg_final_state)

            print(f"Loaded segmentation decoder: {segmentation_ckpt}")