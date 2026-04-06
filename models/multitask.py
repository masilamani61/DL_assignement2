"""Unified multi-task model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.
    
    Single forward pass yields:
    - Classification logits [B, num_breeds]
    - Bounding box [B, 4]
    - Segmentation mask [B, seg_classes, H, W]
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()

        # Shared encoder backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)
        import gdown
        gdown.download(id="https://drive.google.com/file/d/12jZS3yhMiiEVdmFT4vfL6tU5nZT4mqdx/view?usp=sharing", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="https://drive.google.com/file/d/1XLSGHdySbVP2zZLhfrs0TzCbC53VklJc/view?usp=sharing", output=unet_path, quiet=False)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # --- Task 1: Classification Head ---
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

        # --- Task 2: Localization Head ---
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

        # --- Task 3: Segmentation Decoder ---
        self.dec5 = DecoderBlock(1024, 512, 512)
        self.dec4 = DecoderBlock(512,  512, 256)
        self.dec3 = DecoderBlock(256,  256, 128)
        self.dec2 = DecoderBlock(128,  128, 64)
        self.dec1 = DecoderBlock(64,   64,  32)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Single forward pass for all 3 tasks.

        Returns:
            dict with keys:
            - 'classification': [B, num_breeds]
            - 'localization':   [B, 4]
            - 'segmentation':   [B, seg_classes, H, W]
        """
        # Shared encoder
        bottleneck, feats = self.encoder(x, return_features=True)

        # Bottleneck
        b = self.bottleneck(bottleneck)         # [B, 1024, 7, 7]

        # Task 1: Classification
        cls_out = self.cls_head(b)              # [B, 37]

        # Task 2: Localization
        loc_out = self.loc_head(b)              # [B, 4]

        # Task 3: Segmentation
        s = self.dec5(b, feats["s5"])           # [B, 512, 14, 14]
        s = self.dec4(s, feats["s4"])           # [B, 256, 28, 28]
        s = self.dec3(s, feats["s3"])           # [B, 128, 56, 56]
        s = self.dec2(s, feats["s2"])           # [B, 64,  112, 112]
        s = self.dec1(s, feats["s1"])           # [B, 32,  224, 224]
        seg_out = self.seg_final(s)             # [B, 3,   224, 224]

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
        """Load pretrained weights from individual task checkpoints."""

        def load_encoder(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt.get("state_dict", ckpt)
            return {k.replace("encoder.", ""): v
                    for k, v in sd.items() if k.startswith("encoder.")}

        if classifier_ckpt:
            self.encoder.load_state_dict(load_encoder(classifier_ckpt))
            print(f"Loaded encoder from classifier: {classifier_ckpt}")

        if segmentation_ckpt:
            ckpt = torch.load(segmentation_ckpt, map_location=device)
            sd = ckpt.get("state_dict", ckpt)
            # Load decoder weights
            for name, module in [("dec5", self.dec5), ("dec4", self.dec4),
                                  ("dec3", self.dec3), ("dec2", self.dec2),
                                  ("dec1", self.dec1)]:
                dec_state = {k.replace(f"{name}.", ""): v
                             for k, v in sd.items() if k.startswith(f"{name}.")}
                if dec_state:
                    module.load_state_dict(dec_state)
            print(f"Loaded segmentation decoder: {segmentation_ckpt}")