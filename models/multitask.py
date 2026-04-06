"""Unified multi-task model"""

import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import DecoderBlock
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Unified multi-task model.
    
    Uses 3 individually trained models for each task.
    Single forward pass yields all 3 outputs.
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path: str = "checkpoints/localizer.pth",
                 unet_path: str = "checkpoints/unet.pth"):
        super().__init__()

        # Download checkpoints from Google Drive
        import gdown
        os.makedirs("checkpoints", exist_ok=True)
        if not os.path.exists(classifier_path):
            gdown.download(id="12jZS3yhMiiEVdmFT4vfL6tU5nZT4mqdx",
                          output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1wqL0HoYnYNfLSxZopt69c2D5HsPfMlZ1",
                          output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1XLSGHdySbVP2zZLhfrs0TzCbC53VklJc",
                          output=unet_path, quiet=False)

        # Load all 3 models
        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer  = VGG11Localizer()
        self.segmenter  = VGG11UNet(num_classes=seg_classes)

        # Load weights
        if os.path.exists(classifier_path):
            ckpt = torch.load(classifier_path, map_location="cpu")
            self.classifier.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded classifier from: {classifier_path}")

        if os.path.exists(localizer_path):
            ckpt = torch.load(localizer_path, map_location="cpu")
            self.localizer.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded localizer from: {localizer_path}")

        if os.path.exists(unet_path):
            ckpt = torch.load(unet_path, map_location="cpu")
            self.segmenter.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded segmenter from: {unet_path}")

    def forward(self, x: torch.Tensor):
        """Single forward pass for all 3 tasks.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            dict with keys:
            - 'classification': [B, num_breeds]
            - 'localization':   [B, 4]
            - 'segmentation':   [B, seg_classes, H, W]
        """
        cls_out = self.classifier(x)    # [B, 37]
        loc_out = self.localizer(x)     # [B, 4]
        seg_out = self.segmenter(x)     # [B, 3, 224, 224]

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
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            ckpt = torch.load(classifier_ckpt, map_location=device)
            self.classifier.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded classifier: {classifier_ckpt}")

        if localizer_ckpt and os.path.exists(localizer_ckpt):
            ckpt = torch.load(localizer_ckpt, map_location=device)
            self.localizer.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded localizer: {localizer_ckpt}")

        if segmentation_ckpt and os.path.exists(segmentation_ckpt):
            ckpt = torch.load(segmentation_ckpt, map_location=device)
            self.segmenter.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"Loaded segmenter: {segmentation_ckpt}")