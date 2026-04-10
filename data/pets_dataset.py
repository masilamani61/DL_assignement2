"""Dataset loader for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root_dir, split="train", transform=None, image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.bboxes_dir = os.path.join(root_dir, "annotations", "xmls")
        self.list_file = os.path.join(root_dir, "annotations", "list.txt")

        self.samples = []
        self.class_names = []

        self._load_split()

        if self.transform is None:
            self.transform = self._default_transform()

    def _load_split(self):
        """Parse list.txt and build train/val/test splits."""
        all_samples = []
        class_set = set()

        with open(self.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                image_name = parts[0]
                class_id = int(parts[1]) - 1
                breed_name = "_".join(image_name.split("_")[:-1])
                class_set.add((class_id, breed_name))
                all_samples.append((image_name, class_id))

        self.class_names = [name for _, name in sorted(class_set)]

        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        n = len(all_samples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if self.split == "train":
            selected = indices[:train_end]
        elif self.split == "val":
            selected = indices[train_end:val_end]
        else:
            selected = indices[val_end:]

        self.samples = [all_samples[i] for i in selected]

    def _load_bbox(self, image_name):
        """Load bounding box from XML. Returns [xc, yc, w, h] normalized."""
        xml_path = os.path.join(self.bboxes_dir, image_name + ".xml")
        if not os.path.exists(xml_path):
            return None

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        obj = root.find("object")
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        return [xc, yc, w, h]

    def _default_transform(self):
        """Default albumentations transform pipeline."""
        if self.split == "train":
            return A.Compose([
            A.Resize(self.image_size, self.image_size),
            
            # Flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            
            # Color
            A.ColorJitter(brightness=0.3, contrast=0.3, 
                         saturation=0.3, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            
            # Blur and noise
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.2),
            
            # Geometric
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                              rotate_limit=15, p=0.5),
            A.GridDistortion(p=0.2),
            
            # Cutout
            A.CoarseDropout(max_holes=8, max_height=32, 
                           max_width=32, p=0.3),
            
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, class_idx = self.samples[idx]

        # Load image
        img_path = os.path.join(self.images_dir, image_name + ".jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        mask_path = os.path.join(self.masks_dir, image_name + ".png")
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            mask = mask - 1
            mask = np.clip(mask, 0, 2).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()

        # Load bbox
        bbox = self._load_bbox(image_name)
        if bbox is None:
            # Use center of image as fallback - better than full image
            bbox = [0.5, 0.5, 0.5, 0.5]
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return {
            "image": image,
            "label": torch.tensor(class_idx, dtype=torch.long),
            "bbox": bbox,
            "mask": mask,
        }
class OxfordIIITPetBBoxDataset(OxfordIIITPetDataset):
    """Dataset that only includes images with real bbox annotations."""

    def __init__(self, root_dir, split="train", transform=None, image_size=224):
        super().__init__(root_dir, split, transform, image_size)
        # Filter only images with real bbox annotations
        self.samples = [
            s for s in self.samples
            if os.path.exists(os.path.join(self.bboxes_dir, s[0] + ".xml"))
        ]
        print(f"BBox dataset split={split}: {len(self.samples)} images with real annotations")