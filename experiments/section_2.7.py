"""Section 2.7 - Final Pipeline Showcase on Wild Images"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from PIL import Image
import urllib.request

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from inference import load_image, denormalize


# ── Download wild images ──
WILD_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "name": "wild_dog_labrador.jpg",
        "description": "Yellow Labrador - clear background"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
        "name": "wild_cat_tabby.jpg",
        "description": "Tabby Cat - indoor lighting"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/1200px-Dog_Breeds.jpg",
        "name": "wild_dog_mixed.jpg",
        "description": "Mixed breed dog - complex background"
    },
]


import os

def load_all_images(save_dir="wild_images"):
    """Load all images from folder automatically."""
    loaded = []

    for file_name in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file_name)

        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            loaded.append({
                "path": file_path,
                "description": file_name
            })

    return loaded


def run_full_pipeline(image_path, classifier, localizer,
                       seg_model, class_names, device):
    """Run complete pipeline on a single image."""

    # Load image
    image_tensor, _ = load_image(image_path)
    original = np.array(
        Image.open(image_path).convert("RGB").resize((224, 224))
    )

    with torch.no_grad():
        # Classification
        logits = classifier(image_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)
        conf, pred_class = probs.max(1)
        confidence  = conf.item()
        class_name  = class_names[pred_class.item()] \
                      if pred_class.item() < len(class_names) \
                      else f"Class {pred_class.item()}"

        # Localization
        pred_bbox = localizer(image_tensor.to(device))[0].cpu().numpy()

        # Segmentation
        seg_logits = seg_model(image_tensor.to(device))
        pred_mask  = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

    return original, class_name, confidence, pred_bbox, pred_mask


def visualize_pipeline_result(original, class_name, confidence,
                               pred_bbox, pred_mask, description):
    """Create comprehensive pipeline visualization."""

    colors = np.array([
        [0,   255, 0  ],   # FG = Green
        [0,   0,   255],   # BG = Blue
        [255, 0,   0  ],   # Boundary = Red
    ], dtype=np.uint8)

    mask_rgb = colors[pred_mask]
    h, w     = original.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Wild Image Pipeline: {description}",
        fontsize=13, fontweight="bold"
    )

    # 1. Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # 2. Bounding box prediction
    axes[1].imshow(original)
    xc, yc, bw, bh = pred_bbox
    x1 = (xc - bw/2) * w
    y1 = (yc - bh/2) * h
    rect = patches.Rectangle(
        (x1, y1), bw * w, bh * h,
        linewidth=3, edgecolor="red", facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].set_title(
        f"Localization\n{class_name} ({confidence:.2f})",
        fontweight="bold"
    )
    axes[1].axis("off")

    # 3. Segmentation mask
    axes[2].imshow(mask_rgb)
    axes[2].set_title(
        "Segmentation\n(Green=FG, Blue=BG, Red=Boundary)",
        fontweight="bold"
    )
    axes[2].axis("off")

    # 4. Full overlay
    axes[3].imshow(original)
    axes[3].imshow(mask_rgb, alpha=0.4)
    rect2 = patches.Rectangle(
        (x1, y1), bw * w, bh * h,
        linewidth=3, edgecolor="red", facecolor="none"
    )
    axes[3].add_patch(rect2)
    axes[3].set_title(
        "Full Pipeline Overlay",
        fontweight="bold"
    )
    axes[3].axis("off")

    plt.tight_layout()
    return fig


def analyze_pipeline_quality(pred_bbox, pred_mask, confidence):
    """Analyze quality of pipeline predictions."""
    analysis = []

    # Bbox quality
    xc, yc, w, h = pred_bbox
    area = w * h
    if area < 0.05:
        analysis.append("⚠️ BBox too small — may have missed subject")
    elif area > 0.9:
        analysis.append("⚠️ BBox too large — covers whole image")
    else:
        analysis.append("✅ BBox size looks reasonable")

    if not (0.1 < xc < 0.9 and 0.1 < yc < 0.9):
        analysis.append("⚠️ BBox center near edge — possible localization failure")
    else:
        analysis.append("✅ BBox center in reasonable position")

    # Confidence
    if confidence > 0.7:
        analysis.append(f"✅ High confidence: {confidence:.3f}")
    elif confidence > 0.4:
        analysis.append(f"⚠️ Medium confidence: {confidence:.3f}")
    else:
        analysis.append(f"❌ Low confidence: {confidence:.3f}")

    # Segmentation quality
    fg_ratio  = (pred_mask == 0).mean()
    bg_ratio  = (pred_mask == 1).mean()
    bnd_ratio = (pred_mask == 2).mean()

    if fg_ratio < 0.05:
        analysis.append("⚠️ Very little foreground detected")
    elif fg_ratio > 0.7:
        analysis.append("⚠️ Too much foreground — background confused with pet")
    else:
        analysis.append(f"✅ FG ratio looks good: {fg_ratio:.2f}")

    return analysis


def run_experiment(checkpoints_dir="checkpoints",
                   device="cuda",
                   wild_images_dir="wild_images",
                   data_dir="data"):
    """Run pipeline showcase on wild images."""

    wandb.init(
        project="da6401-assignment2",
        name="section2.7-wild-pipeline",
        config={"experiment": "wild_images_showcase"}
    )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── Load models ──
    print("Loading models...")

    classifier = VGG11Classifier(num_classes=37).to(device_obj)
    ckpt = torch.load(
        os.path.join(checkpoints_dir, "classifier.pth"),
        map_location=device_obj
    )
    classifier.load_state_dict(ckpt.get("state_dict", ckpt))
    classifier.eval()

    localizer = VGG11Localizer().to(device_obj)
    ckpt = torch.load(
        os.path.join(checkpoints_dir, "localizer.pth"),
        map_location=device_obj
    )
    localizer.load_state_dict(ckpt.get("state_dict", ckpt))
    localizer.eval()

    seg_model = VGG11UNet(num_classes=3).to(device_obj)
    ckpt = torch.load(
        os.path.join(checkpoints_dir, "unet.pth"),
        map_location=device_obj
    )
    seg_model.load_state_dict(ckpt.get("state_dict", ckpt))
    seg_model.eval()

    # ── Load class names ──
    ds          = OxfordIIITPetDataset(data_dir, split="test")
    class_names = ds.class_names

    # ── Download wild images ──
    print("Downloading wild images...")
    wild_images = load_all_images(wild_images_dir)

    # if not wild_images:
    #     print("No wild images found! Please add images manually to wild_images/")
    #     return

    # ── W&B Table ──
    table = wandb.Table(columns=[
        "Image", "Pipeline Output", "Predicted Class",
        "Confidence", "Analysis"
    ])
    print(wild_images)
    # ── Run pipeline on each wild image ──
    for i, img_info in enumerate(wild_images):

        print(f"\nProcessing: {img_info['description']}")

        try:
            original, class_name, confidence, pred_bbox, pred_mask = \
                run_full_pipeline(
                    img_info["path"],
                    classifier, localizer, seg_model,
                    class_names, device_obj
                )

            # Visualize
            fig = visualize_pipeline_result(
                original, class_name, confidence,
                pred_bbox, pred_mask, img_info["description"]
            )

            # Analyze quality
            analysis = analyze_pipeline_quality(
                pred_bbox, pred_mask, confidence
            )

            print(f"  Class: {class_name} | Conf: {confidence:.3f}")
            for a in analysis:
                print(f"  {a}")

            # Add to table
            table.add_data(
                wandb.Image(original),
                wandb.Image(fig),
                class_name,
                round(confidence, 4),
                "\n".join(analysis)
            )

            # Log individual result
            wandb.log({
                f"wild_image_{i+1}": wandb.Image(
                    fig,
                    caption=f"{img_info['description']} | "
                            f"{class_name} ({confidence:.2f})"
                )
            })

            plt.close()

        except Exception as e:
            print(f"  Error: {e}")
            continue

    wandb.log({"wild_images_table": table})

    wandb.finish()
    print("\nSection 2.7 DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         type=str, default="data")
    parser.add_argument("--checkpoints_dir",  type=str, default="checkpoints")
    parser.add_argument("--wild_images_dir",  type=str, default="wild_images")
    parser.add_argument("--device",           type=str, default="cuda")
    args = parser.parse_args()

    run_experiment(
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        wild_images_dir=args.wild_images_dir,
        data_dir=args.data_dir
    )