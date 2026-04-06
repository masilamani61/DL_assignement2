"""Section 2.5 - Object Detection: Confidence & IoU"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from torch.utils.data import DataLoader
from PIL import Image

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from models.classification import VGG11Classifier
from inference import load_image, denormalize


def compute_iou(pred_box, gt_box):
    """Compute IoU between two boxes in xc,yc,w,h format."""
    # Convert to x1,y1,x2,y2
    px1 = pred_box[0] - pred_box[2] / 2
    py1 = pred_box[1] - pred_box[3] / 2
    px2 = pred_box[0] + pred_box[2] / 2
    py2 = pred_box[1] + pred_box[3] / 2

    gx1 = gt_box[0] - gt_box[2] / 2
    gy1 = gt_box[1] - gt_box[3] / 2
    gx2 = gt_box[0] + gt_box[2] / 2
    gy2 = gt_box[1] + gt_box[3] / 2

    # Intersection
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area   = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union     = pred_area + gt_area - inter + 1e-6

    return inter / union


def draw_boxes(image, pred_box, gt_box, iou, confidence, class_name):
    """Draw GT (green) and predicted (red) boxes on image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)

    h, w = image.shape[:2]

    # Draw GT box - Green
    gx1 = (gt_box[0] - gt_box[2] / 2) * w
    gy1 = (gt_box[1] - gt_box[3] / 2) * h
    gw  = gt_box[2] * w
    gh  = gt_box[3] * h
    gt_rect = patches.Rectangle(
        (gx1, gy1), gw, gh,
        linewidth=3, edgecolor="green",
        facecolor="none", label="Ground Truth"
    )
    ax.add_patch(gt_rect)
    ax.text(gx1, gy1 - 5, "GT",
            color="green", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7))

    # Draw Predicted box - Red
    px1 = (pred_box[0] - pred_box[2] / 2) * w
    py1 = (pred_box[1] - pred_box[3] / 2) * h
    pw  = pred_box[2] * w
    ph  = pred_box[3] * h
    pred_rect = patches.Rectangle(
        (px1, py1), pw, ph,
        linewidth=3, edgecolor="red",
        facecolor="none", label="Prediction"
    )
    ax.add_patch(pred_rect)
    ax.text(px1, py1 - 5, f"Pred IoU={iou:.2f}",
            color="red", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7))

    # Color code title based on IoU
    if iou >= 0.5:
        title_color = "green"
        status = "GOOD"
    elif iou >= 0.3:
        title_color = "orange"
        status = "PARTIAL"
    else:
        title_color = "red"
        status = "POOR"

    ax.set_title(
        f"{class_name}\nConf: {confidence:.3f} | IoU: {iou:.3f} | {status}",
        fontsize=11, color=title_color, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


def run_experiment(data_dir, checkpoints_dir="checkpoints",
                   device="cuda", num_images=15):
    """Run bbox detection visualization."""

    wandb.init(
        project="da6401-assignment2",
        name="section2.5-bbox-detection",
        config={"experiment": "bbox_detection", "num_images": num_images}
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    localizer  = VGG11Localizer().to(device)
    ckpt = torch.load(
        os.path.join(checkpoints_dir, "localizer.pth"),
        map_location=device
    )
    localizer.load_state_dict(ckpt.get("state_dict", ckpt))
    localizer.eval()

    classifier = VGG11Classifier(num_classes=37).to(device)
    ckpt = torch.load(
        os.path.join(checkpoints_dir, "classifier.pth"),
        map_location=device
    )
    classifier.load_state_dict(ckpt.get("state_dict", ckpt))
    classifier.eval()

    # Dataset
    ds = OxfordIIITPetDataset(data_dir, split="test")

    # W&B Table
    table = wandb.Table(columns=[
        "ID", "Image", "Class", "Confidence",
        "IoU", "Status", "Pred Box", "GT Box"
    ])

    failure_cases = []
    good_cases    = []

    print(f"Running inference on {num_images} images...")

    for i in range(num_images):
        sample   = ds[i]
        img_name = ds.samples[i][0]
        img_path = os.path.join(data_dir, "images", img_name + ".jpg")

        # Load image
        image_tensor, original = load_image(img_path)

        # Resize original to 224x224 for display
        original_resized = np.array(
            Image.open(img_path).convert("RGB").resize((224, 224))
        )

        # Predict bbox
        with torch.no_grad():
            pred_bbox = localizer(image_tensor.to(device))
            pred_bbox = pred_bbox[0].cpu().numpy()

        # Predict class + confidence
        with torch.no_grad():
            logits = classifier(image_tensor.to(device))
            probs  = torch.softmax(logits, dim=1)
            conf, pred_class = probs.max(1)
            confidence = conf.item()

        # Ground truth
        gt_bbox    = sample["bbox"].numpy()
        class_name = ds.class_names[sample["label"].item()] \
                     if sample["label"].item() < len(ds.class_names) \
                     else f"Class {sample['label'].item()}"

        # Compute IoU
        iou = compute_iou(pred_bbox, gt_bbox)

        # Status
        if iou >= 0.5:
            status = "GOOD"
            good_cases.append(i)
        elif iou < 0.3:
            status = "POOR"
            failure_cases.append((i, confidence, iou, img_name))
        else:
            status = "PARTIAL"

        # Draw boxes
        fig = draw_boxes(
            original_resized, pred_bbox, gt_bbox,
            iou, confidence, class_name
        )

        # Add to table
        table.add_data(
            i + 1,
            wandb.Image(fig),
            class_name,
            round(confidence, 4),
            round(iou, 4),
            status,
            str(pred_bbox.round(3).tolist()),
            str(gt_bbox.round(3).tolist())
        )

        plt.close()
        print(f"  [{i+1}/{num_images}] {img_name} | "
              f"Class: {class_name} | Conf: {confidence:.3f} | "
              f"IoU: {iou:.3f} | {status}")

    # Log table
    wandb.log({"bbox_predictions_table": table})

    # ── Failure Case Analysis ──
    if failure_cases:
        print("\n=== FAILURE CASES ===")
        # Find worst case - high confidence but low IoU
        failure_cases.sort(key=lambda x: x[1] - x[2], reverse=True)
        worst_idx, worst_conf, worst_iou, worst_name = failure_cases[0]

        print(f"Worst failure: {worst_name}")
        print(f"  Confidence: {worst_conf:.3f} (HIGH)")
        print(f"  IoU: {worst_iou:.3f} (LOW)")

        # Plot failure case
        sample   = ds[worst_idx]
        img_path = os.path.join(data_dir, "images", worst_name + ".jpg")
        image_tensor, _ = load_image(img_path)
        original_resized = np.array(
            Image.open(img_path).convert("RGB").resize((224, 224))
        )

        with torch.no_grad():
            pred_bbox  = localizer(image_tensor.to(device))[0].cpu().numpy()
            logits     = classifier(image_tensor.to(device))
            probs      = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

        gt_bbox    = sample["bbox"].numpy()
        class_name = ds.class_names[sample["label"].item()]
        iou        = compute_iou(pred_bbox, gt_bbox)

        fig = draw_boxes(
            original_resized, pred_bbox, gt_bbox,
            iou, conf.item(), class_name
        )

        wandb.log({
            "failure_case": wandb.Image(
                fig,
                caption=f"HIGH CONF ({worst_conf:.3f}) but LOW IoU ({worst_iou:.3f})"
            )
        })
        plt.close()

    # ── Summary Stats ──
    all_ious = []
    for row in range(num_images):
        sample  = ds[row]
        img_path = os.path.join(
            data_dir, "images", ds.samples[row][0] + ".jpg"
        )
        image_tensor, _ = load_image(img_path)
        with torch.no_grad():
            pred_bbox = localizer(image_tensor.to(device))[0].cpu().numpy()
        gt_bbox = sample["bbox"].numpy()
        all_ious.append(compute_iou(pred_bbox, gt_bbox))

    fig_summary, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, num_images + 1), all_ious,
           color=["green" if iou >= 0.5
                  else "orange" if iou >= 0.3
                  else "red"
                  for iou in all_ious])
    ax.axhline(y=0.5, color="black", linestyle="--", label="IoU=0.5 threshold")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("IoU Score")
    ax.set_title("IoU Scores per Image\n(Green≥0.5, Orange≥0.3, Red<0.3)")
    ax.legend()
    plt.tight_layout()

    wandb.log({
        "iou_summary":  wandb.Image(fig_summary),
        "mean_iou":     np.mean(all_ious),
        "good_cases":   sum(1 for iou in all_ious if iou >= 0.5),
        "poor_cases":   sum(1 for iou in all_ious if iou < 0.3),
    })
    plt.close()

    print(f"\nMean IoU: {np.mean(all_ious):.4f}")
    print(f"Good cases (IoU≥0.5): {sum(1 for iou in all_ious if iou >= 0.5)}")
    print(f"Poor cases (IoU<0.3): {sum(1 for iou in all_ious if iou < 0.3)}")

    wandb.finish()
    print("Section 2.5 DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--num_images",      type=int, default=15)
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        num_images=args.num_images
    )