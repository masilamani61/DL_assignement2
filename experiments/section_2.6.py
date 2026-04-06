"""Section 2.6 - Segmentation Evaluation: Dice vs Pixel Accuracy"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from PIL import Image

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet
from inference import load_image, denormalize


def compute_dice_per_class(pred_masks, target_masks, num_classes=3, eps=1e-6):
    """Compute Dice score per class."""
    pred = torch.argmax(pred_masks, dim=1)
    dice_scores = {}
    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target_masks == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores[c] = dice.item()
    return dice_scores


def compute_pixel_acc(pred_masks, target_masks):
    """Compute pixel accuracy."""
    pred = torch.argmax(pred_masks, dim=1)
    return (pred == target_masks).float().mean().item()


def compute_class_pixel_acc(pred_masks, target_masks, num_classes=3):
    """Compute per class pixel accuracy."""
    pred = torch.argmax(pred_masks, dim=1)
    accs = {}
    for c in range(num_classes):
        mask = (target_masks == c)
        if mask.sum() > 0:
            accs[c] = (pred[mask] == c).float().mean().item()
        else:
            accs[c] = 0.0
    return accs


def visualize_segmentation(image, gt_mask, pred_mask, idx):
    """Visualize original, GT mask and predicted mask."""

    # Color map: 0=FG(green), 1=BG(blue), 2=Boundary(red)
    colors = np.array([
        [0,   255, 0  ],   # 0 = Foreground = Green
        [0,   0,   255],   # 1 = Background = Blue
        [255, 0,   0  ],   # 2 = Boundary   = Red
    ], dtype=np.uint8)

    gt_rgb   = colors[gt_mask]
    pred_rgb = colors[pred_mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title(
        "Ground Truth Trimap\n(Green=FG, Blue=BG, Red=Boundary)",
        fontsize=12, fontweight="bold"
    )
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title(
        "Predicted Trimap Mask\n(Green=FG, Blue=BG, Red=Boundary)",
        fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    plt.suptitle(f"Sample {idx+1}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_pixel_distribution(data_dir):
    """Analyze and plot pixel class distribution in dataset."""

    print("Analyzing pixel distribution...")
    ds = OxfordIIITPetDataset(data_dir, split="train")

    fg_count  = 0
    bg_count  = 0
    bnd_count = 0
    total     = 0

    # Sample 100 images
    for i in range(min(100, len(ds))):
        sample = ds[i]
        mask   = sample["mask"].numpy()
        fg_count  += (mask == 0).sum()
        bg_count  += (mask == 1).sum()
        bnd_count += (mask == 2).sum()
        total     += mask.size

    fg_pct  = fg_count  / total * 100
    bg_pct  = bg_count  / total * 100
    bnd_pct = bnd_count / total * 100

    print(f"Foreground pixels:  {fg_pct:.1f}%")
    print(f"Background pixels:  {bg_pct:.1f}%")
    print(f"Boundary pixels:    {bnd_pct:.1f}%")

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    axes[0].pie(
        [fg_pct, bg_pct, bnd_pct],
        labels=["Foreground", "Background", "Boundary"],
        colors=["green", "blue", "red"],
        autopct="%1.1f%%",
        startangle=90
    )
    axes[0].set_title("Pixel Class Distribution\n(100 training images)")

    # Bar chart
    axes[1].bar(
        ["Foreground", "Background", "Boundary"],
        [fg_pct, bg_pct, bnd_pct],
        color=["green", "blue", "red"],
        alpha=0.7
    )
    axes[1].set_title("Pixel Class Distribution")
    axes[1].set_ylabel("Percentage (%)")
    for i, v in enumerate([fg_pct, bg_pct, bnd_pct]):
        axes[1].text(i, v + 0.5, f"{v:.1f}%",
                    ha="center", fontweight="bold")

    plt.tight_layout()
    return fig, fg_pct, bg_pct, bnd_pct


def plot_dice_vs_pixacc_comparison(dice_scores, pixel_accs, epochs):
    """Plot Dice vs Pixel Accuracy over epochs."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlay plot
    axes[0].plot(epochs, dice_scores,   color="blue",
                 label="Dice Score",    linewidth=2, marker="o")
    axes[0].plot(epochs, pixel_accs,    color="orange",
                 label="Pixel Accuracy", linewidth=2, marker="s")
    axes[0].set_title("Dice Score vs Pixel Accuracy\nover Training Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gap plot
    gaps = [pa - ds for pa, ds in zip(pixel_accs, dice_scores)]
    axes[1].fill_between(epochs, gaps, alpha=0.3, color="red")
    axes[1].plot(epochs, gaps, color="red", linewidth=2)
    axes[1].set_title("Gap: Pixel Accuracy - Dice Score\n(Shows class imbalance effect)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_experiment(data_dir, checkpoints_dir="checkpoints",
                   device="cuda", num_samples=5):
    """Run segmentation evaluation experiment."""

    wandb.init(
        project="da6401-assignment2",
        name="section2.6-dice-vs-pixacc",
        config={"experiment": "dice_vs_pixacc"}
    )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print("Loading segmentation model...")
    model = VGG11UNet(num_classes=3).to(device_obj)
    ckpt  = torch.load(
        os.path.join(checkpoints_dir, "unet.pth"),
        map_location=device_obj
    )
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # ── Dataset ──
    ds = OxfordIIITPetDataset(data_dir, split="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    # ── Pixel Distribution Analysis ──
    print("Analyzing pixel distribution...")
    fig_dist, fg_pct, bg_pct, bnd_pct = plot_pixel_distribution(data_dir)
    wandb.log({"pixel_distribution": wandb.Image(fig_dist)})
    plt.close()

    # ── Full Test Set Evaluation ──
    print("Evaluating on test set...")
    all_dice_fg,  all_dice_bg,  all_dice_bnd = [], [], []
    all_pixel_acc = []
    all_acc_fg,   all_acc_bg,   all_acc_bnd  = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device_obj)
            masks  = batch["mask"].to(device_obj)
            pred   = model(images)

            # Dice per class
            dice = compute_dice_per_class(pred, masks)
            all_dice_fg.append(dice[0])
            all_dice_bg.append(dice[1])
            all_dice_bnd.append(dice[2])

            # Pixel accuracy
            all_pixel_acc.append(compute_pixel_acc(pred, masks))

            # Per class accuracy
            acc = compute_class_pixel_acc(pred, masks)
            all_acc_fg.append(acc[0])
            all_acc_bg.append(acc[1])
            all_acc_bnd.append(acc[2])

    mean_dice_fg  = np.mean(all_dice_fg)
    mean_dice_bg  = np.mean(all_dice_bg)
    mean_dice_bnd = np.mean(all_dice_bnd)
    mean_dice     = (mean_dice_fg + mean_dice_bg + mean_dice_bnd) / 3
    mean_pix_acc  = np.mean(all_pixel_acc)

    print(f"\n===== TEST RESULTS =====")
    print(f"Mean Pixel Accuracy: {mean_pix_acc:.4f}")
    print(f"Mean Dice Score:     {mean_dice:.4f}")
    print(f"  FG Dice:           {mean_dice_fg:.4f}")
    print(f"  BG Dice:           {mean_dice_bg:.4f}")
    print(f"  Boundary Dice:     {mean_dice_bnd:.4f}")
    print(f"Gap (PixAcc-Dice):   {mean_pix_acc - mean_dice:.4f}")

    # ── Metrics Bar Chart ──
    fig_metrics, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall comparison
    axes[0].bar(
        ["Pixel Accuracy", "Mean Dice"],
        [mean_pix_acc, mean_dice],
        color=["orange", "blue"], alpha=0.7
    )
    axes[0].set_title("Pixel Accuracy vs Dice Score\n(Overall)")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([mean_pix_acc, mean_dice]):
        axes[0].text(i, v + 0.01, f"{v:.4f}",
                    ha="center", fontweight="bold")

    # Per class dice
    axes[1].bar(
        ["FG Dice", "BG Dice", "Boundary Dice"],
        [mean_dice_fg, mean_dice_bg, mean_dice_bnd],
        color=["green", "blue", "red"], alpha=0.7
    )
    axes[1].set_title("Dice Score per Class")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate([mean_dice_fg, mean_dice_bg, mean_dice_bnd]):
        axes[1].text(i, v + 0.01, f"{v:.4f}",
                    ha="center", fontweight="bold")

    plt.tight_layout()
    wandb.log({"metrics_comparison": wandb.Image(fig_metrics)})
    plt.close()

    # ── Log 5 Sample Images ──
    print(f"\nVisualizing {num_samples} sample images...")

    seg_table = wandb.Table(columns=[
        "ID", "Original", "Ground Truth", "Prediction",
        "Dice FG", "Dice BG", "Dice Boundary", "Pixel Acc"
    ])

    colors = np.array([
        [0,   255, 0  ],
        [0,   0,   255],
        [255, 0,   0  ],
    ], dtype=np.uint8)

    for i in range(num_samples):
        sample   = ds[i]
        img_name = ds.samples[i][0]
        img_path = os.path.join(data_dir, "images", img_name + ".jpg")

        # Load
        image_tensor, _ = load_image(img_path)
        original = np.array(
            Image.open(img_path).convert("RGB").resize((224, 224))
        )
        gt_mask = sample["mask"].numpy()

        # Predict
        with torch.no_grad():
            pred_logits = model(image_tensor.to(device_obj))
            pred_mask   = torch.argmax(pred_logits, dim=1)[0].cpu().numpy()

        # Compute metrics
        pred_t   = pred_logits
        target_t = sample["mask"].unsqueeze(0).to(device_obj)
        dice     = compute_dice_per_class(pred_t, target_t)
        pix_acc  = compute_pixel_acc(pred_t, target_t)

        # Visualization
        fig = visualize_segmentation(original, gt_mask, pred_mask, i)
        wandb.log({f"segmentation_sample_{i+1}": wandb.Image(fig)})
        plt.close()

        # Table row
        seg_table.add_data(
            i + 1,
            wandb.Image(original),
            wandb.Image(colors[gt_mask]),
            wandb.Image(colors[pred_mask]),
            round(dice[0], 4),
            round(dice[1], 4),
            round(dice[2], 4),
            round(pix_acc, 4)
        )

        print(f"  Sample {i+1}: Dice FG={dice[0]:.3f} "
              f"BG={dice[1]:.3f} Boundary={dice[2]:.3f} "
              f"PixAcc={pix_acc:.3f}")

    wandb.log({"segmentation_samples_table": seg_table})

    # ── Log Summary Metrics ──
    wandb.log({
        "test/mean_pixel_accuracy": mean_pix_acc,
        "test/mean_dice":           mean_dice,
        "test/dice_fg":             mean_dice_fg,
        "test/dice_bg":             mean_dice_bg,
        "test/dice_boundary":       mean_dice_bnd,
        "test/gap_pixacc_dice":     mean_pix_acc - mean_dice,
        "pixel_distribution/fg":    fg_pct,
        "pixel_distribution/bg":    bg_pct,
        "pixel_distribution/bnd":   bnd_pct,
    })

    wandb.finish()
    print("\nSection 2.6 DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--num_samples",     type=int, default=5)
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        num_samples=args.num_samples
    )