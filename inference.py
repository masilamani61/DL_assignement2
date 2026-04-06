"""Inference and evaluation"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from train import compute_iou_score, compute_dice


# ─────────────────────────────────────────
# Transform for single image
# ─────────────────────────────────────────

def get_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_image(image_path, image_size=224):
    """Load and preprocess a single image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    original = image.copy()
    transform = get_transform(image_size)
    transformed = transform(image=image)
    tensor = transformed["image"].unsqueeze(0)  # [1, 3, H, W]
    return tensor, original


# ─────────────────────────────────────────
# Load models
# ─────────────────────────────────────────

def load_classifier(checkpoint_path, device="cpu"):
    model = VGG11Classifier(num_classes=37).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


def load_localizer(checkpoint_path, device="cpu"):
    model = VGG11Localizer().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model
def evaluate_on_test(data_dir, checkpoints_dir="checkpoints", device="cpu"):
    """Evaluate all models on test set and print final metrics."""
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from data.pets_dataset import OxfordIIITPetDataset

    print("Loading test dataset...")
    test_ds = OxfordIIITPetDataset(data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)
    device = torch.device(device)
    # Warm up GPU
    # ── Task 1: Classifier ──
    print("\nEvaluating Task 1: Classifier...")
    cls_model = load_classifier(
        os.path.join(checkpoints_dir, "classifier.pth"), device
    )
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = cls_model(images)
            preds  = logits.argmax(1).cpu()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
    f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = sum(1 for a, b in zip(all_labels, all_preds) if a == b) / len(all_labels)
    print(f"Classification Accuracy:  {accuracy:.4f}")
    print(f"Classification Macro F1:  {f1:.4f}")
    print(f"Classification Macro F1: {f1:.4f}")

    # ── Task 2: Localizer ──
    print("\nEvaluating Task 2: Localizer...")
    loc_model = load_localizer(
        os.path.join(checkpoints_dir, "localizer.pth"), device
    )
    all_ious = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            pred   = loc_model(images)
            iou    = compute_iou_score(pred, bboxes)
            all_ious.append(iou)
    mean_iou = np.mean(all_ious)
    print(f"Localization Mean IoU: {mean_iou:.4f}")

    # ── Task 3: Segmentation ──
    print("\nEvaluating Task 3: Segmentation...")
    seg_model = load_segmentation(
        os.path.join(checkpoints_dir, "unet.pth"), device
    )
    all_dices = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            pred   = seg_model(images)
            dice   = compute_dice(pred, masks)
            all_dices.append(dice)
    mean_dice = np.mean(all_dices)
    print(f"Segmentation Dice: {mean_dice:.4f}")

    # ── Task 4: Multitask ──
    print("\nEvaluating Task 4: Multitask...")
    multi_model = load_multitask(
        os.path.join(checkpoints_dir, "multitask.pth"), device
    )
    all_labels, all_preds, all_ious, all_dices = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)
            out    = multi_model(images)
            preds  = out["classification"].argmax(1).cpu()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.numpy())
            all_ious.append(compute_iou_score(out["localization"], bboxes))
            all_dices.append(compute_dice(out["segmentation"], masks))

    multi_f1   = f1_score(all_labels, all_preds, average="macro")
    accuracy = sum(1 for a, b in zip(all_labels, all_preds) if a == b) / len(all_labels)
    print(f"Classification Accuracy:  {accuracy:.4f}")
    print(f"Classification Macro F1:  {f1:.4f}")
    multi_iou  = np.mean(all_ious)
    multi_dice = np.mean(all_dices)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Task 1 - Classification F1:  {f1:.4f}")
    print(f"Task 2 - Localization IoU:   {mean_iou:.4f}")
    print(f"Task 3 - Segmentation Dice:  {mean_dice:.4f}")
    print(f"Task 4 - Multitask F1:       {multi_f1:.4f}")
    print(f"Task 4 - Multitask IoU:      {multi_iou:.4f}")
    print(f"Task 4 - Multitask Dice:     {multi_dice:.4f}")
    print("=========================================")

    return f1, mean_iou, mean_dice

def load_segmentation(checkpoint_path, device="cpu"):
    model = VGG11UNet(num_classes=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


def load_multitask(checkpoint_path, device="cpu"):
    model = MultiTaskPerceptionModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


# ─────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────

def predict_class(model, image_tensor, class_names, device="cpu"):
    """Predict breed class."""
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return pred.item(), conf.item(), class_names[pred.item()]


def predict_bbox(model, image_tensor, device="cpu"):
    """Predict bounding box [xc, yc, w, h] normalized."""
    with torch.no_grad():
        bbox = model(image_tensor.to(device))
    return bbox[0].cpu().numpy()


def predict_mask(model, image_tensor, device="cpu"):
    """Predict segmentation mask."""
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        mask   = torch.argmax(logits, dim=1)
    return mask[0].cpu().numpy()


def predict_multitask(model, image_tensor, device="cpu"):
    """Run full pipeline - all 3 outputs."""
    with torch.no_grad():
        out = model(image_tensor.to(device))
    cls_probs = torch.softmax(out["classification"], dim=1)
    conf, pred = cls_probs.max(1)
    bbox = out["localization"][0].cpu().numpy()
    mask = torch.argmax(out["segmentation"], dim=1)[0].cpu().numpy()
    return pred.item(), conf.item(), bbox, mask


# ─────────────────────────────────────────
# Visualization functions
# ─────────────────────────────────────────
def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze().permute(1, 2, 0).numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
def draw_bbox(ax, bbox, image_size=224, color="red", label=""):
    """Draw bounding box on axis."""
    xc, yc, w, h = bbox
    x1 = (xc - w/2) * image_size
    y1 = (yc - h/2) * image_size
    bw = w * image_size
    bh = h * image_size
    rect = patches.Rectangle(
        (x1, y1), bw, bh,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1, y1-5, label, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))


def visualize_pipeline(image_path, class_names, checkpoints_dir="checkpoints",
                        device="cpu", save_path=None):
    """Run full pipeline and visualize results."""
    image_tensor, original = load_image(image_path)
    h, w = original.shape[:2]

    # Load models
    multitask_model = load_multitask(
        os.path.join(checkpoints_dir, "multitask.pth"), device
    )

    # Predict
    pred_class, conf, bbox, mask = predict_multitask(
        multitask_model, image_tensor, device
    )

    class_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original + bbox
    axes[0].imshow(original)
    draw_bbox(axes[0], bbox, image_size=w,
              color="red", label=f"{class_name} ({conf:.2f})")
    axes[0].set_title(f"Detection: {class_name}\nConf: {conf:.2f}")
    axes[0].axis("off")

    # Segmentation mask
    colors = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]])
    mask_rgb = colors[mask]
    axes[1].imshow(original)
    axes[1].imshow(mask_rgb, alpha=0.5)
    axes[1].set_title("Segmentation Mask\n(Green=FG, Blue=BG, Red=Boundary)")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(original)
    draw_bbox(axes[2], bbox, image_size=w, color="red")
    axes[2].imshow(mask_rgb, alpha=0.3)
    axes[2].set_title("Full Pipeline Output")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return pred_class, conf, bbox, mask


def visualize_bbox_table(image_paths, gt_bboxes, class_names,
                          checkpoints_dir="checkpoints", device="cpu"):
    """Visualize bbox predictions vs ground truth for W&B table."""
    import wandb

    localizer = load_localizer(
        os.path.join(checkpoints_dir, "localizer.pth"), device
    )

    table = wandb.Table(columns=[
        "Image", "Predicted Box", "GT Box", "IoU", "Confidence"
    ])

    for img_path, gt_bbox in zip(image_paths, gt_bboxes):
        image_tensor, original = load_image(img_path)
        pred_bbox = predict_bbox(localizer, image_tensor, device)

        # Compute IoU
        pred_t  = torch.tensor(pred_bbox).unsqueeze(0)
        gt_t    = torch.tensor(gt_bbox).unsqueeze(0)
        from losses.iou_loss import IoULoss
        iou = 1 - IoULoss()(pred_t, gt_t).item()

        # Draw
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(original)
        draw_bbox(ax, gt_bbox,   image_size=224, color="green", label="GT")
        draw_bbox(ax, pred_bbox, image_size=224, color="red",   label=f"Pred IoU={iou:.2f}")
        ax.axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(fig),
            str(pred_bbox.tolist()),
            str(gt_bbox),
            round(iou, 4),
            "N/A"
        )
        plt.close()

    wandb.log({"bbox_predictions": table})
    print("Logged bbox table to W&B!")


def visualize_segmentation_samples(image_paths, gt_masks,
                                    checkpoints_dir="checkpoints", device="cpu"):
    """Visualize segmentation predictions for W&B."""
    import wandb

    seg_model = load_segmentation(
        os.path.join(checkpoints_dir, "unet.pth"), device
    )

    colors = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]])

    table = wandb.Table(columns=[
        "Original", "Ground Truth", "Prediction"
    ])

    for img_path, gt_mask in zip(image_paths, gt_masks):
        image_tensor, original = load_image(img_path)
        pred_mask = predict_mask(seg_model, image_tensor, device)

        gt_rgb   = colors[gt_mask]
        pred_rgb = colors[pred_mask]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original);        axes[0].set_title("Original");       axes[0].axis("off")
        axes[1].imshow(gt_rgb);          axes[1].set_title("Ground Truth");   axes[1].axis("off")
        axes[2].imshow(pred_rgb);        axes[2].set_title("Prediction");     axes[2].axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(original),
            wandb.Image(gt_rgb),
            wandb.Image(pred_rgb)
        )
        plt.close()

    wandb.log({"segmentation_samples": table})
    print("Logged segmentation table to W&B!")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data.pets_dataset import OxfordIIITPetDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, default="data")
    parser.add_argument("--checkpoints_dir",type=str, default="checkpoints")
    parser.add_argument("--image_path",     type=str, default=None)
    parser.add_argument("--task",           type=str, default="evaluate",
                        choices=["pipeline", "bbox_table", "seg_samples",'evaluate'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset to get class names
    ds = OxfordIIITPetDataset(args.data_dir, split="test")
    class_names = ds.class_names

    if args.task == "pipeline":
        if args.image_path:
            visualize_pipeline(
                args.image_path, class_names,
                checkpoints_dir=args.checkpoints_dir,
                device=device
            )
        else:
            # Run on random test images
            for i in range(3):
                sample = ds[i]
                img_name = ds.samples[i][0]
                img_path = os.path.join(args.data_dir, "images", img_name + ".jpg")
                print(f"\nImage: {img_name}")
                visualize_pipeline(
                    img_path, class_names,
                    checkpoints_dir=args.checkpoints_dir,
                    device=device,
                    save_path=f"output_{i}.png"
                )

    elif args.task == "bbox_table":
        import wandb
        wandb.init(project="da6401-assignment2", name="bbox-table")
        image_paths, gt_bboxes = [], []
        for i in range(10):
            sample  = ds[i]
            img_name = ds.samples[i][0]
            img_path = os.path.join(args.data_dir, "images", img_name + ".jpg")
            image_paths.append(img_path)
            gt_bboxes.append(sample["bbox"].numpy().tolist())
        visualize_bbox_table(
            image_paths, gt_bboxes, class_names,
            checkpoints_dir=args.checkpoints_dir, device=device
        )
        wandb.finish()
    elif args.task == "evaluate":
        evaluate_on_test(
            args.data_dir,
            checkpoints_dir=args.checkpoints_dir,
            device=device
        )

    elif args.task == "seg_samples":
        import wandb
        wandb.init(project="da6401-assignment2", name="seg-samples")
        image_paths, gt_masks = [], []
        for i in range(5):
            sample   = ds[i]
            img_name = ds.samples[i][0]
            img_path = os.path.join(args.data_dir, "images", img_name + ".jpg")
            image_paths.append(img_path)
            gt_masks.append(sample["mask"].numpy())
        visualize_segmentation_samples(
            image_paths, gt_masks,
            checkpoints_dir=args.checkpoints_dir, device=device
        )
        wandb.finish()