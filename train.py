"""Training entrypoint"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────

def compute_iou_score(pred_boxes, target_boxes, eps=1e-6):
    # Normalize to 0-1 if in pixel space
    if pred_boxes.max() > 1.0:
        pred_boxes   = pred_boxes   / 224
        target_boxes = target_boxes / 224

    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_area = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(0) * \
                 (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(0)
    pred_area  = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    tgt_area   = (tgt_x2 - tgt_x1).clamp(0)  * (tgt_y2 - tgt_y1).clamp(0)
    union_area = pred_area + tgt_area - inter_area + eps

    return (inter_area / union_area).mean().item()


def compute_dice(pred_masks, target_masks, num_classes=3, eps=1e-6):
    """Compute mean Dice score across classes."""
    pred = torch.argmax(pred_masks, dim=1)
    dice_scores = []
    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target_masks == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores.append(dice.item())
    return sum(dice_scores) / num_classes


# ─────────────────────────────────────────
# Task 1: Train Classifier
# ─────────────────────────────────────────

def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on {device}")

    wandb.init(project=args.wandb_project, name="task1-classifier", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_dir, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model     = VGG11Classifier(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    best_acc = 0
    # Resume from previous checkpoint if exists
    if os.path.exists("checkpoints/classifier.pth"):
        checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Resumed from checkpoint (best acc={checkpoint['best_metric']:.4f})")
        best_acc = checkpoint['best_metric']

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total
        train_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss    += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss": val_loss,     "val/acc": val_acc,
        })

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_acc,
            }, "checkpoints/classifier.pth")
            print(f"  Saved best classifier (acc={best_acc:.4f})")
            wandb.save("checkpoints/classifier.pth")

    wandb.finish()
    print(f"Classifier training done. Best val acc: {best_acc:.4f}")


# ─────────────────────────────────────────
# Task 2: Train Localizer
# ─────────────────────────────────────────

def train_localizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training localizer on {device}")

    wandb.init(project=args.wandb_project, name="task2-localizer", config=vars(args))

    # Use bbox only dataset
    from data.pets_dataset import OxfordIIITPetBBoxDataset
    train_ds = OxfordIIITPetBBoxDataset(args.data_dir, split="train")
    val_ds   = OxfordIIITPetBBoxDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = VGG11Localizer(freeze_encoder=False).to(device)

    # Combined loss
    from losses.iou_loss import CombinedBoxLoss
    criterion = CombinedBoxLoss(iou_weight=1.0, l1_weight=1.0)

    # Better optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    best_iou = 0.0

    # Load pretrained encoder
    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth", device=str(device))
        print("Loaded encoder weights from classifier checkpoint")

    # Resume from previous localizer checkpoint
    if os.path.exists("checkpoints/localizer.pth"):
        ckpt = torch.load("checkpoints/localizer.pth", map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        best_iou = ckpt.get("best_metric", 0.0)
        print(f"Resumed localizer from checkpoint (iou={best_iou:.4f})")

    

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_iou = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)*224
            
            # Scale GT to pixel space to match model output
            

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, bboxes)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_iou  += compute_iou_score(pred.detach(), bboxes)
        train_loss= train_loss / len(train_loader)  
        train_iou = train_iou / len(train_loader)
        # Validate
        model.eval()
        val_loss, val_iou = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)*224
                
                # Scale GT to pixel spa
                
                pred     = model(images)
                loss     = criterion(pred, bboxes)
                val_loss += loss.item()
                val_iou  += compute_iou_score(pred, bboxes)

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/iou": train_iou,
            "val/loss": val_loss,     "val/iou": val_iou,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_iou,
            }, "checkpoints/localizer.pth")
            wandb.save("checkpoints/localizer.pth")
            print(f"  Saved best localizer (iou={best_iou:.4f})")

    wandb.finish()
    print(f"Localizer done. Best IoU: {best_iou:.4f}")
# ─────────────────────────────────────────
# Task 3: Train Segmentation
# ─────────────────────────────────────────

def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training segmentation on {device}")

    wandb.init(project=args.wandb_project, name="task3-segmentation", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_dir, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model     = VGG11UNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Load pretrained encoder
    if os.path.exists("checkpoints/classifier.pth"):
        model.load_encoder_weights("checkpoints/classifier.pth", device=str(device))

    best_dice = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss, train_dice = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += compute_dice(pred.detach(), masks)

        train_loss = train_loss / len(train_loader)
        train_dice = train_dice / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss, val_dice = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                pred   = model(images)
                loss   = criterion(pred, masks)
                val_loss += loss.item()
                val_dice += compute_dice(pred, masks)

        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/dice": train_dice,
            "val/loss": val_loss,     "val/dice": val_dice,
        })

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_dice,
            }, "checkpoints/unet.pth")
            print(f"  Saved best segmentation (dice={best_dice:.4f})")
            wandb.save("checkpoints/unet.pth")

    wandb.finish()
    print(f"Segmentation training done. Best val Dice: {best_dice:.4f}")


# ─────────────────────────────────────────
# Task 4: Train Unified Model
# ─────────────────────────────────────────

def train_multitask(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training multitask model on {device}")

    wandb.init(project=args.wandb_project, name="task4-multitask", config=vars(args))

    train_ds = OxfordIIITPetDataset(args.data_dir, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiTaskPerceptionModel().to(device)

    # Load pretrained weights from individual tasks
    model.load_pretrained_weights(
        classifier_ckpt="checkpoints/classifier.pth"   if os.path.exists("checkpoints/classifier.pth")   else None,
        segmentation_ckpt="checkpoints/unet.pth"       if os.path.exists("checkpoints/unet.pth")         else None,
        device=str(device),
    )

    cls_criterion = nn.CrossEntropyLoss()
    iou_criterion = IoULoss()
    seg_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_combined = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss, train_acc, train_iou, train_dice = 0, 0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            optimizer.zero_grad()
            out = model(images)

            loss_cls = cls_criterion(out["classification"], labels)
            loss_loc = iou_criterion(out["localization"],   bboxes)
            loss_seg = seg_criterion(out["segmentation"],   masks)

            # Combined loss with equal weights
            loss = loss_cls + loss_loc + loss_seg
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += (out["classification"].argmax(1) == labels).float().mean().item()
            train_iou  += compute_iou_score(out["localization"].detach(), bboxes)
            train_dice += compute_dice(out["segmentation"].detach(), masks)

        n = len(train_loader)
        train_loss /= n; train_acc /= n; train_iou /= n; train_dice /= n

        # --- Validate ---
        model.eval()
        val_loss, val_acc, val_iou, val_dice = 0, 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks  = batch["mask"].to(device)

                out = model(images)

                loss_cls = cls_criterion(out["classification"], labels)
                loss_loc = iou_criterion(out["localization"],   bboxes)
                loss_seg = seg_criterion(out["segmentation"],   masks)
                loss     = loss_cls + loss_loc + loss_seg

                val_loss += loss.item()
                val_acc  += (out["classification"].argmax(1) == labels).float().mean().item()
                val_iou  += compute_iou_score(out["localization"], bboxes)
                val_dice += compute_dice(out["segmentation"], masks)

        n = len(val_loader)
        val_loss /= n; val_acc /= n; val_iou /= n; val_dice /= n

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"IoU: {val_iou:.4f} Dice: {val_dice:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "train/acc": train_acc,
            "train/iou": train_iou,   "train/dice": train_dice,
            "val/loss": val_loss,     "val/acc": val_acc,
            "val/iou": val_iou,       "val/dice": val_dice,
        })

        combined = val_acc + val_iou + val_dice
        if combined > best_combined:
            best_combined = combined
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_combined,
            }, "checkpoints/multitask.pth")
            print(f"  Saved best multitask model")
            wandb.save("checkpoints/multitask.pth")

    wandb.finish()
    print("Multitask training done!")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,   default="classifier",
                        choices=["classifier", "localizer", "segmentation", "multitask"])
    parser.add_argument("--data_dir",       type=str,   default="data")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--wandb_project",  type=str,   default="da6401-assignment2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "classifier":
        train_classifier(args)
    elif args.task == "localizer":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    elif args.task == "multitask":
        train_multitask(args)