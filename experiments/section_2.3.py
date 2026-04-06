"""Section 2.3 - Transfer Learning Showdown"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet


def compute_dice(pred_masks, target_masks, num_classes=3, eps=1e-6):
    pred = torch.argmax(pred_masks, dim=1)
    dice_scores = []
    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target_masks == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores.append(dice.item())
    return sum(dice_scores) / num_classes


def compute_pixel_acc(pred_masks, target_masks):
    pred = torch.argmax(pred_masks, dim=1)
    return (pred == target_masks).float().mean().item()


def freeze_strategy(model, strategy):
    """Apply freezing strategy to model encoder."""

    if strategy == "strict":
        # Freeze entire encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Strategy: Strict - entire encoder frozen")

    elif strategy == "partial":
        # Freeze early blocks (block1, block2, block3)
        # Unfreeze late blocks (block4, block5)
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.block4.parameters():
            param.requires_grad = True
        for param in model.encoder.block5.parameters():
            param.requires_grad = True
        print("Strategy: Partial - blocks 1-3 frozen, blocks 4-5 unfrozen")

    elif strategy == "full":
        # Unfreeze everything
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Strategy: Full - entire network unfrozen")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")


def train_one_strategy(data_dir, strategy, epochs=20,
                        batch_size=16, device="cuda",
                        classifier_ckpt="checkpoints/classifier.pth"):
    """Train segmentation with specific transfer learning strategy."""

    wandb.init(
        project="da6401-assignment2",
        name=f"section2.3-{strategy}-finetune",
        config={
            "strategy": strategy,
            "epochs": epochs,
            "batch_size": batch_size,
            "experiment": "transfer_learning"
        },
        group="section2.3-transfer-learning"
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds     = OxfordIIITPetDataset(data_dir, split="train")
    val_ds       = OxfordIIITPetDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # Model
    model = VGG11UNet(num_classes=3).to(device)

    # Load pretrained encoder
    if os.path.exists(classifier_ckpt):
        model.load_encoder_weights(classifier_ckpt, device=str(device))
        print(f"Loaded encoder from {classifier_ckpt}")
    else:
        print("No pretrained weights found - training from scratch!")

    # Apply freezing strategy
    freeze_strategy(model, strategy)

    # Optimizer - only update trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_dice = 0.0

    for epoch in range(epochs):
        import time
        start = time.time()

        # ── Train ──
        model.train()
        train_loss, train_dice, train_acc = 0, 0, 0

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
            train_acc  += compute_pixel_acc(pred.detach(), masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_acc  /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss, val_dice, val_acc = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                pred   = model(images)
                loss   = criterion(pred, masks)
                val_loss += loss.item()
                val_dice += compute_dice(pred, masks)
                val_acc  += compute_pixel_acc(pred, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_acc  /= len(val_loader)

        epoch_time = time.time() - start
        scheduler.step()

        print(f"[{strategy}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} | "
              f"Time: {epoch_time:.1f}s")

        wandb.log({
            "epoch":           epoch + 1,
            "train/loss":      train_loss,
            "train/dice":      train_dice,
            "train/pixel_acc": train_acc,
            "val/loss":        val_loss,
            "val/dice":        val_dice,
            "val/pixel_acc":   val_acc,
            "epoch_time":      epoch_time,
        })

        if val_dice > best_dice:
            best_dice = val_dice
            print(f"  Best dice: {best_dice:.4f}")

    wandb.log({"best_val_dice": best_dice})
    wandb.finish()
    print(f"Done: {strategy} | Best Dice: {best_dice:.4f}\n")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, default="data")
    parser.add_argument("--epochs",          type=int, default=20)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="checkpoints/classifier.pth")
    args = parser.parse_args()

    results = {}

    # Run all 3 strategies
    for strategy in ["strict", "partial", "full"]:
        print("="*50)
        print(f"Strategy: {strategy.upper()}")
        print("="*50)
        best_dice = train_one_strategy(
            data_dir=args.data_dir,
            strategy=strategy,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            classifier_ckpt=args.classifier_ckpt
        )
        results[strategy] = best_dice

    # Print summary
    print("\n========== RESULTS SUMMARY ==========")
    for strategy, dice in results.items():
        print(f"{strategy:10s} → Best Val Dice: {dice:.4f}")
    print("=====================================")
    print("Section 2.3 ALL DONE!")