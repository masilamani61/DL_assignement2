"""Section 2.2 - Internal Dynamics: Dropout Effect"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def train_one_run(data_dir, dropout_p, epochs=20, batch_size=32, device="cuda"):
    """Train classifier with specific dropout probability."""

    run_name = f"dropout_p={dropout_p}" if dropout_p > 0 else "no_dropout"

    wandb.init(
        project="da6401-assignment2",
        name=f"section2.2-{run_name}",
        config={
            "dropout_p": dropout_p,
            "epochs": epochs,
            "batch_size": batch_size,
            "experiment": "dropout_effect"
        },
        group="section2.2-dropout-comparison"
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds     = OxfordIIITPetDataset(data_dir, split="train")
    val_ds       = OxfordIIITPetDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model with specified dropout
    model     = VGG11Classifier(num_classes=37, dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # ── Train ──
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

            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = correct / total

        # ── Validate ──
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

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total

        # Generalization gap
        gap = train_loss - val_loss

        scheduler.step()

        print(f"[{run_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Gap: {gap:.4f}")

        wandb.log({
            "epoch":       epoch + 1,
            "train/loss":  train_loss,
            "train/acc":   train_acc,
            "val/loss":    val_loss,
            "val/acc":     val_acc,
            "gap":         gap,
        })

    wandb.finish()
    print(f"Done: {run_name}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    # Run all 3 conditions sequentially
    for dropout_p in [0.0, 0.2, 0.5]:
        print("="*50)
        print(f"Running with dropout_p={dropout_p}")
        print("="*50)
        train_one_run(
            data_dir=args.data_dir,
            dropout_p=dropout_p,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

    print("Section 2.2 ALL DONE!")