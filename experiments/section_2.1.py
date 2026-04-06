"""Section 2.1 - Regularization Effect of BatchNorm"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier, VGG11ClassifierNoBN


def get_activation(name, activations):
    """Hook to capture layer activations."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def plot_activation_distribution(activations_bn, activations_nobn, epoch):
    """Plot activation distributions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # With BN
    act_bn = activations_bn["conv3"].cpu().numpy().flatten()
    axes[0].hist(act_bn, bins=100, color="blue", alpha=0.7)
    axes[0].set_title(f"With BatchNorm (Epoch {epoch})\n3rd Conv Layer Activations")
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(x=0, color='red', linestyle='--')

    # Without BN
    act_nobn = activations_nobn["conv3"].cpu().numpy().flatten()
    axes[1].hist(act_nobn, bins=100, color="orange", alpha=0.7)
    axes[1].set_title(f"Without BatchNorm (Epoch {epoch})\n3rd Conv Layer Activations")
    axes[1].set_xlabel("Activation Value")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(x=0, color='red', linestyle='--')

    plt.tight_layout()
    return fig


def run_experiment(data_dir, epochs=20, batch_size=32, device="cuda"):
    """Run BN vs No-BN experiment."""

    wandb.init(
        project="da6401-assignment2",
        name="section2.1-batchnorm-comparison",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "experiment": "batchnorm_effect"
        }
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = OxfordIIITPetDataset(data_dir, split="train")
    val_ds   = OxfordIIITPetDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Models
    model_bn   = VGG11Classifier(num_classes=37).to(device)
    model_nobn = VGG11ClassifierNoBN(num_classes=37).to(device)

    # Optimizers
    opt_bn   = torch.optim.Adam(model_bn.parameters(),   lr=1e-4)
    opt_nobn = torch.optim.Adam(model_nobn.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Activation hooks
    activations_bn   = {}
    activations_nobn = {}

    # Hook on 3rd conv layer (block3 first conv)
    model_bn.encoder.block3[0].register_forward_hook(
        get_activation("conv3", activations_bn)
    )
    model_nobn.encoder.block3[0].register_forward_hook(
        get_activation("conv3", activations_nobn)
    )

    # Get fixed batch for visualization
    fixed_batch = next(iter(val_loader))
    fixed_images = fixed_batch["image"].to(device)

    bn_train_losses,   bn_val_losses   = [], []
    nobn_train_losses, nobn_val_losses = [], []

    for epoch in range(epochs):
        # ── Train both models ──
        model_bn.train();   model_nobn.train()
        bn_loss = 0;        nobn_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # With BN
            opt_bn.zero_grad()
            out  = model_bn(images)
            loss = criterion(out, labels)
            loss.backward()
            opt_bn.step()
            bn_loss += loss.item()

            # Without BN
            opt_nobn.zero_grad()
            out  = model_nobn(images)
            loss = criterion(out, labels)
            loss.backward()
            opt_nobn.step()
            nobn_loss += loss.item()

        bn_loss   /= len(train_loader)
        nobn_loss /= len(train_loader)

        # ── Validate both models ──
        model_bn.eval();   model_nobn.eval()
        bn_val = 0;        nobn_val = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bn_val   += criterion(model_bn(images),   labels).item()
                nobn_val += criterion(model_nobn(images), labels).item()

        bn_val   /= len(val_loader)
        nobn_val /= len(val_loader)

        bn_train_losses.append(bn_loss)
        nobn_train_losses.append(nobn_loss)
        bn_val_losses.append(bn_val)
        nobn_val_losses.append(nobn_val)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"BN: train={bn_loss:.4f} val={bn_val:.4f} | "
              f"NoBN: train={nobn_loss:.4f} val={nobn_val:.4f}")

        # ── Visualize activations every 5 epochs ──
        with torch.no_grad():
            model_bn(fixed_images)
            model_nobn(fixed_images)

        if (epoch + 1) % 5 == 0:
            fig = plot_activation_distribution(
                activations_bn, activations_nobn, epoch + 1
            )
            wandb.log({
                "activation_distribution": wandb.Image(fig),
                "epoch": epoch + 1,
                "bn/train_loss":   bn_loss,
                "bn/val_loss":     bn_val,
                "nobn/train_loss": nobn_loss,
                "nobn/val_loss":   nobn_val,
            })
            plt.close()
        else:
            wandb.log({
                "epoch": epoch + 1,
                "bn/train_loss":   bn_loss,
                "bn/val_loss":     bn_val,
                "nobn/train_loss": nobn_loss,
                "nobn/val_loss":   nobn_val,
            })

    # ── Final activation plot ──
    with torch.no_grad():
        model_bn(fixed_images)
        model_nobn(fixed_images)

    fig = plot_activation_distribution(activations_bn, activations_nobn, epochs)
    wandb.log({"final_activation_distribution": wandb.Image(fig)})
    plt.close()

    # ── Loss curves plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(bn_train_losses,   label="BN Train",   color="blue")
    axes[0].plot(nobn_train_losses, label="NoBN Train", color="orange")
    axes[0].set_title("Training Loss: BN vs No BN")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(bn_val_losses,   label="BN Val",   color="blue")
    axes[1].plot(nobn_val_losses, label="NoBN Val", color="orange")
    axes[1].set_title("Validation Loss: BN vs No BN")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    wandb.log({"loss_curves": wandb.Image(fig)})
    plt.close()

    wandb.finish()
    print("Section 2.1 experiment done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    run_experiment(args.data_dir, args.epochs, args.batch_size, args.device)