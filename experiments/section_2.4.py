"""Section 2.4 - Feature Maps Visualization"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image

from models.classification import VGG11Classifier
from inference import load_image, denormalize


def visualize_feature_maps(model, image_tensor, image_original, device="cuda"):
    """Extract and visualize feature maps from first and last conv layers."""

    model.eval()
    activations = {}

    # Register hooks on first and last conv layers
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # First conv layer (block1 first conv)
    hook1 = model.encoder.block1[0].register_forward_hook(
        get_hook("first_conv")
    )

    # Last conv layer (block5 last conv - before pool5)
    hook2 = model.encoder.block5[3].register_forward_hook(
        get_hook("last_conv")
    )

    # Forward pass
    with torch.no_grad():
        model(image_tensor.to(device))

    # Remove hooks
    hook1.remove()
    hook2.remove()

    return activations


def plot_feature_maps(activations, image_original, num_filters=16):
    """Plot feature maps from first and last conv layers."""

    fig = plt.figure(figsize=(20, 12))

    # ── Original Image ──
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(image_original)
    ax.set_title("Original Image", fontsize=14, fontweight="bold")
    ax.axis("off")

    # ── First Conv Layer Feature Maps ──
    first_maps = activations["first_conv"][0].cpu().numpy()  # [64, H, W]
    num_show   = min(num_filters, first_maps.shape[0])

    fig2, axes = plt.subplots(4, num_show//4, figsize=(20, 8))
    fig2.suptitle("First Conv Layer Feature Maps\n(Low-level: edges, colors, gradients)",
                  fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < num_show:
            fmap = first_maps[i]
            # Normalize for display
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Filter {i+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    # ── Last Conv Layer Feature Maps ──
    last_maps = activations["last_conv"][0].cpu().numpy()   # [512, H, W]
    num_show  = min(num_filters, last_maps.shape[0])

    fig3, axes = plt.subplots(4, num_show//4, figsize=(20, 8))
    fig3.suptitle("Last Conv Layer Feature Maps\n(High-level: semantic shapes, parts)",
                  fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < num_show:
            fmap = last_maps[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Filter {i+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    return fig2, fig3


def plot_activation_statistics(activations):
    """Plot statistics comparing first vs last layer activations."""

    first_maps = activations["first_conv"][0].cpu().numpy().flatten()
    last_maps  = activations["last_conv"][0].cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribution comparison
    axes[0].hist(first_maps, bins=100, alpha=0.7,
                 color="blue", label="First Conv", density=True)
    axes[0].hist(last_maps,  bins=100, alpha=0.7,
                 color="red",  label="Last Conv",  density=True)
    axes[0].set_title("Activation Distribution\nFirst vs Last Layer")
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Sparsity comparison
    first_sparsity = (first_maps == 0).mean() * 100
    last_sparsity  = (last_maps  == 0).mean() * 100
    axes[1].bar(["First Conv", "Last Conv"],
                [first_sparsity, last_sparsity],
                color=["blue", "red"], alpha=0.7)
    axes[1].set_title("Sparsity (%)\n(% of zero activations)")
    axes[1].set_ylabel("Sparsity %")

    # Mean activation per filter
    first_means = activations["first_conv"][0].cpu().numpy().mean(axis=(1, 2))
    last_means  = activations["last_conv"][0].cpu().numpy().mean(axis=(1, 2))
    axes[2].plot(sorted(first_means, reverse=True),
                 color="blue", label="First Conv", alpha=0.7)
    axes[2].plot(sorted(last_means,  reverse=True),
                 color="red",  label="Last Conv",  alpha=0.7)
    axes[2].set_title("Mean Activation per Filter\n(sorted descending)")
    axes[2].set_xlabel("Filter Index")
    axes[2].set_ylabel("Mean Activation")
    axes[2].legend()

    plt.tight_layout()
    return fig


def run_experiment(data_dir, checkpoints_dir="checkpoints",
                   device="cuda", image_path=None):
    """Run feature map visualization experiment."""

    wandb.init(
        project="da6401-assignment2",
        name="section2.4-feature-maps",
        config={"experiment": "feature_maps"}
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading classifier...")
    from inference import load_classifier
    model = load_classifier(
        os.path.join(checkpoints_dir, "classifier.pth"), device
    )

    # Load image
    if image_path and os.path.exists(image_path):
        print(f"Using provided image: {image_path}")
        image_tensor, original = load_image(image_path)
    else:
        # Use a random dog image from dataset
        print("Finding a dog image from dataset...")
        from data.pets_dataset import OxfordIIITPetDataset
        ds = OxfordIIITPetDataset(data_dir, split="test")
        image_tensor, original = None, None
        # Find a dog image (species=2 means dog)
        for i in range(len(ds)):
            img_name = ds.samples[i][0]
            # Dogs have uppercase first letter in name
            if img_name[0].isupper() and not any(
                cat in img_name for cat in
                ["Abyssinian", "Bengal", "Birman", "Bombay",
                 "British", "Egyptian", "Maine", "Persian",
                 "Ragdoll", "Russian", "Siamese", "Sphynx"]
            ):
                img_path = os.path.join(
                    data_dir, "images", img_name + ".jpg"
                )
                image_tensor, original = load_image(img_path)
                print(f"Using image: {img_name}")
                break

    # Extract feature maps
    print("Extracting feature maps...")
    activations = visualize_feature_maps(
        model, image_tensor, original, device
    )

    # Plot feature maps
    print("Plotting feature maps...")
    fig_first, fig_last = plot_feature_maps(
        activations, original, num_filters=16
    )

    # Plot statistics
    fig_stats = plot_activation_statistics(activations)

    # Log to W&B
    wandb.log({
        "original_image":         wandb.Image(original),
        "first_conv_feature_maps": wandb.Image(fig_first),
        "last_conv_feature_maps":  wandb.Image(fig_last),
        "activation_statistics":   wandb.Image(fig_stats),
    })

    # Print layer info
    first_shape = activations["first_conv"].shape
    last_shape  = activations["last_conv"].shape
    print(f"\nFirst conv output shape: {first_shape}")
    print(f"Last conv output shape:  {last_shape}")
    print(f"\nFirst conv - num filters: {first_shape[1]}")
    print(f"Last conv  - num filters: {last_shape[1]}")
    print(f"\nFirst conv - spatial size: {first_shape[2]}x{first_shape[3]}")
    print(f"Last conv  - spatial size: {last_shape[2]}x{last_shape[3]}")

    plt.close("all")
    wandb.finish()
    print("Section 2.4 DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--image_path",      type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        image_path=args.image_path
    )