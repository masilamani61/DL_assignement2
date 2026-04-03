"""Reusable custom layers"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implemented from scratch.
    
    Uses inverted dropout scaling so outputs have
    consistent expected values during training and inference.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During eval mode, dropout is disabled
        if not self.training:
            return x

        # If p=0, nothing to drop
        if self.p == 0.0:
            return x

        # Create binary mask: 1 = keep, 0 = drop
        # Each element kept with probability (1 - p)
        mask = torch.bernoulli(
            torch.full(x.shape, 1 - self.p, device=x.device, dtype=x.dtype)
        )

        # Inverted dropout scaling: divide by (1-p) to maintain
        # expected value of activations at test time
        return x * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"