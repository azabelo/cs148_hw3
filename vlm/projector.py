"""Vision-Language Projector — §5.

You implement: VisionLanguageProjector.

Rationale (why not a single linear layer?): A lone linear map can only rotate/scale
projections between spaces; the two-layer MLP with GELU adds nonlinear capacity so the
projector can learn richer alignments between frozen ViT features and frozen decoder
embeddings when it is the only trainable bridge during VLM pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionLanguageProjector(nn.Module):
    """2-layer MLP that maps image features into the decoder's embedding space.

    Architecture:
        Linear(d_image, expansion * d_image) -> GELU -> Linear(expansion * d_image, d_decoder)

    Must handle both:
      - A single pooled image vector:  input (B, d_image)         -> output (B, 1, d_decoder)
      - A sequence of patch vectors:   input (B, N_vis, d_image)  -> output (B, N_vis, d_decoder)

    Args:
        d_image:   Image-encoder embedding dim (your ViT's d_model).
        d_decoder: Decoder embedding dim (e.g., 960 for SmolLM2-360M).
        expansion: MLP hidden expansion factor (4 by default, à la LLaVA).
    """

    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = expansion * d_image
        self.fc1 = nn.Linear(d_image, hidden)
        self.fc2 = nn.Linear(hidden, d_decoder)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(1)

        x = self.fc1(image_features)
        x = nn.functional.gelu(x)
        return self.fc2(x)
