"""LoRA adapters — §4.

You implement: LoRALinear, apply_lora_to_attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer.

    Computes:  W' x = base_layer(x) + (alpha / rank) * (B A x)

    where:
      - base_layer is the frozen pretrained linear (its weights are not trained).
      - A in R^{rank x d_in}  is initialized with kaiming_uniform_.
      - B in R^{d_out x rank} is initialized to zero (so the adapted layer
        starts equal to the base layer).

    Only A and B receive gradients; base_layer's parameters are frozen.

    Args:
        base_layer: Existing nn.Linear to wrap.
        rank:       Adapter rank `r` (typically 4..32).
        alpha:      Scaling factor; effective scale is `alpha / rank`.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_layer = base_layer
        _freeze_module(self.base_layer)

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.empty(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base_layer(x) == x @ W^T (+ bias); LoRA delta: (alpha/rank) * x @ A^T @ B^T
        return self.base_layer(x) + self.scaling * (x @ self.A.T @ self.B.T)


def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Replace `q_proj` and `v_proj` linear layers in every attention head
    with LoRA-wrapped versions.

    The HW writeup recommends adapting only Q and V projections (per the
    original LoRA paper). Walk the module tree and wherever you find an
    nn.Linear named `q_proj` or `v_proj` inside a Head, swap it for a
    LoRALinear.

    The function modifies `model` in place AND returns it for convenience.

    Args:
        model: A module containing one or more `basics.model.Head` instances
               (e.g., a ViT).
        rank, alpha: Forwarded to LoRALinear.
    """
    from basics.model import Head

    for module in model.modules():
        if isinstance(module, Head):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)

    # Train only LoRA matrices; keep the rest of the backbone frozen (§4.2).
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(("A", "B"))

    return model
