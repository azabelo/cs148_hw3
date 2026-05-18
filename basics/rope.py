"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE along the last dimension for each token position.

        Args:
            x: (B, num_heads, T, head_dim).
            positions: (T,) integer positions.

        Returns:
            (B, num_heads, T, head_dim): x with RoPE applied.
        """
        if positions.ndim != 1:
            raise ValueError("positions must be 1D of shape (T,)")
        t = x.shape[2]
        if positions.shape[0] != t:
            raise ValueError("positions must have length T matching x.shape[2]")

        cos = self.cos_cached[positions.to(self.cos_cached.device)].to(dtype=x.dtype)
        sin = self.sin_cached[positions.to(self.sin_cached.device)].to(dtype=x.dtype)
        cos = cos.view(1, 1, t, -1)
        sin = sin.view(1, 1, t, -1)

        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack((y0, y1), dim=-1).reshape_as(x)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        # TODO: precompute (cos, sin) for x and y separately, each of shape
        # (grid_size, head_dim // 4). Register as buffers.
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: split x along head_dim into two halves; apply 1D RoPE to the
        # first half with x_coords and to the second half with y_coords;
        # concatenate.
        raise NotImplementedError
