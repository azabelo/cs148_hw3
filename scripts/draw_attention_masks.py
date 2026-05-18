#!/usr/bin/env python3
"""Draw 7x7 attention mask diagrams for §5 masking problem (4 vis + 3 text tokens)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlm.masking import build_causal_mask, build_image_bidir_mask


def allowed_matrix(n_vis: int, n_text: int, mode: str) -> np.ndarray:
    dtype = torch.float32
    if mode == "M1":
        m = build_causal_mask(n_vis + n_text, torch.device("cpu"), dtype)[0, 0]
    else:
        m = build_image_bidir_mask(n_vis, n_text, torch.device("cpu"), dtype)[0, 0]
    return (m == 0).cpu().numpy().astype(int)


def plot_mask(mat: np.ndarray, title: str, labels: list[str], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    n_vis, n_text = 4, 3
    labels = [f"v{i+1}" for i in range(n_vis)] + [f"t{i+1}" for i in range(n_text)]
    out_dir = Path("runs/mask_diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)

    m1 = allowed_matrix(n_vis, n_text, "M1")
    m2 = allowed_matrix(n_vis, n_text, "M2")
    plot_mask(m1, "M1: Fully causal", labels, str(out_dir / "mask_M1_causal.png"))
    plot_mask(m2, "M2: Image-block bidirectional", labels, str(out_dir / "mask_M2_image_bidir.png"))
    print(f"Saved diagrams to {out_dir}/")


if __name__ == "__main__":
    main()
