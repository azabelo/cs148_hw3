#!/usr/bin/env python3
"""Benchmark ViT forward-pass latency vs patch size (HW2 timing exercise).

Measures wall-clock time for one forward pass on a batch of 16 images,
for patch sizes P ∈ {8, 16, 32}, with d_model=384, num_heads=6, num_blocks=6.
Uses CUDA synchronization around the timed region; 5 warmup steps, 20 timed
steps; reports mean and sample standard deviation per P.

Run from the repo root (login node has no GPU; use your interactive GPU job):

    srun --jobid=<JOBID> --overlap bash -lc \
      'export PYTHONUNBUFFERED=1; cd /path/to/this/repo && .venv/bin/python -u benchmark_vit_forward.py'

After the environment exists, `.venv/bin/python` avoids a long `uv run` sync on shared
filesystems. First-time setup: `uv sync` from the repo root, then the command above.
"""

from __future__ import annotations

import sys
import time

import torch

from basics.vit import ViT

BATCH = 16
IMG_SIZE = 224  # divisible by 8, 16, 32
PATCH_SIZES = (8, 16, 32)
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
DROPOUT = 0.0
WARMUP = 5
STEPS = 20


def bench_patch_size(patch_size: int, device: torch.device) -> tuple[float, float]:
    model = ViT(
        img_size=IMG_SIZE,
        patch_size=patch_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(device)
    model.eval()
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)
    mean = t.mean().item()
    std = t.std(unbiased=True).item() if STEPS > 1 else 0.0
    return mean, std


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark (torch.cuda.synchronize()).", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(device)
    print(
        f"ViT forward benchmark | device={device} ({dev_name}) | batch={BATCH} | "
        f"img_size={IMG_SIZE} | d_model={D_MODEL} | num_heads={NUM_HEADS} | "
        f"num_blocks={NUM_BLOCKS} | dropout={DROPOUT} | warmup={WARMUP} | steps={STEPS}\n",
        flush=True,
    )

    rows: list[tuple[int, float, float, int]] = []
    for p in PATCH_SIZES:
        mean_ms, std_ms = bench_patch_size(p, device)
        n_patches = (IMG_SIZE // p) ** 2
        rows.append((p, mean_ms, std_ms, n_patches))

    col_w = (10, 14, 14, 12)
    headers = ("P", "mean (ms)", "std (ms)", "N patches")
    sep = (
        "-" * col_w[0]
        + "-+-"
        + "-" * col_w[1]
        + "-+-"
        + "-" * col_w[2]
        + "-+-"
        + "-" * col_w[3]
    )
    header_line = (
        f"{headers[0]:^{col_w[0]}} | "
        f"{headers[1]:^{col_w[1]}} | "
        f"{headers[2]:^{col_w[2]}} | "
        f"{headers[3]:^{col_w[3]}}"
    )
    print(header_line)
    print(sep)
    for p, mean_ms, std_ms, n_patches in rows:
        print(
            f"{p:^{col_w[0]}} | "
            f"{mean_ms:^{col_w[1]}.4f} | "
            f"{std_ms:^{col_w[2]}.4f} | "
            f"{n_patches:^{col_w[3]}}"
        )


if __name__ == "__main__":
    main()
