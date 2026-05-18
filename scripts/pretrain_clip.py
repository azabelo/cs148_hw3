"""§3 — CLIP-style pretraining on EuroSAT.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
from vlm.eval import zeroshot_classification_accuracy


def _fix_ssl_for_hf() -> None:
    """Unbreak TLS on clusters that set SSL_CERT_DIR to a file (e.g. *.pem).

    OpenSSL expects SSL_CERT_DIR to be a directory of certs. If it points at a
    single bundle file, verification fails for Hugging Face. We drop the bad
    value and, when no bundle is set, point Python/requests at certifi.
    """
    cert_dir = os.environ.get("SSL_CERT_DIR")
    if cert_dir and not os.path.isdir(cert_dir):
        os.environ.pop("SSL_CERT_DIR", None)
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = bundle
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="If set, overrides train.num_epochs in the config (useful for smoke tests).",
    )
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def _class_prompts() -> list[str]:
    return [f"a satellite image of {name}" for name in EUROSAT_CLASSES]


def _train_one_epoch(
    vit: ViT,
    text_encoder: FrozenTextEncoder,
    projection_heads: ProjectionHeads,
    logit_scale: nn.Parameter,
    train_loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    log_every: int,
    global_step: int,
) -> tuple[float, int]:
    vit.train()
    projection_heads.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc="train", leave=False)
    for batch_idx, (images, captions) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        # Clone: FrozenTextEncoder uses inference tensors; Linear needs plain tensors for autograd.
        text_embeds = text_encoder(captions).to(device, non_blocking=True).clone()

        optimizer.zero_grad(set_to_none=True)
        image_feats = vit(images)
        image_proj, text_proj = projection_heads(image_feats, text_embeds)
        loss = clip_loss(image_proj, text_proj, logit_scale)
        loss.backward()
        optimizer.step()
        scheduler.step()

        logit_scale.data.clamp_(max=math.log(100.0))

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(num_batches, 1), global_step


def _plot_curves(
    history: list[dict],
    out_dir: Path,
) -> None:
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    accs = [h["val_zs_acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.plot(epochs, losses, color="tab:blue", marker="o", label="train loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("zero-shot val acc", color="tab:orange")
    ax2.plot(epochs, accs, color="tab:orange", marker="s", label="val zs acc")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0.0, 1.0)

    fig.suptitle("CLIP pretraining on EuroSAT")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "train_loss.png", dpi=150)
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, accs, marker="s", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("zero-shot val accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "val_zeroshot_acc.png", dpi=150)
    plt.close(fig3)


def main() -> None:
    args = parse_args()
    _fix_ssl_for_hf()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vit_cfg = cfg["vit"]
    train_dl, val_dl, _test_dl = build_eurosat_loaders(
        img_size=int(vit_cfg["img_size"]),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    vit = ViT(
        img_size=int(vit_cfg["img_size"]),
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg.get("dropout", 0.1)),
    ).to(device)

    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"]).to(
        device
    )
    projection_heads = ProjectionHeads(
        d_image=vit.d_model,
        d_text=text_encoder.embedding_dim,
        d_proj=int(cfg["projection"]["d_proj"]),
    ).to(device)

    logit_scale = init_logit_scale(device=device)

    opt_cfg = cfg["optim"]
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg["weight_decay"])
    betas = tuple(float(b) for b in opt_cfg.get("betas", (0.9, 0.999)))

    optimizer = AdamW(
        [
            {"params": vit.parameters(), "weight_decay": wd},
            {"params": projection_heads.parameters(), "weight_decay": wd},
            {"params": [logit_scale], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )

    num_epochs = (
        int(args.num_epochs)
        if args.num_epochs is not None
        else int(cfg["train"]["num_epochs"])
    )
    warmup_steps = int(opt_cfg.get("warmup_steps", 0))
    steps_per_epoch = len(train_dl)
    total_steps = max(1, num_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    class_prompts = _class_prompts()
    class_indices = list(range(len(class_prompts)))
    log_every = int(cfg["train"].get("log_every", 50))

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="cs148-hw3-clip", config=cfg)

    history: list[dict] = []
    global_step = 0
    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        avg_loss, global_step = _train_one_epoch(
            vit,
            text_encoder,
            projection_heads,
            logit_scale,
            train_dl,
            optimizer,
            scheduler,
            device,
            log_every,
            global_step,
        )

        val_acc = zeroshot_classification_accuracy(
            vit,
            projection_heads,
            text_encoder,
            val_dl,
            class_prompts,
            class_indices,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_zs_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{num_epochs}  train_loss={avg_loss:.4f}  "
            f"val_zero_shot_acc={val_acc:.4f}",
            flush=True,
        )

        with open(args.output_dir / "metrics.jsonl", "a") as mf:
            mf.write(json.dumps(row) + "\n")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": avg_loss, "val_zs_acc": val_acc})

        ckpt = {
            "epoch": epoch,
            "vit": vit.state_dict(),
            "projection_heads": projection_heads.state_dict(),
            "logit_scale": logit_scale.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": avg_loss,
            "val_zs_acc": val_acc,
            "config": cfg,
        }
        torch.save(ckpt, args.output_dir / "last.pt")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, args.output_dir / "best.pt")

    _plot_curves(history, args.output_dir)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
