"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt

LoRA rank sweep (r in {1,2,4,8,16,32,64}, 10 epochs each, alpha = 2r, W&B per rank + summary):
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --pretrained runs/clip_eurosat/best.pt --lora-rank-sweep --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from basics.lora import apply_lora_to_attention
from basics.vit import ViT
from vlm.data import build_resisc45_loaders


def _fix_ssl_for_hf() -> None:
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
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override train.num_epochs (e.g. for smoke tests).",
    )
    p.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument(
        "--wandb-project", default="cs148-hw3-resisc", help="W&B project name"
    )
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    p.add_argument(
        "--wandb-group",
        default=None,
        help="W&B group (e.g. to group rank-sweep child runs in the UI).",
    )
    p.add_argument(
        "--lora-rank-sweep",
        action="store_true",
        help="Run LoRA for each rank in --sweep-ranks (10 epochs each); "
        "alpha = --lora-alpha-per-rank * rank. Logs a summary plot to disk and W&B.",
    )
    p.add_argument(
        "--sweep-ranks",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated LoRA ranks (used with --lora-rank-sweep).",
    )
    p.add_argument(
        "--lora-alpha-per-rank",
        type=float,
        default=2.0,
        help="LoRA alpha = this value * rank (keeps alpha/rank constant; default 2 => alpha=2r).",
    )
    p.add_argument(
        "--sweep-output-dir",
        type=Path,
        default=None,
        help="Directory for sweep plot and sweep_results.json (default: runs/lora_rank_sweep_resisc).",
    )
    return p.parse_args()


class ViTClassifier(nn.Module):
    """ViT backbone + 45-way linear classification head on the CLS embedding."""

    def __init__(self, vit: ViT, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit(x)
        return self.head(feats)


def _count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _apply_method(
    model: ViTClassifier, method: str, rank: int, alpha: float
) -> None:
    """Mutate `model` in place to set requires_grad according to the strategy."""
    if method == "linear_probe":
        for p in model.vit.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif method == "lora":
        apply_lora_to_attention(model.vit, rank=rank, alpha=alpha)
        for p in model.head.parameters():
            p.requires_grad = True
    elif method == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def _evaluate(model: ViTClassifier, loader, device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.item())
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


def _parse_sweep_ranks(s: str) -> list[int]:
    ranks = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not ranks:
        raise ValueError("empty --sweep-ranks")
    return ranks


def _run_lora_rank_sweep(args: argparse.Namespace) -> None:
    """Train one run per rank via subprocess, then plot best test accuracy vs rank."""
    ranks = _parse_sweep_ranks(args.sweep_ranks)
    script_path = Path(__file__).resolve()
    sweep_dir = args.sweep_output_dir or (Path("runs") / "lora_rank_sweep_resisc")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    wb_group = args.wandb_group
    if args.wandb and wb_group is None:
        wb_group = "resisc_lora_rank_sweep"

    scale = float(args.lora_alpha_per_rank)
    for rank in ranks:
        alpha = scale * float(rank)
        cmd: list[str] = [
            sys.executable,
            str(script_path),
            "--config",
            str(args.config),
            "--method",
            "lora",
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
            "--pretrained",
            str(args.pretrained),
            "--num-epochs",
            "10",
            "--device",
            str(args.device),
        ]
        if args.lr is not None:
            cmd.extend(["--lr", str(args.lr)])
        if args.wandb:
            cmd.append("--wandb")
            cmd.extend(["--wandb-project", args.wandb_project])
            cmd.extend(["--wandb-run-name", f"lora_r{rank}_a{int(alpha)}"])
            if wb_group is not None:
                cmd.extend(["--wandb-group", wb_group])
        print(f"[lora-rank-sweep] rank={rank} alpha={alpha}\n  -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    results: list[dict] = []
    for rank in ranks:
        alpha_int = int(scale * float(rank))
        metrics_path = Path("runs") / f"resisc_lora_r{rank}_a{alpha_int}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics after sweep: {metrics_path}")
        with open(metrics_path) as f:
            summary = json.load(f)
        results.append(
            {
                "rank": rank,
                "alpha": scale * float(rank),
                "best_test_acc": summary["best_test_acc"],
                "final_test_acc": summary.get("final_test_acc"),
            }
        )

    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["rank"] for r in results]
    ys = [r["best_test_acc"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", color="tab:blue")
    plt.xlabel("LoRA rank r")
    plt.ylabel("Best test accuracy")
    plt.title(
        f"RESISC45 LoRA: test acc vs rank (α = {scale:g}·r, 10 epochs each)"
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(xs)
    plot_path = sweep_dir / "lora_rank_vs_test_acc.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[lora-rank-sweep] wrote {plot_path}", flush=True)

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "lora_rank_sweep_summary",
            job_type="sweep_summary",
            group=wb_group,
            config={
                "sweep_ranks": ranks,
                "lora_alpha_per_rank": scale,
                "num_epochs": 10,
                "pretrained_ckpt": str(args.pretrained),
            },
        )
        table = wandb.Table(
            columns=["rank", "alpha", "best_test_acc", "final_test_acc"]
        )
        for row in results:
            table.add_data(
                row["rank"],
                row["alpha"],
                row["best_test_acc"],
                row["final_test_acc"],
            )
        run.log(
            {
                "summary/plot": wandb.Image(str(plot_path)),
                "summary/results_table": table,
            }
        )
        run.finish()


def main() -> None:
    args = parse_args()
    _fix_ssl_for_hf()
    if args.lora_rank_sweep:
        if args.method != "lora":
            raise SystemExit("--lora-rank-sweep requires --method lora")
        _run_lora_rank_sweep(args)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}"
        if args.method == "lora":
            args.output_dir = Path("runs") / f"resisc_lora_r{args.rank}_a{int(args.alpha)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pretrained_cfg = ckpt.get("config", {})
    vit_cfg = pretrained_cfg.get("vit", {})
    img_size = int(vit_cfg.get("img_size", 64))

    train_dl, test_dl = build_resisc45_loaders(
        img_size=img_size,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    vit = ViT(
        img_size=img_size,
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg.get("dropout", 0.1)),
    )
    missing, unexpected = vit.load_state_dict(ckpt["vit"], strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing} unexpected={unexpected}", flush=True)

    num_classes = int(cfg["num_classes"])
    model = ViTClassifier(vit, num_classes=num_classes)

    _apply_method(model, args.method, rank=args.rank, alpha=args.alpha)
    model.to(device)

    trainable, total_params = _count_trainable_params(model)
    print(
        f"[{args.method}] trainable={trainable:,} / total={total_params:,} "
        f"({100.0 * trainable / max(total_params, 1):.3f}%)",
        flush=True,
    )

    method_overrides = cfg.get("methods", {}).get(args.method, {})
    base_lr = float(cfg["optim"]["lr"])
    lr = float(args.lr) if args.lr is not None else float(method_overrides.get("lr", base_lr))
    wd = float(cfg["optim"]["weight_decay"])
    betas = tuple(float(b) for b in cfg["optim"].get("betas", (0.9, 0.999)))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, betas=betas, weight_decay=wd)

    num_epochs = (
        int(args.num_epochs)
        if args.num_epochs is not None
        else int(cfg["train"]["num_epochs"])
    )
    warmup_steps = int(cfg["optim"].get("warmup_steps", 0))
    steps_per_epoch = len(train_dl)
    total_steps = max(1, num_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    log_every = int(cfg["train"].get("log_every", 25))

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or (
            f"{args.method}"
            + (f"_r{args.rank}_a{int(args.alpha)}" if args.method == "lora" else "")
        )
        init_kw: dict = {
            "project": args.wandb_project,
            "name": run_name,
            "config": {
                "method": args.method,
                "rank": args.rank,
                "alpha": args.alpha,
                "lr": lr,
                "weight_decay": wd,
                "betas": list(betas),
                "warmup_steps": warmup_steps,
                "num_epochs": num_epochs,
                "batch_size": int(cfg["train"]["batch_size"]),
                "img_size": img_size,
                "num_classes": num_classes,
                "pretrained_ckpt": str(args.pretrained),
                "trainable_params": trainable,
                "total_params": total_params,
                "vit": vit_cfg,
            },
        }
        if args.wandb_group:
            init_kw["group"] = args.wandb_group
        wandb_run = wandb.init(**init_kw)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict] = []
    global_step = 0
    best_test_acc = -1.0
    train_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        # For linear probe + LoRA the frozen backbone still needs eval-mode behavior
        # (dropout off) since we don't update those parameters.
        if args.method in ("linear_probe", "lora"):
            model.vit.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.perf_counter()
        pbar = tqdm(train_dl, desc=f"train e{epoch}", leave=False)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                running_correct += int((preds == labels).sum().item())
                running_total += int(labels.numel())
            running_loss += float(loss.item())
            global_step += 1

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "step": global_step,
                            "train/loss_step": float(loss.item()),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                    )

        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_loss / max(len(train_dl), 1)
        train_acc = running_correct / max(running_total, 1)
        test_acc, test_loss = _evaluate(model, test_dl, device)

        peak_mem_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_s": epoch_time,
            "peak_gpu_mem_bytes": peak_mem_bytes,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[{args.method}] epoch {epoch}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
            f"time={epoch_time:.1f}s  peak_mem={peak_mem_bytes / (1024 ** 2):.1f}MiB",
            flush=True,
        )

        with open(args.output_dir / "metrics.jsonl", "a") as mf:
            mf.write(json.dumps(row) + "\n")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "epoch_time_s": epoch_time,
                    "peak_gpu_mem_mib": peak_mem_bytes / (1024**2),
                }
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

    total_time = time.perf_counter() - train_start
    peak_mem_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    summary = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "alpha": args.alpha if args.method == "lora" else None,
        "lr": lr,
        "num_epochs": num_epochs,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "best_test_acc": best_test_acc,
        "trainable_params": trainable,
        "total_params": total_params,
        "peak_gpu_mem_bytes": peak_mem_bytes,
        "peak_gpu_mem_mib": peak_mem_bytes / (1024**2),
        "wall_clock_train_seconds": total_time,
        "pretrained_ckpt": str(args.pretrained),
    }
    with open(args.output_dir / "metrics.json", "w") as mf:
        json.dump(summary, mf, indent=2)

    print(
        f"[{args.method}] DONE  final_test_acc={summary['final_test_acc']}  "
        f"best_test_acc={best_test_acc:.4f}  "
        f"trainable_params={trainable:,}  "
        f"peak_mem_mib={summary['peak_gpu_mem_mib']:.1f}  "
        f"wall_clock_s={total_time:.1f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
