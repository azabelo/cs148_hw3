"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.lora import LoRALinear
from basics.vit import ViT
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy
from vlm.model import IGNORE_INDEX, VisionLanguageModel
from vlm.projector import VisionLanguageProjector


def _decoder_attn_implementation(requested: str) -> str:
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[warn] flash_attn not found; using sdpa", flush=True)
        return "sdpa"
    return requested


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
    p.add_argument(
        "--pretrained-vit",
        type=Path,
        required=True,
        help="Path to CLIP-pretrained ViT checkpoint from §3",
    )
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="image_bidir",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
        "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override train.num_steps (e.g. short smoke runs).",
    )
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument("--wandb-project", default="cs148-hw3-vlm", help="W&B project")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    p.add_argument(
        "--wandb-group",
        default=None,
        help="W&B group (e.g. injection_compare).",
    )
    p.add_argument(
        "--eval-max-examples",
        type=int,
        default=None,
        help="Override train.eval_max_examples for validation.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override train.gradient_accumulation_steps.",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override train.eval_every_steps (0 = final eval only).",
    )
    return p.parse_args()


IMAGE_PLACEHOLDER = "<image>"


def _build_prompt(question: str, injection: str) -> str:
    if injection == "interleaved":
        return f"Question: {IMAGE_PLACEHOLDER} {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def _build_train_text(question: str, answer: str, injection: str) -> str:
    return _build_prompt(question, injection) + f" {answer}"


def _extract_answer(prediction: str) -> str:
    """Keep only the model's answer span for CLEVR exact-match grading."""
    text = prediction.strip()
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1]
    # First line / sentence only.
    text = text.split("\n", 1)[0].strip()
    return text


def _tokenize_batch(
    tokenizer,
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def _labels_for_sft(
    tokenizer,
    input_ids: torch.Tensor,
    prompt_texts: list[str],
) -> torch.Tensor:
    """Mask prompt tokens; supervise only the answer span."""
    labels = input_ids.clone()
    pad_id = tokenizer.pad_token_id
    for i, prompt in enumerate(prompt_texts):
        prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        labels[i, :prompt_len] = IGNORE_INDEX
    if pad_id is not None:
        labels[input_ids == pad_id] = IGNORE_INDEX
    return labels


def _apply_decoder_lora(decoder: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    """Wrap SmolLM2 q_proj / v_proj with LoRA (§5.6 config B)."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in decoder.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in ("q_proj", "v_proj"):
            parent_name, child_name = name.rsplit(".", 1)
            parent = decoder.get_submodule(parent_name)
            replacements.append((parent, child_name, module))
    for parent, child_name, module in replacements:
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))


def _apply_freeze_config(model: VisionLanguageModel, freeze_config: str) -> None:
    for p in model.vit.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False

    if freeze_config == "A":
        for p in model.projector.parameters():
            p.requires_grad = True
    elif freeze_config == "B":
        for p in model.projector.parameters():
            p.requires_grad = True
        _apply_decoder_lora(model.decoder, rank=8, alpha=16.0)
        for name, p in model.decoder.named_parameters():
            if name.endswith((".A", ".B")):
                p.requires_grad = True
    elif freeze_config == "C":
        for p in model.projector.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif freeze_config == "D":
        for p in model.vit.parameters():
            p.requires_grad = True
        for p in model.projector.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    else:
        raise ValueError(freeze_config)


def visual_tokens_per_example(injection: str, vit: ViT) -> int:
    if injection == "cls":
        return 1
    return vit.num_patches + 1


@torch.no_grad()
def _evaluate_clevr(
    model: VisionLanguageModel,
    val_loader,
    device: torch.device,
    injection: str,
    mask_mode: str,
    max_examples: int,
    gen_cfg: dict,
    eval_batch_size: int = 4,
) -> dict[str, float]:
    model.eval()
    model.vit.eval()

    predictions: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []
    seen = 0

    for batch in tqdm(val_loader, desc="val", leave=False):
        if seen >= max_examples:
            break
        images = batch["image"]
        questions = batch["question"]
        answers = batch["answer"]
        q_type_batch = batch["q_type"]

        for start in range(0, len(questions), eval_batch_size):
            if seen >= max_examples:
                break
            end = min(start + eval_batch_size, len(questions))
            img_chunk = images[start:end].to(device, non_blocking=True)
            q_chunk = questions[start:end]
            a_chunk = answers[start:end]
            qt_chunk = q_type_batch[start:end]

            prompts = [_build_prompt(q, injection) for q in q_chunk]
            gens = model.generate(
                img_chunk,
                prompts,
                injection=injection,
                mask_mode=mask_mode,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
                temperature=float(gen_cfg.get("temperature", 1.0)),
                top_p=float(gen_cfg.get("top_p", 1.0)),
            )

            for pred, gold, qt in zip(gens, a_chunk, qt_chunk):
                if seen >= max_examples:
                    break
                predictions.append(_extract_answer(pred))
                golds.append(gold)
                q_types.append(qt)
                seen += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return batch_clevr_accuracy(predictions, golds, q_types)


def main() -> None:
    args = parse_args()
    _fix_ssl_for_hf()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs")
            / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_cfg = cfg["train"]
    optim_cfg = cfg["optim"]
    gen_cfg = cfg.get("generation", {})

    ckpt = torch.load(args.pretrained_vit, map_location="cpu", weights_only=False)
    vit_cfg = ckpt.get("config", {}).get("vit", {})
    img_size = int(vit_cfg.get("img_size", 64))

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))
    train_dl, val_dl = build_clevr_loaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Smaller val batches reduce dataloader memory; generation uses micro-batches.
    _, val_dl = build_clevr_loaders(
        img_size=img_size,
        batch_size=min(batch_size, 8),
        num_workers=num_workers,
    )

    vit = ViT(
        img_size=img_size,
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg.get("dropout", 0.1)),
    )
    vit.load_state_dict(ckpt["vit"], strict=False)
    vit.to(device)

    dec_cfg = cfg["decoder"]
    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id: int | None = None
    if args.injection == "interleaved":
        special = {IMAGE_PLACEHOLDER}
        tokenizer.add_special_tokens({"additional_special_tokens": list(special)})
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)

    dtype = getattr(torch, dec_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = dec_cfg.get("attn_implementation", "flash_attention_2")
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            dec_cfg["model_name"],
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
    except ImportError as exc:
        if "flash" in str(exc).lower() or attn_impl == "flash_attention_2":
            print(
                f"[decoder] FlashAttention unavailable ({exc}); using sdpa.",
                flush=True,
            )
            decoder = AutoModelForCausalLM.from_pretrained(
                dec_cfg["model_name"],
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        else:
            raise
    if image_token_id is not None:
        decoder.resize_token_embeddings(len(tokenizer))

    d_decoder = decoder.get_input_embeddings().embedding_dim
    projector = VisionLanguageProjector(
        d_image=vit.d_model,
        d_decoder=d_decoder,
        expansion=int(cfg.get("projector", {}).get("expansion", 4)),
    )

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)

    _apply_freeze_config(model, args.freeze_config)
    model.vit.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[freeze {args.freeze_config}] trainable={trainable:,} / {total:,}",
        flush=True,
    )

    lr = float(optim_cfg["lr"])
    wd = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(float(b) for b in optim_cfg.get("betas", (0.9, 0.95)))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, betas=betas, weight_decay=wd)

    num_steps = (
        int(args.num_steps)
        if args.num_steps is not None
        else int(train_cfg["num_steps"])
    )
    grad_accum = (
        int(args.grad_accum)
        if args.grad_accum is not None
        else int(train_cfg.get("gradient_accumulation_steps", 1))
    )
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))

    n_vis_tokens = visual_tokens_per_example(args.injection, vit)
    log_every = int(train_cfg.get("log_every", 25))
    if args.eval_every is not None:
        eval_every = int(args.eval_every)
    else:
        eval_every = int(train_cfg.get("eval_every_steps", 200))
        if num_steps < eval_every:
            eval_every = max(num_steps // 2, 1) if num_steps > 1 else 0
    eval_max = (
        int(args.eval_max_examples)
        if args.eval_max_examples is not None
        else int(train_cfg.get("eval_max_examples", 500))
    )
    eval_microbatch = 4 if args.injection == "cls" else 2

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run_name or (
            f"{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
        init_kw: dict = {
            "project": args.wandb_project,
            "name": run_name,
            "config": {
                **cfg,
                "injection": args.injection,
                "mask_mode": args.mask_mode,
                "freeze_config": args.freeze_config,
                "num_steps": num_steps,
                "visual_tokens_per_example": n_vis_tokens,
                "pretrained_vit": str(args.pretrained_vit),
            },
        }
        if args.wandb_group:
            init_kw["group"] = args.wandb_group
        wandb_run = wandb.init(**init_kw)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    opt_step = 0
    best_val_acc = -1.0
    step_times: list[float] = []
    train_iter = iter(train_dl)
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(num_steps), desc=f"train {args.injection}")

    def lr_lambda(opt_step_idx: int) -> float:
        if warmup_steps > 0 and opt_step_idx < warmup_steps:
            return float(opt_step_idx + 1) / float(warmup_steps)
        t = float(opt_step_idx - warmup_steps) / float(
            max(1, (num_steps // grad_accum) - warmup_steps)
        )
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for step in pbar:
        step_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        images = batch["image"].to(device, non_blocking=True)
        questions = batch["question"]
        answers = batch["answer"]

        full_texts = [
            _build_train_text(q, a, args.injection) for q, a in zip(questions, answers)
        ]
        prompt_texts = [_build_prompt(q, args.injection) for q in questions]

        input_ids, attention_mask = _tokenize_batch(tokenizer, full_texts, device)
        labels = _labels_for_sft(tokenizer, input_ids, prompt_texts)

        model.projector.train()
        if args.freeze_config in ("B", "C", "D"):
            model.decoder.train()
        else:
            model.decoder.eval()

        out = model(
            images,
            input_ids,
            attention_mask,
            labels=labels,
            injection=args.injection,
            mask_mode=args.mask_mode,
        )
        loss = out["loss"] / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == num_steps:
            grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            opt_step += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        global_step = step + 1
        if log_every > 0 and global_step % log_every == 0:
            pbar.set_postfix(
                loss=f"{(loss.item() * grad_accum):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                step_s=f"{step_time:.3f}",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": global_step,
                        "train/loss": float(loss.item() * grad_accum),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": float(grad_norm),
                        "train/step_time_s": step_time,
                    }
                )

        if eval_every > 0 and global_step % eval_every == 0:
            metrics = _evaluate_clevr(
                model,
                val_dl,
                device,
                args.injection,
                args.mask_mode,
                eval_max,
                gen_cfg,
                eval_batch_size=eval_microbatch,
            )
            val_acc = metrics["overall"]
            print(
                f"[step {global_step}] val_exact_match={val_acc:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"step": global_step, "val/exact_match": val_acc, **{
                        f"val/{k}": v for k, v in metrics.items() if k != "overall"
                    }},
                )
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "config": cfg,
                        "injection": args.injection,
                        "mask_mode": args.mask_mode,
                        "freeze_config": args.freeze_config,
                        "val_exact_match": val_acc,
                        "vit_cfg": vit_cfg,
                        "image_token_id": image_token_id,
                    },
                    args.output_dir / "best.pt",
                )

    # Final validation on up to eval_max examples.
    final_metrics = _evaluate_clevr(
        model,
        val_dl,
        device,
        args.injection,
        args.mask_mode,
        eval_max,
        gen_cfg,
        eval_batch_size=eval_microbatch,
    )
    final_val_acc = final_metrics["overall"]

    peak_mem_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    avg_step_time = sum(step_times) / max(len(step_times), 1)

    summary = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "visual_tokens_per_example": n_vis_tokens,
        "val_exact_match": final_val_acc,
        "val_metrics": final_metrics,
        "best_val_exact_match": max(best_val_acc, final_val_acc),
        "peak_gpu_mem_bytes": peak_mem_bytes,
        "peak_gpu_mem_mib": peak_mem_bytes / (1024**2),
        "avg_step_time_s": avg_step_time,
        "pretrained_vit": str(args.pretrained_vit),
    }
    with open(args.output_dir / "metrics.json", "w") as mf:
        json.dump(summary, mf, indent=2)

    torch.save(
        {
            "step": global_step,
            "model": model.state_dict(),
            "config": cfg,
            "injection": args.injection,
            "mask_mode": args.mask_mode,
            "freeze_config": args.freeze_config,
            "val_exact_match": final_val_acc,
            "vit_cfg": vit_cfg,
            "image_token_id": image_token_id,
        },
        args.output_dir / "last.pt",
    )

    print(
        f"[{args.injection}] DONE  val_exact_match={final_val_acc:.4f}  "
        f"visual_tokens={n_vis_tokens}  peak_mem_mib={summary['peak_gpu_mem_mib']:.1f}  "
        f"avg_step_s={avg_step_time:.3f}",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
