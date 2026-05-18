"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from basics.vit import ViT
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy, clevr_exact_match
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector

IMAGE_PLACEHOLDER = "<image>"


def _decoder_attn_implementation(requested: str) -> str:
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401
    except ImportError:
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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to dump for qualitative inspection",
    )
    p.add_argument(
        "--max-eval",
        type=int,
        default=500,
        help="Number of examples to use for accuracy computation",
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help="Save the example images alongside the JSON output",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_prompt(question: str, injection: str) -> str:
    if injection == "interleaved":
        return f"Question: {IMAGE_PLACEHOLDER} {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    vit_cfg = ckpt.get("vit_cfg", cfg.get("vit", {}))
    injection = ckpt.get("injection", "all_patches")
    mask_mode = ckpt.get("mask_mode", "image_bidir")
    image_token_id = ckpt.get("image_token_id")

    img_size = int(vit_cfg.get("img_size", 64))
    vit = ViT(
        img_size=img_size,
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg.get("dropout", 0.1)),
    )

    dec_cfg = cfg.get("decoder", {})
    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if injection == "interleaved" and image_token_id is None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [IMAGE_PLACEHOLDER]}
        )
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)

    attn_impl = _decoder_attn_implementation(
        dec_cfg.get("attn_implementation", "flash_attention_2")
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        dec_cfg["model_name"],
        torch_dtype=getattr(torch, dec_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=attn_impl,
    )
    if image_token_id is not None and decoder.get_input_embeddings().num_embeddings < len(
        tokenizer
    ):
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    model.vit.eval()

    gen_cfg = cfg.get("generation", {})
    return model, injection, mask_mode, gen_cfg, img_size


def main() -> None:
    args = parse_args()
    _fix_ssl_for_hf()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)

    model, injection, mask_mode, gen_cfg, img_size = _load_model(
        args.checkpoint, device
    )

    _, val_dl = build_clevr_loaders(
        img_size=img_size,
        batch_size=16,
        num_workers=2,
    )

    predictions: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []
    pool: list[dict] = []
    seen = 0

    for batch in tqdm(val_dl, desc="eval"):
        if seen >= args.max_eval:
            break
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        q_type_batch = batch["q_type"]

        prompts = [_build_prompt(q, injection) for q in questions]
        gens = model.generate(
            images,
            prompts,
            injection=injection,
            mask_mode=mask_mode,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
        )

        for i, (pred, gold, qt, q) in enumerate(
            zip(gens, answers, q_type_batch, questions)
        ):
            if seen >= args.max_eval:
                break
            pred_clean = pred.strip()
            correct = clevr_exact_match(pred_clean, gold)
            predictions.append(pred_clean)
            golds.append(gold)
            q_types.append(qt)
            pool.append(
                {
                    "index": seen,
                    "question": q,
                    "gold": gold,
                    "prediction": pred_clean,
                    "q_type": qt,
                    "correct": correct,
                    "image_tensor": batch["image"][i].cpu(),
                }
            )
            seen += 1

    metrics = batch_clevr_accuracy(predictions, golds, q_types)
    print(f"Overall exact-match accuracy ({seen} examples): {metrics['overall']:.4f}")
    for k, v in sorted(metrics.items()):
        if k != "overall":
            print(f"  {k}: {v:.4f}")

    correct_pool = [ex for ex in pool if ex["correct"]]
    wrong_pool = [ex for ex in pool if not ex["correct"]]
    n_correct = min(len(correct_pool), max(1, args.num_examples // 2))
    n_wrong = args.num_examples - n_correct
    if len(wrong_pool) < n_wrong:
        n_wrong = len(wrong_pool)
        n_correct = args.num_examples - n_wrong
    selected = random.sample(correct_pool, min(n_correct, len(correct_pool)))
    selected += random.sample(wrong_pool, min(n_wrong, len(wrong_pool)))
    random.shuffle(selected)

    img_dir = args.output_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    examples_path = args.output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for rank, ex in enumerate(selected):
            row = {
                "rank": rank,
                "index": ex["index"],
                "question": ex["question"],
                "gold": ex["gold"],
                "prediction": ex["prediction"],
                "q_type": ex["q_type"],
                "correct": ex["correct"],
            }
            if args.save_images:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = ex["image_tensor"] * std + mean
                img = img.clamp(0, 1)
                pil = to_pil_image(img)
                fname = f"example_{rank:02d}.png"
                pil.save(img_dir / fname)
                row["image_file"] = str(img_dir / fname)
            f.write(json.dumps(row) + "\n")

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_eval": seen,
        "metrics": metrics,
        "num_qualitative": len(selected),
        "examples_path": str(examples_path),
    }
    with open(args.output_dir / "summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)

    print(f"\nWrote {len(selected)} qualitative examples to {examples_path}")
    print(f"{'rank':<5} {'ok':<5} {'q_type':<12} gold -> pred")
    for row in selected:
        mark = "✓" if row["correct"] else "✗"
        print(
            f"{row['index']:<5} {mark:<5} {row['q_type']:<12} "
            f"{row['gold']!r} -> {row['prediction']!r}"
        )


if __name__ == "__main__":
    main()
