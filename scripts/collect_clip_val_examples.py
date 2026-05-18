"""Save 5 correct + 5 wrong EuroSAT val predictions from a CLIP checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads
from vlm.data import EUROSAT_CLASSES, IMAGENET_MEAN, IMAGENET_STD, build_eurosat_loaders


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("runs/clip_eurosat/best.pt"))
    p.add_argument("--out", type=Path, default=Path("qualitative_analysis_images"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _fix_ssl_for_hf()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vit_cfg = cfg["vit"]

    _, val_dl, _ = build_eurosat_loaders(
        img_size=int(vit_cfg["img_size"]),
        batch_size=128,
        num_workers=0,
    )

    vit = ViT(
        img_size=int(vit_cfg["img_size"]),
        patch_size=int(vit_cfg["patch_size"]),
        d_model=int(vit_cfg["d_model"]),
        num_heads=int(vit_cfg["num_heads"]),
        num_blocks=int(vit_cfg["num_blocks"]),
        dropout=float(vit_cfg.get("dropout", 0.1)),
    ).to(device)
    text_encoder = FrozenTextEncoder(model_name=cfg["text_encoder"]["model_name"]).to(device)
    projection_heads = ProjectionHeads(
        d_image=vit.d_model,
        d_text=text_encoder.embedding_dim,
        d_proj=int(cfg["projection"]["d_proj"]),
    ).to(device)
    vit.load_state_dict(ckpt["vit"])
    projection_heads.load_state_dict(ckpt["projection_heads"])
    vit.eval()
    projection_heads.eval()

    class_prompts = [f"a satellite image of {name}" for name in EUROSAT_CLASSES]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    def tensor_to_pil(x: torch.Tensor) -> Image.Image:
        x = x.detach().cpu().clamp(0, 1)
        arr = (x * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    correct_rows: list[dict] = []
    wrong_rows: list[dict] = []
    global_i = 0

    with torch.no_grad():
        text_embeds = text_encoder(class_prompts)
        _, class_proj = projection_heads(
            torch.zeros(len(class_prompts), vit.d_model, device=device),
            text_embeds,
        )
        class_proj = F.normalize(class_proj, dim=-1)

        for images, captions in val_dl:
            images = images.to(device)
            labels = torch.tensor([class_prompts.index(c) for c in captions], device=device)
            feats = vit(images)
            img_proj, _ = projection_heads(feats, torch.zeros_like(text_embeds[:1]))
            img_proj = F.normalize(img_proj, dim=-1)
            sims = img_proj @ class_proj.T
            top3 = sims.topk(3, dim=-1)
            preds = sims.argmax(dim=-1)
            for b in range(images.size(0)):
                row = {
                    "idx": global_i,
                    "true": int(labels[b].item()),
                    "pred": int(preds[b].item()),
                    "top3_idx": top3.indices[b].tolist(),
                    "top3_sim": [float(x) for x in top3.values[b].tolist()],
                    "img_cpu": ((images[b] * std + mean).cpu().clamp(0, 1)),
                }
                if preds[b] == labels[b]:
                    correct_rows.append(row)
                else:
                    wrong_rows.append(row)
                global_i += 1

    random.seed(args.seed)
    correct_pick = random.sample(correct_rows, min(5, len(correct_rows)))
    wrong_pick = random.sample(wrong_rows, min(5, len(wrong_rows)))

    summary: dict = {
        "checkpoint": str(args.ckpt.resolve()),
        "epoch_saved": ckpt.get("epoch"),
        "val_zs_acc_ckpt": float(ckpt.get("val_zs_acc", 0.0)),
        "examples": [],
    }

    for i, row in enumerate(correct_pick):
        pil = tensor_to_pil(row["img_cpu"])
        path = args.out / f"correct_{i + 1}_idx{row['idx']}.png"
        pil.save(path)
        summary["examples"].append(
            {"file": path.name, "kind": "correct", **{k: v for k, v in row.items() if k != "img_cpu"}}
        )

    for i, row in enumerate(wrong_pick):
        pil = tensor_to_pil(row["img_cpu"])
        path = args.out / f"wrong_{i + 1}_idx{row['idx']}.png"
        pil.save(path)
        summary["examples"].append(
            {"file": path.name, "kind": "wrong", **{k: v for k, v in row.items() if k != "img_cpu"}}
        )

    summary_path = Path("runs/clip_eurosat/val_qualitative_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
