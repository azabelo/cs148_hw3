"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_causal_mask, build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]

IGNORE_INDEX = -100


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _encode_visual(
        self, images: torch.Tensor, injection: InjectionMode
    ) -> torch.Tensor:
        """Projected visual token embeddings, shape (B, N_vis, d_decoder)."""
        if injection == "cls":
            feats = self.vit(images)
            if feats.ndim == 2:
                feats = feats.unsqueeze(1)
        else:
            feats = self.vit(images, return_all_tokens=True)
        return self.projector(feats)

    def _text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.decoder.get_input_embeddings()
        return embed(input_ids)

    def _prepend_visual(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        """Prefix layout: [visual..., text...]."""
        n_vis = visual_embeds.shape[1]
        n_text = text_embeds.shape[1]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        vis_mask = torch.ones(
            attention_mask.shape[0],
            n_vis,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attn = torch.cat([vis_mask, attention_mask], dim=1)

        adj_labels = None
        if labels is not None:
            vis_labels = torch.full(
                (labels.shape[0], n_vis),
                IGNORE_INDEX,
                device=labels.device,
                dtype=labels.dtype,
            )
            adj_labels = torch.cat([vis_labels, labels], dim=1)

        return inputs_embeds, attn, adj_labels, n_vis, n_text

    def _interleave_visual(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        """Replace each <image> placeholder with the full visual sequence."""
        if self.image_token_id is None:
            raise ValueError("image_token_id must be set for interleaved injection")

        B, n_vis, _ = visual_embeds.shape
        device = text_embeds.device
        dtype = text_embeds.dtype

        seq_embeds: list[torch.Tensor] = []
        seq_attn: list[torch.Tensor] = []
        seq_labels: list[torch.Tensor] | None = [] if labels is not None else None
        n_text = 0

        vis_m = torch.ones(n_vis, device=device, dtype=attention_mask.dtype)
        label_dtype = labels.dtype if labels is not None else torch.long
        vis_l = torch.full((n_vis,), IGNORE_INDEX, device=device, dtype=label_dtype)

        for b in range(B):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if img_pos.numel() == 0:
                raise ValueError("interleaved mode requires an <image> token in each sample")

            emb_parts: list[torch.Tensor] = []
            attn_parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []
            prev = 0
            for pos in img_pos.tolist():
                emb_parts.append(text_embeds[b, prev:pos])
                emb_parts.append(visual_embeds[b])
                attn_parts.append(attention_mask[b, prev:pos])
                attn_parts.append(vis_m)
                if seq_labels is not None:
                    label_parts.append(labels[b, prev:pos])
                    label_parts.append(vis_l)
                prev = pos + 1

            emb_parts.append(text_embeds[b, prev:])
            attn_parts.append(attention_mask[b, prev:])
            emb = torch.cat(emb_parts, dim=0)
            seq_embeds.append(emb)
            seq_attn.append(torch.cat(attn_parts, dim=0))
            n_text = emb.shape[0] - n_vis * len(img_pos)

            if seq_labels is not None:
                label_parts.append(labels[b, prev:])
                seq_labels.append(torch.cat(label_parts, dim=0))

        max_len = max(s.shape[0] for s in seq_embeds)
        d_dec = text_embeds.shape[-1]
        pad_emb = torch.zeros(d_dec, dtype=dtype, device=device)
        padded_embeds = []
        padded_attn = []
        padded_labels = [] if seq_labels is not None else None

        for b in range(B):
            emb = seq_embeds[b]
            attn = seq_attn[b]
            pad_len = max_len - emb.shape[0]
            if pad_len:
                emb = torch.cat(
                    [emb, pad_emb.unsqueeze(0).expand(pad_len, -1)], dim=0
                )
                attn = torch.cat(
                    [
                        attn,
                        torch.zeros(pad_len, device=device, dtype=attention_mask.dtype),
                    ],
                    dim=0,
                )
            padded_embeds.append(emb)
            padded_attn.append(attn)
            if padded_labels is not None:
                lab = seq_labels[b]
                if pad_len:
                    lab = torch.cat(
                        [
                            lab,
                            torch.full(
                                (pad_len,),
                                IGNORE_INDEX,
                                device=device,
                                dtype=labels.dtype,
                            ),
                        ],
                        dim=0,
                    )
                padded_labels.append(lab)

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        attn = torch.stack(padded_attn, dim=0)
        adj_labels = (
            torch.stack(padded_labels, dim=0) if padded_labels is not None else None
        )
        return inputs_embeds, attn, adj_labels, n_vis, n_text

    def _build_attention_mask(
        self,
        mask_mode: MaskMode,
        injection: InjectionMode,
        attention_mask: torch.Tensor,
        n_vis: int,
        n_text: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """4D additive mask (B, 1, T, T) for prefix injection; 2D pad mask otherwise."""
        if injection == "interleaved":
            return attention_mask

        seq_len = n_vis + n_text
        if mask_mode == "causal":
            base = build_causal_mask(seq_len, attention_mask.device, dtype)
        else:
            base = build_image_bidir_mask(
                n_vis, n_text, attention_mask.device, dtype
            )

        min_val = torch.finfo(dtype).min
        pad = (1.0 - attention_mask.to(dtype)) * min_val
        return base + pad.unsqueeze(1).unsqueeze(1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        visual_embeds = self._encode_visual(images, injection)
        text_embeds = self._text_embeds(input_ids)

        if injection == "interleaved":
            inputs_embeds, attn, adj_labels, n_vis, n_text = self._interleave_visual(
                visual_embeds, text_embeds, input_ids, attention_mask, labels
            )
        else:
            inputs_embeds, attn, adj_labels, n_vis, n_text = self._prepend_visual(
                visual_embeds, text_embeds, attention_mask, labels
            )

        decoder_dtype = next(self.decoder.parameters()).dtype
        inputs_embeds = inputs_embeds.to(decoder_dtype)

        attn_arg = self._build_attention_mask(
            mask_mode, injection, attn, n_vis, n_text, decoder_dtype
        )

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_arg,
            labels=adj_labels,
            use_cache=False,
        )

        result: dict = {"logits": outputs.logits}
        if adj_labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        device = images.device
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        visual_embeds = self._encode_visual(images, injection)
        text_embeds = self._text_embeds(input_ids)

        if injection == "interleaved":
            inputs_embeds, attn, _, n_vis, n_text = self._interleave_visual(
                visual_embeds, text_embeds, input_ids, attention_mask, None
            )
        else:
            inputs_embeds, attn, _, n_vis, n_text = self._prepend_visual(
                visual_embeds, text_embeds, attention_mask, None
            )

        decoder_dtype = next(self.decoder.parameters()).dtype
        inputs_embeds = inputs_embeds.to(decoder_dtype)

        attn_arg = self._build_attention_mask(
            mask_mode, injection, attn, n_vis, n_text, decoder_dtype
        )
        if attn_arg.ndim == 4:
            attn_arg = attn

        gen_defaults = {"do_sample": False, "pad_token_id": self.tokenizer.pad_token_id}
        gen_defaults.update(gen_kwargs)

        token_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_arg,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **gen_defaults,
        )
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
