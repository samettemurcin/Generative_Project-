"""
decoder.py — M2 Caption Generation Module
==========================================
Responsibilities:
  - PrefixProjection: maps CLIP image embeddings → GPT-2 prefix tokens
  - build_inputs_embeds: constructs teacher-forcing inputs for training
  - generate_caption: inference-time prefix-conditioned caption generation
  - run_generation: pipeline.py entry point for mode="generate"

Architecture:
  CLIP image emb [B, clip_dim]
    → PrefixProjection
    → K prefix tokens [B, K, gpt2_dim]
    → prepend to GPT-2 input_embeds
    → GPT-2 generates caption tokens autoregressively

Design decisions:
  - PrefixProjection is a single Linear layer (no activation, no dropout).
    On 1k samples a deeper MLP overfits in epoch 1. Add complexity only
    if val_loss plateaus after 3 epochs with LoRA.
  - generate_caption infers device from image_embedding.device so that
    Streamlit can call it without explicitly passing device.
  - GPT-2's word embedding table (wte) is accessed directly for teacher-
    forcing — no special setup, same table GPT-2 uses internally.
  - Labels: prefix positions → -100 (loss ignores them), pad tokens → -100,
    real caption tokens kept as-is. Forgetting either mask produces silent
    training failures that only appear in generated caption quality.

Milestone compatibility:
  M2: full implementation
  M3: unchanged — train.py calls these functions directly
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PrefixProjection
# ---------------------------------------------------------------------------

class PrefixProjection(nn.Module):
    """
    Projects a CLIP image embedding into K prefix tokens for GPT-2.

    The projection maps [B, clip_dim] → [B, K, gpt2_dim].
    These K tokens are prepended to the GPT-2 input_embeds,
    conditioning the decoder on the image content.

    Args:
        clip_dim   : CLIP output dimension (512 for ViT-B/32, 768 for ViT-L/14)
        gpt2_dim   : GPT-2 hidden dimension (768 for gpt2/gpt2-medium)
        num_prefix : Number of prefix tokens K. Default: 10.
                     More tokens = more conditioning capacity, more VRAM.
                     10 is the standard ClipCap value.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        gpt2_dim: int = 768,
        num_prefix: int = 10,
    ) -> None:
        super().__init__()
        self.num_prefix = num_prefix
        self.gpt2_dim   = gpt2_dim
        # 2-layer MLP: clip_dim → hidden → num_prefix * gpt2_dim
        # GELU + LayerNorm gives the projection enough capacity to discriminate
        # between different images (a single Linear collapses all images to the
        # same average caption — the MLP prevents this).
        hidden_dim = clip_dim * 2           # 1024 for ViT-B/32, 1536 for ViT-L/14
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_prefix * gpt2_dim, bias=True),
        )

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embedding: [B, clip_dim] — L2-normalised CLIP image embedding

        Returns:
            prefix_tokens: [B, K, gpt2_dim] — prepend to GPT-2 input_embeds
        """
        B   = clip_embedding.shape[0]
        out = self.projection(clip_embedding)              # [B, K * gpt2_dim]
        return out.view(B, self.num_prefix, self.gpt2_dim) # [B, K, gpt2_dim]


# ---------------------------------------------------------------------------
# Training helper: build_inputs_embeds
# ---------------------------------------------------------------------------

def build_inputs_embeds(
    clip_emb:    torch.Tensor,      # [B, clip_dim] — L2-normalised CLIP image embedding
    caption_ids: torch.Tensor,      # [B, seq_len]  — tokenised caption (padded)
    attn_mask:   torch.Tensor,      # [B, seq_len]  — 1=real token, 0=padding
    prefix_proj: PrefixProjection,
    gpt2_model,                     # GPT2LMHeadModel — accessed for wte only
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the full input to GPT-2 for teacher-forcing training.

    Concatenates prefix tokens (from CLIP image) with caption token embeddings.
    Constructs the labels tensor with -100 on prefix and padding positions so
    the cross-entropy loss only supervises real caption tokens.

    Returns:
        inputs_embeds : [B, K+seq_len, gpt2_dim]
        attention_mask: [B, K+seq_len]
        labels        : [B, K+seq_len]  — -100 on prefix and pad positions
    """
    B, seq_len = caption_ids.shape
    K          = prefix_proj.num_prefix

    # 1. Compute prefix tokens from image
    prefix_tokens = prefix_proj(clip_emb)                          # [B, K, 768]

    # 2. Convert caption token IDs to embeddings via GPT-2's embedding table
    word_embeds = gpt2_model.transformer.wte(caption_ids)          # [B, seq_len, 768]

    # 3. Concatenate prefix + caption embeddings
    inputs_embeds = torch.cat([prefix_tokens, word_embeds], dim=1) # [B, K+seq_len, 768]

    # 4. Build attention mask: prefix tokens are all real (all 1s)
    prefix_attn    = torch.ones(B, K, dtype=torch.long, device=device)
    attention_mask = torch.cat([prefix_attn, attn_mask], dim=1)    # [B, K+seq_len]

    # 5. Build labels — CRITICAL: mask prefix and padding positions with -100
    #    Loss must only flow through real caption tokens, not prefix or pad.
    prefix_labels  = torch.full((B, K), -100, dtype=torch.long, device=device)
    caption_labels = caption_ids.clone()
    caption_labels[caption_labels == tokenizer.pad_token_id] = -100
    labels = torch.cat([prefix_labels, caption_labels], dim=1)     # [B, K+seq_len]

    return inputs_embeds, attention_mask, labels


# ---------------------------------------------------------------------------
# Inference: generate_caption
# ---------------------------------------------------------------------------

def generate_caption(
    image_embedding: torch.Tensor,     # [1, clip_dim] — single image, L2-normalised
    prefix_proj:     PrefixProjection,
    gpt2_model,                        # GPT2LMHeadModel (eval mode)
    tokenizer,
    config:          dict,             # full config or {"generation": {...}} section
    device:          torch.device | None = None,
) -> str:
    """
    Full prefix-conditioned caption generation.

    device is inferred from image_embedding.device if not provided — this
    allows Streamlit to call this function without explicitly passing device.

    Steps:
      1. Project image embedding → K prefix tokens [1, K, 768]
      2. Get GPT-2 word embedding for BOS token [1, 1, 768]
      3. Concatenate: [prefix | BOS] = [1, K+1, 768]
      4. GPT-2 .generate() uses inputs_embeds (not input_ids)
      5. Decode output_ids → string (no slicing — inputs_embeds excluded from output)

    Returns:
        Plain string caption, stripped, no special tokens.
    """
    if device is None:
        device = image_embedding.device

    K       = prefix_proj.num_prefix
    gen_cfg = config.get("generation", config)  # accept both full config and sub-dict

    with torch.no_grad():
        # Step 1: prefix tokens from image
        prefix = prefix_proj(image_embedding)                          # [1, K, 768]

        # Step 2: BOS token embedding
        # GPT-2 doesn't have a separate BOS — use eos_token_id as the start signal.
        bos_id  = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos_emb = gpt2_model.transformer.wte(
            torch.tensor([[bos_id]], device=device)
        )                                                              # [1, 1, 768]

        # Step 3: concatenate prefix + BOS
        inputs_embeds = torch.cat([prefix, bos_emb], dim=1)           # [1, K+1, 768]

        # Step 4: attention mask — all K+1 positions are real (prefix + BOS, no padding)
        attention_mask = torch.ones(1, K + 1, dtype=torch.long, device=device)

        # Step 5: build generation kwargs from config
        strategy = gen_cfg.get("decoding_strategy", "greedy")
        gen_kwargs: dict[str, Any] = {
            "inputs_embeds"       : inputs_embeds,
            "attention_mask"      : attention_mask,
            "max_new_tokens"      : gen_cfg.get("max_new_tokens", 50),
            "pad_token_id"        : tokenizer.eos_token_id,
            "eos_token_id"        : tokenizer.eos_token_id,
            "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 3),
            "length_penalty"      : gen_cfg.get("length_penalty", 1.0),
        }

        if strategy == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1

        elif strategy == "beam":
            gen_kwargs["do_sample"]      = False
            gen_kwargs["num_beams"]      = gen_cfg.get("beam_width", 5)
            gen_kwargs["early_stopping"] = True

        elif strategy == "nucleus":
            gen_kwargs["do_sample"]   = True
            gen_kwargs["top_p"]       = gen_cfg.get("top_p", 0.9)
            gen_kwargs["temperature"] = gen_cfg.get("temperature", 1.0)
            gen_kwargs["num_beams"]   = 1

        else:
            raise ValueError(f"Unknown decoding strategy: '{strategy}'. "
                             f"Choose from: greedy, beam, nucleus")

        # Step 6: generate
        output_ids = gpt2_model.generate(**gen_kwargs)

        # Step 7: decode output
        # When inputs_embeds is used, generate() does NOT include the input
        # embedding positions in the output — output_ids contains only the
        # newly generated token IDs. No slicing needed.
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Pipeline entry point: run_generation
# ---------------------------------------------------------------------------

def run_generation(
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    M2 batch generation pipeline — called by pipeline.py when mode="generate".

    Requires:
      - outputs/embeddings.npz from M1 must exist
      - outputs/weights/best/checkpoint_best.pt must exist (trained model)

    Writes:
      - outputs/captions.jsonl — one JSON line per sample

    Returns:
        Result dict: mode, output_path, total_samples, summary
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from src.embeddings_io import load_embeddings

    output_dir   = Path(config["output"]["dir"])
    weights_dir  = Path(config["output"].get("weights_dir", "outputs/weights"))
    ckpt_path    = weights_dir / "best" / "checkpoint_best.pt"
    captions_out = output_dir / "captions.jsonl"

    # Load embeddings from M1
    emb_path = output_dir / config["output"]["filename"]
    if not emb_path.exists():
        raise FileNotFoundError(
            f"M1 embeddings not found at {emb_path}. "
            "Run pipeline with mode='extract' first."
        )

    logger.info("Loading M1 embeddings from %s", emb_path)
    data = load_embeddings(str(emb_path), config)
    image_embeddings = torch.tensor(data["image_embeddings"], dtype=torch.float32)
    ids    = data["ids"]
    labels = data["labels"]

    # Load trained model
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at {ckpt_path}. "
            "Run train.py first: python train.py --run_id run_001"
        )

    logger.info("Loading checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    clip_dim   = image_embeddings.shape[-1]
    gpt2_name  = config["models"]["gpt2"]

    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_name).to(device).eval()
    tokenizer  = AutoTokenizer.from_pretrained(gpt2_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpt2_dim   = gpt2_model.config.n_embd
    num_prefix = config["generation"].get("num_prefix_tokens", 10)

    prefix_proj = PrefixProjection(
        clip_dim=clip_dim, gpt2_dim=gpt2_dim, num_prefix=num_prefix
    ).to(device)
    prefix_proj.load_state_dict(ckpt["prefix_proj"])

    if ckpt.get("lora_adapter") is not None:
        gpt2_model.load_state_dict(ckpt["lora_adapter"], strict=False)

    prefix_proj.eval()
    logger.info("Models loaded. Generating captions for %d samples...", len(ids))

    # Generate captions
    output_dir.mkdir(parents=True, exist_ok=True)
    n_generated = 0

    with open(captions_out, "w", encoding="utf-8") as f:
        for i, (img_id, label) in enumerate(zip(ids, labels)):
            if i % 50 == 0:
                logger.info("Generating %d/%d", i, len(ids))

            img_emb = image_embeddings[i].unsqueeze(0).to(device)
            caption = generate_caption(
                image_embedding=img_emb,
                prefix_proj=prefix_proj,
                gpt2_model=gpt2_model,
                tokenizer=tokenizer,
                config=config,
                device=device,
            )

            record = {
                "image_id"  : str(img_id),
                "label"     : str(label),
                "generated" : caption,
            }
            f.write(json.dumps(record) + "\n")
            n_generated += 1

    summary = f"Generation complete | samples={n_generated} | output={captions_out}"
    logger.info(summary)

    return {
        "mode"         : "generate",
        "output_path"  : str(captions_out),
        "total_samples": n_generated,
        "summary"      : summary,
    }
