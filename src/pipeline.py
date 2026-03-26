"""
pipeline.py — Orchestrator
===========================
Responsibilities:
  - Load config, initialize all modules
  - Route execution based on pipeline.mode (extract | generate | evaluate)
  - M1 (extract): run CLIP + GPT-2 forward pass, save embeddings
  - M2 (generate): prefix-conditioned caption generation via decoder.py
  - M3 (evaluate): BLEU/CIDEr/ROUGE-L/METEOR evaluation via metrics.py
  - Memory management: batch processing, CUDA cache clearing
  - Checkpointed saves via EmbeddingCheckpointer

Design decisions:
  - pipeline.py is the ONLY entry point. Never call data_loader or
    preprocessor directly from a notebook — always go through here.
  - torch.no_grad() wraps ALL forward passes. This is not optional —
    retaining gradients for 1k samples at CLIP scale will OOM on 8GB VRAM.
  - Batch size comes from config, not hardcoded. Reduce batch_size in
    config.yaml to 16 if you hit OOM on the RTX 4060.
  - CLIP image encoder and GPT-2 are kept on the same device.
    Mixed-device runs are not supported in M1.

Milestone compatibility:
  M1: extract mode — full implementation
  M2: generate mode — implemented in decoder.py
  M3: evaluate mode — implemented in metrics.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from transformers import CLIPModel, GPT2Model

from src.data_loader import build_candidate_pool, flatten_candidates, get_dataloader_splits
from src.embeddings_io import EmbeddingCheckpointer, load_embeddings
from src.preprocessor import Preprocessor
from src.utils import (
    get_device,
    get_output_path,
    get_pipeline_mode,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config_path: str | None = None) -> dict[str, Any]:
    """
    Main pipeline entry point. Call this from notebooks or CLI.

    Args:
        config_path: Optional path to config.yaml override.
                     Defaults to configs/config.yaml relative to project root.

    Returns:
        Result dict containing:
            mode         : str — pipeline mode that was executed
            output_path  : str — path to saved output file
            total_samples: int — number of samples processed
            summary      : str — human-readable run summary
    """
    # --- Setup ---
    config = load_config(config_path) if config_path else load_config()
    setup_logging(config)
    set_seed(config["dataset"]["seed"])

    mode   = get_pipeline_mode(config)
    device = get_device()

    logger.info("=" * 60)
    logger.info("Pipeline starting | mode=%s | device=%s", mode, device)
    logger.info("=" * 60)

    if mode == "extract":
        return _run_extract(config, device)

    elif mode == "generate":
        # M2 — batch caption generation using a trained prefix-conditioned GPT-2
        logger.info("Mode 'generate' selected — running decoder module (M2)")
        from src.decoder import run_generation
        return run_generation(config, device)

    elif mode == "evaluate":
        # M3 — compute BLEU/CIDEr/ROUGE-L/METEOR across all runs
        logger.info("Mode 'evaluate' selected — running metrics module (M3)")
        from src.metrics import run_evaluation
        return run_evaluation(config, device)

    else:
        raise ValueError(f"Unknown pipeline mode: '{mode}'")


# ---------------------------------------------------------------------------
# M1: Extract mode
# ---------------------------------------------------------------------------

def _run_extract(
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    M1 extraction pipeline:
      1. Build candidate pool from the configured dataset
      2. Select and preprocess samples
      3. Run CLIP image encoder + GPT-2 text encoder in batches
      4. Save embeddings to .npz with checkpointing
    """
    batch_size = config["hardware"]["batch_size"]

    # ── Step 1: Data loading ──────────────────────────────────────────────
    logger.info("Step 1/4: Building candidate pool")
    buckets  = build_candidate_pool(config)
    selected = get_dataloader_splits(buckets, config)
    flat     = flatten_candidates(selected)

    logger.info("Candidate pool ready: %d raw samples", len(flat))

    # ── Step 2: Preprocessing ─────────────────────────────────────────────
    logger.info("Step 2/4: Preprocessing candidates")
    preprocessor = Preprocessor(config)
    selected_ids = {
        cls: {s["image_id"] for s in samples}
        for cls, samples in selected.items()
    }
    reserve_pool = {
        cls: [s for s in buckets[cls] if s["image_id"] not in selected_ids[cls]]
        for cls in buckets
    }

    result = preprocessor.process_candidates(
        flat_candidates=flat,
        reserve_pool=reserve_pool,  # ← true holdout candidates only
    )

    logger.info(result.summary())

    if result.total_valid == 0:
        raise RuntimeError(
            "No valid samples after preprocessing. "
            "Check your dataset name, classes, and internet connection."
        )

    valid_samples = result.valid_samples

    # ── Step 3: Model loading ─────────────────────────────────────────────
    logger.info("Step 3/4: Loading models")
    clip_model_name = config["models"]["clip"]
    gpt2_model_name = config["models"]["gpt2"]

    logger.info("Loading CLIP model: %s", clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    logger.info("Loading GPT-2 model: %s", gpt2_model_name)
    gpt2_model = GPT2Model.from_pretrained(gpt2_model_name).to(device)
    gpt2_model.eval()

    logger.info("Models loaded and set to eval mode")

    # Dimension alignment: GPT-2 hidden size (768) ≠ CLIP projection dim (512).
    # A frozen linear layer maps GPT-2’s mean-pooled output to CLIP’s embedding
    # space so both image and text embeddings share a common 512-d space in M1.
    # In M2, this projection is replaced by the prefix conditioning layer,
    # which is trained (via LoRA) rather than frozen.
    text_projection = None
    if gpt2_model.config.n_embd != clip_model.config.projection_dim:
        text_projection = torch.nn.Linear(gpt2_model.config.n_embd, clip_model.config.projection_dim, bias=False).to(
            device)
        torch.nn.init.kaiming_uniform_(text_projection.weight)
        text_projection.weight.requires_grad_(False)
        text_projection.eval()

    # ── Step 4: Embedding extraction ─────────────────────────────────────
    logger.info("Step 4/4: Extracting embeddings (%d samples, batch_size=%d)", len(valid_samples), batch_size)
    checkpointer = EmbeddingCheckpointer(config)

    with torch.no_grad():
        for batch_start in range(0, len(valid_samples), batch_size):
            batch = valid_samples[batch_start : batch_start + batch_size]

            if (batch_start // batch_size) % 10 == 0:
                logger.info(
                    "Batch %d/%d | accumulated=%d",
                    batch_start // batch_size + 1,
                    (len(valid_samples) + batch_size - 1) // batch_size,
                    checkpointer.accumulated_count,
                )

            # Stack batch tensors
            pixel_values = torch.stack([s["pixel_values"] for s in batch]).to(device)
            token_ids    = torch.stack([s["token_ids"]    for s in batch]).to(device)
            ids          = [s["image_id"] for s in batch]
            labels       = [s["label"]    for s in batch]

            # CLIP image encoder → pooled image embedding [B, 512]
            image_outputs     = clip_model.get_image_features(pixel_values=pixel_values)
            image_embeddings  = _normalize(image_outputs)

            # GPT-2 text encoder → mean-pooled last hidden state [B, 512]
            text_outputs      = gpt2_model(input_ids=token_ids)
            text_hidden       = text_outputs.last_hidden_state   # [B, 77, 768]
            text_embeddings   = _pool_and_project(text_hidden, projection=text_projection)

            # Move to CPU and convert to numpy before accumulating
            checkpointer.add_batch(
                image_embeddings = image_embeddings.cpu().numpy(),
                text_embeddings  = text_embeddings.cpu().numpy(),
                ids              = ids,
                labels           = labels,
            )

            # Clear CUDA cache between batches to prevent OOM on RTX 4060
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Finalize ──────────────────────────────────────────────────────────
    output_path = checkpointer.finalize()

    # Verify the saved file loads correctly
    logger.info("Verifying saved output...")
    verification = load_embeddings(output_path, config)
    n_saved      = len(verification["ids"])

    # Log per-class distribution of saved embeddings
    _log_class_distribution(verification)

    summary = (
        f"Extraction complete | "
        f"samples={n_saved} | "
        f"repaired={result.total_repaired} ({result.repair_rate:.1%}) | "
        f"skipped={result.total_skipped} | "
        f"output={output_path}"
    )
    logger.info(summary)
    logger.info("=" * 60)

    return {
        "mode":          "extract",
        "output_path":   str(output_path),
        "total_samples": n_saved,
        "summary":       summary,
        "repair_rate":   result.repair_rate,
        "skip_rate":     result.skip_rate,
    }


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embedding vectors.
    CLIP embeddings should be normalized before cosine similarity or storage.
    Shape: [B, D] → [B, D]
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


def _pool_and_project(
    hidden_states: torch.Tensor,
    projection: torch.nn.Linear | None = None,
) -> torch.Tensor:
    """
    Mean-pool the last hidden state across the sequence dimension,
    then optionally project to a target dimension and L2-normalise.

    Args:
        hidden_states: [B, seq_len, hidden_dim] — GPT-2 last_hidden_state
        projection:    Optional Linear layer mapping hidden_dim → target_dim.
                       If None, the pooled vector is normalised as-is.

    Returns:
        [B, target_dim] L2-normalised embedding tensor.
    """
    pooled = hidden_states.mean(dim=1)      # [B, hidden_dim]
    if projection is None:
        return _normalize(pooled)
    return _normalize(projection(pooled))   # [B, target_dim]


def _log_class_distribution(data: dict) -> None:
    """Log per-class sample count from saved embeddings."""
    labels  = data["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Per-class distribution in saved embeddings:")
    for cls, count in zip(unique, counts):
        logger.info("  %-15s : %d", cls, count)