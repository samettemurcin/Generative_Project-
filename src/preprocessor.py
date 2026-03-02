"""
preprocessor.py — Transformation, Validation & Repair Layer
=============================================================
Responsibilities:
  - Validate each raw image (mode, minimum size, decodability)
  - Apply repair pipeline on invalid images (mode convert → pad → upscale)
  - Apply CLIPProcessor (handles resize/crop/normalize internally)
  - Tokenize captions with GPT-2 tokenizer
  - Select exactly target_n samples across all classes
  - Handle shortfall: retry from reserve pool, then repair, then log

Design decisions:
  - CLIPProcessor is used instead of manual transforms. It owns the exact
    normalization constants (mean/std) for ViT-B/32. Manual transforms
    risk silent numerical misalignment with the pre-trained weights.
  - Repair is applied ONLY after max_retries exhausted. The repair
    pipeline is ordered: mode_convert → pad → upscale. Each step is
    the minimum intervention needed to produce a valid tensor.
  - Every repaired image is logged to repairs.jsonl via get_repair_logger()
    so M3 evaluation can audit dataset quality programmatically.
  - Token length is hard-capped at 77 (CLIP's positional embedding limit).
    GPT-2's native max is 1024; we truncate to 77 for CLIP alignment.

Milestone compatibility:
  M1: Full implementation — produces image tensors + token ids
  M2: Unchanged — pipeline.py will use the same tensors for decoder input
  M3: Unchanged — repair log is consumed by evaluation analysis
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoTokenizer

from src.utils import (
    get_per_class_target,
    get_repair_logger,
    load_config,
    log_repair_event,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Stateful preprocessor that holds loaded processor/tokenizer instances.
    Instantiate once and reuse across all batches to avoid reloading weights.

    Usage:
        pre = Preprocessor(config)
        result = pre.process_candidates(flat_candidates)
        # result.valid_samples  -> list of processed records
        # result.total_repaired -> int
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config      = config
        self.prep_cfg    = config["preprocessing"]
        self.dataset_cfg = config["dataset"]
        self.max_retries = self.dataset_cfg["max_retries"]

        model_name_clip = config["models"]["clip"]
        model_name_gpt2 = config["models"]["gpt2"]

        logger.info("Loading AutoProcessor (CLIP) from: %s", model_name_clip)
        self.clip_processor = AutoProcessor.from_pretrained(model_name_clip)

        logger.info("Loading AutoTokenizer (GPT-2) from: %s", model_name_gpt2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_gpt2)

        # GPT-2 has no native pad token — set to eos_token
        # This is the standard practice for GPT-2 inference/extraction.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("GPT2Tokenizer: set pad_token = eos_token ('%s')", self.tokenizer.eos_token)

        self.max_token_length = self.prep_cfg["max_token_length"]  # 77
        self.image_size       = self.prep_cfg["image_size"]         # 224
        self.min_w            = self.prep_cfg["min_image_width"]    # 64
        self.min_h            = self.prep_cfg["min_image_height"]   # 64
        self.allowed_modes    = set(self.prep_cfg["allowed_modes"])
        self.repairable_modes = set(self.prep_cfg["repairable_modes"])
        self.repair_order     = self.prep_cfg["repair_pipeline"]

        self._repair_logger   = get_repair_logger(config)

        logger.info(
            "Preprocessor ready | image_size=%d | max_token_length=%d | max_retries=%d",
            self.image_size, self.max_token_length, self.max_retries,
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def process_candidates(
        self,
        flat_candidates: list[dict],
        reserve_pool: dict[str, list[dict]] | None = None,
    ) -> "PreprocessResult":
        """
        Process a flat list of raw candidate records into validated,
        model-ready tensors.

        For each candidate:
          1. Validate image
          2. If invalid → retry from reserve_pool (up to max_retries)
          3. If still invalid → apply repair pipeline
          4. Apply CLIPProcessor → image tensor [3, 224, 224]
          5. Tokenize caption → token ids [77]
          6. Validate output shapes

        Args:
            flat_candidates: Output of data_loader.flatten_candidates()
            reserve_pool:    Per-class reserve dict for retry on failure.
                             If None, repair is the only fallback.

        Returns:
            PreprocessResult with valid_samples and statistics.
        """
        valid_samples: list[dict] = []
        n_repaired   = 0
        n_skipped    = 0

        reserve_iters: dict[str, int] = {}  # tracks index into reserve per class

        for i, record in enumerate(flat_candidates):
            if i % 100 == 0:
                logger.info(
                    "Preprocessing %d/%d | valid=%d | repaired=%d | skipped=%d",
                    i, len(flat_candidates), len(valid_samples), n_repaired, n_skipped,
                )

            image   = record["image"]
            label   = record["label"]
            img_id  = record["image_id"]
            caption = record["captions"][0]  # Use first caption for tokenization

            # --- Validate image ---
            failure = self._validate_image(image)

            if failure is not None:
                # Try reserve pool first
                if reserve_pool is not None:
                    replacement, retry_failure = self._retry_from_reserve(
                        label, reserve_pool, reserve_iters, img_id
                    )
                    if replacement is not None:
                        image   = replacement["image"]
                        img_id  = replacement["image_id"]
                        caption = replacement["captions"][0]
                        failure = retry_failure  # None if replacement was valid

                # If still failing, apply repair pipeline
                if failure is not None:
                    repaired_image, repair_name, repair_success = self._apply_repair(
                        image, failure, img_id
                    )
                    log_repair_event(
                        self._repair_logger,
                        image_id      = img_id,
                        failure_reason= failure,
                        repair_applied= repair_name,
                        success       = repair_success,
                    )
                    if not repair_success:
                        logger.warning("Skipping image_id=%s: repair failed (%s)", img_id, failure)
                        n_skipped += 1
                        continue
                    image = repaired_image
                    n_repaired += 1

            # --- Apply CLIPProcessor ---
            try:
                clip_inputs = self.clip_processor(
                    images=image,
                    return_tensors="pt",
                )
                pixel_values = clip_inputs["pixel_values"].squeeze(0)  # [3, 224, 224]
            except Exception as e:
                logger.warning("CLIPProcessor failed for image_id=%s: %s", img_id, e)
                n_skipped += 1
                continue

            # --- Validate image tensor shape ---
            if pixel_values.shape != torch.Size([3, self.image_size, self.image_size]):
                logger.warning(
                    "Unexpected pixel_values shape %s for image_id=%s — skipping",
                    pixel_values.shape, img_id,
                )
                n_skipped += 1
                continue

            # --- Tokenize caption ---
            token_ids = self._tokenize_caption(caption)
            if token_ids is None:
                n_skipped += 1
                continue

            # --- Validate token shape ---
            assert token_ids.shape == torch.Size([self.max_token_length]), (
                f"Token shape mismatch: got {token_ids.shape}, "
                f"expected [{self.max_token_length}]"
            )

            valid_samples.append({
                "image_id":     img_id,
                "pixel_values": pixel_values,   # [3, 224, 224] float32
                "token_ids":    token_ids,       # [77] int64
                "caption":      caption,
                "label":        label,
            })

        logger.info(
            "Preprocessing complete | valid=%d | repaired=%d | skipped=%d",
            len(valid_samples), n_repaired, n_skipped,
        )

        return PreprocessResult(
            valid_samples  = valid_samples,
            total_repaired = n_repaired,
            total_skipped  = n_skipped,
            total_input    = len(flat_candidates),
        )

    # -----------------------------------------------------------------------
    # Image validation
    # -----------------------------------------------------------------------

    def _validate_image(self, image: Any) -> str | None:
        """
        Validate a PIL image against config thresholds.

        Returns:
            None if image is valid.
            str  if invalid — the failure reason string used in repair log.
        """
        if image is None:
            return "image_is_none"

        if not isinstance(image, Image.Image):
            return f"not_pil_image:{type(image).__name__}"

        # Mode check
        if image.mode not in self.allowed_modes:
            if image.mode in self.repairable_modes:
                return f"mode_{image.mode}"  # Repairable
            return f"mode_unsupported:{image.mode}"  # Not repairable

        # Size check
        w, h = image.size
        if w < self.min_w or h < self.min_h:
            return f"too_small:{w}x{h}"

        return None  # Valid

    # -----------------------------------------------------------------------
    # Retry from reserve pool
    # -----------------------------------------------------------------------

    def _retry_from_reserve(
        self,
        label:          str,
        reserve_pool:   dict[str, list[dict]],
        reserve_iters:  dict[str, int],
        original_id:    str,
    ) -> tuple[dict | None, str | None]:
        """
        Attempt to find a valid replacement from the reserve pool for the
        given class label.

        Tries up to max_retries candidates from the reserve pool.

        Returns:
            (replacement_record, failure_reason)
            If a valid replacement is found: (record, None)
            If all retries exhausted:        (None, last_failure_reason)
        """
        class_reserve = reserve_pool.get(label, [])
        start_idx     = reserve_iters.get(label, 0)

        last_failure = f"no_reserve_for_{label}"

        for attempt in range(self.max_retries):
            idx = start_idx + attempt
            if idx >= len(class_reserve):
                logger.debug(
                    "Reserve exhausted for class '%s' at index %d (attempt %d/%d)",
                    label, idx, attempt + 1, self.max_retries,
                )
                break

            candidate = class_reserve[idx]
            failure   = self._validate_image(candidate.get("image"))

            if failure is None:
                reserve_iters[label] = idx + 1  # Advance reserve pointer
                logger.debug(
                    "image_id=%s: replaced with reserve image_id=%s (attempt %d)",
                    original_id, candidate["image_id"], attempt + 1,
                )
                return candidate, None

            last_failure = failure

        reserve_iters[label] = start_idx + self.max_retries
        return None, last_failure

    # -----------------------------------------------------------------------
    # Repair pipeline
    # -----------------------------------------------------------------------

    def _apply_repair(
        self,
        image:          Any,
        failure_reason: str,
        image_id:       str,
    ) -> tuple[Image.Image | None, str, bool]:
        """
        Apply the configured repair pipeline in priority order.

        Repair order from config (repair_pipeline):
          1. mode_convert — convert to RGB
          2. pad          — pad to minimum size with black border
          3. upscale      — bicubic upscale to minimum size

        Returns:
            (repaired_image_or_none, repair_name_applied, success)
        """
        if not isinstance(image, Image.Image):
            return None, "cannot_repair_non_pil", False

        repaired  = image
        last_name = "none"

        for repair_name in self.repair_order:

            if repair_name == "mode_convert" and "mode_" in failure_reason:
                try:
                    repaired  = repaired.convert("RGB")
                    last_name = "mode_convert"
                    logger.debug("image_id=%s: mode_convert applied (%s → RGB)", image_id, image.mode)
                except Exception as e:
                    logger.warning("mode_convert failed for image_id=%s: %s", image_id, e)
                    return None, "mode_convert_failed", False

            elif repair_name == "pad" and "too_small" in failure_reason:
                try:
                    repaired  = self._pad_to_minimum(repaired)
                    last_name = "pad"
                    logger.debug("image_id=%s: pad applied", image_id)
                except Exception as e:
                    logger.warning("pad failed for image_id=%s: %s", image_id, e)

            elif repair_name == "upscale" and "too_small" in failure_reason:
                try:
                    repaired  = repaired.resize(
                        (max(self.min_w, repaired.width), max(self.min_h, repaired.height)),
                        Image.BICUBIC,
                    )
                    last_name = "upscale"
                    logger.debug("image_id=%s: upscale applied to %s", image_id, repaired.size)
                except Exception as e:
                    logger.warning("upscale failed for image_id=%s: %s", image_id, e)
                    return None, "upscale_failed", False

        # Re-validate after all repairs
        final_failure = self._validate_image(repaired)
        if final_failure is not None:
            logger.warning(
                "image_id=%s: repair '%s' did not resolve failure '%s' → new failure: '%s'",
                image_id, last_name, failure_reason, final_failure,
            )
            return repaired, last_name, False

        return repaired, last_name, True

    def _pad_to_minimum(self, image: Image.Image) -> Image.Image:
        """Pad image to minimum required size with black border (constant padding)."""
        w, h       = image.size
        new_w      = max(w, self.min_w)
        new_h      = max(h, self.min_h)
        padded     = Image.new("RGB", (new_w, new_h), color=(0, 0, 0))
        padded.paste(image, (0, 0))
        return padded

    # -----------------------------------------------------------------------
    # Caption tokenization
    # -----------------------------------------------------------------------

    def _tokenize_caption(self, caption: str) -> torch.Tensor | None:
        """
        Tokenize a single caption string using the GPT-2 tokenizer.

        Output is padded/truncated to exactly max_token_length (77) tokens.
        Returns a 1D int64 tensor of shape [max_token_length].

        Returns None if tokenization fails (empty caption, etc.).
        """
        if not caption or not caption.strip():
            logger.warning("Empty caption encountered — skipping")
            return None

        try:
            encoded = self.tokenizer(
                caption,
                padding        = "max_length",
                truncation     = True,
                max_length     = self.max_token_length,
                return_tensors = "pt",
            )
            return encoded["input_ids"].squeeze(0)  # [77] int64

        except Exception as e:
            logger.warning("Tokenization failed for caption '%s...': %s", caption[:50], e)
            return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class PreprocessResult:
    """
    Container for the output of Preprocessor.process_candidates().

    Attributes:
        valid_samples  : list of processed record dicts, each containing:
                           image_id, pixel_values [3,224,224], token_ids [77],
                           caption, label
        total_repaired : number of images that required repair
        total_skipped  : number of images that were discarded entirely
        total_input    : number of raw candidates that were input
    """

    def __init__(
        self,
        valid_samples:  list[dict],
        total_repaired: int,
        total_skipped:  int,
        total_input:    int,
    ) -> None:
        self.valid_samples  = valid_samples
        self.total_repaired = total_repaired
        self.total_skipped  = total_skipped
        self.total_input    = total_input

    @property
    def total_valid(self) -> int:
        return len(self.valid_samples)

    @property
    def repair_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_repaired / self.total_input

    @property
    def skip_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_skipped / self.total_input

    def summary(self) -> str:
        return (
            f"PreprocessResult | input={self.total_input} | "
            f"valid={self.total_valid} | "
            f"repaired={self.total_repaired} ({self.repair_rate:.1%}) | "
            f"skipped={self.total_skipped} ({self.skip_rate:.1%})"
        )

    def __repr__(self) -> str:
        return self.summary()