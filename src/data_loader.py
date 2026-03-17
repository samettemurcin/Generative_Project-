"""
data_loader.py — Data Ingestion Layer
======================================
Responsibilities:
  - Build a stratified candidate pool (target_n x pool_multiplier)
  - Return per-class raw candidate lists — no filtering, no transforms
  - Filtering and repair belong to preprocessor.py

Design decisions:
  - Class matching is done via caption text scanning since the current
    dataset (nlphuji/flickr30k) provides captions only, not object
    category annotations.
  - Streaming mode is used to avoid downloading the full dataset.
  - Per-class candidate pools are built independently so shortfall handling
    in preprocessor.py has a clean reserve per class to draw from.

Milestone compatibility:
  M1: Full implementation used
  M2+: Unchanged — decoder and evaluator consume the same pool
"""

from __future__ import annotations

import logging
import random
from typing import Any

from datasets import load_dataset

from src.utils import (
    get_candidate_pool_size,
    get_per_class_target,
    set_seed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caption-based class aliases
# ---------------------------------------------------------------------------

# Maps config class names to caption text aliases for matching.
# All current classes (RISK-1 replacement set) are single-word exact-match
# terms, so the fallback `CLASS_CAPTION_ALIASES.get(cls, [cls])` handles
# them natively.  Add entries here only for classes with non-obvious
# caption aliases  (e.g., "motorbike": ["motorbike", "motorcycle"]).
CLASS_CAPTION_ALIASES: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_candidate_pool(
    config: dict[str, Any],
) -> dict[str, list[dict]]:
    """
    Build a stratified candidate pool from the configured dataset.

    Fetches (target_n * pool_multiplier) samples total, distributed evenly
    across all configured classes. Returns a dict mapping each class name
    to its list of raw candidate records.

    Each record is a plain dict with keys:
        image_id  : str   — dataset image identifier
        image     : PIL.Image.Image — raw PIL image (not yet processed)
        captions  : list[str] — all reference captions for this image
        label     : str   — matched class name from config

    Args:
        config: Master config dict from load_config().

    Returns:
        dict[class_name -> list[raw_record]]
        Each list has at least per_class_target * pool_multiplier candidates
        (subject to dataset availability).

    Raises:
        RuntimeError: If a class has zero candidates after full scan.
    """
    dataset_cfg = config["dataset"]
    classes     = dataset_cfg["classes"]
    split       = dataset_cfg["split"]
    seed        = dataset_cfg["seed"]
    pool_size   = get_candidate_pool_size(config)
    per_class   = get_per_class_target(config)
    multiplier  = dataset_cfg["pool_multiplier"]

    # Per-class pool target: we want pool_multiplier * per_class candidates
    # per class to give preprocessor.py enough reserve for filtering + repair.
    per_class_pool_target = per_class * multiplier

    set_seed(seed)

    logger.info(
        "Building candidate pool | classes=%d | per_class_pool_target=%d | total_pool=%d",
        len(classes), per_class_pool_target, pool_size,
    )

    # Load dataset in streaming mode — avoids downloading the full dataset locally.
    # Shuffled with a fixed seed for reproducibility.
    logger.info("Loading HuggingFace dataset: %s (split=%s, streaming=True)", dataset_cfg["name"], split)
    try:
        ds_kwargs = {
            "split": split,
            "streaming": True,
        }
        if "revision" in dataset_cfg:
            ds_kwargs["revision"] = dataset_cfg["revision"]
        if "year" in dataset_cfg:
            ds_kwargs["year"] = dataset_cfg["year"]
        if "coco_task" in dataset_cfg:
            ds_kwargs["coco_task"] = dataset_cfg["coco_task"]

        ds = load_dataset(dataset_cfg["name"], **ds_kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_cfg['name']}'. "
            f"Check your internet connection and dataset name.\nOriginal error: {e}"
        ) from e

    # Per-class buckets — we fill until each reaches per_class_pool_target
    buckets: dict[str, list[dict]] = {cls: [] for cls in classes}
    scanned  = 0
    # Safety cap: for flickr30k (~31k samples) we need to scan most of the
    # dataset if classes overlap heavily. Set to 15x to cover the full split
    # without silently truncating. This cap exists to prevent infinite loops
    # on malformed streams, not to limit throughput.
    max_scan = pool_size * 15

    logger.info("Scanning dataset stream to fill per-class buckets...")

    for sample in ds:
        if scanned >= max_scan:
            logger.warning(
                "Reached max_scan limit (%d). Underfilled classes: %s. "
                "Consider reducing target_n, changing classes, or increasing pool_multiplier.",
                max_scan,
                {c: len(buckets[c]) for c in classes if len(buckets[c]) < per_class_pool_target},
            )
            break

        # Stop early if all buckets are full
        if all(len(buckets[c]) >= per_class_pool_target for c in classes):
            logger.info("All class buckets full after scanning %d samples.", scanned)
            break

        scanned += 1
        matched_class = _match_sample_to_class(sample, classes)

        if matched_class is None:
            continue

        if len(buckets[matched_class]) >= per_class_pool_target:
            continue  # This class bucket is already full

        record = _build_record(sample, matched_class)
        if record is not None:
            buckets[matched_class].append(record)

        # Progress log every 5000 samples
        if scanned % 5000 == 0:
            counts = {c: len(buckets[c]) for c in classes}
            logger.info("Scanned %d | bucket counts: %s", scanned, counts)

    # Final report
    _log_pool_summary(buckets, per_class_pool_target, scanned)

    # Validate — raise if any class has zero candidates
    empty_classes = [c for c, records in buckets.items() if len(records) == 0]
    if empty_classes:
        raise RuntimeError(
            f"Classes with zero candidates after scanning {scanned} samples: "
            f"{empty_classes}. "
            f"Check that these class names exist in the '{split}' split of "
            f"'{dataset_cfg['name']}'."
        )

    return buckets


def get_dataloader_splits(
    buckets: dict[str, list[dict]],
    config: dict[str, Any],
) -> dict[str, list[dict]]:
    """
    From the candidate pool, select exactly per_class_target samples per class
    using random sampling. This is the final selection step before
    preprocessor.py applies quality filtering.

    The returned dict has the same structure as build_candidate_pool() output
    but each list is trimmed to per_class_target length.

    NOTE: This intentionally does NOT filter for quality — that is
    preprocessor.py's job. We select here, repair/reject there.

    Args:
        buckets: Output of build_candidate_pool().
        config:  Master config dict.

    Returns:
        dict[class_name -> list[raw_record]] with exactly per_class_target
        records per class (or fewer if pool was smaller than target).
    """
    per_class = get_per_class_target(config)
    seed      = config["dataset"]["seed"]
    random.seed(seed)

    selected: dict[str, list[dict]] = {}

    for cls, records in buckets.items():
        if len(records) <= per_class:
            logger.warning(
                "Class '%s': pool has %d candidates but target is %d. "
                "Using all available — preprocessor.py will handle shortfall.",
                cls, len(records), per_class,
            )
            selected[cls] = list(records)
        else:
            selected[cls] = random.sample(records, per_class)

        logger.info(
            "Class '%s': selected %d / %d candidates",
            cls, len(selected[cls]), len(records),
        )

    total = sum(len(v) for v in selected.values())
    logger.info("Total selected across all classes: %d / %d target", total, config["dataset"]["target_n"])

    return selected


def flatten_candidates(
    selected: dict[str, list[dict]],
) -> list[dict]:
    """
    Flatten the per-class selected dict into a single list for
    preprocessor.py to iterate over.

    Shuffles the final list so class order doesn't bias batch statistics.

    Args:
        selected: Output of get_dataloader_splits().

    Returns:
        Flat list of raw records in random order.
    """
    flat = [record for records in selected.values() for record in records]
    random.shuffle(flat)
    logger.info("Flattened candidate pool: %d total records", len(flat))
    return flat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_sample_to_class(
    sample: dict,
    classes: list[str],
) -> str | None:
    """
    Determine if a sample belongs to any of our target classes.

    Matching is done by scanning caption text for class keyword presence.
    This works for any caption-only dataset (flickr30k, COCO captions task)
    because caption text reliably describes the dominant visual subject.

    NOTE: Object-level annotation matching (e.g., COCO detection task) is
    not used here because the current dataset (nlphuji/flickr30k) only
    provides captions, not bounding box annotations.

    Returns the first matching class (in config.yaml order), or None.
    """
    captions = _extract_captions(sample)
    if not captions:
        return None

    caption_text = " ".join(captions).lower()
    for cls in classes:
        aliases = CLASS_CAPTION_ALIASES.get(cls, [cls])
        if any(alias in caption_text for alias in aliases):
            return cls

    return None


def _extract_captions(sample: dict) -> list[str]:
    """
    Extract caption strings from a dataset sample.

    Active code path for current dataset (nlphuji/flickr30k):
      → Format 0: sample["caption"] = ["str1", "str2", ...]

    Formats 1–5 below are compatibility fallbacks for other dataset
    variants (shunk031/MSCOCO, HuggingFaceM4/COCO). They are not
    triggered by flickr30k but are retained so this function works
    without modification if the dataset is swapped in config.yaml.
    """
    # Format 0: nlphuji/flickr30k — ACTIVE PATH
    caption_field = sample.get("caption", None)
    if isinstance(caption_field, list) and caption_field and isinstance(caption_field[0], str):
        return [c.strip() for c in caption_field if c.strip()]
    # Format 1: shunk031/MSCOCO — INACTIVE for flickr30k, retained for compatibility
    annotations = sample.get("annotations", [])
    if isinstance(annotations, list) and annotations:
        extracted = [a.get("caption", "") for a in annotations if isinstance(a, dict)]
        extracted = [c.strip() for c in extracted if c]
        if extracted:
            return extracted

    # Format 2: sample["captions"]["raw"] -> list of strings (HuggingFaceM4/COCO legacy)
    captions_field = sample.get("captions", {})
    if isinstance(captions_field, dict):
        raw = captions_field.get("raw", captions_field.get("text", []))
        if isinstance(raw, list) and raw:
            return [str(c) for c in raw if c]

    # Format 3: sample["captions"] -> list of strings directly
    if isinstance(captions_field, list) and captions_field:
        return [str(c) for c in captions_field if c]

    # Format 4: sample["caption"] -> single string
    single = sample.get("caption", "")
    if single:
        return [str(single)]

    # Format 5: sample["sentences"] -> list of dicts with "raw" key
    sentences = sample.get("sentences", [])
    if isinstance(sentences, list):
        extracted = [s.get("raw", "") for s in sentences if isinstance(s, dict)]
        extracted = [c for c in extracted if c]
        if extracted:
            return extracted

    return []


def _build_record(sample: dict, label: str) -> dict | None:
    """
    Build a clean record dict from a raw dataset sample.

    Returns None if the sample is missing a required field (image or captions),
    which signals to the caller to skip this sample silently.

    The returned record contains only what preprocessor.py needs:
        image_id : str
        image    : PIL.Image.Image  (raw, not yet processed)
        captions : list[str]
        label    : str
    """
    # Extract image
    image = sample.get("image", None)
    if image is None:
        return None

    # Extract image ID — try common HuggingFace dataset field names
    # (image_id, img_id, id) with fallback to "unknown"
    image_id = str(sample.get("image_id", sample.get("img_id", sample.get("id", "unknown"))))

    # Extract captions
    captions = _extract_captions(sample)
    if not captions:
        logger.debug("Skipping image_id=%s: no captions found", image_id)
        return None

    return {
        "image_id": image_id,
        "image":    image,
        "captions": captions,
        "label":    label,
    }


def _log_pool_summary(
    buckets: dict[str, list[dict]],
    per_class_pool_target: int,
    scanned: int,
) -> None:
    """Log a summary table of per-class candidate counts after pool building."""
    logger.info("=" * 60)
    logger.info("Candidate pool summary (scanned %d samples)", scanned)
    logger.info("%-15s %10s %10s %10s", "Class", "Target", "Actual", "Status")
    logger.info("-" * 60)

    total_actual = 0
    total_target = 0

    for cls, records in buckets.items():
        actual  = len(records)
        target  = per_class_pool_target
        status  = "OK" if actual >= target else f"SHORT by {target - actual}"
        total_actual += actual
        total_target += target
        logger.info("%-15s %10d %10d %10s", cls, target, actual, status)

    logger.info("-" * 60)
    logger.info("%-15s %10d %10d", "TOTAL", total_target, total_actual)
    logger.info("=" * 60)