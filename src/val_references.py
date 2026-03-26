"""
val_references.py — Validation Reference Cache
================================================
Responsibilities:
  - build_val_references_cache: streams Flickr30k and saves image_id → captions
    as a JSON lookup file for Streamlit and evaluation scripts
  - load_val_references: loads the cached JSON → {image_id: [captions]}

Why a separate cache?
  During training, all_captions are stored inline in each sample dict.
  For inference-time evaluation (Streamlit demo, post-training eval), we
  need to look up reference captions by image_id without re-running the full
  data pipeline. This cache file is that lookup table.

Usage:
  # Build once after M1 pipeline completes:
  from src.val_references import build_val_references_cache
  build_val_references_cache(config)

  # Load in evaluation scripts or Streamlit:
  from src.val_references import load_val_references
  refs = load_val_references("outputs/val_references.json")
  # refs["1234567890"] → ["A dog runs...", "The brown dog...", ...]

Milestone compatibility:
  M2: build_val_references_cache called once after M1 run
  M3: load_val_references used by eval scripts and Streamlit
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build cache
# ---------------------------------------------------------------------------

def build_val_references_cache(
    config: dict[str, Any],
    output_path: Path | str = Path("outputs/val_references.json"),
    n_samples: int = 200,
) -> Path:
    """
    Stream Flickr30k and save image_id → reference captions as JSON.

    Saves at most n_samples records. These are drawn from the same dataset
    split used by the pipeline so image_ids should overlap with training data.

    Args:
        config      : project config dict (used for dataset name/split)
        output_path : destination path for the JSON cache
        n_samples   : number of images to cache (default 200 = val split size)

    Returns:
        Path to the written JSON file.
    """
    from datasets import load_dataset
    from src.data_loader import _extract_captions  # reuse existing extractor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds_name  = config["dataset"]["name"]
    split    = config["dataset"]["split"]
    revision = config["dataset"].get("revision")

    logger.info(
        "Streaming %s (split=%s) for val references cache...", ds_name, split
    )

    load_kwargs: dict[str, Any] = {"streaming": True, "split": split}
    if revision:
        load_kwargs["revision"] = revision

    dataset = load_dataset(ds_name, **load_kwargs)

    records: dict[str, list[str]] = {}
    scanned = 0

    for sample in dataset:
        scanned += 1
        img_id   = str(sample.get("img_id", sample.get("image_id", sample.get("id", f"sample_{scanned}"))))
        captions = _extract_captions(sample)

        if not captions:
            continue

        records[img_id] = captions[:5]  # at most 5 reference captions per image

        if len(records) >= n_samples:
            break

    logger.info(
        "Cached %d val reference records (scanned %d samples)", len(records), scanned
    )

    payload = {
        "image_ids" : list(records.keys()),
        "references": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Saved val references to %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------

def load_val_references(
    path: Path | str = Path("outputs/val_references.json"),
) -> dict[str, list[str]]:
    """
    Load the validation reference cache.

    Returns:
        {image_id: [caption1, caption2, ..., caption5]}
        Empty dict if file does not exist.
    """
    path = Path(path)

    if not path.exists():
        logger.warning(
            "Val references cache not found at %s. "
            "Run build_val_references_cache(config) to create it.",
            path,
        )
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        refs    = payload.get("references", {})
        logger.info("Loaded %d val references from %s", len(refs), path)
        return refs
    except Exception as e:
        logger.error("Failed to load val references from %s: %s", path, e)
        return {}
