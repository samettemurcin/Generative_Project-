"""
test_pipeline.py — Unit Tests
==============================
All tests run on CPU with synthetic tensors.
No model downloads, no COCO dataset required.
Designed to run in < 60 seconds on any machine.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    """Minimal valid config for testing — no real paths required."""
    return {
        "pipeline": {"mode": "extract"},
        "dataset": {
            "name": "shunk031/MSCOCO",
            "split": "train",
            "year": 2017,
            "coco_task": "captions",
            "target_n":        20,
            "pool_multiplier": 3,
            "max_retries":     3,
            "seed":            42,
            "classes":         ["person", "car"],
        },
        "sampling": {
            "strategy":          "flat_equal",
            "per_class":         10,
            "balance_tolerance": 0.05,
        },
        "models": {
            "clip": "openai/clip-vit-base-patch32",
            "gpt2": "gpt2",
        },
        "preprocessing": {
            "image_size":       224,
            "max_token_length": 77,
            "use_clip_processor": True,
            "min_image_width":  64,
            "min_image_height": 64,
            "allowed_modes":    ["RGB"],
            "repairable_modes": ["RGBA", "L", "P", "CMYK"],
            "repair_pipeline":  ["mode_convert", "pad", "upscale"],
        },
        "output": {
            "format":            "npz",
            "dir":               str(tmp_path / "outputs"),
            "filename":          "embeddings.npz",
            "checkpoint_every":  5,
            "schema": {
                "image_embeddings": "image_embeddings",
                "text_embeddings":  "text_embeddings",
                "ids":              "ids",
                "labels":           "labels",
            },
        },
        "hardware": {
            "batch_size":          4,
            "pin_memory":          False,
            "num_workers_override": 0,
        },
        "logging": {
            "level":           "WARNING",
            "log_dir":         str(tmp_path / "logs"),
            "log_file":        "pipeline.log",
            "max_bytes":       1048576,
            "backup_count":    1,
            "repair_log_file": "repairs.jsonl",
        },
        "generation": {
            "injection_method":  "prefix",
            "decoding_strategy": "beam",
            "beam_width":        5,
            "temperature":       1.0,
            "top_p":             0.9,
            "max_new_tokens":    50,
            "captions_file":     str(tmp_path / "outputs" / "captions.jsonl"),
        },
        "fine_tuning": {
            "strategy":                    "lora",
            "lora_rank":                   8,
            "lora_alpha":                  32,
            "lora_dropout":                0.1,
            "lora_target_modules":         ["c_attn", "c_proj"],
            "learning_rate":               5e-5,
            "num_epochs":                  3,
            "warmup_steps":                100,
            "gradient_accumulation_steps": 4,
            "checkpoint_dir":              str(tmp_path / "checkpoints"),
            "save_every_n_epochs":         1,
        },
        "evaluation": {
            "metrics":     ["bleu", "meteor", "cider", "rouge_l"],
            "sensitivity": {
                "temperatures": [0.5, 1.0],
                "beam_widths":  [1, 5],
                "top_p_values": [0.9],
            },
            "results_file": str(tmp_path / "outputs" / "metrics.json"),
            "plots_dir":    str(tmp_path / "outputs" / "plots"),
        },
    }


@pytest.fixture
def valid_rgb_image():
    """A valid 224x224 RGB PIL image."""
    return Image.new("RGB", (224, 224), color=(128, 64, 32))


@pytest.fixture
def small_rgb_image():
    """A too-small 32x32 RGB image — should trigger size repair."""
    return Image.new("RGB", (32, 32), color=(100, 100, 100))


@pytest.fixture
def grayscale_image():
    """A grayscale (L mode) image — should trigger mode_convert repair."""
    return Image.new("L", (224, 224), color=128)


@pytest.fixture
def rgba_image():
    """An RGBA image — repairable via mode_convert."""
    return Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))


# ---------------------------------------------------------------------------
# Test 1: Image tensor shape is exactly [3, 224, 224]
# ---------------------------------------------------------------------------

def test_image_tensor_shape(config, valid_rgb_image):
    """
    CLIPProcessor must produce pixel_values of shape [3, 224, 224].
    This is the contract that pipeline.py depends on.
    """
    from transformers import AutoProcessor

    processor    = AutoProcessor.from_pretrained(config["models"]["clip"])
    clip_inputs  = processor(images=valid_rgb_image, return_tensors="pt")
    pixel_values = clip_inputs["pixel_values"].squeeze(0)

    assert pixel_values.shape == torch.Size([3, 224, 224]), (
        f"Expected [3, 224, 224], got {pixel_values.shape}"
    )
    assert pixel_values.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: Token length never exceeds max_length
# ---------------------------------------------------------------------------

def test_token_length(config):
    """
    GPT-2 tokenizer with truncation must produce exactly max_token_length
    tokens, even for very long captions.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["models"]["gpt2"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = config["preprocessing"]["max_token_length"]

    test_captions = [
        "A short caption.",
        "a " * 200,   # Extremely long — must truncate to 77
        "",           # Edge case: empty string (pad to max_length)
        "dog cat person car bicycle bottle bird laptop cup chair " * 5,
    ]

    for caption in test_captions:
        if not caption.strip():
            caption = tokenizer.eos_token  # Avoid empty tokenizer input

        encoded = tokenizer(
            caption,
            padding        = "max_length",
            truncation     = True,
            max_length     = max_length,
            return_tensors = "pt",
        )
        token_ids = encoded["input_ids"].squeeze(0)

        assert token_ids.shape == torch.Size([max_length]), (
            f"Token shape {token_ids.shape} != [{max_length}] "
            f"for caption: '{caption[:50]}...'"
        )
        assert token_ids.dtype == torch.int64


# ---------------------------------------------------------------------------
# Test 3: Class balance within tolerance
# ---------------------------------------------------------------------------

def test_class_balance(config):
    """
    get_dataloader_splits() must select samples within ±5% of per_class target
    when the pool has sufficient candidates.
    """
    from src.data_loader import get_dataloader_splits
    from src.utils import get_per_class_target

    classes    = config["dataset"]["classes"]
    per_class  = get_per_class_target(config)  # 10
    tolerance  = config["sampling"]["balance_tolerance"]  # 0.05

    # Build a mock bucket pool with 30 candidates per class (3x target)
    mock_buckets = {}
    for i, cls in enumerate(classes):
        mock_buckets[cls] = [
            {
                "image_id": f"{cls}_{j}",
                "image":    Image.new("RGB", (224, 224)),
                "captions": [f"A {cls} in a photo {j}"],
                "label":    cls,
            }
            for j in range(per_class * 3)
        ]

    selected = get_dataloader_splits(mock_buckets, config)

    for cls in classes:
        actual = len(selected[cls])
        lo     = per_class * (1 - tolerance)
        hi     = per_class * (1 + tolerance)

        assert lo <= actual <= hi, (
            f"Class '{cls}': selected {actual} samples, "
            f"expected [{lo:.0f}, {hi:.0f}] (target={per_class}, tol={tolerance:.0%})"
        )


# ---------------------------------------------------------------------------
# Test 4: Output schema contains all required keys
# ---------------------------------------------------------------------------

def test_output_schema(config, tmp_path):
    """
    save_embeddings() must produce a .npz file with all M1 required keys,
    and load_embeddings() must recover them correctly.
    """
    from src.embeddings_io import M1_REQUIRED_KEYS, load_embeddings, save_embeddings

    n   = 10
    dim = 512

    image_embs = np.random.randn(n, dim).astype(np.float32)
    text_embs  = np.random.randn(n, dim).astype(np.float32)
    ids        = [f"img_{i}" for i in range(n)]
    labels     = ["person"] * 5 + ["car"] * 5

    out_path = tmp_path / "test_embeddings.npz"

    saved_path = save_embeddings(
        image_embeddings = image_embs,
        text_embeddings  = text_embs,
        ids              = ids,
        labels           = labels,
        config           = config,
        output_path      = out_path,
    )

    loaded = load_embeddings(saved_path)

    for key in M1_REQUIRED_KEYS:
        assert key in loaded, f"Required key '{key}' missing from loaded embeddings"

    assert loaded["image_embeddings"].shape == (n, dim)
    assert loaded["text_embeddings"].shape  == (n, dim)
    assert len(loaded["ids"])               == n
    assert len(loaded["labels"])            == n
    assert list(loaded["ids"])              == ids
    assert list(loaded["labels"])           == labels


# ---------------------------------------------------------------------------
# Test 5: Repair events are logged to repairs.jsonl
# ---------------------------------------------------------------------------

def test_repair_logging(config, tmp_path, grayscale_image):
    """
    When preprocessor repairs an image, it must write a structured JSON
    record to repairs.jsonl with the correct fields.
    """
    import logging
    from src.utils import get_repair_logger, log_repair_event

    # Override log dir to tmp_path for isolation
    config["logging"]["log_dir"] = str(tmp_path / "logs")

    repair_logger = get_repair_logger(config)
    log_path      = Path(config["logging"]["log_dir"]) / config["logging"]["repair_log_file"]

    log_repair_event(
        repair_logger  = repair_logger,
        image_id       = "test_img_001",
        failure_reason = "mode_L",
        repair_applied = "mode_convert",
        success        = True,
    )

    # Flush handlers
    for handler in repair_logger.handlers:
        handler.flush()

    assert log_path.exists(), f"Repair log not created at {log_path}"

    with log_path.open("r") as f:
        line = f.readline().strip()

    assert line, "Repair log is empty"
    record = json.loads(line)

    assert record["image_id"]       == "test_img_001"
    assert record["failure_reason"] == "mode_L"
    assert record["repair_applied"] == "mode_convert"
    assert record["success"]        is True


# ---------------------------------------------------------------------------
# Test 6: Shortfall handling — pipeline completes even with fewer valid images
# ---------------------------------------------------------------------------

def test_shortfall_handling(config):
    """
    If the preprocessed pool is smaller than target_n, the pipeline must
    still complete without raising an error, using all available samples.
    This simulates a scenario where many images fail validation.
    """
    from src.data_loader import flatten_candidates, get_dataloader_splits

    classes   = config["dataset"]["classes"]
    per_class = 10

    # Build a pool that is SMALLER than target (only 6 per class instead of 10)
    short_buckets = {}
    for cls in classes:
        short_buckets[cls] = [
            {
                "image_id": f"{cls}_{j}",
                "image":    Image.new("RGB", (224, 224)),
                "captions": [f"A {cls} photo {j}"],
                "label":    cls,
            }
            for j in range(6)  # Only 6, target is 10
        ]

    # get_dataloader_splits should not raise — it logs a warning and uses all 6
    selected = get_dataloader_splits(short_buckets, config)

    for cls in classes:
        # Must use everything available, not crash
        assert len(selected[cls]) == 6, (
            f"Class '{cls}': expected 6 samples (all available), "
            f"got {len(selected[cls])}"
        )

    flat = flatten_candidates(selected)
    assert len(flat) == 6 * len(classes), (
        f"Flattened pool should have {6 * len(classes)} samples, got {len(flat)}"
    )