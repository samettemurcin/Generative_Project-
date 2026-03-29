"""
utils.py — Infrastructure Layer
================================
Responsibilities:
  - Hardware detection and device resolution (CUDA → MPS → CPU)
  - Dynamic DataLoader worker allocation (OS-aware)
  - Centralized logger factory (file + console, rotating)
  - Config loading with structural validation across all milestones
  - Pipeline mode resolution (extract | generate | evaluate)

All other modules import from here. Nothing in this file is model-specific.

Milestone compatibility:
  M1: get_device, get_num_workers, setup_logging, load_config, set_seed
  M2: get_pipeline_mode (activate generation section)
  M3: get_pipeline_mode (activate evaluation + fine_tuning sections)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import multiprocessing
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

import torch
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"

# Valid pipeline modes — extend here when adding new milestones
PipelineMode = Literal["extract", "generate", "evaluate"]
VALID_MODES: set[str] = {"extract", "generate", "evaluate"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """
    Load and return the master config as a plain dict.

    Args:
        path: Path to config.yaml. Defaults to configs/config.yaml
              relative to the project root.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError:    If the config file is malformed.
        ValueError:        If required keys are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'.\n"
            f"Expected location: {CONFIG_PATH}"
        )
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """
    Structural validation of config keys.
    Validates M1 required keys always.
    Validates M2/M3 keys only when pipeline.mode requires them.
    Fails fast with a clear message rather than a cryptic KeyError later.
    """
    # --- Always required (M1 baseline) ---
    required = {"pipeline", "dataset", "models", "preprocessing", "output", "hardware", "logging"}
    missing = required - set(config.keys())
    if missing:
        raise ValueError(f"Config is missing required top-level keys: {missing}")

    mode = config.get("pipeline", {}).get("mode", "extract")
    if mode not in VALID_MODES:
        raise ValueError(
            f"pipeline.mode '{mode}' is not valid. "
            f"Must be one of: {sorted(VALID_MODES)}"
        )

    classes  = config.get("dataset", {}).get("classes", [])
    target_n = config.get("dataset", {}).get("target_n", 0)

    if not classes:
        raise ValueError("config.dataset.classes must contain at least one class name.")

    if target_n < len(classes):
        raise ValueError(
            f"target_n ({target_n}) must be >= number of classes ({len(classes)}) "
            f"to guarantee at least 1 sample per class."
        )

    # --- M2: generation section required ---
    if mode == "generate" and "generation" not in config:
        raise ValueError(
            "pipeline.mode is 'generate' but config is missing 'generation' section. "
            "Add the generation block to config.yaml."
        )

    # --- M3: evaluation + fine_tuning sections required ---
    if mode == "evaluate":
        missing_m3 = {"evaluation", "fine_tuning"} - set(config.keys())
        if missing_m3:
            raise ValueError(
                f"pipeline.mode is 'evaluate' but config is missing sections: {missing_m3}"
            )


def get_pipeline_mode(config: dict[str, Any]) -> PipelineMode:
    """
    Return the active pipeline mode from config.

    Returns:
        One of: 'extract' | 'generate' | 'evaluate'

    Usage:
        mode = get_pipeline_mode(config)
        if mode == "extract":
            ...
        elif mode == "generate":   # M2
            ...
        elif mode == "evaluate":   # M3
            ...
    """
    return config["pipeline"]["mode"]

# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Resolve the best available compute device in priority order:
      1. CUDA (NVIDIA GPU)
      2. MPS  (Apple Silicon)
      3. CPU  (fallback)

    Returns:
        torch.device instance ready for use in .to(device) calls.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _log_device_info(device)
        return device

    # MPS: is_built() confirms compiled support beyond just driver availability
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
        _log_device_info(device)
        return device

    device = torch.device("cpu")
    _log_device_info(device)
    return device


def _log_device_info(device: torch.device) -> None:
    """Log device details for reproducibility audit trail."""
    logger = logging.getLogger(__name__)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "Device selected: CUDA | %s | VRAM: %.1f GB | CUDA %s",
            props.name,
            props.total_memory / 1e9,
            torch.version.cuda,
        )
    elif device.type == "mps":
        logger.info("Device selected: MPS (Apple Silicon)")
    else:
        logger.info(
            "Device selected: CPU — consider using a GPU for large-scale extraction"
        )


def get_pin_memory(device: torch.device, config: dict[str, Any]) -> bool:
    """
    pin_memory=True accelerates CPU→GPU transfers but is unsupported on MPS.
    Always returns False for non-CUDA devices regardless of config setting.
    """
    if device.type != "cuda":
        return False
    return config.get("hardware", {}).get("pin_memory", True)

# ---------------------------------------------------------------------------
# Worker Allocation
# ---------------------------------------------------------------------------

def get_num_workers(config: dict[str, Any]) -> int:
    """
    Resolve the number of DataLoader worker processes.

    Priority order:
      1. num_workers_override in config → use that value directly.
      2. Windows → 0 (multiprocessing spawn + HF dataset pickling is
         unreliable in PyCharm; avoids BrokenPipeError on first run).
      3. Linux/macOS → min(cpu_count // 2, 8).

    Returns:
        int: Number of worker processes for DataLoader.
    """
    override = config.get("hardware", {}).get("num_workers_override")
    if override is not None:
        return int(override)

    if platform.system() == "Windows":
        return 0

    return min(multiprocessing.cpu_count() // 2, 8)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """
    Configure root logger with:
      - RotatingFileHandler → logs/pipeline.log
      - StreamHandler       → stdout

    Call once at pipeline entry point. All subsequent
    logging.getLogger(__name__) calls in other modules inherit this config.

    Args:
        config: Master config dict.

    Returns:
        Root logger instance.
    """
    log_cfg  = config.get("logging", {})
    level    = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_dir  = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        filename    = log_dir / log_cfg.get("log_file", "pipeline.log"),
        maxBytes    = log_cfg.get("max_bytes", 5 * 1024 * 1024),
        backupCount = log_cfg.get("backup_count", 3),
        encoding    = "utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear ALL handlers on root and named loggers to prevent duplicate
    # log lines when running multiple experiments in a notebook session.
    root.handlers.clear()
    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        if hasattr(lg, 'handlers'):
            lg.handlers.clear()

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return root


def get_repair_logger(config: dict[str, Any]) -> logging.Logger:
    """
    Dedicated logger for repaired image records.
    Writes structured JSON lines to logs/repairs.jsonl.

    Each line is a valid JSON object:
        {"image_id": "...", "failure_reason": "...", "repair_applied": "...", "success": true}

    Repair logs are consumed by M3 evaluation to audit dataset quality.
    """
    log_cfg    = config.get("logging", {})
    log_dir    = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    repair_logger = logging.getLogger("repair")
    repair_logger.setLevel(logging.WARNING)

    if not repair_logger.handlers:
        handler = logging.FileHandler(
            log_dir / log_cfg.get("repair_log_file", "repairs.jsonl"),
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        repair_logger.addHandler(handler)
        repair_logger.propagate = False  # Don't duplicate to root logger

    return repair_logger


def log_repair_event(
    repair_logger: logging.Logger,
    image_id:      str,
    failure_reason: str,
    repair_applied: str,
    success:        bool,
) -> None:
    """
    Write one structured repair event to repairs.jsonl.

    Args:
        repair_logger:  Logger from get_repair_logger().
        image_id:       Dataset image identifier.
        failure_reason: Why the image failed validation.
        repair_applied: Which repair was attempted.
        success:        Whether the repair produced a valid result.
    """
    repair_logger.warning(json.dumps({
        "image_id":       image_id,
        "failure_reason": failure_reason,
        "repair_applied": repair_applied,
        "success":        success,
    }))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    Call once at pipeline startup before any data loading or model init.
    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.getLogger(__name__).info("Random seed set to %d", seed)

# ---------------------------------------------------------------------------
# Derived Config Helpers
# (Always derive at runtime — never read stale pre-computed values from config)
# ---------------------------------------------------------------------------

def get_per_class_target(config: dict[str, Any]) -> int:
    """
    Per-class sample target = target_n // num_classes.
    Derived at runtime to stay consistent if target_n or classes change.
    """
    return config["dataset"]["target_n"] // len(config["dataset"]["classes"])


def get_candidate_pool_size(config: dict[str, Any]) -> int:
    """Total candidate pool = target_n × pool_multiplier."""
    return config["dataset"]["target_n"] * config["dataset"]["pool_multiplier"]


def get_output_path(config: dict[str, Any]) -> Path:
    """Resolve absolute output path for the embeddings .npz file."""
    out_dir = PROJECT_ROOT / config["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / config["output"]["filename"]


def get_captions_output_path(config: dict[str, Any]) -> Path:
    """
    [M2] Resolve absolute output path for the generated captions JSONL file.
    Only call this when pipeline.mode == 'generate'.
    """
    path = PROJECT_ROOT / config["generation"]["captions_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_metrics_output_path(config: dict[str, Any]) -> Path:
    """
    [M3] Resolve absolute output path for the evaluation metrics JSON file.
    Only call this when pipeline.mode == 'evaluate'.
    """
    path = PROJECT_ROOT / config["evaluation"]["results_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_dir(config: dict[str, Any]) -> Path:
    """
    [M3] Resolve absolute path for model checkpoints directory.
    Only call this when pipeline.mode == 'evaluate'.
    """
    ckpt_dir = PROJECT_ROOT / config["fine_tuning"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir