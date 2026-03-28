"""
train.py — M2/M3 Training CLI
==============================
Single entry point for all captioning experiments.

Usage examples:
  # Run 1 — frozen baseline (greedy)
  python train.py --run_id run_001 --finetune frozen --decoder greedy

  # Run 2 — LoRA fine-tuning (beam)
  python train.py --run_id run_002 --finetune lora --lora_rank 8 --decoder beam

  # Run 3 — prefix tuning
  python train.py --run_id run_003 --finetune prefix_tuning --decoder greedy

  # Run 5 — ViT-L/14 encoder
  python train.py --run_id run_005 --encoder openai/clip-vit-large-patch14 --finetune lora

  # Smoke test (< 3 min, verifies correctness)
  python train.py --run_id smoke_test --finetune frozen --max_samples 50 --epochs 1 --batch_size 4

Output structure:
  outputs/runs/{run_id}/
      config.json          ← all hyperparams at run start
      metrics.json         ← final metric scores
      captions.jsonl       ← one line per val sample
      training_log.csv     ← epoch, step, train_loss, val_loss, lr
  outputs/weights/{run_id}/
      checkpoint_epoch1.pt
      checkpoint_epoch2.pt
      checkpoint_best.pt

Notes:
  - Only prefix_proj + LoRA adapter weights are saved (NOT full GPT-2).
    Full GPT-2 is always reloaded from HuggingFace Hub. Keeps checkpoints
    under ~17MB instead of ~500MB.
  - CLIP embeddings are cached to outputs/cache/clip_cache_{run_id}.pt after
    first run. Subsequent runs load in ~2s instead of ~8 min.
  - DataLoader uses a custom collate_fn that tokenizes captions on-the-fly
    so the Dataset only stores raw strings (not pre-tokenized tensors).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

# Project root on sys.path so `src` imports resolve
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decoder import PrefixProjection, build_inputs_embeds, generate_caption
from src.metrics import compute_all_metrics, compute_single_sample_metrics
from src.utils import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP + GPT-2 prefix captioning — train and evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="Unique run identifier (e.g. run_001). Creates outputs/runs/{run_id}/",
    )

    # Model config
    parser.add_argument("--encoder",    type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP encoder variant")
    parser.add_argument("--gpt2",       type=str, default="gpt2",
                        help="GPT-2 model name or path")
    parser.add_argument("--injection",  type=str, default="prefix",
                        choices=["prefix"],
                        help="Image injection method (prefix only in M2)")
    parser.add_argument("--finetune",   type=str, default="frozen",
                        choices=["frozen", "lora", "prefix_tuning"],
                        help="GPT-2 fine-tuning strategy")
    parser.add_argument("--lora_rank",  type=int, default=8,
                        help="LoRA rank (r). Used when --finetune=lora")
    parser.add_argument("--num_prefix", type=int, default=10,
                        help="Number of prefix tokens K")

    # Decoding
    parser.add_argument("--decoder",     type=str, default="greedy",
                        choices=["greedy", "beam", "nucleus"],
                        help="Decoding strategy for caption generation")
    parser.add_argument("--beam_width",  type=int,   default=5,
                        help="Beam width (used when --decoder=beam)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (used when --decoder=nucleus)")
    parser.add_argument("--top_p",       type=float, default=0.9,
                        help="Nucleus sampling threshold")

    # Training
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max training samples. -1 = use full dataset")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--seed",        type=int, default=42)

    # Resumption
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CaptionDataset(Dataset):
    """
    Thin wrapper around pre-computed CLIP embedding cache.

    Each sample: {clip_emb: np.ndarray [clip_dim], caption: str,
                  all_captions: list[str], image_id: str, label: str}
    """

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def make_collate_fn(tokenizer, max_length: int = 77):
    """
    Returns a collate_fn that tokenizes captions on-the-fly.

    Tokenizing in the collate_fn (not __getitem__) avoids storing
    pre-tokenized tensors in the cache, keeping cache files small.
    """
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        clip_embs   = torch.tensor([s["clip_emb"] for s in batch], dtype=torch.float32)
        captions    = [s["caption"] for s in batch]
        image_ids   = [s["image_id"] for s in batch]
        all_captions = [s["all_captions"] for s in batch]

        encoded = tokenizer(
            captions,
            padding        = "max_length",
            truncation     = True,
            max_length     = max_length,
            return_tensors = "pt",
        )

        return {
            "clip_emb"    : clip_embs,
            "caption_ids" : encoded["input_ids"],        # [B, 77]
            "attn_mask"   : encoded["attention_mask"],   # [B, 77]
            "image_ids"   : image_ids,
            "all_captions": all_captions,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    args:   argparse.Namespace,
    config: dict,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """
    Load and split the dataset. Pre-computes CLIP embeddings once and caches
    them to outputs/cache/clip_cache_{run_id}.pt for fast subsequent runs.

    Returns:
        (train_samples, val_samples) — each sample dict has:
            image_id    : str
            clip_emb    : list[float] [clip_dim]  — pre-computed, cached
            caption     : str                     — first caption (for training)
            all_captions: list[str]               — all refs (for evaluation)
            label       : str
    """
    from transformers import CLIPModel, CLIPProcessor
    from src.data_loader import build_candidate_pool, flatten_candidates, get_dataloader_splits
    from src.preprocessor import Preprocessor

    cache_dir  = Path(config["output"].get("cache_dir", "outputs/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"clip_cache_{args.run_id}.pt"

    if cache_path.exists():
        logger.info("Loading cached CLIP embeddings from %s", cache_path)
        cache       = torch.load(cache_path, map_location="cpu")
        all_samples = cache["samples"]
    else:
        logger.info("Computing CLIP embeddings (will be cached to %s)", cache_path)

        # 1. Build candidate pool — reuses existing data_loader.py.
        #    Inject max_samples so the flat (non-stratified) path can stop
        #    streaming early instead of loading all 31k images unnecessarily.
        config["dataset"]["max_samples"] = args.max_samples
        buckets = build_candidate_pool(config)

        if isinstance(buckets, list):
            # Non-stratified (full-dataset) mode: build_candidate_pool already
            # capped the stream at args.max_samples; no reserve pool needed.
            flat         = buckets
            reserve_pool: dict = {}
        else:
            # Stratified mode: select per-class targets then build reserve pool.
            selected = get_dataloader_splits(buckets, config)
            flat     = flatten_candidates(selected)
            selected_ids = {
                cls: {s["image_id"] for s in samples}
                for cls, samples in selected.items()
            }
            reserve_pool = {
                cls: [s for s in buckets[cls] if s["image_id"] not in selected_ids[cls]]
                for cls in buckets
            }

        # 2. Preprocess — reuses existing preprocessor.py
        preprocessor = Preprocessor(config)
        result = preprocessor.process_candidates(flat, reserve_pool=reserve_pool)

        if result.total_valid == 0:
            raise RuntimeError(
                "No valid samples after preprocessing. "
                "Check dataset name, classes, and internet connection."
            )

        logger.info("Preprocessing done: %d valid samples", result.total_valid)

        # 3. Encode ALL images with CLIP once — this is the bottleneck.
        #    Process in batches of 32 to maximise GPU utilisation (~20x faster
        #    than encoding one image at a time).
        logger.info("Loading CLIP encoder: %s", args.encoder)
        clip_model     = CLIPModel.from_pretrained(args.encoder).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(args.encoder)

        clip_batch_size = 32
        all_valid       = result.valid_samples
        all_samples: list[dict] = []

        with torch.no_grad():
            for batch_start in range(0, len(all_valid), clip_batch_size):
                batch        = all_valid[batch_start : batch_start + clip_batch_size]
                pixel_values = torch.stack(
                    [s["pixel_values"] for s in batch]
                ).to(device)                                         # [B, 3, H, W]
                embs = clip_model.get_image_features(pixel_values=pixel_values)
                if not isinstance(embs, torch.Tensor):              # transformers ≥4.x compat
                    embs = embs.pooler_output
                embs = embs / embs.norm(dim=-1, keepdim=True)       # L2-normalise
                embs_cpu = embs.cpu().numpy()                        # [B, clip_dim]

                for i, sample in enumerate(batch):
                    all_samples.append({
                        "image_id"    : sample["image_id"],
                        "clip_emb"    : embs_cpu[i].tolist(),
                        "caption"     : sample["caption"],
                        "all_captions": sample.get("all_captions", [sample["caption"]]),
                        "label"       : sample["label"],
                    })

        torch.save({"samples": all_samples}, cache_path)
        logger.info("Cached %d CLIP embeddings to %s", len(all_samples), cache_path)

    # 4. Optional sample limit (for smoke tests)
    if args.max_samples > 0 and len(all_samples) > args.max_samples:
        all_samples = all_samples[:args.max_samples]
        logger.info("Limited to %d samples (--max_samples)", len(all_samples))

    # 5. Deterministic 80/20 train/val split
    random.seed(args.seed)
    shuffled = all_samples[:]
    random.shuffle(shuffled)
    split         = int(0.8 * len(shuffled))
    train_samples = shuffled[:split]
    val_samples   = shuffled[split:]

    logger.info("Dataset: %d train / %d val", len(train_samples), len(val_samples))
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    args:   argparse.Namespace,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Load GPT-2 (LMHead) + PrefixProjection + tokenizer.

    CLIP is NOT returned here — it was already used in build_dataset
    to pre-compute and cache embeddings.

    Returns:
        (gpt2_model, prefix_proj, tokenizer)
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer

    logger.info("Loading GPT-2: %s", args.gpt2)
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2).to(device)
    gpt2_model.train()  # apply_finetuning will freeze what's needed

    tokenizer = AutoTokenizer.from_pretrained(args.gpt2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine CLIP dim from encoder name
    clip_dim_map = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-large-patch14": 768,
    }
    clip_dim = clip_dim_map.get(args.encoder, 512)
    gpt2_dim = gpt2_model.config.n_embd  # 768 for gpt2-small, 1024 for gpt2-medium

    prefix_proj = PrefixProjection(
        clip_dim   = clip_dim,
        gpt2_dim   = gpt2_dim,
        num_prefix = args.num_prefix,
    ).to(device)

    logger.info(
        "GPT-2 total params: %d | PrefixProjection params: %d",
        sum(p.numel() for p in gpt2_model.parameters()),
        sum(p.numel() for p in prefix_proj.parameters()),
    )
    return gpt2_model, prefix_proj, tokenizer


# ---------------------------------------------------------------------------
# Fine-tuning strategy
# ---------------------------------------------------------------------------

def apply_finetuning(
    gpt2_model,
    args: argparse.Namespace,
) -> tuple:
    """
    Apply the requested fine-tuning strategy to GPT-2.

    frozen:        All GPT-2 params frozen. Only prefix_proj trains.
    lora:          LoRA adapters on c_attn + c_proj. prefix_proj also trains.
    prefix_tuning: Same as frozen for our ClipCap implementation.
                   "Prefix tuning" here means the prefix_proj is trained,
                   not the Lester et al. per-layer learnable vectors.

    Returns:
        (gpt2_model, n_trainable_gpt2_params)
    """
    if args.finetune == "frozen":
        for param in gpt2_model.parameters():
            param.requires_grad = False
        trainable = 0
        logger.info("Fine-tune: frozen — only prefix_proj trains")

    elif args.finetune == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            inference_mode = False,
            r              = args.lora_rank,
            lora_alpha     = args.lora_rank * 2,   # standard: alpha = 2r
            lora_dropout   = 0.05,
            target_modules = ["c_attn", "c_proj"],  # GPT-2 attention projections
            bias           = "none",
        )
        gpt2_model = get_peft_model(gpt2_model, lora_config)
        gpt2_model.print_trainable_parameters()
        trainable = sum(p.numel() for p in gpt2_model.parameters() if p.requires_grad)
        logger.info("Fine-tune: LoRA r=%d | GPT-2 trainable params: %d",
                    args.lora_rank, trainable)

    elif args.finetune == "prefix_tuning":
        for param in gpt2_model.parameters():
            param.requires_grad = False
        trainable = 0
        logger.info("Fine-tune: prefix_tuning — GPT-2 frozen, prefix_proj trains")

    else:
        raise ValueError(f"Unknown finetune strategy: {args.finetune}")

    return gpt2_model, trainable


# ---------------------------------------------------------------------------
# Optimiser + scheduler
# ---------------------------------------------------------------------------

def build_optimiser(
    gpt2_model,
    prefix_proj:   PrefixProjection,
    args:          argparse.Namespace,
    n_train_steps: int,
) -> tuple:
    """
    AdamW optimiser over all trainable params (prefix_proj + LoRA if applicable).
    Cosine LR schedule with linear warmup.

    Returns:
        (optimizer, scheduler)
    """
    from transformers import get_cosine_schedule_with_warmup

    # Always include prefix_proj. Add LoRA params only if they exist.
    trainable_params = list(prefix_proj.parameters())
    trainable_params += [p for p in gpt2_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = args.lr,
        weight_decay = 0.01,
        betas        = (0.9, 0.999),
    )

    warmup_steps = min(100, n_train_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = n_train_steps,
    )

    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(
        "Optimiser: AdamW | lr=%.2e | warmup=%d steps | total trainable params=%d",
        args.lr, warmup_steps, total_trainable,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    epoch:        int,
    train_loader: DataLoader,
    gpt2_model,
    prefix_proj:  PrefixProjection,
    optimizer,
    scheduler,
    tokenizer,
    device:       torch.device,
) -> float:
    """
    Run one training epoch.

    Returns:
        Mean training loss for this epoch.
    """
    gpt2_model.train()
    prefix_proj.train()

    # Build gradient clipping target list once — params don't change between steps.
    all_trainable = (
        list(prefix_proj.parameters()) +
        [p for p in gpt2_model.parameters() if p.requires_grad]
    )

    total_loss = 0.0
    n_batches  = 0

    for step, batch in enumerate(train_loader):
        clip_emb    = batch["clip_emb"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        attn_mask   = batch["attn_mask"].to(device)

        inputs_embeds, attention_mask, labels = build_inputs_embeds(
            clip_emb, caption_ids, attn_mask,
            prefix_proj, gpt2_model, tokenizer, device,
        )

        outputs = gpt2_model(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = outputs.loss

        loss.backward()

        # Gradient clipping — prevents exploding gradients in early training
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches  += 1

        if step % 10 == 0:
            logger.info(
                "Epoch %d | step %d/%d | loss=%.4f | lr=%.2e",
                epoch, step, len(train_loader),
                loss.item(), scheduler.get_last_lr()[0],
            )

    mean_loss = total_loss / max(n_batches, 1)
    logger.info("Epoch %d complete | mean_train_loss=%.4f", epoch, mean_loss)
    return mean_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    val_samples:    list[dict],
    gpt2_model,
    prefix_proj:    PrefixProjection,
    tokenizer,
    device:         torch.device,
    args:           argparse.Namespace,
    clip_model      = None,
    clip_processor  = None,
) -> tuple[dict, dict, dict]:
    """
    Generate captions for all val samples and compute metrics.

    Returns:
        (scores, hypotheses, references)
        scores     : full metrics dict
        hypotheses : {image_id: [generated_caption]}
        references : {image_id: [ref1, ref2, ...]}
    """
    gpt2_model.eval()
    prefix_proj.eval()

    hypotheses: dict[str, list[str]] = {}
    references:  dict[str, list[str]] = {}
    clip_sims:   list[float] = []

    gen_config = {
        "generation": {
            "decoding_strategy": args.decoder,
            "beam_width"       : args.beam_width,
            "temperature"      : args.temperature,
            "top_p"            : args.top_p,
            "max_new_tokens"   : 50,
        }
    }

    with torch.no_grad():
        for i, sample in enumerate(val_samples):
            if i % 50 == 0:
                logger.info("Evaluating %d/%d val samples", i, len(val_samples))

            img_emb = torch.tensor(
                sample["clip_emb"], dtype=torch.float32
            ).unsqueeze(0).to(device)

            caption = generate_caption(
                image_embedding = img_emb,
                prefix_proj     = prefix_proj,
                gpt2_model      = gpt2_model,
                tokenizer       = tokenizer,
                config          = gen_config,
                device          = device,
            )

            img_id               = sample["image_id"]
            hypotheses[img_id]   = [caption]
            references[img_id]   = sample.get("all_captions", [sample["caption"]])

            # CLIP cosine similarity (image vs generated caption text)
            if clip_model is not None and clip_processor is not None:
                try:
                    txt_inputs = clip_processor(
                        text=[caption], return_tensors="pt",
                        padding=True, truncation=True, max_length=77,
                    ).to(device)
                    txt_emb = clip_model.get_text_features(**txt_inputs)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    sim     = float((img_emb * txt_emb).sum())
                    clip_sims.append(sim)
                except Exception:
                    pass

    # Corpus-level metrics
    logger.info("Computing corpus metrics for %d val samples", len(hypotheses))
    scores = compute_all_metrics(hypotheses, references)

    if clip_sims:
        clip_tensor           = torch.tensor(clip_sims)
        scores["clip_sim_mean"] = round(float(clip_tensor.mean()), 4)
        scores["clip_sim_std"]  = round(float(clip_tensor.std()), 4)
    else:
        scores["clip_sim_mean"] = -1.0
        scores["clip_sim_std"]  = -1.0

    logger.info("Metrics: %s", scores)
    return scores, hypotheses, references


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    weight_dir:  Path,
    epoch:       int,
    prefix_proj: PrefixProjection,
    gpt2_model,
    val_loss:    float,
    args:        argparse.Namespace,
    is_best:     bool = False,
) -> Path:
    """
    Save prefix_proj + LoRA adapter weights (NOT full GPT-2).

    Full GPT-2 weights are always reloaded from HuggingFace Hub — this
    keeps each checkpoint under ~17MB instead of ~500MB.

    Returns:
        Path to the saved checkpoint file.
    """
    weight_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "run_id"      : args.run_id,
        "epoch"       : epoch,
        "val_loss"    : val_loss,
        "prefix_proj" : prefix_proj.state_dict(),
        # With PEFT, state_dict() returns only the LoRA adapter weights,
        # not the full GPT-2 base model (~17MB vs ~500MB). None for frozen/prefix_tuning.
        "lora_adapter": gpt2_model.state_dict() if args.finetune == "lora" else None,
        "config"      : vars(args),
    }

    path = weight_dir / f"checkpoint_epoch{epoch}.pt"
    torch.save(ckpt, path)
    logger.info("Saved checkpoint: %s (val_loss=%.4f)", path, val_loss)

    if is_best:
        best_path = weight_dir / "checkpoint_best.pt"
        torch.save(ckpt, best_path)
        logger.info("New best checkpoint: %s", best_path)

    return path


# ---------------------------------------------------------------------------
# Run output writing
# ---------------------------------------------------------------------------

def write_run_outputs(
    run_dir:    Path,
    scores:     dict,
    args:       argparse.Namespace,
    hypotheses: dict[str, list[str]],
    references: dict[str, list[str]],
    train_log:  list[dict],
) -> None:
    """
    Write all per-run artifacts to outputs/runs/{run_id}/.

    Written files:
      config.json       — all hyperparams at run start
      metrics.json      — final scores (Streamlit reads this)
      captions.jsonl    — one JSON line per val image
      training_log.csv  — epoch, step, train_loss, val_loss, lr
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    # metrics.json — enriched with run metadata for Streamlit
    metrics_payload = {
        "run_id"      : args.run_id,
        "encoder"     : args.encoder,
        "injection"   : args.injection,
        "fine_tune"   : args.finetune,
        "decoding"    : args.decoder,
        "lora_rank"   : args.lora_rank if args.finetune == "lora" else None,
        "timestamp"   : datetime.now(timezone.utc).isoformat(),
        **scores,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    # captions.jsonl — one JSON line per val image
    # Instantiate RougeScorer once outside the loop (stemmer init is non-trivial).
    from rouge_score import rouge_scorer as rouge_lib
    _rouge_scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    with open(run_dir / "captions.jsonl", "w", encoding="utf-8") as f:
        for img_id, gen_list in hypotheses.items():
            generated   = gen_list[0]
            ref_list    = references.get(img_id, [])
            per_sample  = compute_single_sample_metrics(
                generated, ref_list or [""], rouge_scorer=_rouge_scorer
            )
            record = {
                "image_id"  : img_id,
                "generated" : generated,
                "references": ref_list,
                "bleu_4"    : per_sample["bleu_4"],
                "rouge_l"   : per_sample["rouge_l"],
            }
            f.write(json.dumps(record) + "\n")

    # training_log.csv
    if train_log:
        fieldnames = list(train_log[0].keys())
        with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(train_log)

    logger.info("Run outputs written to %s", run_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config()
    setup_logging(config)
    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 60)
    logger.info("Training run: %s | device=%s | finetune=%s",
                args.run_id, device, args.finetune)
    logger.info("=" * 60)

    # Create output directories
    run_dir    = Path(config["output"].get("runs_dir",    "outputs/runs"))    / args.run_id
    weight_dir = Path(config["output"].get("weights_dir", "outputs/weights")) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_samples, val_samples = build_dataset(args, config, device)

    # ── Models ──────────────────────────────────────────────────────────────
    gpt2_model, prefix_proj, tokenizer = load_models(args, config, device)
    gpt2_model, _ = apply_finetuning(gpt2_model, args)

    # Optionally load CLIP model for CLIP-sim metric during evaluation
    clip_model_eval, clip_proc_eval = None, None
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model_eval = CLIPModel.from_pretrained(args.encoder).to(device).eval()
        clip_proc_eval  = CLIPProcessor.from_pretrained(args.encoder)
    except Exception as e:
        logger.warning("Could not load CLIP for eval CLIP-sim: %s", e)

    # Resume from checkpoint if requested
    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        prefix_proj.load_state_dict(ckpt["prefix_proj"])
        if ckpt.get("lora_adapter") is not None:
            gpt2_model.load_state_dict(ckpt["lora_adapter"], strict=False)

    # ── DataLoader ───────────────────────────────────────────────────────────
    collate_fn   = make_collate_fn(tokenizer, max_length=config["preprocessing"]["max_token_length"])
    train_loader = DataLoader(
        CaptionDataset(train_samples),
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,    # safe default on Windows
        pin_memory  = device.type == "cuda",
    )

    # ── Optimiser ────────────────────────────────────────────────────────────
    n_train_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimiser(gpt2_model, prefix_proj, args, n_train_steps)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    train_log: list[dict] = []
    final_scores:    dict = {}
    final_hypotheses: dict = {}
    final_references: dict = {}

    for epoch in range(1, args.epochs + 1):
        logger.info("--- Epoch %d/%d ---", epoch, args.epochs)

        train_loss = train_one_epoch(
            epoch, train_loader, gpt2_model, prefix_proj,
            optimizer, scheduler, tokenizer, device,
        )

        # Evaluate after each epoch
        scores, hypotheses, references = evaluate(
            val_samples, gpt2_model, prefix_proj, tokenizer, device, args,
            clip_model_eval, clip_proc_eval,
        )

        # Invert BLEU-4 so that "best checkpoint" = lowest val_loss = highest BLEU-4.
        val_loss = 1.0 - scores.get("bleu_4", 0.0)

        train_log.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 4),
            "bleu_4"    : scores.get("bleu_4",  -1),
            "cider"     : scores.get("cider",   -1),
            "rouge_l"   : scores.get("rouge_l", -1),
            "meteor"    : scores.get("meteor",  -1),
            "lr"        : round(scheduler.get_last_lr()[0], 8),
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss     = val_loss
            final_scores      = scores
            final_hypotheses  = hypotheses
            final_references  = references

        save_checkpoint(weight_dir, epoch, prefix_proj, gpt2_model, val_loss, args, is_best)

    # Use final epoch results if we never got a "best"
    if not final_scores:
        final_scores     = scores
        final_hypotheses = hypotheses
        final_references = references

    # Also symlink best weights to outputs/weights/best/ for Streamlit
    best_global_dir = Path(config["output"].get("weights_dir", "outputs/weights")) / "best"
    best_global_dir.mkdir(parents=True, exist_ok=True)
    best_src  = weight_dir / "checkpoint_best.pt"
    best_link = best_global_dir / "checkpoint_best.pt"
    if best_src.exists():
        try:
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            # On Windows, copy instead of symlink to avoid privilege requirements
            import shutil
            shutil.copy2(best_src, best_link)
            logger.info("Copied best checkpoint to %s", best_link)
        except Exception as e:
            logger.warning("Could not copy best checkpoint to %s: %s", best_link, e)

    # ── Write run outputs ────────────────────────────────────────────────────
    write_run_outputs(
        run_dir, final_scores, args,
        final_hypotheses, final_references, train_log,
    )

    logger.info("=" * 60)
    logger.info(
        "Run complete: %s | BLEU-4=%.4f | CIDEr=%.4f | ROUGE-L=%.4f",
        args.run_id,
        final_scores.get("bleu_4",  0),
        final_scores.get("cider",   0),
        final_scores.get("rouge_l", 0),
    )
    logger.info("Artifacts: %s", run_dir)
    logger.info("Weights:   %s", weight_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
