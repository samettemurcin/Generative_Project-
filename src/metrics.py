"""
metrics.py — M3 Evaluation Metrics
=====================================
Responsibilities:
  - compute_all_metrics: corpus-level BLEU, CIDEr, ROUGE-L, METEOR
  - compute_single_sample_metrics: per-image BLEU-4 + ROUGE-L (no CIDEr)
  - load_captions_jsonl: load captions.jsonl → (hypotheses, references)
  - run_evaluation: pipeline.py entry point for mode="evaluate"

Library choices:
  - BLEU:   sacrebleu.corpus_bleu — standard, handles multi-ref correctly
  - CIDEr:  pycocoevalcap.cider   — official COCO implementation
  - ROUGE-L: rouge-score          — fast, stemmer-aware
  - METEOR:  nltk                 — no Java required (unlike pycocoevalcap Meteor)

CIDEr note:
  CIDEr is NOT bounded to 0–1. Scores can exceed 1.0 because TF-IDF weights
  are corpus-dependent. On a 200-sample val set, scores < 1.0 are expected
  and correct. Good full-dataset models score 0.8–1.1 CIDEr.

METEOR note:
  Requires NLTK wordnet and punkt_tab data. Downloaded automatically on first
  call via nltk.download(..., quiet=True). No network access needed after that.

Milestone compatibility:
  M3: full implementation
  train.py calls compute_all_metrics after each epoch
  Streamlit reads metrics.json produced by write_run_outputs in train.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK wordnet download (lazy, once)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK datasets on first use."""
    import nltk
    for pkg in ("wordnet", "punkt_tab", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # offline environments — proceed, METEOR may fail gracefully


# ---------------------------------------------------------------------------
# Corpus-level metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    hypotheses: dict[str, list[str]],  # {image_id: [generated_caption]}
    references: dict[str, list[str]],  # {image_id: [ref1, ref2, ..., ref5]}
) -> dict[str, float]:
    """
    Compute all M3 corpus-level metrics for a set of generated captions.

    Both dicts must be keyed by the same image_id strings.
    References should have ALL reference captions per image (up to 5 for
    Flickr30k) — this is why the all_captions BUG-2 fix was critical.

    Returns:
        {
            "bleu_1"   : float,  # 0–1
            "bleu_4"   : float,  # 0–1 (strict, expect 0.05–0.20 for 1k samples)
            "cider"    : float,  # 0–~2 (corpus-dependent TF-IDF weights)
            "rouge_l"  : float,  # 0–1
            "meteor"   : float,  # 0–1 (or -1.0 if NLTK unavailable)
            "n_samples": int,
        }
    """
    import sacrebleu
    from rouge_score import rouge_scorer as rouge_lib

    sorted_ids = sorted(hypotheses.keys())

    if set(sorted_ids) != set(references.keys()):
        missing_in_refs = set(sorted_ids) - set(references.keys())
        missing_in_hyps = set(references.keys()) - set(sorted_ids)
        raise ValueError(
            f"ID mismatch between hypotheses and references. "
            f"Missing in refs: {len(missing_in_refs)}, "
            f"Missing in hyps: {len(missing_in_hyps)}"
        )

    hyps_list = [hypotheses[k][0] for k in sorted_ids]
    refs_list  = [references[k]   for k in sorted_ids]  # list of lists

    # ── BLEU via sacrebleu ────────────────────────────────────────────────
    # sacrebleu.corpus_bleu expects refs as [list_of_ref1s, list_of_ref2s, ...]
    # We need to transpose from per-image ref lists to per-position ref lists.
    max_refs = max(len(r) for r in refs_list)
    refs_transposed = []
    for i in range(max_refs):
        refs_transposed.append([
            r[i] if i < len(r) else r[0]   # fallback to first ref if fewer than max
            for r in refs_list
        ])

    bleu_result = sacrebleu.corpus_bleu(hyps_list, refs_transposed)
    bleu_1 = bleu_result.precisions[0] / 100.0   # sacrebleu returns 0–100
    bleu_4 = bleu_result.score        / 100.0

    # ── CIDEr via pycocoevalcap ───────────────────────────────────────────
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        # pycocoevalcap expects {id: [list]} for both args
        cider_score, _ = cider_scorer.compute_score(references, hypotheses)
        cider = round(float(cider_score), 4)
    except ImportError:
        logger.warning("pycocoevalcap not installed — CIDEr will be -1.0. "
                       "Install with: pip install pycocoevalcap")
        cider = -1.0

    # ── ROUGE-L via rouge-score ───────────────────────────────────────────
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for img_id in sorted_ids:
        # Score against first reference (standard practice for corpus ROUGE-L)
        score = scorer.score(references[img_id][0], hypotheses[img_id][0])
        rouge_scores.append(score["rougeL"].fmeasure)
    rouge_l = round(float(sum(rouge_scores) / len(rouge_scores)), 4)

    # ── METEOR via NLTK ───────────────────────────────────────────────────
    try:
        _ensure_nltk_data()
        from nltk.translate.meteor_score import meteor_score as nltk_meteor

        meteor_scores = []
        for img_id in sorted_ids:
            hyp_tokens  = hypotheses[img_id][0].split()
            ref_tokens_list = [r.split() for r in references[img_id]]
            score = nltk_meteor(ref_tokens_list, hyp_tokens)
            meteor_scores.append(score)
        meteor = round(float(sum(meteor_scores) / len(meteor_scores)), 4)
    except Exception as e:
        logger.warning("METEOR computation failed: %s — setting meteor=-1.0", e)
        meteor = -1.0

    return {
        "bleu_1"   : round(bleu_1, 4),
        "bleu_4"   : round(bleu_4, 4),
        "cider"    : cider,
        "rouge_l"  : rouge_l,
        "meteor"   : meteor,
        "n_samples": len(sorted_ids),
    }


# ---------------------------------------------------------------------------
# Per-sample metrics (no CIDEr)
# ---------------------------------------------------------------------------

def compute_single_sample_metrics(
    hypothesis: str,
    references: list[str],
    rouge_scorer=None,
) -> dict[str, float]:
    """
    Sentence-level BLEU-4 and ROUGE-L for a single image.

    NOTE: CIDEr is intentionally NOT computed here — CIDEr requires TF-IDF
    weights from the full corpus. Per-image CIDEr is undefined and misleading.

    Args:
        hypothesis   : generated caption string
        references   : list of reference caption strings
        rouge_scorer : optional pre-built RougeScorer instance. Pass one when
                       calling this function in a loop to avoid re-instantiating
                       the scorer (and its stemmer) on every call.

    Returns:
        {"bleu_4": float, "rouge_l": float}
    """
    import sacrebleu
    from rouge_score import rouge_scorer as rouge_lib

    # Sentence-level BLEU-4
    bleu    = sacrebleu.sentence_bleu(hypothesis, references)
    bleu_4  = round(bleu.score / 100.0, 4)

    # ROUGE-L against first reference
    # Reuse a caller-supplied scorer if available; otherwise create one.
    if rouge_scorer is None:
        rouge_scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = round(
        rouge_scorer.score(references[0], hypothesis)["rougeL"].fmeasure, 4
    )

    return {
        "bleu_4" : bleu_4,
        "rouge_l": rouge_l,
    }


# ---------------------------------------------------------------------------
# Load captions.jsonl
# ---------------------------------------------------------------------------

def load_captions_jsonl(
    path: Path | str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Load a captions.jsonl file and return (hypotheses, references).

    Expected JSON line format:
        {"image_id": "...", "generated": "...", "references": ["...", ...]}

    Returns:
        hypotheses: {image_id: [generated_caption]}
        references: {image_id: [ref1, ref2, ...]}
    """
    hypotheses: dict[str, list[str]] = {}
    references: dict[str, list[str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec    = json.loads(line)
            img_id = rec["image_id"]
            hypotheses[img_id] = [rec["generated"]]
            if "references" in rec:
                references[img_id] = rec["references"]

    logger.info("Loaded %d captions from %s", len(hypotheses), path)
    return hypotheses, references


# ---------------------------------------------------------------------------
# Pipeline entry point: run_evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    config: dict[str, Any],
    device: "torch.device",  # noqa: F821 — avoid hard torch import at module level
) -> dict[str, Any]:
    """
    M3 evaluation pipeline — called by pipeline.py when mode="evaluate".

    Reads all runs/*/captions.jsonl under outputs/runs/, computes metrics
    for each run, writes a combined outputs/metrics.json.

    Returns:
        Result dict: mode, output_path, total_runs, summary
    """
    runs_dir    = Path(config["output"].get("runs_dir", "outputs/runs"))
    results_out = Path(config["output"]["dir"]) / "metrics.json"

    if not runs_dir.exists():
        raise FileNotFoundError(
            f"No runs directory found at {runs_dir}. "
            "Run train.py first: python train.py --run_id run_001"
        )

    caption_files = sorted(runs_dir.glob("*/captions.jsonl"))
    if not caption_files:
        raise FileNotFoundError(
            f"No captions.jsonl files found under {runs_dir}. "
            "Run train.py with at least one run_id first."
        )

    logger.info("Found %d caption files under %s", len(caption_files), runs_dir)

    all_results: list[dict] = []
    for cap_file in caption_files:
        run_id = cap_file.parent.name
        logger.info("Evaluating run: %s", run_id)

        try:
            hypotheses, references = load_captions_jsonl(cap_file)
            if not references:
                logger.warning(
                    "Run %s has no references in captions.jsonl — skipping metrics",
                    run_id,
                )
                continue

            scores = compute_all_metrics(hypotheses, references)
            scores["run_id"] = run_id

            # Merge with existing run metrics.json if it exists
            run_metrics_file = cap_file.parent / "metrics.json"
            if run_metrics_file.exists():
                existing = json.loads(run_metrics_file.read_text())
                existing.update(scores)
                scores = existing

            all_results.append(scores)

        except Exception as e:
            logger.error("Failed to evaluate run %s: %s", run_id, e)

    # Write combined metrics
    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(all_results, indent=2))

    summary = (
        f"Evaluation complete | runs={len(all_results)} | output={results_out}"
    )
    logger.info(summary)

    return {
        "mode"       : "evaluate",
        "output_path": str(results_out),
        "total_runs" : len(all_results),
        "summary"    : summary,
    }
