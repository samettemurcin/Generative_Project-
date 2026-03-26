"""
test_M3.py — Unit tests for M3 metrics module
===============================================
No model downloads required — tests use synthetic hypotheses and references.
Runtime target: < 30 seconds on CPU.

Run with:
    pytest tests/test_M3.py -v
"""

import json
import pytest
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_hyps_refs():
    """Minimal synthetic hypotheses and references for metric tests."""
    hyps = {
        "img1": ["a dog runs in the grass"],
        "img2": ["a bicycle on a street"],
        "img3": ["a person sitting on a bench"],
    }
    refs = {
        "img1": ["a dog is running in a field", "a brown dog playing in grass",
                 "the dog is in the grass", "a dog running outside", "dog in the grass"],
        "img2": ["a bicycle parked on the road", "a bike on the street",
                 "bicycle on pavement", "a parked bicycle", "blue bicycle on a road"],
        "img3": ["a man sitting on a park bench", "someone sitting on a bench",
                 "a person resting on a bench", "a bench with a person on it",
                 "a human seated on a bench"],
    }
    return hyps, refs


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def test_compute_all_metrics_returns_all_keys(sample_hyps_refs):
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    for key in ("bleu_1", "bleu_4", "cider", "rouge_l", "meteor", "n_samples"):
        assert key in scores, f"Missing key: {key}"


def test_compute_all_metrics_bleu_range(sample_hyps_refs):
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    assert 0.0 <= scores["bleu_1"] <= 1.0, f"bleu_1 out of range: {scores['bleu_1']}"
    assert 0.0 <= scores["bleu_4"] <= 1.0, f"bleu_4 out of range: {scores['bleu_4']}"


def test_compute_all_metrics_rouge_range(sample_hyps_refs):
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    assert 0.0 <= scores["rouge_l"] <= 1.0, f"rouge_l out of range: {scores['rouge_l']}"


def test_compute_all_metrics_cider_non_negative(sample_hyps_refs):
    """CIDEr can exceed 1.0 (TF-IDF weighted) but must be non-negative."""
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    # -1.0 means pycocoevalcap not installed — that's acceptable
    assert scores["cider"] >= 0.0 or scores["cider"] == -1.0, \
        f"cider must be >= 0 or -1 (missing lib), got {scores['cider']}"


def test_compute_all_metrics_meteor_range(sample_hyps_refs):
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    # -1.0 means nltk unavailable — acceptable; otherwise must be 0–1
    assert scores["meteor"] == -1.0 or 0.0 <= scores["meteor"] <= 1.0, \
        f"meteor out of range: {scores['meteor']}"


def test_compute_all_metrics_n_samples(sample_hyps_refs):
    from src.metrics import compute_all_metrics
    hyps, refs = sample_hyps_refs
    scores = compute_all_metrics(hyps, refs)
    assert scores["n_samples"] == len(hyps)


def test_compute_all_metrics_id_mismatch_raises():
    from src.metrics import compute_all_metrics
    hyps = {"img1": ["a dog"]}
    refs = {"img2": ["a cat"]}  # different ID
    with pytest.raises(ValueError, match="ID mismatch"):
        compute_all_metrics(hyps, refs)


def test_compute_all_metrics_perfect_match():
    """When hypothesis == reference, BLEU-1 and ROUGE-L should be high."""
    from src.metrics import compute_all_metrics
    text = "a dog runs in the grass"
    hyps = {"img1": [text]}
    refs = {"img1": [text, text]}
    scores = compute_all_metrics(hyps, refs)
    assert scores["bleu_1"] > 0.9,  f"Expected high bleu_1, got {scores['bleu_1']}"
    assert scores["rouge_l"] > 0.9, f"Expected high rouge_l, got {scores['rouge_l']}"


# ---------------------------------------------------------------------------
# compute_single_sample_metrics
# ---------------------------------------------------------------------------

def test_single_sample_metrics_keys():
    from src.metrics import compute_single_sample_metrics
    scores = compute_single_sample_metrics(
        "a dog runs in the field",
        ["a dog is running in a park", "a dog playing outside"],
    )
    assert "bleu_4"  in scores
    assert "rouge_l" in scores
    assert "cider"   not in scores, "CIDEr must NOT be in single-sample metrics"


def test_single_sample_metrics_bleu_range():
    from src.metrics import compute_single_sample_metrics
    scores = compute_single_sample_metrics(
        "a bicycle parked on the road",
        ["a bike on the street", "a bicycle parked outside"],
    )
    assert 0.0 <= scores["bleu_4"]  <= 1.0
    assert 0.0 <= scores["rouge_l"] <= 1.0


def test_single_sample_metrics_perfect_match():
    from src.metrics import compute_single_sample_metrics
    text = "a dog runs in the grass"
    scores = compute_single_sample_metrics(text, [text])
    assert scores["bleu_4"]  > 0.9
    assert scores["rouge_l"] > 0.9


# ---------------------------------------------------------------------------
# load_captions_jsonl
# ---------------------------------------------------------------------------

def test_load_captions_jsonl_basic():
    from src.metrics import load_captions_jsonl

    lines = [
        {"image_id": "img1", "generated": "a dog runs",
         "references": ["a dog is running", "the dog runs"]},
        {"image_id": "img2", "generated": "a bicycle",
         "references": ["a bike on the road"]},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name

    hyps, refs = load_captions_jsonl(tmp_path)

    assert set(hyps.keys()) == {"img1", "img2"}
    assert hyps["img1"] == ["a dog runs"]
    assert refs["img1"] == ["a dog is running", "the dog runs"]
    assert hyps["img2"] == ["a bicycle"]

    Path(tmp_path).unlink()


def test_load_captions_jsonl_no_references():
    """File without references field should still load hypotheses."""
    from src.metrics import load_captions_jsonl

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps({"image_id": "img1", "generated": "a cat"}) + "\n")
        tmp_path = f.name

    hyps, refs = load_captions_jsonl(tmp_path)
    assert "img1" in hyps
    assert hyps["img1"] == ["a cat"]

    Path(tmp_path).unlink()
