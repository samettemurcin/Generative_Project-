# Generative Project 1 — CLIP + GPT-2 Image Captioning

Multimodal image captioning pipeline built on Flickr30k. Extracts CLIP image embeddings, trains a prefix-conditioned GPT-2 decoder, and evaluates caption quality with BLEU, CIDEr, ROUGE-L, and METEOR.

---

## Milestone Status

| Milestone | Description | Status |
|---|---|---|
| M1 | Data pipeline + embedding extraction | Complete — 6/6 tests passing |
| M2 | Prefix projection + GPT-2 decoder + training CLI | Implemented |
| M3 | Evaluation metrics + ablation experiments | Implemented |

---

## What's Built

### M1 — Data Pipeline & Embedding Extraction
- **Data pipeline** — streams 1,000 Flickr30k image–caption pairs across 10 classes, with image validation, repair, and stratified sampling
- **CLIP image encoder** — extracts `[N, 512]` L2-normalized embeddings via `openai/clip-vit-base-patch32`
- **GPT-2 text encoder** — tokenizes captions, extracts mean-pooled `[N, 512]` projected embeddings via `gpt2`
- **Checkpointed output** — saves to `outputs/embeddings.npz` every 100 batches
- **6 unit tests** — all passing, CPU-only

### M2 — Caption Generation
- **PrefixProjection** — maps CLIP image embeddings `[B, 512]` → K prefix tokens `[B, 10, 768]` for GPT-2
- **Teacher-forcing training** — cross-entropy loss on caption tokens, prefix positions masked with `-100`
- **Three decoding strategies** — greedy, beam search (configurable width), nucleus sampling
- **LoRA fine-tuning** — `peft` LoRA adapters on `c_attn` + `c_proj`, ~295k trainable params for r=8
- **`train.py` CLI** — single entry point for all experiments; checkpoints saved to `outputs/weights/{run_id}/`

### M3 — Evaluation & Ablation
- **Corpus metrics** — BLEU-1, BLEU-4 (`sacrebleu`), CIDEr (`pycocoevalcap`), ROUGE-L (`rouge-score`), METEOR (`nltk`)
- **Per-sample metrics** — sentence BLEU-4 + ROUGE-L written to `captions.jsonl` per run
- **Ablation runs** — sweep encoder, fine-tuning strategy, and decoding via CLI flags
- **Streamlit dashboard** — unlocks automatically when ≥3 `outputs/runs/*/metrics.json` files exist

---

## Dataset

| | |
|---|---|
| Source | Flickr30k via `nlphuji/flickr30k` |
| Size | 1,000 samples (configurable in `configs/config.yaml`) |
| Classes | `bicycle, motorcycle, bus, dog, cat, chair, bench, umbrella, skateboard, pizza` |
| Balance | 100 samples per class (±5% tolerance) |
| Candidate pool | 3,000 (3× buffer for filtering and repair) |

---

## Project Structure

```
generative_project_1/
├── configs/
│   └── config.yaml              ← all parameters live here (single source of truth)
├── src/
│   ├── utils.py                 ← hardware detection, logging, config loading
│   ├── data_loader.py           ← Flickr30k streaming, candidate pool
│   ├── preprocessor.py          ← validation, repair, CLIP transforms, tokenization
│   ├── embeddings_io.py         ← save/load .npz, checkpointing
│   ├── pipeline.py              ← orchestrator (extract | generate | evaluate modes)
│   ├── decoder.py               ← [M2] PrefixProjection, generate_caption
│   ├── metrics.py               ← [M3] BLEU, CIDEr, ROUGE-L, METEOR
│   └── val_references.py        ← [M3] reference caption cache builder
├── train.py                     ← [M2/M3] training CLI entry point
├── streamlit_app.py             ← web demo (milestone-gated)
├── notebooks/
│   └── baseline_colab.ipynb     ← full M1 pipeline run on Colab Pro
├── tests/
│   └── test_pipeline.py         ← unit tests (M1: 6 passing)
└── requirements/
    ├── local.txt                ← Windows (CUDA 12.1) + macOS (MPS) / Python 3.11
    └── colab.txt                ← Google Colab Pro / Python 3.12
```

---

## Output Directory Structure

All outputs are organized under `outputs/`:

```
outputs/
├── embeddings.npz              ← M1: CLIP + GPT-2 embeddings [N, 512]
├── captions.jsonl              ← M2: batch generation results
├── metrics.json                ← M3: aggregated metrics across all runs
├── val_references.json         ← M3: validation reference captions lookup
├── weights/                    ← M2/M3: trained model checkpoints
│   ├── best/
│   │   └── checkpoint_best.pt  ← best checkpoint (Streamlit loads this)
│   └── {run_id}/
│       ├── checkpoint_epoch1.pt
│       └── checkpoint_best.pt
├── runs/                       ← M2/M3: per-run training artifacts
│   └── {run_id}/
│       ├── config.json         ← hyperparams at run start
│       ├── metrics.json        ← final scores (Streamlit M3 dashboard)
│       ├── captions.jsonl      ← per-image generated captions + scores
│       └── training_log.csv    ← epoch, loss, lr history
├── cache/                      ← CLIP embedding cache (speeds up re-runs)
│   └── clip_cache_{run_id}.pt
└── plots/                      ← M3 visualization outputs
```

---

## Setup

### Windows — NVIDIA GPU (CUDA 12.1)

```powershell
# Step 1 — Clone the repo
git clone https://github.com/Scofe-C/Generative_Project_1.git
cd Generative_Project_1

# Step 2 — Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Step 3 — PyTorch CUDA
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 4 — Everything else (includes M2/M3 deps: peft, sacrebleu, rouge-score, nltk, pycocoevalcap)
pip install -r requirements\local.txt

# Step 5 — Run tests
pytest tests/ -v

# Step 6 — Run M1 pipeline (extracts embeddings)
python -c "from src.pipeline import run; result = run(); print(result['summary'])"
```

---

### macOS — Apple Silicon (MPS)

```bash
# Step 1 — Clone the repo
git clone https://github.com/Scofe-C/Generative_Project_1.git
cd Generative_Project_1

# Step 2 — Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Step 3 — PyTorch MPS build
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Step 4 — Everything else
pip install -r requirements/local.txt

# Step 5 — Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True on Apple Silicon, False on Intel Mac

# Step 6 — Run tests
pytest tests/ -v

# Step 7 — Run M1 pipeline
python -c "from src.pipeline import run; result = run(); print(result['summary'])"
```

> **Apple Silicon (8GB unified memory):** set `batch_size: 16` in `configs/config.yaml` to avoid memory pressure. 16GB+ machines can use the default `batch_size: 32`.

---

### Google Colab Pro (full pipeline)

1. `Runtime` → `Change runtime type` → **GPU → A100**
2. Open `notebooks/baseline_colab.ipynb` and run all cells

---

## Usage

### M1 — Extract embeddings

```bash
python -c "from src.pipeline import run; run()"
# Output: outputs/embeddings.npz
```

### M2 — Train caption model

```bash
# Run 1 — frozen prefix projection, greedy decoding
python train.py --run_id run_001 --finetune frozen --decoder greedy

# Run 2 — LoRA fine-tuning, beam search
python train.py --run_id run_002 --finetune lora --lora_rank 8 --decoder beam

# Run 3 — prefix tuning only
python train.py --run_id run_003 --finetune prefix_tuning --decoder greedy

# Run 4 — decoding sweep on Run 2's model (nucleus sampling)
python train.py --run_id run_004 --finetune lora --decoder nucleus --temperature 0.9 --top_p 0.9

# Smoke test (< 3 min, verifies end-to-end correctness)
python train.py --run_id smoke_test --finetune frozen --max_samples 50 --epochs 1 --batch_size 4
```

**CLIP embedding cache:** The first run for each `--run_id` pre-computes and caches CLIP embeddings to `outputs/cache/clip_cache_{run_id}.pt`. Subsequent runs load in ~2 seconds instead of ~8 minutes.

### M3 — View results

```bash
# Launch Streamlit demo (M3 dashboard unlocks when ≥3 runs have metrics.json)
streamlit run streamlit_app.py

# Or check metrics directly
python -c "
import json; from pathlib import Path
for f in sorted(Path('outputs/runs').glob('*/metrics.json')):
    m = json.loads(f.read_text())
    print(f\"{m['run_id']:20} BLEU-4={m.get('bleu_4',0):.3f}  CIDEr={m.get('cider',0):.3f}  ROUGE-L={m.get('rouge_l',0):.3f}\")
"
```

### Streamlit demo

```bash
streamlit run streamlit_app.py
# or with iframe embed mode:
# open http://localhost:8501?embed=true
```

Milestone gating:
- **M1** — always active (CLIP zero-shot classification + embedding visualization)
- **M2** — unlocks when `outputs/weights/best/checkpoint_best.pt` exists
- **M3** — unlocks when ≥3 `outputs/runs/*/metrics.json` files exist

Override via env var: `MILESTONE=2 streamlit run streamlit_app.py`

---

## Environment

| | Windows | macOS (Apple Silicon) | Colab Pro |
|---|---|---|---|
| Python | 3.11 | 3.11 | 3.12 |
| PyTorch | 2.5.1 + cu121 | 2.5.1 (MPS) | 2.5.1 + cu12x |
| transformers | 4.44.2 | 4.44.2 | 4.44.2 |
| peft | 0.12.0 | 0.12.0 | 0.12.0 |
| Accelerator | RTX 4060 8GB | Apple MPS | A100 40GB |

---

## Tests

```bash
pytest tests/ -v   # M1: 6/6 passing
```

| Test | Validates |
|---|---|
| `test_image_tensor_shape` | CLIPProcessor output is `[3, 224, 224]` |
| `test_token_length` | Tokens never exceed `max_length=77` |
| `test_class_balance` | Each class within ±5% of target |
| `test_output_schema` | `.npz` has all required keys and shapes |
| `test_repair_logging` | Repair events written to `logs/repairs.jsonl` |
| `test_shortfall_handling` | Pipeline completes when pool < target |

---

## Key Configuration

All parameters are in `configs/config.yaml`. Never hardcode values in `src/`.

```yaml
pipeline:
  mode: "extract"      # extract | generate | evaluate

dataset:
  target_n: 1000       # total samples
  classes: [bicycle, motorcycle, bus, dog, cat, ...]

generation:
  decoding_strategy: "beam"   # greedy | beam | nucleus
  beam_width: 5
  max_new_tokens: 50

fine_tuning:
  lora_rank: 8
  learning_rate: 5.0e-5
  num_epochs: 3
```

---

