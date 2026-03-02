# Generative Project 1 — CLIP & GPT-2 Extraction Pipeline

Milestone 1 of a multimodal image captioning project. This milestone builds the data pipeline and extracts image and text embeddings from MS COCO Captions using CLIP and GPT-2.

---

## What's Built

- **Data pipeline** — streams 1,000 COCO image–caption pairs across 10 classes, with image validation, repair, and stratified sampling
- **CLIP image encoder** — extracts `[N, 512]` L2-normalized embeddings via `openai/clip-vit-base-patch32`
- **GPT-2 text encoder** — tokenizes captions and extracts mean-pooled `[N, 512]` projected embeddings via `gpt2`
- **Checkpointed output** — saves to `outputs/embeddings.npz` every 100 batches
- **6 unit tests** — all passing, CPU-only, no GPU required

---

## Dataset

| | |
|---|---|
| Source | MS COCO Captions via `HuggingFaceM4/COCO` |
| Size | 1,000 samples (configurable in `configs/config.yaml`) |
| Classes | `person, car, dog, cat, chair, bottle, bicycle, bird, laptop, cup` |
| Balance | 100 samples per class |
| Candidate pool | 3,000 (3× buffer for filtering and repair) |

---

## Project Structure

```
generative_project_1/
├── configs/config.yaml          ← all parameters live here
├── src/
│   ├── utils.py                 ← hardware detection, logging, config
│   ├── data_loader.py           ← COCO streaming, candidate pool
│   ├── preprocessor.py          ← validation, repair, transforms, tokenization
│   ├── embeddings_io.py         ← save/load .npz, checkpointing
│   └── pipeline.py              ← orchestrator
├── notebooks/
│   └── baseline_colab.ipynb     ← full pipeline run on Colab Pro
├── tests/
│   └── test_pipeline.py         ← 6 unit tests
├── requirements/
│   ├── local.txt                ← Windows (CUDA 12.1) + macOS (MPS) / Python 3.11
│   └── colab.txt                ← Google Colab Pro / Python 3.12
└── SETUP_GUIDE.md               ← full setup and workflow instructions
```

---

## Setup

### Windows — NVIDIA GPU (CUDA 12.1)

```powershell
# Step 1 — PyTorch CUDA (single line)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 2 — Everything else
pip install -r requirements\local.txt

# Step 3 — Run tests
pytest tests/ -v
```

### macOS — Apple Silicon (MPS)

```bash
# Step 1 — PyTorch MPS build
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Step 2 — Everything else
pip install -r requirements/local.txt

# Step 3 — Run tests
pytest tests/ -v
```

> **Intel Mac:** MPS is not available. Pipeline runs on CPU — functional but slower.  
> **pycocotools on macOS** requires Xcode command line tools: `xcode-select --install`

### Google Colab Pro (full pipeline)

1. `Runtime` → `Change runtime type` → **GPU → A100**
2. Open `notebooks/baseline_colab.ipynb` and run all cells

See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for detailed instructions and troubleshooting.

---

## Environment

| | Windows | macOS (Apple Silicon) | Colab Pro |
|---|---|---|---|
| Python | 3.11 | 3.11 | 3.12 |
| PyTorch | 2.5.1 + cu121 | 2.5.1 (MPS) | 2.10.0 + cu12x |
| transformers | 4.44.2 | 4.44.2 | 5.0.0 |
| Accelerator | RTX 4060 8GB | Apple MPS | A100 40GB |

---

## Tests

```
pytest tests/ -v   # 6/6 passing
```

| Test | Validates |
|---|---|
| `test_image_tensor_shape` | CLIPProcessor output is `[3, 224, 224]` |
| `test_token_length` | Tokens never exceed `max_length=77` |
| `test_class_balance` | Each class within ±5% of target |
| `test_output_schema` | `.npz` has all required keys and shapes |
| `test_repair_logging` | Repair events written to `logs/repairs.jsonl` |
| `test_shortfall_handling` | Pipeline completes when pool < target |