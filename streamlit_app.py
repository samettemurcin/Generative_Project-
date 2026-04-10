"""
streamlit_app.py — Image Captioning Web Demo
=============================================
Milestone-gated Streamlit interface for the CLIP + GPT-2 captioning pipeline.

Run
---
    streamlit run streamlit_app.py

Requirements
------------
    pip install streamlit>=1.32 torch transformers pillow matplotlib numpy

Milestone gating
----------------
    M1  Always active — CLIP zero-shot classification + embedding viz
    M2  Unlocks when a trained checkpoint exists (runs/best/checkpoint_best.pt)
    M3  Unlocks when ≥3 run metrics.json files exist under runs/

Override via environment variable:
    MILESTONE=1 streamlit run streamlit_app.py

Design rules (read before editing)
------------------------------------
1. @st.cache_resource on ALL model loaders.
   Models load ONCE at startup and persist across reruns.
   Removing this causes 30+ second reloads on every slider touch.

2. st.session_state for pipeline results.
   Results are stored after pipeline runs and retrieved on rerender.
   Without this, changing any sidebar widget re-runs the full pipeline.

3. st.spinner on every model call > 0.5s.
   Streamlit freezes the UI during blocking calls.
   The spinner is the only signal to the user that work is happening.

4. Never call torch inside a Streamlit callback directly.
   Always call a function decorated with @st.cache_resource or
   stored results via st.session_state. Bare torch calls in button
   callbacks are fine as long as they're wrapped in try/except.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # must be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_device

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "Image Captioning Demo",
    page_icon   = "🖼",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Iframe embed detection
# ---------------------------------------------------------------------------
# When the website loads Streamlit via <iframe src="...?embed=true">,
# we collapse the sidebar and hide chrome for a cleaner embedded look.
# Standalone usage (no query param) is completely unaffected.
# ---------------------------------------------------------------------------
EMBED_MODE = (
    st.query_params.get("embed", "").lower() == "true"
    or "hide_sidebar" in st.query_params.get_all("embed_options")
)

# ---------------------------------------------------------------------------
# Global CSS theme — matches web/index.html design tokens
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Apply Inter to everything */
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', sans-serif !important;
}

/* Headings use Newsreader serif */
h1, h2, h3, .stTitle > div, [data-testid="stHeadingWithActionElements"] {
    font-family: 'Newsreader', serif !important;
    font-style: italic;
    font-weight: 600;
}

/* ── Color palette (from web/index.html CSS vars) ───────────── */
.stApp {
    background-color: #fbf9f6 !important;
}

/* Primary buttons → warm brown */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: #99420d !important;
    border-color: #99420d !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #b95925 !important;
    border-color: #b95925 !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    border-color: rgba(220,193,181,0.3) !important;
    color: #99420d !important;
}

/* File uploader border */
[data-testid="stFileUploader"] {
    border-color: rgba(220,193,181,0.3) !important;
}

/* Tabs — active tab uses primary color */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #99420d !important;
    border-bottom-color: #99420d !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #376847 !important;
    font-variant-numeric: tabular-nums;
}

/* Expander headers */
details summary span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #56433a !important;
}

/* Caption box styling */
.caption-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    line-height: 1.65;
    color: #1b1c1a;
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid rgba(220,193,181,0.2);
    box-shadow: 0 0 0 1px rgba(220,193,181,0.1);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #fff4ed !important;
}
section[data-testid="stSidebar"] *:not(code):not(pre):not([data-testid="stCodeBlock"] *):not([data-testid="stSidebarCollapsedControl"] *):not(i):not(.material-symbols-rounded) {
    color: #1b1c1a !important;
}
section[data-testid="stSidebar"] code,
section[data-testid="stSidebar"] pre {
    color: #e0e0e0 !important;
}
/* Hide the sidebar collapse button entirely */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* Fix expander arrow icon rendering as text */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"] {
    font-size: 0 !important;
}
[data-testid="stExpander"] summary span.material-symbols-rounded,
details summary span {
    font-size: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary p {
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #1b1c1a !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
    color: #1b1c1a !important;
    font-family: 'Inter', sans-serif !important;
}

/* Dividers */
hr {
    border-color: rgba(220,193,181,0.2) !important;
}

/* Dataframe header */
[data-testid="stDataFrame"] th {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

if EMBED_MODE:
    st.markdown("""
    <style>
    /* ── Embed-mode overrides ──────────────────────────────────
       Applied only when ?embed=true is in the URL.
       Hides sidebar, toolbar, footer, and hamburger menu
       so the Streamlit app looks native inside the website iframe.
       ────────────────────────────────────────────────────────── */

    /* Collapse sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Hide the hamburger menu / toolbar / footer */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }

    /* Remove default padding — fill the iframe fully */
    .stApp > div:first-child {
        padding-top: 0 !important;
    }
    .block-container {
        padding: 1rem 1.5rem !important;
        max-width: 100% !important;
    }
    /* Tighter file uploader */
    [data-testid="stFileUploader"] section {
        padding: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Tighten standalone layout too
st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config and milestone detection
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_config() -> dict:
    return load_config()

config = _load_config()

CLIP_NAME  = config["models"]["clip"]
GPT2_NAME  = config["models"]["gpt2"]
CLASSES    = config["dataset"]["classes"]

_forced      = int(os.environ.get("MILESTONE", "0"))
_ckpt_path   = Path(os.environ.get("HF_CHECKPOINT", "outputs/weights/best/checkpoint_best.pt"))
_runs_dir    = PROJECT_ROOT / "outputs" / "runs"
_n_runs      = len(list(_runs_dir.glob("*/metrics.json"))) if _runs_dir.exists() else 0

M2_READY = _forced == 2 or (_forced == 0 and _ckpt_path.exists())
M3_READY = _forced == 3 or (_forced == 0 and _n_runs >= 3)

ACTIVE_MILESTONE = 3 if M3_READY else (2 if M2_READY else 1)

# ---------------------------------------------------------------------------
# Model loaders — @st.cache_resource ensures each loads ONCE across ALL reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip():
    """Load CLIP model and processor. Cached for the lifetime of the app."""
    from transformers import CLIPModel, CLIPProcessor
    device = get_device()
    model  = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc   = CLIPProcessor.from_pretrained(CLIP_NAME)
    return model, proc, device


@st.cache_resource(show_spinner="Loading GPT-2 + checkpoint…")
def load_m2_model(checkpoint_path: str):
    """
    Load GPT-2 + trained prefix projection + optional LoRA weights.
    Only called when M2_READY is True.

    Returns (gpt2_model, prefix_proj, tokenizer, device) or raises on failure.
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from src.decoder import PrefixProjection  # implemented in M2

    from transformers import CLIPModel, CLIPProcessor

    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer architecture from checkpoint weights so any checkpoint works,
    # regardless of which CLIP/GPT-2 variant was used to train it.
    proj_weights    = ckpt["prefix_proj"]
    ckpt_clip_dim   = proj_weights["projection.0.weight"].shape[1]
    ckpt_gpt2_dim   = proj_weights["projection.3.bias"].shape[0]  # output = num_prefix * gpt2_dim ... need gpt2_dim
    # gpt2_dim comes from the last Linear's output divided by num_prefix;
    # we recover it from the saved config or fall back to a dim→name lookup.
    _dim_to_clip = {512: "openai/clip-vit-base-patch32", 768: "openai/clip-vit-large-patch14"}
    _dim_to_gpt2 = {768: "gpt2", 1024: "gpt2-medium", 1280: "gpt2-large"}

    ckpt_clip_name = ckpt.get("encoder_name") or _dim_to_clip.get(ckpt_clip_dim, CLIP_NAME)

    # Infer GPT-2 variant from checkpoint: try each known dim, pick the one where
    # projection.3.weight.shape[0] divides evenly AND yields a plausible num_prefix (1-20).
    proj3_out = proj_weights["projection.3.weight"].shape[0]
    if ckpt.get("gpt2_name"):
        ckpt_gpt2_name = ckpt["gpt2_name"]
        _inferred_gpt2_dim = None  # will be set after model load
    else:
        ckpt_gpt2_name = None
        for _gdim, _gname in sorted(_dim_to_gpt2.items()):
            if proj3_out % _gdim == 0 and 1 <= proj3_out // _gdim <= 20:
                ckpt_gpt2_name = _gname
                _inferred_gpt2_dim = _gdim
                break
        if ckpt_gpt2_name is None:
            ckpt_gpt2_name = GPT2_NAME  # last-resort fallback

    # Reload gpt2_dim from the actual gpt2 model config (authoritative)
    gpt2_model = GPT2LMHeadModel.from_pretrained(ckpt_gpt2_name).to(device).eval()
    real_gpt2_dim = gpt2_model.config.n_embd
    num_prefix    = proj3_out // real_gpt2_dim

    tokenizer = AutoTokenizer.from_pretrained(ckpt_gpt2_name, clean_up_tokenization_spaces=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the CLIP model that matches the checkpoint
    clip_model_ckpt = CLIPModel.from_pretrained(ckpt_clip_name).to(device).eval()
    clip_proc_ckpt  = CLIPProcessor.from_pretrained(ckpt_clip_name)

    # Infer MLP depth from checkpoint: depth=2 has keys 0,1,2,3;
    # depth=3 also has keys 4,5,6,7 (the extra hidden layer block).
    ckpt_depth = 2
    if any(k.startswith("projection.4.") for k in proj_weights):
        ckpt_depth = 3

    prefix_proj = PrefixProjection(
        clip_dim=ckpt_clip_dim,
        gpt2_dim=real_gpt2_dim,
        num_prefix=num_prefix,
        depth=ckpt_depth,
        dropout=0.0,  # inference — no dropout needed
    ).to(device)

    prefix_proj.load_state_dict(proj_weights)

    if ckpt.get("lora_adapter") is not None:
        # LoRA weights have PEFT-prefixed key names — must wrap the model
        # with PEFT before loading, otherwise strict=False silently skips them.
        from peft import LoraConfig, get_peft_model
        ckpt_cfg = ckpt.get("config", {})
        lora_rank = ckpt_cfg.get("lora_rank", 8)
        lora_config = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_rank * 2,
            lora_dropout   = 0.0,
            target_modules = ["c_attn", "c_proj"],
            inference_mode = True,
        )
        gpt2_model = get_peft_model(gpt2_model, lora_config)
        gpt2_model.load_state_dict(ckpt["lora_adapter"], strict=False)
        gpt2_model.eval()

    return gpt2_model, prefix_proj, tokenizer, clip_model_ckpt, clip_proc_ckpt, device


# ---------------------------------------------------------------------------
# Pipeline functions — pure, no Streamlit calls inside
# ---------------------------------------------------------------------------

def clip_image_embedding(
    pil_image: Image.Image,
    clip_model,
    clip_processor,
    device,
) -> torch.Tensor:
    """[1, clip_dim] L2-normalised CLIP image embedding."""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)


def clip_text_embedding(
    text: str,
    clip_model,
    clip_processor,
    device,
) -> torch.Tensor:
    """[1, clip_dim] L2-normalised CLIP text embedding."""
    inputs = clip_processor(
        text=[text], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)


# Broad label set for dynamic zero-shot classification (top 10 shown)
_DYNAMIC_LABELS = [
    # People & activities
    "person", "man", "woman", "child", "baby", "crowd",
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear",
    "zebra", "giraffe", "fish", "butterfly", "rabbit",
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat",
    "train", "helicopter",
    # Food
    "pizza", "cake", "sandwich", "fruit", "banana", "apple", "salad",
    "hot dog", "donut", "ice cream", "coffee", "wine",
    # Furniture & indoor
    "chair", "couch", "bed", "table", "desk", "lamp", "television",
    "laptop", "phone", "book", "clock", "vase", "toilet", "sink",
    # Outdoor & nature
    "tree", "flower", "mountain", "ocean", "beach", "river", "lake",
    "sky", "sunset", "snow", "rain", "forest", "field", "garden",
    # Objects
    "umbrella", "backpack", "handbag", "suitcase", "skateboard",
    "surfboard", "tennis racket", "ball", "kite", "frisbee",
    "bench", "fire hydrant", "stop sign", "traffic light",
    # Scenes
    "kitchen", "bedroom", "bathroom", "restaurant", "street",
    "park", "stadium", "office", "bridge", "building", "church",
]


def run_m1_pipeline(
    pil_image: Image.Image,
    clip_model,
    clip_processor,
    device,
) -> dict:
    """
    M1: CLIP zero-shot classification.

    Scores image against a broad set of 80+ labels and returns top 10.
    """
    img_emb = clip_image_embedding(pil_image, clip_model, clip_processor, device)

    # Batch all text embeddings at once for speed
    prompts = [f"a photo of a {cls}" for cls in _DYNAMIC_LABELS]
    inputs = clip_processor(
        text=prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)
    with torch.no_grad():
        txt_embs = clip_model.get_text_features(**inputs)
    txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

    # Compute all similarities at once
    sims = (img_emb @ txt_embs.T).squeeze(0)
    scores = [(cls, float(sims[i])) for i, cls in enumerate(_DYNAMIC_LABELS)]
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top 10
    top_scores = scores[:10]
    top_cls, top_sim = top_scores[0]

    return {
        "scores":      top_scores,
        "top_class":   top_cls,
        "top_sim":     top_sim,
        "img_emb_np":  img_emb.cpu().numpy().flatten(),
    }


def run_m2_pipeline(
    pil_image:          Image.Image,
    gpt2_model,
    prefix_proj,
    tokenizer,
    clip_model,
    clip_processor,
    device,
    decoding_strategy:  str,
    beam_width:         int,
    temperature:        float,
    top_p:              float,
    repetition_penalty: float = 1.5,
) -> dict:
    """
    M2: Full caption generation.

    Returns dict with generated caption + CLIP cosine similarity.
    """
    from src.decoder import generate_caption

    img_emb = clip_image_embedding(pil_image, clip_model, clip_processor, device)

    gen_cfg = dict(config["generation"])
    gen_cfg["decoding_strategy"]  = decoding_strategy
    gen_cfg["beam_width"]         = beam_width
    gen_cfg["temperature"]        = temperature
    gen_cfg["top_p"]              = top_p
    gen_cfg["repetition_penalty"] = repetition_penalty

    caption = generate_caption(
        image_embedding = img_emb,
        prefix_proj     = prefix_proj,
        gpt2_model      = gpt2_model,
        tokenizer       = tokenizer,
        config          = {"generation": gen_cfg},
    )

    # Cosine similarity — image vs generated caption (CLIP-to-CLIP, correct)
    txt_emb = clip_text_embedding(caption, clip_model, clip_processor, device)
    cos_sim = float((img_emb * txt_emb).sum())

    return {
        "caption":      caption,
        "cos_sim":      cos_sim,
        "strategy":     decoding_strategy,
        "beam_width":   beam_width,
        "temperature":  temperature,
    }


# ---------------------------------------------------------------------------
# Matplotlib helpers — return Figure objects for st.pyplot()
# ---------------------------------------------------------------------------

def make_embedding_figure(embedding: np.ndarray, n_dims: int = 64) -> plt.Figure:
    """Horizontal bar chart of first n_dims embedding values."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    vals    = embedding[:n_dims]
    colors  = ["#1D9E75" if v >= 0 else "#D85A30" for v in vals]
    ax.barh(range(n_dims), vals, color=colors, linewidth=0, height=0.8)
    ax.axvline(0, color="#888780", linewidth=0.6)
    ax.set_yticks([0, n_dims // 4, n_dims // 2, n_dims - 1])
    ax.set_yticklabels([f"dim 0", f"dim {n_dims//4}", f"dim {n_dims//2}", f"dim {n_dims-1}"], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("value", fontsize=8)
    ax.set_title(f"Image embedding — first {n_dims} dims  (teal=positive, coral=negative)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def make_scores_figure(scores: list[tuple[str, float]]) -> plt.Figure:
    """Horizontal bar chart of CLIP class scores."""
    classes = [s[0] for s in scores]
    sims    = [s[1] for s in scores]
    colors  = [
        "#1D9E75" if s >= 0.25 else "#BA7517" if s >= 0.15 else "#B4B2A9"
        for s in sims
    ]
    fig, ax = plt.subplots(figsize=(6, max(3, len(classes) * 0.4)))
    y_pos   = range(len(classes))
    ax.barh(y_pos, sims, color=colors, linewidth=0, height=0.7)
    ax.axvline(0.25, color="#1D9E75", linewidth=0.8, linestyle="--", alpha=0.5, label="Strong (0.25)")
    ax.axvline(0.15, color="#BA7517", linewidth=0.8, linestyle="--", alpha=0.5, label="Moderate (0.15)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("CLIP cosine similarity", fontsize=8)
    ax.set_title("Class similarity scores\n(CLIP image emb ↔ CLIP text emb)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _alignment_badge(cos_sim: float) -> str:
    """Return a coloured Markdown badge string for a cosine similarity value."""
    if cos_sim >= 0.25:
        return f":green[**Strong alignment** ({cos_sim:.4f})]"
    elif cos_sim >= 0.15:
        return f":orange[**Moderate alignment** ({cos_sim:.4f})]"
    else:
        return f":red[**Weak alignment** ({cos_sim:.4f})]"


def _milestone_badge(milestone: int) -> str:
    if milestone == 1:
        return ":gray[M1 — embedding extraction]"
    elif milestone == 2:
        return ":green[M2 — caption generation]"
    else:
        return ":violet[M3 — ablation evaluation]"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Controls")

    st.markdown(f"**Active milestone:** {_milestone_badge(ACTIVE_MILESTONE)}")

    if M2_READY:
        st.divider()
        st.markdown("**Caption generation**")
        decoding_strategy = st.radio(
            "Decoding strategy",
            options=["greedy", "beam", "nucleus"],
            index=1,
            help=(
                "greedy: fastest, deterministic\n"
                "beam: best quality, slower\n"
                "nucleus: most diverse, stochastic"
            ),
        )
        beam_width = st.slider(
            "Beam width",
            min_value=1, max_value=10, value=5, step=1,
            disabled=(decoding_strategy != "beam"),
            help="Only used when strategy = beam",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.5, max_value=1.5, value=1.0, step=0.1,
            disabled=(decoding_strategy != "nucleus"),
            help="Only used when strategy = nucleus. Higher = more random.",
        )
        top_p = st.slider(
            "Top-p",
            min_value=0.5, max_value=1.0, value=0.9, step=0.05,
            disabled=(decoding_strategy != "nucleus"),
            help="Nucleus sampling threshold. Only used when strategy = nucleus.",
        )
        repetition_penalty = st.slider(
            "Repetition penalty",
            min_value=1.0, max_value=2.0, value=1.5, step=0.1,
            help="Penalise any previously generated token. 1.0 = off, 1.5 = recommended.",
        )
    else:
        decoding_strategy  = "beam"
        beam_width         = 5
        temperature        = 1.0
        top_p              = 0.9
        repetition_penalty = 1.5

    st.divider()
    st.markdown("**Model info**")
    st.code(f"CLIP: {CLIP_NAME.split('/')[-1]}\nGPT-2: {GPT2_NAME}", language=None)
    if M2_READY:
        st.success(f"Checkpoint: `{_ckpt_path.name}`")
    else:
        st.warning("No checkpoint — M1 only")
    if M3_READY:
        st.info(f"{_n_runs} training runs found")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if EMBED_MODE:
    st.markdown("#### Upload an image to generate a caption")
else:
    st.title("Image Captioning Demo")
    st.markdown(
        "Upload any image. CLIP encodes it and — if a trained model is available "
        "— GPT-2 generates a natural language description."
    )

# Image upload
uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif", "ico", "heic", "avif"],
    help="Any image format up to 10 MB. The model works best with clear, well-lit photos.",
)

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

# Validate file size (10 MB limit)
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
if uploaded.size > _MAX_FILE_SIZE:
    st.error(f"Image too large ({uploaded.size / 1024 / 1024:.1f} MB). Maximum is 10 MB.")
    st.stop()

# Load and normalize image: any mode → RGB, handle EXIF rotation
try:
    from PIL import ImageOps
    pil_image = Image.open(uploaded)
    pil_image = ImageOps.exif_transpose(pil_image)  # fix phone rotation
    pil_image = pil_image.convert("RGB")
except Exception as e:
    st.error(f"Cannot read this image: {e}")
    st.stop()

# ── Run pipeline (before layout so results are ready) ──────────────────
pipeline_key = (
    uploaded.name,
    uploaded.size,
    decoding_strategy,
    beam_width if decoding_strategy == "beam" else 0,
    round(temperature, 1) if decoding_strategy == "nucleus" else 0,
    round(top_p, 2) if decoding_strategy == "nucleus" else 0,
    round(repetition_penalty, 1),
)

if st.session_state.get("pipeline_key") != pipeline_key:
    with st.spinner("Running CLIP image encoder…"):
        try:
            clip_model, clip_processor, device = load_clip()
            m1_result = run_m1_pipeline(pil_image, clip_model, clip_processor, device)
            st.session_state["m1_result"] = m1_result
            st.session_state["m1_error"]  = None
        except Exception as e:
            st.session_state["m1_result"] = None
            st.session_state["m1_error"]  = str(e)

    if M2_READY:
        with st.spinner("Generating caption…"):
            try:
                gpt2_model, prefix_proj, tokenizer, m2_clip, m2_clip_proc, device = load_m2_model(str(_ckpt_path))
                m2_result = run_m2_pipeline(
                    pil_image, gpt2_model, prefix_proj, tokenizer,
                    m2_clip, m2_clip_proc, device,
                    decoding_strategy, beam_width, temperature, top_p,
                    repetition_penalty,
                )
                st.session_state["m2_result"] = m2_result
                st.session_state["m2_error"]  = None
            except Exception as e:
                st.session_state["m2_result"] = None
                st.session_state["m2_error"]  = str(e)

    st.session_state["pipeline_key"] = pipeline_key

m1_result = st.session_state.get("m1_result")
m1_error  = st.session_state.get("m1_error")
m2_result = st.session_state.get("m2_result")
m2_error  = st.session_state.get("m2_error")

# ── Layout: Image (center) | Caption + Metrics (right) ────────────────
col_image, col_results = st.columns([1, 1], gap="large")

with col_image:
    st.image(pil_image, caption=uploaded.name, width="stretch")

with col_results:
    # ── Caption ──────────────────────────────────────────────────────
    if M2_READY:
        if m2_error:
            st.error(f"Caption generation failed: {m2_error}")
        elif m2_result:
            caption = m2_result["caption"]
            cos_sim = m2_result["cos_sim"]
            strat   = m2_result["strategy"]

            st.markdown(
                "<p style='font-size:0.7rem; text-transform:uppercase; "
                "letter-spacing:0.08em; font-weight:600; color:#99420d; "
                "margin-bottom:0.25rem;'>Generated Caption</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='caption-box'>{caption}</div>",
                unsafe_allow_html=True,
            )

            # ── Metrics rows (no Detected Class) ─────────────────────
            st.markdown(
                f"""
                <div style="margin-top:1rem;">
                <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;">
                <tr style="border-bottom:1px solid rgba(220,193,181,0.2);">
                    <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">CLIP Alignment</td>
                    <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{cos_sim:.4f}</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(220,193,181,0.2);">
                    <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">Strategy</td>
                    <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{strat}{f' (k={m2_result["beam_width"]})' if strat == 'beam' else ''}</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">Caption Length</td>
                    <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{len(caption.split())} words</td>
                </tr>
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Optional reference captions for NLG metrics ───────────────
            st.markdown(
                "<p style='font-size:0.7rem; text-transform:uppercase; "
                "letter-spacing:0.08em; font-weight:600; color:#99420d; "
                "margin-top:1.25rem; margin-bottom:0.25rem;'>Reference Captions "
                "<span style=\"font-weight:400; color:#625b55;\">(optional — one per line)</span></p>",
                unsafe_allow_html=True,
            )
            ref_text = st.text_area(
                label="Reference captions",
                label_visibility="collapsed",
                placeholder="Paste 1–5 reference captions here to compute BLEU-4, ROUGE-L, METEOR …",
                height=90,
                key="ref_captions",
            )

            if ref_text.strip():
                ref_lines = [ln.strip() for ln in ref_text.strip().splitlines() if ln.strip()]
                if ref_lines:
                    from src.metrics import compute_single_sample_metrics
                    per_img_scores = compute_single_sample_metrics(
                        hypothesis=caption,
                        references=ref_lines,
                    )
                    meteor_val = per_img_scores["meteor"]
                    meteor_display = f"{meteor_val:.4f}" if meteor_val >= 0 else "N/A"
                    st.markdown(
                        f"""
                        <div style="margin-top:0.75rem;">
                        <p style="font-size:0.7rem; text-transform:uppercase;
                           letter-spacing:0.08em; font-weight:600; color:#99420d;
                           margin-bottom:0.25rem;">NLG Metrics</p>
                        <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;">
                        <tr style="border-bottom:1px solid rgba(220,193,181,0.2);">
                            <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">BLEU-4</td>
                            <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{per_img_scores['bleu_4']:.4f}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(220,193,181,0.2);">
                            <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">ROUGE-L</td>
                            <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{per_img_scores['rouge_l']:.4f}</td>
                        </tr>
                        <tr>
                            <td style="padding:0.5rem 0; color:#625b55; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">METEOR</td>
                            <td style="padding:0.5rem 0; text-align:right; font-size:1rem; font-weight:700; color:#376847;">{meteor_display}</td>
                        </tr>
                        </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info(
            "Caption generation not yet available. "
            f"Train a model and place checkpoint at `{_ckpt_path}`."
        )

# ── Collapsible detail sections (below main layout, standalone only) ────
if not EMBED_MODE:
    if m1_result:
        with st.expander("CLIP Image Analysis", expanded=False):
            import pandas as pd
            scores  = m1_result["scores"]
            img_emb = m1_result["img_emb_np"]

            df = pd.DataFrame(scores, columns=["class", "cosine_similarity"])
            df["alignment"] = df["cosine_similarity"].apply(
                lambda s: "Strong" if s >= 0.25 else "Moderate" if s >= 0.15 else "Weak"
            )
            df["cosine_similarity"] = df["cosine_similarity"].round(4)
            st.dataframe(
                df.style.background_gradient(
                    subset=["cosine_similarity"],
                    cmap="RdYlGn",
                    vmin=0.0, vmax=0.35,
                ),
                width="stretch",
                hide_index=True,
            )

            st.pyplot(make_scores_figure(scores), width="stretch")
            st.pyplot(make_embedding_figure(img_emb), width="stretch")

# ── M3 comparison table (collapsible, standalone only) ──────────────────
if M3_READY and not EMBED_MODE:
    with st.expander("Run Comparison — All Experiments", expanded=False):
        rows = []
        for metrics_file in sorted(_runs_dir.glob("*/metrics.json")):
            try:
                m = json.loads(metrics_file.read_text())
                rows.append({
                    "Run":       m.get("run_id", metrics_file.parent.name),
                    "Encoder":   m.get("encoder", "?").split("/")[-1],
                    "Fine-tune": m.get("fine_tune", "?"),
                    "Decoding":  m.get("decoding",  "?"),
                    "BLEU-4":    round(m.get("bleu_4",  0), 3),
                    "CIDEr":     round(m.get("cider",   0), 3),
                    "METEOR":    round(m.get("meteor",  0), 3),
                    "ROUGE-L":   round(m.get("rouge_l", 0), 3),
                })
            except Exception:
                continue

        if rows:
            import pandas as pd
            df_runs = pd.DataFrame(rows)
            metric_cols = ["BLEU-4", "CIDEr", "METEOR", "ROUGE-L"]
            st.dataframe(
                df_runs.style.background_gradient(
                    subset=metric_cols, cmap="YlGn",
                ).highlight_max(
                    subset=metric_cols, color="#d4edda",
                ),
                width="stretch",
                hide_index=True,
            )