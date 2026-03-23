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
EMBED_MODE = st.query_params.get("embed", "").lower() == "true"

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

    /* Remove default top padding that Streamlit adds for the header */
    .stApp > div:first-child {
        padding-top: 0 !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
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
_ckpt_path   = Path(os.environ.get("HF_CHECKPOINT", "runs/best/checkpoint_best.pt"))
_runs_dir    = PROJECT_ROOT / "runs"
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

    clip_model, _, device = load_clip()

    gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_NAME).to(device).eval()
    tokenizer  = AutoTokenizer.from_pretrained(GPT2_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clip_dim   = clip_model.config.projection_dim    # 512 for ViT-B/32
    gpt2_dim   = gpt2_model.config.n_embd             # 768 for gpt2
    num_prefix = config["generation"].get("num_prefix_tokens", 10)

    prefix_proj = PrefixProjection(
        clip_dim=clip_dim,
        gpt2_dim=gpt2_dim,
        num_prefix=num_prefix,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    prefix_proj.load_state_dict(ckpt["prefix_proj"])
    if "lora_adapter" in ckpt:
        gpt2_model.load_state_dict(ckpt["lora_adapter"], strict=False)

    return gpt2_model, prefix_proj, tokenizer, device


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


def run_m1_pipeline(
    pil_image: Image.Image,
    clip_model,
    clip_processor,
    device,
) -> dict:
    """
    M1: CLIP zero-shot classification.

    Scores image against each class using "a photo of a {class}" prompts.
    Returns dict with ranked scores, top class, cosine sim.
    """
    img_emb = clip_image_embedding(pil_image, clip_model, clip_processor, device)

    scores = []
    for cls in CLASSES:
        prompt  = f"a photo of a {cls}"
        txt_emb = clip_text_embedding(prompt, clip_model, clip_processor, device)
        sim     = float((img_emb * txt_emb).sum())
        scores.append((cls, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_cls, top_sim = scores[0]

    return {
        "scores":      scores,
        "top_class":   top_cls,
        "top_sim":     top_sim,
        "img_emb_np":  img_emb.cpu().numpy().flatten(),
    }


def run_m2_pipeline(
    pil_image:         Image.Image,
    gpt2_model,
    prefix_proj,
    tokenizer,
    clip_model,
    clip_processor,
    device,
    decoding_strategy: str,
    beam_width:        int,
    temperature:       float,
    top_p:             float,
) -> dict:
    """
    M2: Full caption generation.

    Returns dict with generated caption + CLIP cosine similarity.
    """
    from src.decoder import generate_caption

    img_emb = clip_image_embedding(pil_image, clip_model, clip_processor, device)

    gen_cfg = dict(config["generation"])
    gen_cfg["decoding_strategy"] = decoding_strategy
    gen_cfg["beam_width"]        = beam_width
    gen_cfg["temperature"]       = temperature
    gen_cfg["top_p"]             = top_p

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

    st.divider()

    show_embedding = st.checkbox("Show embedding chart", value=True)
    show_scores    = st.checkbox("Show class scores chart", value=True)

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
    else:
        decoding_strategy = "beam"
        beam_width        = 5
        temperature       = 1.0
        top_p             = 0.9

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

st.title("Image Captioning Demo")
st.markdown(
    "Upload any image. CLIP encodes it and — if a trained model is available "
    "— GPT-2 generates a natural language description."
)

# Image upload
uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Drag and drop or click to browse. Any image works — not limited to training classes.",
)

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()  # nothing to show yet — halt cleanly

# Load image
pil_image = Image.open(uploaded).convert("RGB")

# Two-column layout: image on left, outputs on right
col_img, col_out = st.columns([1, 2], gap="large")

with col_img:
    st.image(pil_image, caption=uploaded.name, use_container_width=True)
    st.caption(f"Size: {pil_image.width}×{pil_image.height}px")

with col_out:

    # ── Run pipeline ────────────────────────────────────────────────────
    # Use session_state to avoid re-running the full pipeline when a
    # sidebar slider changes. Only re-run when image or M2 settings change.

    pipeline_key = (
        uploaded.name,
        uploaded.size,
        decoding_strategy,
        beam_width if decoding_strategy == "beam" else 0,
        round(temperature, 1) if decoding_strategy == "nucleus" else 0,
        round(top_p, 2) if decoding_strategy == "nucleus" else 0,
    )

    if st.session_state.get("pipeline_key") != pipeline_key:
        # ── M1 ────────────────────────────────────────────────────────
        with st.spinner("Running CLIP image encoder…"):
            try:
                clip_model, clip_processor, device = load_clip()
                m1_result = run_m1_pipeline(pil_image, clip_model, clip_processor, device)
                st.session_state["m1_result"] = m1_result
                st.session_state["m1_error"]  = None
            except Exception as e:
                st.session_state["m1_result"] = None
                st.session_state["m1_error"]  = str(e)

        # ── M2 ────────────────────────────────────────────────────────
        if M2_READY:
            with st.spinner("Generating caption…"):
                try:
                    gpt2_model, prefix_proj, tokenizer, device = load_m2_model(str(_ckpt_path))
                    m2_result = run_m2_pipeline(
                        pil_image, gpt2_model, prefix_proj, tokenizer,
                        clip_model, clip_processor, device,
                        decoding_strategy, beam_width, temperature, top_p,
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

    # ── M1 outputs ──────────────────────────────────────────────────────
    st.subheader("CLIP image analysis")

    if m1_error:
        st.error(f"M1 pipeline failed: {m1_error}")

    elif m1_result:
        top_cls  = m1_result["top_class"]
        top_sim  = m1_result["top_sim"]
        scores   = m1_result["scores"]
        img_emb  = m1_result["img_emb_np"]

        # Top class badge
        st.markdown(
            f"**Best matching class:** `{top_cls}` — "
            + _alignment_badge(top_sim)
        )
        st.caption(
            "Similarity = CLIP image embedding · CLIP text embedding  "
            "(both in CLIP's contrastive latent space — this number is meaningful)"
        )

        # Scores table + chart in tabs
        tab_table, tab_chart = st.tabs(["Scores table", "Scores chart"])

        with tab_table:
            import pandas as pd
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
                use_container_width=True,
                hide_index=True,
            )

        with tab_chart:
            if show_scores:
                st.pyplot(make_scores_figure(scores), use_container_width=True)

        if show_embedding:
            with st.expander("Embedding visualisation", expanded=False):
                st.pyplot(make_embedding_figure(img_emb), use_container_width=True)
                st.caption(
                    f"Shape: {img_emb.shape[0]}-d | "
                    f"Norm: {np.linalg.norm(img_emb):.4f} | "
                    f"Mean: {img_emb.mean():.4f} | "
                    f"Std: {img_emb.std():.4f}"
                )

    # ── M2 outputs ──────────────────────────────────────────────────────
    if M2_READY:
        st.divider()
        st.subheader("Generated caption")

        if m2_error:
            st.error(f"M2 pipeline failed: {m2_error}")

        elif m2_result:
            caption = m2_result["caption"]
            cos_sim = m2_result["cos_sim"]
            strat   = m2_result["strategy"]

            # Caption display — full text, large font
            st.markdown(
                f"<p style='font-size:1.2rem; line-height:1.7; "
                f"font-family:monospace; padding:1rem; "
                f"background:#f9f9f9; border-radius:8px; border:1px solid #e0e0e0;'>"
                f"{caption}</p>",
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "CLIP alignment",
                f"{cos_sim:.4f}",
                help="Cosine similarity: CLIP image emb ↔ CLIP text emb on generated caption",
            )
            col_b.metric("Decoding strategy", strat)
            col_c.metric("Caption length", f"{len(caption.split())} words")

            st.markdown(_alignment_badge(cos_sim))

            with st.expander("Generation details"):
                detail_lines = [f"Strategy: {strat}"]
                if strat == "beam":
                    detail_lines.append(f"Beam width: {m2_result['beam_width']}")
                elif strat == "nucleus":
                    detail_lines.append(f"Temperature: {m2_result['temperature']:.1f}")
                    detail_lines.append(f"Top-p: {top_p:.2f}")
                st.code("\n".join(detail_lines))
    else:
        st.divider()
        st.info(
            "Caption generation not yet available. "
            f"Train a model with `python train.py --run_id run_001` "
            f"and place the checkpoint at `{_ckpt_path}`."
        )

# ── M3 comparison table ─────────────────────────────────────────────────────
if M3_READY:
    st.divider()
    st.subheader("Run comparison — all experiments")
    st.caption(f"Reading from `{_runs_dir}` — {_n_runs} runs found")

    rows = []
    for metrics_file in sorted(_runs_dir.glob("*/metrics.json")):
        try:
            m = json.loads(metrics_file.read_text())
            rows.append({
                "Run":       m.get("run_id", metrics_file.parent.name),
                "Encoder":   m.get("encoder", "?").split("/")[-1],
                "Injection": m.get("injection", "?"),
                "Fine-tune": m.get("fine_tune", "?"),
                "Decoding":  m.get("decoding",  "?"),
                "BLEU-4":    round(m.get("bleu_4",  0), 3),
                "CIDEr":     round(m.get("cider",   0), 3),
                "METEOR":    round(m.get("meteor",  0), 3),
                "ROUGE-L":   round(m.get("rouge_l", 0), 3),
                "CLIP sim":  round(m.get("clip_sim_mean", 0), 3),
            })
        except Exception:
            continue

    if rows:
        import pandas as pd
        df_runs = pd.DataFrame(rows)
        metric_cols = ["BLEU-4", "CIDEr", "METEOR", "ROUGE-L", "CLIP sim"]
        st.dataframe(
            df_runs.style.background_gradient(
                subset=metric_cols, cmap="YlGn",
            ).highlight_max(
                subset=metric_cols, color="#d4edda",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Green shading = higher is better. "
            "Highlighted cell = best score in each column across all runs."
        )
    else:
        st.warning("No valid metrics.json files found in runs/ directory.")