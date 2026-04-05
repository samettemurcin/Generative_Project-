"""
test_M2.py — Unit tests for M2 decoder module
===============================================
All tests run on CPU with no GPU required.
GPT-2 small is loaded from local cache (~500 MB, one-time download).

Run with:
    pytest tests/test_M2.py -v
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Fixtures — loaded once per module, shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@pytest.fixture(scope="module")
def prefix_proj():
    from src.decoder import PrefixProjection
    return PrefixProjection(clip_dim=512, gpt2_dim=768, num_prefix=10)


# ---------------------------------------------------------------------------
# PrefixProjection
# ---------------------------------------------------------------------------

def test_prefix_projection_shape_single(prefix_proj):
    x = torch.randn(1, 512)
    assert prefix_proj(x).shape == (1, 10, 768)


def test_prefix_projection_shape_batch(prefix_proj):
    x = torch.randn(4, 512)
    assert prefix_proj(x).shape == (4, 10, 768)


def test_prefix_projection_param_count():
    from src.decoder import PrefixProjection
    proj = PrefixProjection(clip_dim=512, gpt2_dim=768, num_prefix=10)
    # MLP: Linear(512,1024)=525312  +  LayerNorm(1024)=2048  +  Linear(1024,7680)=7872000  =  8399360
    assert sum(p.numel() for p in proj.parameters()) == 8_399_360


def test_prefix_projection_differentiable(prefix_proj):
    x = torch.randn(2, 512, requires_grad=True)
    assert prefix_proj(x).requires_grad


# ---------------------------------------------------------------------------
# build_inputs_embeds
# ---------------------------------------------------------------------------

def test_build_inputs_embeds_shapes(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import build_inputs_embeds
    gpt2, tok = gpt2_and_tokenizer
    B, seq_len, K = 2, 77, prefix_proj.num_prefix

    embeds, mask, labels = build_inputs_embeds(
        torch.randn(B, 512),
        torch.randint(100, 5000, (B, seq_len)),
        torch.ones(B, seq_len, dtype=torch.long),
        prefix_proj, gpt2, tok, torch.device("cpu"),
    )
    assert embeds.shape == (B, K + seq_len, 768)
    assert mask.shape   == (B, K + seq_len)
    assert labels.shape == (B, K + seq_len)


def test_build_inputs_embeds_prefix_labels_masked(prefix_proj, gpt2_and_tokenizer):
    """First K label positions must all be -100."""
    from src.decoder import build_inputs_embeds
    gpt2, tok = gpt2_and_tokenizer
    K = prefix_proj.num_prefix

    _, _, labels = build_inputs_embeds(
        torch.randn(2, 512),
        torch.randint(100, 5000, (2, 77)),
        torch.ones(2, 77, dtype=torch.long),
        prefix_proj, gpt2, tok, torch.device("cpu"),
    )
    assert (labels[:, :K] == -100).all(), "All prefix positions must be -100"


def test_build_inputs_embeds_caption_labels_not_all_masked(prefix_proj, gpt2_and_tokenizer):
    """Caption label positions must have at least some real token IDs."""
    from src.decoder import build_inputs_embeds
    gpt2, tok = gpt2_and_tokenizer
    K = prefix_proj.num_prefix

    _, _, labels = build_inputs_embeds(
        torch.randn(2, 512),
        torch.randint(100, 5000, (2, 77)),   # non-pad ids
        torch.ones(2, 77, dtype=torch.long),
        prefix_proj, gpt2, tok, torch.device("cpu"),
    )
    assert (labels[:, K:] != -100).any(), "Some caption positions must be real token IDs"


def test_build_inputs_embeds_padding_masked(prefix_proj, gpt2_and_tokenizer):
    """Padding token IDs in caption_ids must map to -100 in labels."""
    from src.decoder import build_inputs_embeds
    gpt2, tok = gpt2_and_tokenizer
    K   = prefix_proj.num_prefix
    pad = tok.pad_token_id

    caption_ids = torch.cat([
        torch.randint(100, 5000, (1, 40)),
        torch.full((1, 37), pad),
    ], dim=1)

    _, _, labels = build_inputs_embeds(
        torch.randn(1, 512), caption_ids,
        torch.ones(1, 77, dtype=torch.long),
        prefix_proj, gpt2, tok, torch.device("cpu"),
    )
    assert (labels[0, K + 40:] == -100).all(), "Pad positions must be -100"


def test_build_inputs_embeds_attention_mask_prefix(prefix_proj, gpt2_and_tokenizer):
    """Prefix positions in attention_mask must all be 1."""
    from src.decoder import build_inputs_embeds
    gpt2, tok = gpt2_and_tokenizer
    K = prefix_proj.num_prefix

    _, mask, _ = build_inputs_embeds(
        torch.randn(2, 512),
        torch.randint(100, 5000, (2, 77)),
        torch.ones(2, 77, dtype=torch.long),
        prefix_proj, gpt2, tok, torch.device("cpu"),
    )
    assert (mask[:, :K] == 1).all(), "Prefix attention mask positions must be 1"


# ---------------------------------------------------------------------------
# generate_caption
# ---------------------------------------------------------------------------

def test_generate_caption_returns_string(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    cfg = {"generation": {"decoding_strategy": "greedy", "max_new_tokens": 10}}
    caption = generate_caption(torch.randn(1, 512), prefix_proj, gpt2, tok, cfg)
    assert isinstance(caption, str) and len(caption) > 0


def test_generate_caption_no_special_tokens(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    caption = generate_caption(
        torch.randn(1, 512), prefix_proj, gpt2, tok,
        {"generation": {"decoding_strategy": "greedy", "max_new_tokens": 15}},
    )
    assert "<|endoftext|>" not in caption


def test_generate_caption_stripped(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    caption = generate_caption(
        torch.randn(1, 512), prefix_proj, gpt2, tok,
        {"generation": {"decoding_strategy": "greedy", "max_new_tokens": 10}},
    )
    assert caption == caption.strip()


def test_generate_caption_device_inferred(prefix_proj, gpt2_and_tokenizer):
    """Should work without explicit device argument."""
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    caption = generate_caption(
        torch.randn(1, 512), prefix_proj, gpt2, tok,
        {"generation": {"decoding_strategy": "greedy", "max_new_tokens": 8}},
    )
    assert isinstance(caption, str) and len(caption) > 0


def test_generate_caption_greedy_deterministic(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    emb = torch.randn(1, 512)
    cfg = {"generation": {"decoding_strategy": "greedy", "max_new_tokens": 10}}
    assert generate_caption(emb, prefix_proj, gpt2, tok, cfg) == \
           generate_caption(emb, prefix_proj, gpt2, tok, cfg)


def test_generate_caption_beam(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    caption = generate_caption(
        torch.randn(1, 512), prefix_proj, gpt2, tok,
        {"generation": {"decoding_strategy": "beam", "beam_width": 2, "max_new_tokens": 10}},
    )
    assert isinstance(caption, str) and len(caption) > 0


def test_generate_caption_nucleus(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    caption = generate_caption(
        torch.randn(1, 512), prefix_proj, gpt2, tok,
        {"generation": {"decoding_strategy": "nucleus", "top_p": 0.9,
                        "temperature": 1.0, "max_new_tokens": 10}},
    )
    assert isinstance(caption, str) and len(caption) > 0


def test_generate_caption_invalid_strategy_raises(prefix_proj, gpt2_and_tokenizer):
    from src.decoder import generate_caption
    gpt2, tok = gpt2_and_tokenizer
    with pytest.raises(ValueError, match="Unknown decoding strategy"):
        generate_caption(
            torch.randn(1, 512), prefix_proj, gpt2, tok,
            {"generation": {"decoding_strategy": "invalid", "max_new_tokens": 5}},
        )
