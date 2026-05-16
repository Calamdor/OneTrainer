"""Standalone LTX-2 text-embeddings connector loader (V1 + V2).

The 22B "sulphur" LTX-2.3 transformer was trained against a **V2 connector**
with wider hidden dims (video=4096, audio=2048), 8 transformer blocks per
modality, gated attention, and per-modality text projections.  Diffusers'
official Lightricks-distilled-13B snapshot (e.g. ``dg845/LTX-2.3-Diffusers``)
ships the **V1 connector** instead: 3840-dim, 2 layers, no gating, single
shared text projection.

OT's default load path reads ``<base_model>/connectors/`` which gives the
V1 weights.  Feeding V1 features into the 22B transformer silently
degrades prompt adherence (the cross-attention's K/V projections expect
4096-dim video / 2048-dim audio — see ``sulphur_dev_bf16.safetensors``
block-0 ``attn2.to_k.weight: (4096, 4096)`` and ``audio_attn2.to_k.weight:
(2048, 2048)``).

This module loads a standalone ``connector.safetensors`` (the 6.3 GB V2
file shipped with the 22B model release) into a freshly-constructed
``LTX2TextConnectors`` instance with the right V2 kwargs, applying a
small key-rename map between the file's Comfy-style names and diffusers'
state-dict names.

Variant auto-detection: inspects the file's text-projection weight shape
to pick V1 (3840) vs V2 (4096) kwargs.  Both code paths use the same
``LTX2TextConnectors`` class -- diffusers parameterises both variants
via constructor kwargs, no new class needed.
"""
from __future__ import annotations

import gc
import os

import torch
from safetensors import safe_open

from diffusers.pipelines.ltx2 import LTX2TextConnectors


# --------------------------------------------------------------------------
# V1 vs V2 kwargs for LTX2TextConnectors
# --------------------------------------------------------------------------
#
# Verified against the 6.3 GB ComfyUI ``connector.safetensors`` (V2) and the
# 22B sulphur transformer's cross-attention K/V projection shapes: V2's
# video_hidden_dim=4096 / audio_hidden_dim=2048 lock-and-key with
# ``attn2.to_k.weight (4096, 4096)`` and ``audio_attn2.to_k.weight (2048, 2048)``.
# V1 (3840 hidden, 2 layers, no gating, single projection) is the default
# pre-22B configuration shipped with ``dg845/LTX-2.3-Diffusers``.

_V2_KWARGS = dict(
    caption_channels=3840,                   # Gemma3-12B hidden
    text_proj_in_factor=49,                  # 48 hidden layers + 1 embed layer
    video_connector_num_attention_heads=32,
    video_connector_attention_head_dim=128,  # 32 * 128 = 4096
    video_connector_num_layers=8,
    video_gated_attn=True,
    video_hidden_dim=4096,
    audio_connector_num_attention_heads=32,
    audio_connector_attention_head_dim=64,   # 32 * 64 = 2048
    audio_connector_num_layers=8,
    audio_gated_attn=True,
    audio_hidden_dim=2048,
    per_modality_projections=True,
    proj_bias=True,
)

_V1_KWARGS = dict(
    caption_channels=3840,
    text_proj_in_factor=49,
    video_connector_num_attention_heads=30,
    video_connector_attention_head_dim=128,  # 30 * 128 = 3840
    video_connector_num_layers=2,
    video_gated_attn=False,
    video_hidden_dim=3840,
    audio_connector_num_attention_heads=30,
    audio_connector_attention_head_dim=128,
    audio_connector_num_layers=2,
    audio_gated_attn=False,
    audio_hidden_dim=2048,
    per_modality_projections=False,
    proj_bias=False,
)


# --------------------------------------------------------------------------
# File-key → diffusers state-dict-key rewrites
# --------------------------------------------------------------------------
#
# Order matters: longer prefixes first.  All replacements are
# substring-style ``str.replace`` (not regex) so each rule applies once
# per matching substring in the key.
#
# Verified by exhaustive test (smoke check): with the rule list below
# plus ``proj_bias=True`` in V2 kwargs, all 262 file keys translate to
# keys present in the constructed model's state_dict, and all 262 model
# keys are covered by translated inputs — 0 unmapped on either side, 0
# shape mismatches.

_RENAMES_V2 = [
    ("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.", "video_connector.transformer_blocks."),
    ("model.diffusion_model.video_embeddings_connector.learnable_registers",    "video_connector.learnable_registers"),
    ("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.", "audio_connector.transformer_blocks."),
    ("model.diffusion_model.audio_embeddings_connector.learnable_registers",    "audio_connector.learnable_registers"),
    # Per-modality text projections (V2 only — V1 has a single text_proj_in).
    ("text_embedding_projection.video_aggregate_embed.",  "video_text_proj_in."),
    ("text_embedding_projection.audio_aggregate_embed.",  "audio_text_proj_in."),
    # ComfyUI uses .q_norm / .k_norm; diffusers uses .norm_q / .norm_k.
    (".attn1.q_norm.", ".attn1.norm_q."),
    (".attn1.k_norm.", ".attn1.norm_k."),
]

_RENAMES_V1 = [
    # V1 connectors saved in Comfy-style use the same block-internal naming
    # but a single shared text projection (no per-modality split).
    ("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.", "video_connector.transformer_blocks."),
    ("model.diffusion_model.video_embeddings_connector.learnable_registers",    "video_connector.learnable_registers"),
    ("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.", "audio_connector.transformer_blocks."),
    ("model.diffusion_model.audio_embeddings_connector.learnable_registers",    "audio_connector.learnable_registers"),
    ("text_embedding_projection.aggregate_embed.", "text_proj_in."),
    (".attn1.q_norm.", ".attn1.norm_q."),
    (".attn1.k_norm.", ".attn1.norm_k."),
]


# --------------------------------------------------------------------------
# Variant detection
# --------------------------------------------------------------------------

def _sniff_variant(connector_path: str) -> str:
    """Inspect the file's text-projection weight to decide V1 vs V2.

    V2 has split ``text_embedding_projection.video_aggregate_embed.weight``
    of shape (4096, 188160) and a similar audio one.
    V1 has a single ``text_embedding_projection.aggregate_embed.weight``
    of shape (3840, 188160).
    """
    with safe_open(connector_path, framework="pt") as f:
        keys = list(f.keys())
        if "text_embedding_projection.video_aggregate_embed.weight" in keys:
            shape = f.get_tensor("text_embedding_projection.video_aggregate_embed.weight").shape
            if shape[0] == 4096:
                return "v2"
        if "text_embedding_projection.aggregate_embed.weight" in keys:
            shape = f.get_tensor("text_embedding_projection.aggregate_embed.weight").shape
            if shape[0] == 3840:
                return "v1"
        raise RuntimeError(
            f"Cannot auto-detect connector variant from {connector_path}. "
            f"Expected either 'text_embedding_projection.video_aggregate_embed.weight' (V2) "
            f"or 'text_embedding_projection.aggregate_embed.weight' (V1)."
        )


# --------------------------------------------------------------------------
# State-dict reader + translator
# --------------------------------------------------------------------------

def _translate_key(name: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        if old in name:
            name = name.replace(old, new)
    return name


def _read_connector_state_dict(
    connector_path: str, renames: list[tuple[str, str]]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(connector_path, framework="pt") as f:
        for k in f.keys():
            new_key = _translate_key(k, renames)
            out[new_key] = f.get_tensor(k)
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def load_ltx2_connector_from_file(
    connector_path: str,
    dtype: torch.dtype,
) -> LTX2TextConnectors:
    """Build an ``LTX2TextConnectors`` matching the variant of the file and
    load the file's weights into it.

    Args:
        connector_path: absolute path to the standalone ``.safetensors`` file
            (e.g. the 6.3 GB ``connector.safetensors`` shipped with the 22B
            sulphur distil model).
        dtype: compute dtype the connector will run in.  The small
            unquantised tensors land at this dtype; ``LTX2Pipeline`` doesn't
            quantise the connector path (unlike the transformer).

    Returns:
        Fully populated ``LTX2TextConnectors`` instance on CPU.
    """
    print(
        f"[Ltx2Connector] custom standalone-file loader engaged "
        f"(file={os.path.basename(connector_path)}, "
        f"size={os.path.getsize(connector_path) / 1e9:.2f} GB)",
        flush=True,
    )

    variant = _sniff_variant(connector_path)
    kwargs = _V2_KWARGS if variant == "v2" else _V1_KWARGS
    renames = _RENAMES_V2 if variant == "v2" else _RENAMES_V1

    # Build the empty shell on the meta device so the initial Linear/RMSNorm
    # allocations are zero-cost.  We immediately overwrite every parameter
    # with the file's weights via ``load_state_dict(assign=True)``.
    with torch.device("meta"):
        model = LTX2TextConnectors(**kwargs)
    model.eval()

    sd = _read_connector_state_dict(connector_path, renames)

    # Cast to the requested compute dtype before assignment.  Keep biases
    # and weights at the same dtype as the destination module's parameters
    # would have used (this matters for RMSNorm fp32-pre-cast paths).
    sd = {k: v.to(dtype) for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    real_unexpected = [k for k in unexpected if k in sd]
    real_missing = [k for k in missing if k not in sd]
    if real_unexpected or real_missing:
        raise RuntimeError(
            f"LTX2 connector load mismatch for {connector_path}: "
            f"{len(real_unexpected)} unexpected, {len(real_missing)} missing. "
            f"Sample unexpected: {real_unexpected[:5]}. "
            f"Sample missing: {real_missing[:5]}."
        )

    del sd
    gc.collect()

    print(
        f"[Ltx2Connector] loaded variant={variant} "
        f"(video_hidden={kwargs['video_hidden_dim']}, "
        f"audio_hidden={kwargs['audio_hidden_dim']}, "
        f"layers={kwargs['video_connector_num_layers']}, "
        f"gated_attn={kwargs['video_gated_attn']})",
        flush=True,
    )

    return model
