"""Runtime-quantized LTX2 transformer loader for community Q6_K / Q8_0 GGUFs.

The diffusers ``from_single_file`` path works for the official "Lightricks/"
LTX2 GGUFs but fails on community-quantised dumps (e.g. ``sulphur_distil-Q6_K.gguf``)
that store tensors under the Comfy-style ``model.diffusion_model.*`` prefix
with ``general.architecture=ltxv``. The ComfyUI_LTX2_SM custom node solves
this by bypassing the diffusers single-file infrastructure and using a raw
``GGUFReader`` + explicit key translation. This module ports the same idea
into OneTrainer.

Pipeline:
    1. Read the GGUF into a flat ``{name: tensor_or_GGUFParameter}`` dict.
    2. Apply the LTX 2.0 → diffusers key renames (same table diffusers uses
       internally in ``convert_ltx2_transformer_to_diffusers``), starting
       with the ``model.diffusion_model.`` prefix strip that's the root
       cause of the standard-loader failure.
    3. Drop keys diffusers explicitly removes (video / audio embeddings
       connectors — handled in separate sub-modules in diffusers).
    4. Build ``LTX2VideoTransformer3DModel`` on the meta device.
    5. Swap every ``nn.Linear`` whose weight is a ``GGUFParameter`` for a
       diffusers ``GGUFLinear`` holding the packed weight (zero dequant
       at load time — happens lazily per matmul).
    6. Copy remaining unquantised tensors (norms, embeddings, biases)
       into the model.

After this the model behaves identically to one loaded via
``LTX2VideoTransformer3DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))``
on a well-formed GGUF, and ``replace_linear_with_quantized_layers`` can
upgrade ``GGUFLinear`` to OneTrainer's ``LinearGGUFA8`` as it does for
the Wan2.2 / Chroma GGUF paths.
"""

import gc

import torch
import torch.nn as nn

import gguf
from gguf import GGUFReader

from diffusers import LTX2VideoTransformer3DModel
from diffusers.quantizers.gguf.utils import (
    SUPPORTED_GGUF_QUANT_TYPES,
    GGUFLinear,
    GGUFParameter,
)


_UNQUANTIZED_TYPES = (
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
)


# Diffusers' LTX 2.0 → diffusers key renames (kept in sync with
# diffusers.loaders.single_file_utils.convert_ltx2_transformer_to_diffusers).
# Order matters: the longer key fragments must come before substrings of them
# so the substring replacement doesn't corrupt the longer key on a later pass.
_RENAME_DICT = {
    # Comfy-prefix strip — the whole point of this loader.
    "model.diffusion_model.": "",
    # Input patchify projections
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    # AV cross-attention modulation parameters (handle these before the
    # bare `adaln_single` rule below — they share the substring).
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single":          "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single":          "av_cross_attn_audio_v2a_gate",
    # Per-block cross-attention modulation
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    # Attention QK norms
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    # OT's _diffusers_patch.py adds two more (kept here so this loader is
    # self-contained even when that patch isn't applied yet):
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "prompt_adaln_single":       "prompt_adaln",
}

# Keys to drop entirely — diffusers' converter `remove_keys_inplace` does
# the same: these live in sibling sub-modules, not the transformer.
_DROP_PREFIXES = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
)


def _is_community_ltx2_gguf(gguf_path: str) -> bool:
    """Cheap header check: does this GGUF carry the Comfy-style prefix?

    Sniffs the first handful of tensor names — if any start with
    ``model.diffusion_model.`` AND ``general.architecture`` is ``ltxv`` (or
    a known LTX2 variant), this is the community format that diffusers'
    single-file converter can usually handle but that we route through
    this loader for robustness on unusual quants like Q6_K.
    """
    try:
        r = GGUFReader(gguf_path)
    except Exception:
        return False
    arch_field = r.fields.get("general.architecture")
    arch = ""
    if arch_field is not None:
        try:
            arch = str(arch_field.contents())
        except Exception:
            arch = ""
    has_prefix = any(
        t.name.startswith("model.diffusion_model.") for t in r.tensors[:20]
    )
    return has_prefix or arch in ("ltxv", "ltx2", "ltx", "ltx_video")


def _read_gguf_state_dict(gguf_path: str) -> dict[str, torch.Tensor]:
    reader = GGUFReader(gguf_path)
    out: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        qt = tensor.tensor_type
        is_quant = qt not in _UNQUANTIZED_TYPES
        if is_quant and qt not in SUPPORTED_GGUF_QUANT_TYPES:
            raise ValueError(
                f"Unsupported GGUF quantization type {qt} for tensor '{tensor.name}'."
            )
        weights = torch.from_numpy(tensor.data.copy())
        out[tensor.name] = (
            GGUFParameter(weights, quant_type=qt) if is_quant else weights
        )
    return out


def _convert_state_dict(
    gguf_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the LTX2 rename dict + drop prefixes diffusers excludes."""
    converted: dict[str, torch.Tensor] = {}
    for old_key, value in gguf_state.items():
        # Drop video/audio embeddings connectors — not part of the transformer.
        bare_key = old_key
        if bare_key.startswith("model.diffusion_model."):
            bare_key = bare_key[len("model.diffusion_model."):]
        if any(bare_key.startswith(p) for p in _DROP_PREFIXES):
            continue

        # adaln_single → time_embed.* / audio_adaln_single → audio_time_embed.*
        # diffusers handles this via a per-key callback because the rename
        # must only apply to weight/bias leaves of layers actually named
        # adaln_single (not the AV cross-attn ones that contain the substring
        # but were already remapped above).  We do the same here.
        new_key = old_key
        for src, dst in _RENAME_DICT.items():
            new_key = new_key.replace(src, dst)

        # Bare-name `adaln_single.` and `audio_adaln_single.` handlers,
        # applied only to weight/bias leaves so we don't accidentally
        # rewrite a different module that contained those substrings.
        if (".weight" in new_key) or (".bias" in new_key):
            if new_key.startswith("adaln_single."):
                new_key = "time_embed." + new_key[len("adaln_single."):]
            elif new_key.startswith("audio_adaln_single."):
                new_key = "audio_time_embed." + new_key[len("audio_adaln_single."):]

        converted[new_key] = value
    return converted


def _swap_quantized_linears(
    root: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
) -> None:
    """Replace every ``nn.Linear`` whose weight is a ``GGUFParameter`` with a
    diffusers ``GGUFLinear`` that holds the packed weight directly.

    Building the new ``GGUFLinear`` on ``device="meta"`` skips the transient
    bf16 allocation the default constructor would do — we immediately
    overwrite that placeholder with the packed ``GGUFParameter``.
    """
    targets: list[tuple[str, nn.Linear]] = []
    for full_name, module in root.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, GGUFLinear):
            wkey = f"{full_name}.weight"
            if wkey in state_dict and isinstance(state_dict[wkey], GGUFParameter):
                targets.append((full_name, module))

    for full_name, old in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root

        new_module = GGUFLinear(
            old.in_features,
            old.out_features,
            old.bias is not None,
            compute_dtype=compute_dtype,
            device=torch.device("meta"),
        )
        new_module.weight = state_dict[f"{full_name}.weight"]
        bkey = f"{full_name}.bias"
        if bkey in state_dict:
            new_module.bias = nn.Parameter(
                state_dict[bkey].to(compute_dtype), requires_grad=False
            )
        else:
            new_module.bias = None
        new_module.source_cls = type(old)
        new_module.requires_grad_(False)

        setattr(parent, attr, new_module)


def load_ltx2_transformer_from_gguf(
    gguf_path: str,
    base_model_name: str,
    dtype: torch.dtype,
) -> nn.Module:
    """Load an LTX2 transformer from a community-format GGUF.

    Args:
        gguf_path: absolute path to the .gguf file (e.g. sulphur_distil-Q6_K.gguf).
        base_model_name: HF repo or local snapshot path containing
            ``transformer/config.json``.  Used purely for the model
            architecture / hyperparameters; the GGUF supplies all weights.
        dtype: compute dtype for ``GGUFLinear`` forward passes and the small
            unquantised tensors (norms, embeddings, biases).
    """
    import os as _os

    print(
        f"[Ltx2TransformerGGUF] custom packed-weight loader engaged "
        f"(file={_os.path.basename(gguf_path)}, "
        f"size={_os.path.getsize(gguf_path) / 1e9:.2f} GB)",
        flush=True,
    )

    # Build the meta-device shell from the base model's transformer config.
    config = LTX2VideoTransformer3DModel.load_config(
        base_model_name, subfolder="transformer"
    )
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(config).to(dtype)
    model.eval()

    gguf_state = _read_gguf_state_dict(gguf_path)
    converted = _convert_state_dict(gguf_state)
    del gguf_state

    # 1) Replace quantised linears.
    _swap_quantized_linears(model, converted, compute_dtype=dtype)

    # 2) Copy remaining (unquantised) tensors into the model: embeddings,
    # norms, biases that weren't swallowed by GGUFLinear.bias above.
    remaining = {
        k: v.to(dtype) for k, v in converted.items()
        if not isinstance(v, GGUFParameter)
    }
    missing, unexpected = model.load_state_dict(
        remaining, strict=False, assign=False
    )

    # Unexpected keys among unquantised tensors signal a real key-name
    # mismatch (renames out of sync with diffusers).  Quantised mismatches
    # would have been caught earlier when _swap_quantized_linears failed
    # to find a target nn.Linear for the key.
    real_unexpected = [k for k in unexpected if k in remaining]
    if real_unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading LTX2 transformer GGUF: "
            f"{real_unexpected[:10]}"
            + (f" (and {len(real_unexpected) - 10} more)" if len(real_unexpected) > 10 else "")
            + ". Extend modules.modelLoader.ltx2.Ltx2TransformerGGUFLoader._RENAME_DICT."
        )

    del converted, remaining
    gc.collect()

    # Diagnostic: weights should remain packed.  packed_bytes ≈ GGUF file
    # size (within tens of MB for the small unquantised tensors), NOT 2×.
    n_packed, n_dequant, packed_bytes, dequant_bytes = 0, 0, 0, 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None or not isinstance(m, nn.Linear):
            continue
        if isinstance(w, GGUFParameter):
            n_packed += 1
            packed_bytes += w.nbytes
        else:
            n_dequant += 1
            dequant_bytes += w.nbytes
    print(
        f"[Ltx2TransformerGGUF] linears packed={n_packed} "
        f"({packed_bytes / 1e9:.2f} GB), "
        f"unpacked={n_dequant} ({dequant_bytes / 1e9:.2f} GB)",
        flush=True,
    )

    return model
