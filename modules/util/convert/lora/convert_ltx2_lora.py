"""LTX-2.3 LoRA key conversion utilities.

The official Lightricks LTX-2.3 LoRA safetensors (e.g.
``ltx-2.3-22b-distilled-lora-384-1.1.safetensors``) ship with key paths that
mostly align to diffusers' ``LTX2VideoTransformer3DModel`` module names — but
diverge at the top level for a small set of modules (patchify_proj, adaln_single,
av_ca_*, prompt_adaln_single, etc.). This module provides the small rename table
needed to translate between the two conventions, plus suffix normalization
between the PEFT (``lora_A``/``lora_B``) and Wan/musubi (``lora_down``/``lora_up``)
naming styles.

Reference: ai-toolkit/extensions_built_in/diffusion_models/ltx2/convert_ltx2_to_diffusers.py
(``LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT`` and ``convert_lora_original_to_diffusers``).
"""

from typing import Mapping

import torch


# 1:1 module-name renames. Order matters: keys are matched as substrings via
# ``str.replace``, so any keys whose left-hand string is a substring of a longer
# key must be processed AFTER the longer one. Entries are kept in declaration
# order, longest-string-first inside the apply function.
LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT: dict[str, str] = {
    # Per-block modulation parameters (LTX-2.0 base + LTX-2.3 inherits).
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",

    # Top-level cross-modality scale/shift adalns. These must be matched
    # BEFORE the bare ``adaln_single`` prefix-special-case below, hence the
    # length-sorted apply.
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single":          "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single":          "av_cross_attn_audio_v2a_gate",

    # LTX-2.3 prompt-conditioning adalns.
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "prompt_adaln_single":       "prompt_adaln",

    # Top-level patchify projections.
    "audio_patchify_proj": "audio_proj_in",
    "patchify_proj":       "proj_in",

    # Attention QK norms (only present in some LoRAs; harmless if absent).
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}


# ``adaln_single``/``audio_adaln_single`` are PREFIXES that must be replaced
# only when they appear at the START of the path (after the
# ``diffusion_model.`` prefix has been stripped). Doing a naive substring
# replace would also clobber the ``av_ca_*_adaln_single`` keys above, so they
# are handled separately below.
_ADALN_PREFIX_RENAMES: dict[str, str] = {
    "adaln_single.":       "time_embed.",
    "audio_adaln_single.": "audio_time_embed.",
}


def _rename_module_path(path: str) -> str:
    """Apply LTX-2.3 module renames to a single path (``diffusion_model.``-stripped).

    Order:
    1. Length-sorted substring replaces from ``LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT``
       (longest first so substrings don't clobber supersets).
    2. Prefix replace for ``adaln_single.`` / ``audio_adaln_single.`` at the
       start of the (already-renamed) path.
    """
    # Step 1: substring renames, longest first
    for old, new in sorted(LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT.items(), key=lambda kv: -len(kv[0])):
        path = path.replace(old, new)

    # Step 2: leading-prefix-only renames for adaln_single (avoids clobbering
    # the av_ca_*_adaln_single names handled in step 1)
    for old, new in _ADALN_PREFIX_RENAMES.items():
        if path.startswith(old):
            path = new + path[len(old):]
            break

    return path


def convert_ltx2_lora_original_to_diffusers(
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an official-format LTX-2.3 LoRA state dict to diffusers module naming.

    Input keys look like ``diffusion_model.<lightricks_path>.lora_A/lora_B.weight``.
    Output keys keep the ``diffusion_model.`` prefix but the inner path is
    rewritten to match ``LTX2VideoTransformer3DModel``'s module structure.

    The ``lora_A``/``lora_B`` suffix is preserved here — see
    ``normalize_lora_ab_to_down_up`` for the suffix normalization step.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        prefix = ""
        rest = key
        if rest.startswith("diffusion_model."):
            prefix = "diffusion_model."
            rest = rest[len(prefix):]
        out[prefix + _rename_module_path(rest)] = value
    return out


def convert_ltx2_lora_diffusers_to_original(
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Inverse of ``convert_ltx2_lora_original_to_diffusers``.

    Used when saving a trained LTX-2.3 LoRA in ComfyUI-compatible (Lightricks
    original) format.

    Implementation uses unique sentinels during replacement to avoid
    substring-collision: the inverse direction can RE-INTRODUCE substrings that
    later iterations would mistakenly re-match (e.g. ``prompt_adaln`` substring
    inside an already-rewritten ``audio_prompt_adaln_single``). The forward
    direction is safe without sentinels because each rename CONSUMES the
    distinguishing suffix; the inverse re-adds it.
    """
    inv_renames = {v: k for k, v in LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT.items()}
    inv_adaln = {v: k for k, v in _ADALN_PREFIX_RENAMES.items()}

    # Build a sentinel for each unique target (longest source first so that the
    # longer matches are replaced before their shorter substring counterparts).
    sentinel_to_final: dict[str, str] = {}
    sentinel_renames: list[tuple[str, str]] = []
    for i, (new, old) in enumerate(sorted(inv_renames.items(), key=lambda kv: -len(kv[0]))):
        sentinel = f"\x00R{i}\x00"
        sentinel_to_final[sentinel] = old
        sentinel_renames.append((new, sentinel))

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        prefix = ""
        rest = key
        if rest.startswith("diffusion_model."):
            prefix = "diffusion_model."
            rest = rest[len(prefix):]

        # Inverse prefix-only renames first (so we don't mangle the
        # av_ca_*_adaln_single rewrites that target the same prefix region).
        for new_prefix, old_prefix in inv_adaln.items():
            if rest.startswith(new_prefix):
                rest = old_prefix + rest[len(new_prefix):]
                break

        # Phase 1: replace each new-side string with a unique sentinel
        for new_str, sentinel in sentinel_renames:
            rest = rest.replace(new_str, sentinel)
        # Phase 2: expand sentinels to the final original-side strings
        for sentinel, final in sentinel_to_final.items():
            rest = rest.replace(sentinel, final)

        out[prefix + rest] = value
    return out


def normalize_lora_ab_to_down_up(
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename PEFT-style suffixes to musubi/Wan-style suffixes.

    ``.lora_A.weight`` → ``.lora_down.weight``
    ``.lora_B.weight`` → ``.lora_up.weight``

    Both conventions describe the same matrices (down-projection then
    up-projection); only the names differ. OT's internal LoRA application
    code path uses ``lora_down``/``lora_up``, so we standardize here at load
    time.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up.")
        out[new_key] = value
    return out


def pair_lora_down_up(
        state_dict: Mapping[str, torch.Tensor],
        prefix_to_strip: str = "diffusion_model.",
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Group a LoRA state dict into ``(module_path, lora_down, lora_up)`` triplets.

    Expects keys of the form ``[<prefix_to_strip>]<module_path>.lora_down.weight``
    and ``[<prefix_to_strip>]<module_path>.lora_up.weight``. The returned
    ``module_path`` has the prefix removed so it can be passed straight to
    ``Module.get_submodule()`` on the transformer.

    Keys missing a matching down/up partner are skipped with a warning.
    """
    downs: dict[str, torch.Tensor] = {}
    ups: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    extras: list[str] = []

    for key, value in state_dict.items():
        rest = key
        if rest.startswith(prefix_to_strip):
            rest = rest[len(prefix_to_strip):]
        if rest.endswith(".lora_down.weight"):
            downs[rest[: -len(".lora_down.weight")]] = value
        elif rest.endswith(".lora_up.weight"):
            ups[rest[: -len(".lora_up.weight")]] = value
        elif rest.endswith(".alpha"):
            # Alpha keys from PEFT/community LoRAs: scale = alpha / rank
            module_path = rest[: -len(".alpha")]
            alphas[module_path] = float(value)
        else:
            extras.append(key)

    paired: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    only_down = sorted(set(downs) - set(ups))
    only_up = sorted(set(ups) - set(downs))
    for path in sorted(set(downs) & set(ups)):
        d, u = downs[path], ups[path]
        if path in alphas:
            rank = d.shape[0]
            scale = alphas[path] / rank
            if abs(scale - 1.0) > 1e-6:
                u = u * scale
        paired.append((path, d, u))

    if extras:
        print(f"[Ltx2 LoRA] Skipping {len(extras)} unrecognized keys (e.g. {extras[0]})")
    if only_down:
        print(f"[Ltx2 LoRA] {len(only_down)} lora_down keys without lora_up partner")
    if only_up:
        print(f"[Ltx2 LoRA] {len(only_up)} lora_up keys without lora_down partner")

    return paired
