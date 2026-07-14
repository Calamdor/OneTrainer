"""Convert a raw Lightricks-format LTX-2.3 LoRA safetensors state dict into
(module_path, down, up) triplets addressable on the live diffusers transformer.

The rename table here is the exact inverse of ``Ltx2Model.diffusers_to_original()`` --
that direction was verified against the real diffusers ``LTX2VideoTransformer3DModel``
module tree (meta-device instantiation, this venv's pinned diffusers commit, matching the
test suite's ``cross_attn_mod=True`` config) earlier in this project. Kept as a single
source of truth split across the two directions rather than duplicated/re-derived, so a
future rename-table fix only needs to happen in one place conceptually (mirror any change
made to ``Ltx2Model.diffusers_to_original()`` here too).

Deliberately NOT covered, same as ``Ltx2Model.diffusers_to_original()``: the
``scale_shift_table_a2v_ca_video``/``audio`` pair from the old branch's rename table --
no matching submodule exists in ``named_modules()`` (those are top-level ``nn.Parameter``s,
not submodules); a LoRA that happens to target them would silently fail to match here,
same as it would with the generic saver hooks.
"""

from typing import Mapping

import torch


# native (Lightricks) substring -> diffusers substring. Longest-first order matters: keys
# are matched as substrings via str.replace, so a key whose left-hand string is a substring
# of a longer key must be processed after the longer one -- enforced by sorting on apply.
_NATIVE_TO_DIFFUSERS: dict[str, str] = {
    "audio_patchify_proj": "audio_proj_in",
    "patchify_proj": "proj_in",
    "audio_adaln_single.": "audio_time_embed.",
    "adaln_single.": "time_embed.",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "prompt_adaln_single": "prompt_adaln",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}


def _rename_native_to_diffusers(path: str) -> str:
    for old, new in sorted(_NATIVE_TO_DIFFUSERS.items(), key=lambda kv: -len(kv[0])):
        path = path.replace(old, new)
    return path


def convert_ltx2_lora_original_to_diffusers(
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite a raw Lightricks-format LoRA state dict's module paths to diffusers naming.

    Input keys look like ``diffusion_model.<lightricks_path>.lora_A/lora_B.weight`` (or
    ``.alpha``). The ``diffusion_model.`` prefix, if present, is preserved.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        prefix = ""
        rest = key
        if rest.startswith("diffusion_model."):
            prefix = "diffusion_model."
            rest = rest[len(prefix):]
        out[prefix + _rename_native_to_diffusers(rest)] = value
    return out


def normalize_lora_ab_to_down_up(
        state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """``.lora_A.weight`` -> ``.lora_down.weight``, ``.lora_B.weight`` -> ``.lora_up.weight``."""
    return {
        key.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up."): value
        for key, value in state_dict.items()
    }


def pair_lora_down_up(
        state_dict: Mapping[str, torch.Tensor],
        prefix_to_strip: str = "diffusion_model.",
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Group a LoRA state dict into (module_path, lora_down, lora_up) triplets.

    ``module_path`` has ``prefix_to_strip`` removed so it can be passed straight to
    ``Module.get_submodule()`` on the transformer. Alpha keys (PEFT/community convention:
    scale = alpha / rank) are folded into ``lora_up`` here so the caller never needs to
    special-case them. Keys missing a matching down/up partner are skipped with a warning.
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
            alphas[rest[: -len(".alpha")]] = float(value)
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
