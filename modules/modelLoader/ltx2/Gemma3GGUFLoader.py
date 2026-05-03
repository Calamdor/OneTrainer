"""Runtime-quantized Gemma3 loader for GGUF text encoders.

HF transformers' `from_pretrained(gguf_file=...)` dequantizes GGUF weights to
the requested dtype at load time, defeating the whole point of GGUF. This
module bypasses that path: it reads the GGUF directly with the `gguf` library,
swaps every quantized `nn.Linear` for a diffusers `GGUFLinear` holding the
packed `GGUFParameter`, and copies the small unquantized tensors (embeddings,
norms) into place. After this, OneTrainer's `replace_linear_with_quantized_layers`
upgrades the `GGUFLinear` modules to `LinearGGUFA8`, matching the path already
used for the LTX2 transformer.
"""

import gc

import torch
import torch.nn as nn

import gguf
from gguf import GGUFReader

from diffusers.quantizers.gguf.utils import (
    SUPPORTED_GGUF_QUANT_TYPES,
    GGUFLinear,
    GGUFParameter,
)
from transformers import (
    AutoConfig,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)


_UNQUANTIZED_TYPES = (
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
)

# GGUF block tensor stem -> HF Gemma3 module path (relative to a decoder layer).
_LINEAR_TENSORS = {
    "attn_q":      "self_attn.q_proj",
    "attn_k":      "self_attn.k_proj",
    "attn_v":      "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate":    "mlp.gate_proj",
    "ffn_up":      "mlp.up_proj",
    "ffn_down":    "mlp.down_proj",
}
_NORM_TENSORS = {
    "attn_norm":           "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm":            "pre_feedforward_layernorm",   # Gemma3-specific override of FFN_NORM
    "post_ffw_norm":       "post_feedforward_layernorm",
    "attn_q_norm":         "self_attn.q_norm",
    "attn_k_norm":         "self_attn.k_norm",
}


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


def _gguf_to_hf_key(name: str, language_prefix: str) -> str | None:
    """Map a GGUF tensor name to an HF state-dict key, or return None to skip."""
    if name == "token_embd.weight":
        return f"{language_prefix}embed_tokens.weight"
    if name == "output_norm.weight":
        return f"{language_prefix}norm.weight"
    if name in ("output.weight",):
        # Tied to embed_tokens; lm_head is replaced with Identity post-load.
        return None
    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) < 4:
            return None
        block, stem = parts[1], parts[2]
        suffix = ".".join(parts[3:])  # weight | bias
        if stem in _LINEAR_TENSORS:
            return f"{language_prefix}layers.{block}.{_LINEAR_TENSORS[stem]}.{suffix}"
        if stem in _NORM_TENSORS:
            return f"{language_prefix}layers.{block}.{_NORM_TENSORS[stem]}.{suffix}"
    return None


def _convert_state_dict(
    gguf_state: dict[str, torch.Tensor], language_prefix: str
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for k, v in gguf_state.items():
        new_key = _gguf_to_hf_key(k, language_prefix)
        if new_key is None:
            if k != "output.weight":
                unmapped.append(k)
            continue
        converted[new_key] = v
    if unmapped:
        head = unmapped[:10]
        more = max(0, len(unmapped) - 10)
        raise RuntimeError(
            f"Unmapped GGUF tensors: {head}"
            + (f" (and {more} more)" if more else "")
            + ". Extend modules.modelLoader.ltx2.Gemma3GGUFLoader._gguf_to_hf_key."
        )
    return converted


def _swap_quantized_linears(
    root: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
) -> None:
    """Replace every nn.Linear whose weight is a GGUFParameter with a GGUFLinear,
    assigning the packed weight directly so the bf16 storage from the original
    Linear becomes garbage-collectable."""
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
        )
        # Overwrite the freshly-allocated bf16 weight with the packed GGUFParameter.
        new_module.weight = state_dict[f"{full_name}.weight"]
        bkey = f"{full_name}.bias"
        if bkey in state_dict:
            new_module.bias = nn.Parameter(
                state_dict[bkey].to(compute_dtype), requires_grad=False
            )
        elif old.bias is None:
            new_module.bias = None
        new_module.source_cls = type(old)
        new_module.requires_grad_(False)

        setattr(parent, attr, new_module)


def load_gemma3_from_gguf(
    gguf_path: str,
    base_model_name: str,
    dtype: torch.dtype,
    is_causal_lm: bool,
) -> nn.Module:
    """Load a Gemma3 text encoder from a GGUF file, keeping quantized weights packed.

    Args:
        gguf_path: absolute path to the .gguf file.
        base_model_name: HF repo or local snapshot path containing `text_encoder/config.json`.
        dtype: compute dtype for the GGUFLinear forward pass and for the small
            unquantized tensors (embeddings, norms).
        is_causal_lm: True for text-only GGUFs (Gemma3ForCausalLM), False for
            multimodal exports loaded into Gemma3ForConditionalGeneration.
    """
    config = AutoConfig.from_pretrained(base_model_name, subfolder="text_encoder")

    if is_causal_lm:
        text_config = getattr(config, "text_config", config)
        # Build on CPU. The transient bf16 weights are released after _swap_quantized_linears.
        model = Gemma3ForCausalLM(text_config).to(dtype)
        language_prefix = "model."
    else:
        model = Gemma3ForConditionalGeneration(config).to(dtype)
        language_prefix = "language_model."

    model.eval()

    gguf_state = _read_gguf_state_dict(gguf_path)
    converted = _convert_state_dict(gguf_state, language_prefix)

    # 1) Replace quantized linears first (frees their original bf16 weights).
    _swap_quantized_linears(model, converted, compute_dtype=dtype)

    # 2) Copy remaining (unquantized) tensors into the model: embeddings, norms.
    remaining = {
        k: v for k, v in converted.items() if not isinstance(v, GGUFParameter)
    }
    missing, unexpected = model.load_state_dict(remaining, strict=False, assign=False)

    # `missing` will include all GGUF-quantized linear weights (because they're
    # not in `remaining`) plus tied lm_head and rotary buffers — those are fine.
    # We only care that none of the unquantized targets failed.
    real_unexpected = [k for k in unexpected if k in remaining]
    if real_unexpected:
        raise RuntimeError(f"Unexpected keys when loading Gemma3 GGUF: {real_unexpected[:10]}")

    # Free the parsed GGUF state dict and any orphaned bf16 weights.
    del gguf_state, converted, remaining
    gc.collect()

    return model
