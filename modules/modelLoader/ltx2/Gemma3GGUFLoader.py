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


def _is_llama_cpp_norm(name: str) -> bool:
    """True for a llama.cpp-flavor Gemma3 RMSNorm tensor.

    llama.cpp's Gemma3 GGUF conversion FOLDS +1.0 into every RMSNorm weight
    (Gemma3 computes ``x * (1 + w)``; llama.cpp stores ``1 + w`` so its plain
    RMSNorm reproduces the result). HF's Gemma3 applies its own ``(1 + w)``,
    so these weights must have 1.0 subtracted on load or every norm is doubled
    and the forward produces garbage (prompt-ignored output). Confirmed
    element-wise: llama.cpp norm − HF(qat) norm = +1.0 exactly across all norm
    types. Mirrors city96's ComfyUI-GGUF ``gemma3_norm_corrections`` (9ecc3c4).
    Only applies to the llama.cpp ``blk.*`` / ``output_norm`` naming — HF-flavor
    GGUFs already carry the un-folded values.
    """
    if name == "output_norm.weight":
        return True
    if name.startswith("blk."):
        parts = name.split(".")
        return len(parts) >= 4 and parts[2] in _NORM_TENSORS
    return False


def _detect_gguf_flavor(gguf_state: dict[str, torch.Tensor]) -> str:
    """Return 'llama_cpp' or 'hf' based on tensor naming.

    Two incompatible Gemma3 GGUF naming conventions exist in the wild:
      - llama.cpp toolchain: 'token_embd.weight', 'blk.N.attn_q.weight', ...
      - HF/diffusers exports: 'language_model.model.embed_tokens.weight',
        'language_model.model.layers.N.self_attn.q_proj.weight', ...
    Sniff the first few keys to decide.
    """
    sample = list(gguf_state.keys())[:20]
    if any(k == "token_embd.weight" or k.startswith("blk.") for k in sample):
        return "llama_cpp"
    if any(k.startswith("language_model.") or k.startswith("vision_tower.")
           or k.startswith("multi_modal_projector.") for k in sample):
        return "hf"
    raise RuntimeError(
        f"Unrecognized GGUF tensor naming. Sample keys: {sample[:5]}"
    )


def _detect_hf_target_prefixes(model: nn.Module) -> dict[str, str]:
    """Probe the constructed Gemma3 model for the prefix under which each
    component lives in its state_dict.

    Returns a dict with keys ``language``, ``vision``, ``multi_modal`` —
    each mapped to the prefix that should be prepended to the GGUF-side
    relative key, OR ``None`` for components that don't exist in this
    model class (e.g. Gemma3ForCausalLM has no vision_tower or
    multi_modal_projector).

    The ``language`` prefix is required; the other two are optional and
    only populated when the model has those sub-modules.  Different
    transformers versions wrap Gemma3 differently:
      - Gemma3ForCausalLM (text-only):       ``model.X``
      - pre-4.57 ForConditionalGeneration:   ``language_model.X``
      - 4.57+    ForConditionalGeneration:   ``model.language_model.X``
    Probing the actual state_dict avoids hardcoding a version table.
    """
    out: dict[str, str | None] = {"language": None, "vision": None, "multi_modal": None}
    keys = list(model.state_dict().keys())

    # Language: find any "embed_tokens.weight" leaf — its prefix is the
    # language sub-tree path.  In a multimodal model that's
    # "model.language_model.embed_tokens.weight"; in causal LM it's just
    # "model.embed_tokens.weight".  Either way, strip the leaf to get the prefix.
    for k in keys:
        if k.endswith(".embed_tokens.weight") or k == "embed_tokens.weight":
            out["language"] = k[:-len("embed_tokens.weight")]
            break

    # Vision tower (optional — only present in Gemma3ForConditionalGeneration).
    for k in keys:
        idx = k.find(".vision_tower.")
        if idx >= 0 and ".embeddings.patch_embedding.weight" in k:
            out["vision"] = k[:idx + 1]
            break
        if k.startswith("vision_tower.") and ".embeddings.patch_embedding.weight" in k:
            out["vision"] = ""
            break

    # Multi-modal projector (optional).
    for k in keys:
        idx = k.find(".multi_modal_projector.")
        if idx >= 0:
            out["multi_modal"] = k[:idx + 1]
            break
        if k.startswith("multi_modal_projector."):
            out["multi_modal"] = ""
            break

    if out["language"] is None:
        raise RuntimeError(
            "Could not locate target prefix for 'language' (no key ending "
            "in '.embed_tokens.weight' found in model state_dict). "
            "transformers may have changed structure; sample state_dict "
            f"keys: {keys[:5]}"
        )
    return out


def _hf_to_hf_key(name: str, prefixes: dict[str, str | None]) -> str | None:
    """Translate an HF-flavor GGUF key to the constructed model's state_dict key.

    HF-flavor GGUFs store keys like 'language_model.model.layers.0.self_attn.q_proj.weight'
    (i.e. the HF *transformers* state-dict form, before any outer wrapping).
    The target model wraps this differently depending on transformers version
    AND model class:
      - Gemma3ForCausalLM (text-only):       ``model.X``
      - pre-4.57 ForConditionalGeneration:   ``language_model.X``
      - 4.57+    ForConditionalGeneration:   ``model.language_model.X``

    Returns ``None`` for keys that don't apply to this model class — e.g. a
    multimodal-format GGUF's ``vision_tower.*`` and ``multi_modal_projector.*``
    keys are dropped when loading into a text-only Gemma3ForCausalLM.
    """
    # output.weight is tied to embed_tokens in Gemma3; lm_head is replaced
    # with nn.Identity by Ltx2ModelLoader, so we skip it entirely.
    if name in ("output.weight", "lm_head.weight"):
        return None

    if name.startswith("language_model.model."):
        rest = name[len("language_model.model."):]
        return prefixes["language"] + rest
    if name.startswith("language_model."):
        # rare: GGUFs that already strip the inner '.model.'
        rest = name[len("language_model."):]
        return prefixes["language"] + rest
    if name.startswith("vision_tower."):
        # Skip silently when target model has no vision tower (causal LM).
        if prefixes.get("vision") is None:
            return None
        return prefixes["vision"] + name
    if name.startswith("multi_modal_projector."):
        if prefixes.get("multi_modal") is None:
            return None
        return prefixes["multi_modal"] + name
    return None


def _convert_state_dict(
    gguf_state: dict[str, torch.Tensor],
    language_prefix: str,
    model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Translate GGUF tensor names to the constructed model's state_dict keys.

    Auto-detects llama.cpp vs HF flavor.  For HF flavor, ``model`` must be
    provided so target prefixes can be probed from its state_dict.
    """
    flavor = _detect_gguf_flavor(gguf_state)
    if flavor == "hf":
        if model is None:
            raise RuntimeError(
                "HF-flavor Gemma3 GGUF detected but no model passed to "
                "_convert_state_dict for prefix probing."
            )
        prefixes = _detect_hf_target_prefixes(model)
        key_fn = lambda k: _hf_to_hf_key(k, prefixes)  # noqa: E731
    else:
        key_fn = lambda k: _gguf_to_hf_key(k, language_prefix)  # noqa: E731

    # When loading a multimodal-format GGUF into a text-only Gemma3ForCausalLM,
    # the vision_tower.* and multi_modal_projector.* keys legitimately have
    # nowhere to land — silence those so they don't trigger the unmapped
    # error.  Detected by the corresponding prefix being None.
    silently_drop_vision = (flavor == "hf"
                            and prefixes.get("vision") is None)
    silently_drop_mmp    = (flavor == "hf"
                            and prefixes.get("multi_modal") is None)

    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    n_dropped_vision = 0
    n_dropped_mmp = 0
    skipped_silently = {"output.weight", "lm_head.weight"}
    for k, v in gguf_state.items():
        new_key = key_fn(k)
        if new_key is None:
            if k in skipped_silently:
                continue
            if silently_drop_vision and k.startswith("vision_tower."):
                n_dropped_vision += 1
                continue
            if silently_drop_mmp and k.startswith("multi_modal_projector."):
                n_dropped_mmp += 1
                continue
            unmapped.append(k)
            continue
        if flavor == "llama_cpp" and _is_llama_cpp_norm(k):
            # Reverse llama.cpp's +1.0 Gemma3 RMSNorm fold (see _is_llama_cpp_norm).
            # Norms are F32/unquantized in these files, so a plain subtract works;
            # guard against the unexpected quantized-norm case rather than corrupt.
            if isinstance(v, GGUFParameter):
                raise RuntimeError(
                    f"Gemma3 norm '{k}' is quantized ({v.quant_type}); the +1.0 "
                    f"fold correction needs an unquantized (F32) norm tensor."
                )
            v = v.float() - 1.0
        converted[new_key] = v

    if n_dropped_vision or n_dropped_mmp:
        print(
            f"[Gemma3GGUF] dropped {n_dropped_vision} vision_tower + "
            f"{n_dropped_mmp} multi_modal_projector tensors "
            f"(target model is text-only Gemma3ForCausalLM)",
            flush=True,
        )

    if unmapped:
        head = unmapped[:10]
        more = max(0, len(unmapped) - 10)
        raise RuntimeError(
            f"Unmapped GGUF tensors (flavor={flavor}): {head}"
            + (f" (and {more} more)" if more else "")
            + ". Extend modules.modelLoader.ltx2.Gemma3GGUFLoader."
        )
    return converted


def _swap_quantized_linears(
    root: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
) -> None:
    """Replace every nn.Linear whose weight is a GGUFParameter with a GGUFLinear.

    Building the new GGUFLinear on ``device="meta"`` means its constructor
    allocates a zero-storage placeholder weight that we immediately overwrite
    with the packed GGUFParameter — saves the transient bf16 alloc that the
    default constructor would otherwise create. The packed weight ends up
    materialized on CPU (whatever device the source data was read from); a
    later ``model.to(cuda)`` will move the packed bytes.
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
    import os as _os

    print(
        f"[Gemma3GGUF] custom packed-weight loader engaged "
        f"(causal_lm={is_causal_lm}, file={_os.path.basename(gguf_path)}, "
        f"size={_os.path.getsize(gguf_path) / 1e9:.2f} GB)",
        flush=True,
    )
    config = AutoConfig.from_pretrained(base_model_name, subfolder="text_encoder")

    # Use ``init_empty_weights(include_buffers=False)``: parameters (the
    # 12B weight tensors that account for ~24 GB of bf16) land on meta,
    # but BUFFERS — including Gemma3's computed ``embed_scale``,
    # ``inv_freq`` and ``original_inv_freq`` — initialize on real CPU
    # memory and keep their config-derived values intact. Without this
    # we used to materialize the full bf16 shell, then GGUF-swap most
    # weights, then drop the bf16 — peak ~24 GB transient. With
    # include_buffers=False the bf16 shell never exists; only the
    # ~600 MB of unquantized embeddings/norms ever materialize on CPU.
    #
    # The earlier attempt that used plain ``init_empty_weights()`` and
    # broke RoPE was correct in spirit but wrong in scope — buffers
    # need real allocation, parameters don't.
    from accelerate import init_empty_weights as _init_empty_weights
    with _init_empty_weights(include_buffers=False):
        if is_causal_lm:
            text_config = getattr(config, "text_config", config)
            model = Gemma3ForCausalLM(text_config)
            language_prefix = "model."
        else:
            model = Gemma3ForConditionalGeneration(config)
            language_prefix = "language_model."

    # Nuke ``lm_head`` immediately. The Ltx2ModelLoader does this after we
    # return, but by then HF's ``tie_weights()`` has already broken
    # (because ``assign=True`` replaces embed_tokens.weight, severing the
    # tied reference) and a fresh ~4 GB bf16 ``lm_head.weight`` materializes
    # on CPU during the load. Replacing the module with Identity here
    # avoids that allocation entirely — LTX-2 never reads ``lm_head``
    # output anyway (it only consumes ``outputs.hidden_states``).
    if hasattr(model, "lm_head"):
        model.lm_head = nn.Identity()

    model.eval()

    gguf_state = _read_gguf_state_dict(gguf_path)
    converted = _convert_state_dict(gguf_state, language_prefix, model=model)

    # Dequantize the token embedding if it was stored quantized (e.g. Q8_0 in
    # llama.cpp-format GGUFs). It feeds an nn.Embedding LOOKUP, not a matmul, so
    # it cannot live as a GGUFLinear/GGUFParameter. And it falls through both
    # load paths otherwise: _swap_quantized_linears skips it (not an nn.Linear)
    # AND it's excluded from `remaining` (GGUFParameter filtered out) — so a
    # quantized embedding silently DROPS, leaving embed_tokens at random init
    # → garbage / prompt-ignored output. (HF-export GGUFs like the QAT-Q4 file
    # keep the embedding unquantized, which is why they happened to work.)
    # Matches city96 ComfyUI-GGUF's "dequantize token embedding" step.
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    for _ek in list(converted.keys()):
        if _ek.endswith("embed_tokens.weight") and isinstance(converted[_ek], GGUFParameter):
            converted[_ek] = dequantize_gguf_tensor(converted[_ek]).to(dtype)
            print(f"[Gemma3GGUF] dequantized quantized token embedding '{_ek}' -> {dtype} "
                  f"(would otherwise be dropped → garbage)", flush=True)

    # 1) Replace quantized linears first. Each new GGUFLinear is built on the
    # meta device (no bf16 alloc) and its weight set to the packed
    # GGUFParameter directly — the original bf16 weight from step above is
    # dropped from the parent module, ref count → 0, GC reclaims.
    _swap_quantized_linears(model, converted, compute_dtype=dtype)

    # 2) Install remaining (unquantized) tensors: embeddings, norms.
    # With ``init_empty_weights(include_buffers=False)`` the destination
    # parameters (embed_tokens.weight, norm.weight, etc.) are meta —
    # ``assign=False`` would call .copy_() and fail on meta. ``assign=True``
    # replaces the meta param with the cast tensor outright, which is what
    # we want.
    remaining = {
        k: v.to(dtype) for k, v in converted.items() if not isinstance(v, GGUFParameter)
    }
    missing, unexpected = model.load_state_dict(remaining, strict=False, assign=True)

    # `missing` will include all GGUF-quantized linear weights (because they're
    # not in `remaining`) plus tied lm_head and rotary buffers — those are fine.
    # We only care that none of the unquantized targets failed.
    real_unexpected = [k for k in unexpected if k in remaining]
    if real_unexpected:
        raise RuntimeError(f"Unexpected keys when loading Gemma3 GGUF: {real_unexpected[:10]}")

    # Free the parsed GGUF state dict and any orphaned bf16 weights.
    del gguf_state, converted, remaining
    gc.collect()

    # Verify weights stayed packed. A correctly-loaded Gemma3 GGUF should leave
    # all decoder linears as GGUFLinear with GGUFParameter weights — total bytes
    # should be close to the .gguf file size (within tens of MB for unquantized
    # embeddings/norms, NOT 2x).
    n_packed, n_dequant, packed_bytes, dequant_bytes = 0, 0, 0, 0
    from diffusers.quantizers.gguf.utils import GGUFLinear as _GL  # noqa: F401
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
        f"[Gemma3GGUF] linears packed={n_packed} ({packed_bytes / 1e9:.2f} GB), "
        f"unpacked={n_dequant} ({dequant_bytes / 1e9:.2f} GB)",
        flush=True,
    )

    return model
