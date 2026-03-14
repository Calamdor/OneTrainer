from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet


def convert_wan2_2_lora_key_sets() -> list[LoraConversionKeySet]:
    """
    Maps OT internal prefix to external musubi/ComfyUI prefix for LOADING.

    Both Wan2.2 experts use "diffusion_model." prefix in ComfyUI.
    Layer name differences (attn1.to_q vs self_attn.q) are NOT handled
    here — those are managed in ot_to_musubi* for saving and in
    _musubi_path_to_diffusers for companion-LoRA hook loading.
    """
    return [
        LoraConversionKeySet("diffusion_model", "lora_transformer"),
    ]


def _musubi_suffix(key: str) -> str:
    """Rename lora_down/up suffixes to lora_A/B."""
    key = key.replace(".lora_down.weight", ".lora_A.weight")
    key = key.replace(".lora_up.weight", ".lora_B.weight")
    return key


def _diffusers_to_musubi_layers(key: str) -> str:
    """
    Map diffusers WanTransformerBlock layer paths to musubi-tuner/ComfyUI naming.

    diffusers (OT internal)     musubi / ComfyUI
    ─────────────────────────   ─────────────────
    .attn1.to_q.             →  .self_attn.q.
    .attn1.to_k.             →  .self_attn.k.
    .attn1.to_v.             →  .self_attn.v.
    .attn1.to_out.0.         →  .self_attn.o.
    .attn2.to_q.             →  .cross_attn.q.
    .attn2.to_k.             →  .cross_attn.k.
    .attn2.to_v.             →  .cross_attn.v.
    .attn2.to_out.0.         →  .cross_attn.o.
    .ffn.net.0.proj.         →  .ffn.0.
    .ffn.net.2.              →  .ffn.2.

    Patterns are surrounded by dots so they only match full path segments.
    Keys for non-block layers (condition_embedder etc.) pass through unchanged;
    ComfyUI will log them as "not loaded" but that is harmless.
    """
    key = key.replace(".attn1.to_q.", ".self_attn.q.")
    key = key.replace(".attn1.to_k.", ".self_attn.k.")
    key = key.replace(".attn1.to_v.", ".self_attn.v.")
    key = key.replace(".attn1.to_out.0.", ".self_attn.o.")
    key = key.replace(".attn2.to_q.", ".cross_attn.q.")
    key = key.replace(".attn2.to_k.", ".cross_attn.k.")
    key = key.replace(".attn2.to_v.", ".cross_attn.v.")
    key = key.replace(".attn2.to_out.0.", ".cross_attn.o.")
    key = key.replace(".ffn.net.0.proj.", ".ffn.0.")
    key = key.replace(".ffn.net.2.", ".ffn.2.")
    return key


def musubi_path_to_diffusers(path: str) -> str:
    """
    Reverse of _diffusers_to_musubi_layers — convert a musubi module path
    to diffusers attribute path for model traversal.

    Used by the companion LoRA hook loader so that external musubi-format
    LoRAs (self_attn.q etc.) can be applied to the diffusers model.
    Also a no-op for paths already in diffusers format.
    """
    path = path.replace("self_attn.q", "attn1.to_q")
    path = path.replace("self_attn.k", "attn1.to_k")
    path = path.replace("self_attn.v", "attn1.to_v")
    path = path.replace("self_attn.o", "attn1.to_out.0")
    path = path.replace("cross_attn.q", "attn2.to_q")
    path = path.replace("cross_attn.k", "attn2.to_k")
    path = path.replace("cross_attn.v", "attn2.to_v")
    path = path.replace("cross_attn.o", "attn2.to_out.0")
    path = path.replace("ffn.0", "ffn.net.0.proj")
    path = path.replace("ffn.2", "ffn.net.2")
    return path


def _convert_one(key: str, tensor) -> tuple[str, object] | None:
    """
    Convert a single OT-internal key (already with diffusion_model. prefix stripped)
    to musubi format.
    """
    key = "diffusion_model." + key
    key = _diffusers_to_musubi_layers(_musubi_suffix(key))
    return key, tensor


def ot_to_musubi(state_dict: dict) -> dict:
    """
    Rename OT-internal LoRA keys to musubi-tuner / ComfyUI format.

    Both experts use "diffusion_model." prefix in ComfyUI — the two
    transformers are loaded as separate model nodes and each node's
    LoRA is applied independently, so there is no collision.

    Only block-level attention/FFN layers are exported; diffusers-specific
    layers (condition_embedder, proj_out, etc.) have no ComfyUI equivalent
    and are silently dropped.

    Prefix : lora_transformer.   → diffusion_model.
             lora_transformer_2. → diffusion_model.
    Layers : attn1.to_q         → self_attn.q  (etc.)
    Suffix : .lora_down.weight   → .lora_A.weight
             .lora_up.weight     → .lora_B.weight
    """
    result = {}
    for key, tensor in state_dict.items():
        if key.startswith("lora_transformer_2."):
            key = key[len("lora_transformer_2."):]
        elif key.startswith("lora_transformer."):
            key = key[len("lora_transformer."):]
        else:
            continue
        out = _convert_one(key, tensor)
        if out is not None:
            result[out[0]] = out[1]
    return result


def ot_to_musubi_high_noise(state_dict: dict) -> dict:
    """Convert only high-noise expert keys (lora_transformer.*) to musubi format."""
    result = {}
    for key, tensor in state_dict.items():
        if key.startswith("lora_transformer."):
            out = _convert_one(key[len("lora_transformer."):], tensor)
            if out is not None:
                result[out[0]] = out[1]
    return result


def ot_to_musubi_low_noise(state_dict: dict) -> dict:
    """Convert only low-noise expert keys (lora_transformer_2.*) to musubi format.
    Uses diffusion_model. prefix — same as high-noise; ComfyUI applies each
    LoRA to its respective transformer node separately."""
    result = {}
    for key, tensor in state_dict.items():
        if key.startswith("lora_transformer_2."):
            out = _convert_one(key[len("lora_transformer_2."):], tensor)
            if out is not None:
                result[out[0]] = out[1]
    return result
