from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert_util import convert as convert_util


def convert_wan2_2_lora_key_sets() -> list[LoraConversionKeySet]:
    """
    Maps OT internal prefix to external musubi/ComfyUI prefix for LOADING.

    Both Wan2.2 experts use "diffusion_model." prefix in ComfyUI.
    Layer name differences (attn1.to_q vs self_attn.q) are NOT handled
    here — those are managed in ot_to_musubi* for saving and in
    musubi_path_to_diffusers for companion-LoRA hook loading.
    """
    return [
        LoraConversionKeySet("diffusion_model", "lora_transformer"),
    ]


# Layer name mapping: diffusers/OT internal → musubi-tuner / ComfyUI.
# A catch-all at the end passes through any other block sub-layers (e.g. norm
# layers) unchanged so they reach the output under the correct diffusion_model prefix.
_BLOCK_LAYER_PATTERNS = [
    ("attn1.to_q",     "self_attn.q"),
    ("attn1.to_k",     "self_attn.k"),
    ("attn1.to_v",     "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"),
    ("attn2.to_q",     "cross_attn.q"),
    ("attn2.to_k",     "cross_attn.k"),
    ("attn2.to_v",     "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"),
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2",      "ffn.2"),
    ("{rest}",         "{rest}"),
]

_WAN_HIGH_NOISE_PATTERNS = [
    ("lora_transformer.blocks.{i}", "diffusion_model.blocks.{i}", _BLOCK_LAYER_PATTERNS),
    ("lora_transformer.{rest}",     "diffusion_model.{rest}"),
]

_WAN_LOW_NOISE_PATTERNS = [
    ("lora_transformer_2.blocks.{i}", "diffusion_model.blocks.{i}", _BLOCK_LAYER_PATTERNS),
    ("lora_transformer_2.{rest}",     "diffusion_model.{rest}"),
]


def _rename_lora_suffixes(state_dict: dict) -> dict:
    """Rename lora_down/up suffixes to lora_A/B (musubi/ComfyUI convention)."""
    result = {}
    for k, v in state_dict.items():
        k = k.replace(".lora_down.weight", ".lora_A.weight")
        k = k.replace(".lora_up.weight", ".lora_B.weight")
        result[k] = v
    return result


def musubi_path_to_diffusers(path: str) -> str:
    """
    Reverse of _BLOCK_LAYER_PATTERNS — convert a musubi module path
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


def ot_to_musubi(state_dict: dict) -> dict:
    """
    Rename OT-internal LoRA keys to musubi-tuner / ComfyUI format.

    Both experts use "diffusion_model." prefix in ComfyUI — the two
    transformers are loaded as separate model nodes and each node's
    LoRA is applied independently, so there is no collision.

    Only lora_transformer.* and lora_transformer_2.* keys are processed;
    all other keys (e.g. text encoder) are dropped.

    Prefix : lora_transformer.   → diffusion_model.
             lora_transformer_2. → diffusion_model.
    Layers : attn1.to_q         → self_attn.q  (etc.)
    Suffix : .lora_down.weight   → .lora_A.weight
             .lora_up.weight     → .lora_B.weight
    """
    high = {k: v for k, v in state_dict.items()
            if k.startswith("lora_transformer.") and not k.startswith("lora_transformer_2.")}
    low = {k: v for k, v in state_dict.items()
           if k.startswith("lora_transformer_2.")}
    result = {}
    if high:
        result |= convert_util(high, _WAN_HIGH_NOISE_PATTERNS, strict=True)
    if low:
        result |= convert_util(low, _WAN_LOW_NOISE_PATTERNS, strict=True)
    return _rename_lora_suffixes(result)


def ot_to_musubi_high_noise(state_dict: dict) -> dict:
    """Convert only high-noise expert keys (lora_transformer.*) to musubi format."""
    filtered = {k: v for k, v in state_dict.items()
                if k.startswith("lora_transformer.") and not k.startswith("lora_transformer_2.")}
    return _rename_lora_suffixes(convert_util(filtered, _WAN_HIGH_NOISE_PATTERNS, strict=True))


def ot_to_musubi_low_noise(state_dict: dict) -> dict:
    """Convert only low-noise expert keys (lora_transformer_2.*) to musubi format.
    Uses diffusion_model. prefix — same as high-noise; ComfyUI applies each
    LoRA to its respective transformer node separately."""
    filtered = {k: v for k, v in state_dict.items()
                if k.startswith("lora_transformer_2.")}
    return _rename_lora_suffixes(convert_util(filtered, _WAN_LOW_NOISE_PATTERNS, strict=True))
