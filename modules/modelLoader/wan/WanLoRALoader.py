from modules.model.BaseModel import BaseModel
from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_wan2_2_lora import comfyui_path_to_diffusers, convert_wan2_2_lora_key_sets
from modules.util.ModelNames import ModelNames


def _remap_comfyui(state_dict: dict) -> dict:
    """Convert ComfyUI-format layer names (self_attn.q etc.) to OT-internal format.

    Single-pass: detect and remap in one walk. If no ComfyUI marker is seen,
    the original dict is returned unchanged.
    """
    if not state_dict:
        return state_dict
    remapped = {}
    seen_comfyui = False
    for k, v in state_dict.items():
        if not seen_comfyui and ("self_attn." in k or "cross_attn." in k):
            seen_comfyui = True
        remapped[comfyui_path_to_diffusers(k)] = v
    if not seen_comfyui:
        return state_dict
    print("[WanLoRA] ComfyUI-format LoRA detected — remapping layer names to OT format.")
    return remapped


class WanLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_wan2_2_lora_key_sets()

    def load(
            self,
            model: WanModel,
            model_names: ModelNames,
    ):
        self._load(model, model_names)
        # Remap ComfyUI layer names for the first (high-noise) file immediately,
        # before touching lora_model_name_2, so the two dicts never collide.
        sd1 = _remap_comfyui(model.lora_state_dict or {})

        if model_names.lora_2:
            model.lora_state_dict = None
            self._load(model, ModelNames(lora=model_names.lora_2))
            sd2_raw = _remap_comfyui(model.lora_state_dict or {})

            # lora_model_name_2 always targets the low-noise expert.
            # ComfyUI files for both experts share the diffusion_model. prefix, so
            # _load() converts them all to lora_transformer.* — re-prefix to
            # lora_transformer_2.* so WanLoRASetup routes them to the right wrapper.
            sd2 = {}
            for k, v in sd2_raw.items():
                if k.startswith("lora_transformer.") and not k.startswith("lora_transformer_2."):
                    k = "lora_transformer_2." + k[len("lora_transformer."):]
                sd2[k] = v

            model.lora_state_dict = {**sd1, **sd2}
        else:
            model.lora_state_dict = sd1 or None
