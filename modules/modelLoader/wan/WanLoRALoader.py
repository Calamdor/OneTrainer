from modules.model.BaseModel import BaseModel
from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_wan2_2_lora import convert_wan2_2_lora_key_sets
from modules.util.ModelNames import ModelNames


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

        if model.lora_state_dict and any(
            "self_attn." in k or "cross_attn." in k
            for k in model.lora_state_dict
        ):
            print(
                "[WanLoRA] WARNING: the provided LoRA appears to be in ComfyUI format "
                "(keys contain 'self_attn.'/'cross_attn.' layer names).\n"
                "  Only the prefix was converted — layer names are NOT remapped, so "
                "weights will not be loaded into the training wrappers.\n"
                "  To resume Wan2.2 LoRA training, point 'lora' at the internal OT "
                "checkpoint directory (the folder containing meta.json), not a "
                "ComfyUI-export .safetensors file."
            )
