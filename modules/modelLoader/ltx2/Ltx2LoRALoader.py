from modules.model.BaseModel import BaseModel
from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class Ltx2LoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        # Map external "diffusion_model." prefix to OT's internal "lora_transformer." prefix.
        # Note: this handles only prefix remapping. Lightricks-format LoRAs also require module
        # path renaming (patchify_proj → proj_in, adaln_single → time_embed, etc.) which is
        # handled by convert_ltx2_lora_original_to_diffusers. For now, external format loading
        # is best-effort — OT-trained LoRAs (already in diffusers format) round-trip cleanly.
        return [LoraConversionKeySet("diffusion_model", "lora_transformer")]

    def load(
            self,
            model: Ltx2Model,
            model_names: ModelNames,
    ):
        self._load(model, model_names)

        if model.lora_state_dict and any(
            "patchify_proj" in k or "adaln_single" in k
            for k in model.lora_state_dict
        ):
            print(
                "[Ltx2LoRA] WARNING: the provided LoRA appears to be in Lightricks original format "
                "(keys contain 'patchify_proj'/'adaln_single' module names).\n"
                "  Only the prefix was converted — module paths are NOT remapped, so many weights\n"
                "  will not load into the training wrappers.\n"
                "  To resume LTX-2.3 LoRA training, point 'lora' at the internal OT checkpoint "
                "directory (the folder containing meta.json)."
            )
