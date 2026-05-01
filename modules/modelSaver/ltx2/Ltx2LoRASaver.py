import os
from pathlib import Path

from modules.model.Ltx2Model import Ltx2Model
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_ltx2_lora import convert_ltx2_lora_diffusers_to_original
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class Ltx2LoRASaver(LoRASaverMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: Ltx2Model) -> list[LoraConversionKeySet] | None:
        # Key conversion is handled in save() directly for SAFETENSORS outputs.
        # INTERNAL format uses OT keys as-is for checkpoint resume.
        return None

    def _get_state_dict(self, model: Ltx2Model) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    @staticmethod
    def _to_comfyui_format(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert OT diffusers-format LoRA keys to Lightricks/ComfyUI format.

        OT internal:  lora_transformer.transformer_blocks.0.attn1.to_q.lora_down.weight
        ComfyUI:      diffusion_model.transformer_blocks.0.attn.q.lora_A.weight
        """
        # Step 1: swap prefix lora_transformer. → diffusion_model.
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("lora_transformer."):
                remapped["diffusion_model." + key[len("lora_transformer."):]] = value
            else:
                remapped[key] = value

        # Step 2: apply per-module renames (proj_in → patchify_proj, time_embed → adaln_single, …)
        remapped = convert_ltx2_lora_diffusers_to_original(remapped)

        # Step 3: lora_down/lora_up → lora_A/lora_B  (PEFT convention ComfyUI expects)
        out = {}
        for key, value in remapped.items():
            key = key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
            out[key] = value
        return out

    def save(
            self,
            model: Ltx2Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        if output_model_format in (ModelFormat.SAFETENSORS, ModelFormat.LEGACY_SAFETENSORS):
            raw = self._get_state_dict(model)
            save_state_dict = self._convert_state_dict_dtype(raw, dtype)
            save_state_dict = self._to_comfyui_format(save_state_dict)
            os.makedirs(Path(output_model_destination).parent.absolute(), exist_ok=True)
            save_file(
                save_state_dict,
                output_model_destination,
                self._create_safetensors_header(model, save_state_dict),
            )
        else:
            # INTERNAL format — keep OT keys for checkpoint resume
            self._save(model, output_model_format, output_model_destination, dtype)
