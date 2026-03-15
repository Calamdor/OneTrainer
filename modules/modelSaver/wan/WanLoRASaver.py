import os
from pathlib import Path

from modules.model.WanModel import WanModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_wan2_2_lora import (
    ot_to_comfyui_high_noise,
    ot_to_comfyui_low_noise,
)
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class WanLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: WanModel) -> list[LoraConversionKeySet] | None:
        # Key conversion is handled in save() directly; bypass standard machinery.
        return None

    def _get_state_dict(
            self,
            model: WanModel,
    ) -> dict[str, Tensor]:
        # Not used directly — save() drives per-expert state dicts.
        # Kept for compatibility with internal save path.
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.transformer_2_lora is not None:
            state_dict |= model.transformer_2_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    def _build_expert_destination(self, base_destination: str, suffix: str) -> str:
        """Insert _high_noise or _low_noise before the file extension."""
        p = Path(base_destination)
        return str(p.with_stem(p.stem + suffix))

    def _save_expert_safetensors(
            self,
            model: WanModel,
            state_dict: dict[str, Tensor],
            destination: str,
            dtype: torch.dtype | None,
    ):
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def save(
            self,
            model: WanModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        has_high = model.transformer_lora is not None
        has_low = model.transformer_2_lora is not None

        # Build the full raw state dict (both experts combined)
        raw = {}
        if has_high:
            raw |= model.transformer_lora.state_dict()
        if has_low:
            raw |= model.transformer_2_lora.state_dict()
        if model.lora_state_dict is not None:
            raw |= model.lora_state_dict

        if output_model_format in (ModelFormat.SAFETENSORS, ModelFormat.LEGACY_SAFETENSORS):
            if has_high:
                dest = self._build_expert_destination(output_model_destination, "_high_noise")
                self._save_expert_safetensors(model, ot_to_comfyui_high_noise(raw), dest, dtype)
                print(f"[WanLoRASaver] Saved high-noise expert LoRA → {dest}")
            if has_low:
                dest = self._build_expert_destination(output_model_destination, "_low_noise")
                self._save_expert_safetensors(model, ot_to_comfyui_low_noise(raw), dest, dtype)
                print(f"[WanLoRASaver] Saved low-noise expert LoRA → {dest}")
        else:
            # INTERNAL format — fall back to base class behaviour
            self._save(model, output_model_format, output_model_destination, dtype)
