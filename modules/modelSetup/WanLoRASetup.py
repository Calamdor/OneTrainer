import math
import os

from modules.model.WanModel import WanModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert.lora.convert_wan2_2_lora import comfyui_path_to_diffusers
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.WanExpertMode import WanExpertMode
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F


class WanLoRASetup(BaseWanSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def __setup_requires_grad(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.transformer_2.requires_grad_(False)

        if model.transformer_lora is not None:
            self._setup_model_part_requires_grad(
                "transformer_lora", model.transformer_lora, config.transformer, model.train_progress,
            )
        if model.transformer_2_lora is not None:
            self._setup_model_part_requires_grad(
                "transformer_2_lora", model.transformer_2_lora, config.transformer, model.train_progress,
            )

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Default to "blocks" for Wan2.2 so only transformer block layers (attention/FFN)
        # are trained — these are the only layers ComfyUI can load via its LoRA mechanism.
        layer_filter = config.layer_filter.split(",") if config.layer_filter else ["blocks"]
        expert_mode = config.wan_expert_mode

        if expert_mode != WanExpertMode.LOW_NOISE:
            model.transformer_lora = LoRAModuleWrapper(
                model.transformer, "lora_transformer", config, layer_filter,
            )
        if expert_mode != WanExpertMode.HIGH_NOISE:
            model.transformer_2_lora = LoRAModuleWrapper(
                model.transformer_2, "lora_transformer_2", config, layer_filter,
            )

        if model.lora_state_dict:
            if model.transformer_lora is not None:
                if any(k.startswith("lora_transformer.") for k in model.lora_state_dict):
                    model.transformer_lora.load_state_dict(model.lora_state_dict, strict=False)
            if model.transformer_2_lora is not None:
                if any(k.startswith("lora_transformer_2.") for k in model.lora_state_dict):
                    model.transformer_2_lora.load_state_dict(model.lora_state_dict, strict=False)
            model.lora_state_dict = None

        if model.transformer_lora is not None:
            model.transformer_lora.set_dropout(config.dropout_probability)
            model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.transformer_lora.hook_to_module()

        if model.transformer_2_lora is not None:
            model.transformer_2_lora.set_dropout(config.dropout_probability)
            model.transformer_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.transformer_2_lora.hook_to_module()

        # Remove any stale companion patches from a previous setup_model call
        for handle in model.companion_lora_handles:
            m, orig_fwd = handle[0], handle[1]
            m.forward = orig_fwd
        model.companion_lora_handles = []
        model.companion_lora_expert = None  # 1 = high-noise, 2 = low-noise

        # Load companion LoRA (frozen) for the non-trained expert — improves validation sample quality
        companion_path = getattr(config, 'wan_companion_lora_path', '')
        if companion_path and os.path.isfile(companion_path) and expert_mode != WanExpertMode.BOTH:
            if expert_mode == WanExpertMode.HIGH_NOISE:
                target = model.transformer_2
                model.companion_lora_expert = 2
            else:  # LOW_NOISE
                target = model.transformer
                model.companion_lora_expert = 1
            model.companion_lora_handles = _apply_companion_lora_hooks(target, companion_path)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Text encoder and VAE stay on temp during training (latent caching used).
        model.text_encoder_to(self.temp_device)
        model.vae_to(self.temp_device)

        expert_mode = config.wan_expert_mode
        if expert_mode == WanExpertMode.HIGH_NOISE:
            # Only the high-noise expert (transformer) is trained; keep transformer_2 on temp
            model.transformer_1_to(self.train_device)
            model.transformer_2_to(self.temp_device)
        elif expert_mode == WanExpertMode.LOW_NOISE:
            # Only the low-noise expert (transformer_2) is trained; keep transformer on temp
            model.transformer_1_to(self.temp_device)
            model.transformer_2_to(self.train_device)
        else:
            model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            if expert_mode != WanExpertMode.LOW_NOISE:
                model.transformer.train()
            if expert_mode != WanExpertMode.HIGH_NOISE:
                model.transformer_2.train()
        else:
            model.transformer.eval()
            model.transformer_2.eval()

    def after_optimizer_step(
            self,
            model: WanModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)


def _apply_companion_lora_hooks(transformer, lora_path: str) -> list:
    """
    Load a pre-trained Wan2.2 LoRA (musubi/ComfyUI or OT format) and apply it as
    frozen forward hooks on the given transformer.  Hooks dynamically cast their
    tensors to the input's device/dtype so no explicit device tracking is needed.
    Returns a list of hook handles that can be removed later.
    """
    from safetensors.torch import load_file

    sd = load_file(lora_path)

    # Group keys: {module_dotpath: {"down": Tensor, "up": Tensor, "alpha": float}}
    modules: dict[str, dict] = {}
    for key, tensor in sd.items():
        # Strip any known root prefix (keep diffusion_model_2. for backwards compat)
        for pfx in ("diffusion_model_2.", "diffusion_model.", "lora_transformer_2.", "lora_transformer."):
            if key.startswith(pfx):
                key = key[len(pfx):]
                break
        if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight"):
            sfx = ".lora_A.weight" if key.endswith(".lora_A.weight") else ".lora_down.weight"
            modules.setdefault(key[:-len(sfx)], {})["down"] = tensor.detach()
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight"):
            sfx = ".lora_B.weight" if key.endswith(".lora_B.weight") else ".lora_up.weight"
            modules.setdefault(key[:-len(sfx)], {})["up"] = tensor.detach()
        elif key.endswith(".alpha"):
            modules.setdefault(key[:-len(".alpha")], {})["alpha"] = tensor.item()
        elif key.endswith(".oft_R.weight"):
            modules.setdefault(key[:-len(".oft_R.weight")], {})["oft_R"] = tensor.detach()

    handles = []
    applied = 0
    for mod_path, weights in modules.items():
        is_lora = "down" in weights and "up" in weights
        is_oft  = "oft_R" in weights
        if not is_lora and not is_oft:
            continue

        # Convert ComfyUI layer names (self_attn.q etc.) to diffusers names (attn1.to_q)
        # so both OT-format and ComfyUI-format companion LoRAs can be applied.
        diffusers_path = comfyui_path_to_diffusers(mod_path)

        # OffloadCheckpointLayer-aware traversal
        try:
            m = transformer
            for part in diffusers_path.split("."):
                m = getattr(m, part)
                if hasattr(m, "checkpoint") and isinstance(m.checkpoint, torch.nn.Module):
                    m = m.checkpoint
        except AttributeError:
            continue

        # Patch m.forward directly (same mechanism as LoRAModuleWrapper.hook_to_module)
        # rather than register_forward_hook, because torch.compile(fullgraph=True)
        # on the parent block inlines module calls and bypasses register_forward_hook.
        if is_lora:
            down = weights["down"]
            up = weights["up"]
            rank = down.shape[0]
            scale = weights.get("alpha", float(rank)) / rank

            def _make_lora_patched_forward(orig_fwd, d, u, s):
                def patched_forward(x):
                    return orig_fwd(x) + F.linear(F.linear(x, d.to(x.device, x.dtype)),
                                                  u.to(x.device, x.dtype)) * s
                return patched_forward

            orig_forward = m.forward
            m.forward = _make_lora_patched_forward(orig_forward, down, up, scale)
        else:  # OFT
            from modules.module.oft_utils import OFTRotationModule
            oft_weight = weights["oft_R"]
            r, n_elements = oft_weight.shape
            # Invert: block_size*(block_size-1)/2 = n_elements
            block_size = int((1 + math.sqrt(1 + 8 * n_elements)) / 2)
            in_features = r * block_size

            rot_module = OFTRotationModule(
                r=r,
                n_elements=n_elements,
                block_size=block_size,
                in_features=in_features,
                use_cayley_neumann=True,
                num_cayley_neumann_terms=5,
            )
            rot_module.weight = torch.nn.Parameter(oft_weight)
            rot_module.eval()

            def _make_oft_patched_forward(orig_fwd, rot_mod):
                def patched_forward(x):
                    # rot_mod is moved to the correct device by transformer_1_to/transformer_2_to
                    # before sampling — do NOT call rot_mod.to() here; module-level .to() returns
                    # a non-Tensor and is untraceable by torch.compile(fullgraph=True).
                    rotated_x = rot_mod(x)
                    return orig_fwd(rotated_x)
                return patched_forward

            orig_forward = m.forward
            m.forward = _make_oft_patched_forward(orig_forward, rot_module)

        handles.append((m, orig_forward, rot_module if is_oft else None))
        applied += 1

    print(f"[WanLoRA] Companion LoRA: {applied} patches applied from {os.path.basename(lora_path)}")
    return handles


factory.register(BaseModelSetup, WanLoRASetup, ModelType.WAN2_2_T2V, TrainingMethod.LORA)
