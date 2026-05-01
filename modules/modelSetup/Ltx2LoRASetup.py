from modules.model.Ltx2Model import Ltx2Model
from modules.modelSetup.BaseLtx2Setup import BaseLtx2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import init_model_parameters
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress

import torch


class Ltx2LoRASetup(BaseLtx2Setup):
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
            model: Ltx2Model,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        if model.vae is not None:
            model.vae.requires_grad_(False)
        if model.audio_vae is not None:
            model.audio_vae.requires_grad_(False)
        if model.connectors is not None:
            model.connectors.requires_grad_(False)
        if model.vocoder is not None:
            model.vocoder.requires_grad_(False)
        model.transformer.requires_grad_(False)

        if model.transformer_lora is not None:
            self._setup_model_part_requires_grad(
                "transformer_lora", model.transformer_lora, config.transformer, model.train_progress,
            )

    def create_parameters(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        collection = NamedParameterGroupCollection()
        if model.transformer_lora is not None:
            self._create_model_part_parameters(
                collection, "transformer_lora", model.transformer_lora, config.transformer,
            )
        return collection

    def setup_model(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # Apply distilled LoRA patches first (BaseLtx2Setup.setup_model loads
        # them from disk and wires up the forward patches).  We immediately
        # pause the patches so they don't interfere with training gradients,
        # but keep the ~7.6 GB weight tensors alive on CPU so the sampler can
        # re-apply them cheaply without a disk reload each sample window.
        super().setup_model(model, config)
        model._pause_distilled_lora_hooks()

        # Default to "blocks" so only transformer_block layers are targeted —
        # matches ComfyUI's LoRA key scope. Comma-separated overrides allowed.
        layer_filter = config.layer_filter.split(",") if config.layer_filter else ["transformer_blocks"]

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "lora_transformer", config, layer_filter,
        )

        if model.lora_state_dict:
            lora_sd = {k: v for k, v in model.lora_state_dict.items()
                       if k.startswith("lora_transformer.")}
            if lora_sd:
                model.transformer_lora.load_state_dict(lora_sd, strict=False)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # Connector outputs are cached during the text-caching pass and consumed
        # from the batch in predict(). Connectors stay on temp_device (CPU) during
        # training — they're never called per-step, saving ~1.4 GB VRAM.
        model.text_encoder_to(self.temp_device)
        model.vae_to(self.temp_device)
        model.connectors_to(self.temp_device)
        model.vocoder_to(self.temp_device)
        model.latent_upsampler_to(self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder is not None:
            model.text_encoder.eval()
        if model.vae is not None:
            model.vae.eval()
        if model.audio_vae is not None:
            model.audio_vae.eval()
        if model.connectors is not None:
            model.connectors.eval()
        if model.vocoder is not None:
            model.vocoder.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: Ltx2Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        n_corrupt = 0
        if model.transformer_lora is not None:
            for mod in model.transformer_lora.lora_modules.values():
                for p in mod.parameters():
                    if p.isnan().any() or p.isinf().any():
                        n_corrupt += 1
                        with torch.no_grad():
                            p.zero_()
        if n_corrupt:
            print(
                f"[Ltx2 LoRA reset] step={train_progress.global_step}: "
                f"{n_corrupt} params had NaN/Inf — zeroed"
            )


factory.register(BaseModelSetup, Ltx2LoRASetup, ModelType.LTX_2_3, TrainingMethod.LORA)
