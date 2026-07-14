from modules.model.Ltx2Model import Ltx2Model
from modules.modelSetup.BaseLtx2Setup import BaseLtx2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.LTX_2_3, TrainingMethod.LORA)
class Ltx2LoRASetup(
    BaseLtx2Setup,
):
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

    def create_parameters(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "text_encoder_lora", model.text_encoder_lora, config.text_encoder)
        self._create_model_part_parameters(parameter_group_collection, "transformer_lora", model.transformer_lora, config.transformer)

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        if model.audio_vae is not None:
            model.audio_vae.requires_grad_(False)
        if model.connectors is not None:
            model.connectors.requires_grad_(False)
        if model.vocoder is not None:
            model.vocoder.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder_lora", model.text_encoder_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer_lora", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # LTX-2.3 T2V LoRA trains the transformer only by default; text-encoder LoRA is
        # created only when explicitly requested or when resuming a state dict that has one
        # (matches every other model's create-on-demand pattern).
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "text_encoder")

        if model.text_encoder is not None:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, "text_encoder", config
            ) if create_te else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(","),
            fusion_spec=model.fusion_groups(), fuse=config.output_model_format.needs_qkv_fusion(),
        )

        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self._setup_distilled_lora(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)
        # connectors/vocoder/audio_vae are never trained and their outputs are cached during
        # the text-caching pass, so they stay off the train device during training proper.
        model.connectors_to(self.temp_device)
        model.vocoder_to(self.temp_device)
        if model.audio_vae is not None:
            model.audio_vae_to(self.temp_device)

        if model.text_encoder is not None:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        model.vae.eval()

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
        self.__setup_requires_grad(model, config)
