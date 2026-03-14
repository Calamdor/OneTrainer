import copy
from abc import ABCMeta

import modules.util.multi_gpu_util as multi
from modules.model.WanModel import WanModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import enable_checkpointing_for_wan_transformer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.WanExpertMode import WanExpertMode
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseWanSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ffn"],
        "attn-only": ["attn"],
        "blocks": ["blocks"],
        "full": [],
    }

    def create_parameters(
            self,
            model: WanModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        expert_mode = getattr(config, 'wan_expert_mode', WanExpertMode.BOTH)
        if expert_mode != WanExpertMode.LOW_NOISE and model.transformer_lora is not None:
            self._create_model_part_parameters(
                parameter_group_collection, "transformer_lora", model.transformer_lora, config.transformer,
            )
        if expert_mode != WanExpertMode.HIGH_NOISE and model.transformer_2_lora is not None:
            self._create_model_part_parameters(
                parameter_group_collection, "transformer_2_lora", model.transformer_2_lora, config.transformer,
            )
        return parameter_group_collection

    def setup_optimizations(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_wan_transformer(model.transformer, config)
            model.transformer_2_offload_conductor = \
                enable_checkpointing_for_wan_transformer(model.transformer_2, config)

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        model.transformer_autocast_context, model.transformer_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().transformer,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        # Quantize each expert separately and return it to temp_device before quantizing
        # the next one.  GGUF_A8_INT layers call .quantize(device=train_device) which
        # moves the whole transformer to GPU; doing both at once doubles peak VRAM.
        quantize_layers(model.transformer, self.train_device, model.transformer_train_dtype, config)
        model.transformer_1_to(self.temp_device)
        quantize_layers(model.transformer_2, self.train_device, model.transformer_train_dtype, config)
        model.transformer_2_to(self.temp_device)

    def setup_model(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Overridden by WanLoRASetup for training.
        pass

    def setup_train_device(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        # Put everything on temp_device; WanSampler handles device moves during sampling.
        model.eval()
        model.to(self.temp_device)

    def on_validation_start(self):
        self._wan_val_toggle = 0

    def validation_predict_all(
            self,
            model: WanModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> list[dict]:
        expert_mode = getattr(config, 'wan_expert_mode', WanExpertMode.BOTH)
        if expert_mode != WanExpertMode.BOTH:
            return [self.predict(model, batch, config, train_progress, deterministic=True)]
        results = []
        for override_mode in (WanExpertMode.HIGH_NOISE, WanExpertMode.LOW_NOISE):
            _cfg = copy.copy(config)
            _cfg.wan_expert_mode = override_mode
            results.append(self.predict(model, batch, _cfg, train_progress, deterministic=True))
        return results

    def prepare_text_caching(
            self,
            model: WanModel,
            config: TrainConfig,
    ):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()

    def predict(
            self,
            model: WanModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)

            # --- text embeddings (always cached; text encoder is never trained) ---
            text_encoder_output = batch['text_encoder_1_hidden_state']
            tokens_mask = batch.get('tokens_mask_1')
            if tokens_mask is not None:
                # Zero out padding positions to match WanModel.encode_text behaviour
                text_encoder_output = text_encoder_output * \
                    tokens_mask.unsqueeze(-1).to(text_encoder_output.dtype)

            # --- latents: normalize per-channel from raw VAE mean ---
            vae = model.vae
            latents_mean = (
                torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=self.train_device)
                .view(1, vae.config.z_dim, 1, 1, 1)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std, dtype=torch.float32, device=self.train_device)
                .view(1, vae.config.z_dim, 1, 1, 1)
            )

            latent_image = batch['latent_image']
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)   # (B,C,H,W) → (B,C,1,H,W)

            # Cast to float32 for numerical stability during normalization
            normalized_latent = (latent_image.float() - latents_mean) / latents_std

            # --- noise & timestep ---
            latent_noise = self._create_noise(normalized_latent, config, generator)

            num_train_timesteps = model.noise_scheduler.config.num_train_timesteps
            expert_mode = getattr(config, 'wan_expert_mode', WanExpertMode.BOTH)
            boundary_t = int(model.boundary_ratio * num_train_timesteps)

            # In deterministic (validation) mode, explicitly target each expert in turn so
            # both are evaluated.  In sequential modes the batch always hits the active expert;
            # in BOTH mode we alternate: even calls → high-noise midpoint, odd → low-noise midpoint.
            validation_expert_label = None
            if deterministic:
                toggle = getattr(self, '_wan_val_toggle', 0)
                self._wan_val_toggle = 1 - toggle
                if expert_mode == WanExpertMode.HIGH_NOISE or (expert_mode == WanExpertMode.BOTH and toggle == 0):
                    det_t = boundary_t + (num_train_timesteps - boundary_t) // 2
                    validation_expert_label = 'high_noise'
                else:
                    det_t = boundary_t // 2
                    validation_expert_label = 'low_noise'
                timestep = torch.tensor([det_t], dtype=torch.long, device=self.train_device)
                # Expand to batch size
                if normalized_latent.shape[0] > 1:
                    timestep = timestep.expand(normalized_latent.shape[0])
            else:
                # In sequential expert training, restrict the noising-strength range
                # to the active expert's timestep window BEFORE sampling so the chosen
                # distribution (logit_normal, uniform, …) is applied correctly within
                # the window rather than pile-up occurring at the boundary from clamping.
                if expert_mode != WanExpertMode.BOTH:
                    _cfg = copy.copy(config)
                    if expert_mode == WanExpertMode.HIGH_NOISE:
                        _cfg.min_noising_strength = model.boundary_ratio
                        _cfg.max_noising_strength = 1.0
                    else:  # LOW_NOISE
                        _cfg.min_noising_strength = 0.0
                        _cfg.max_noising_strength = model.boundary_ratio
                else:
                    _cfg = config
                timestep = self._get_timestep_discrete(
                    num_train_timesteps,
                    deterministic,
                    generator,
                    normalized_latent.shape[0],
                    _cfg,
                )

            # Build a linear sigma schedule tensor for _add_noise_discrete
            training_timesteps = torch.arange(
                1, num_train_timesteps + 1, dtype=torch.long, device=self.train_device,
            )

            noisy_latent, sigma = self._add_noise_discrete(
                normalized_latent,
                latent_noise,
                timestep,
                training_timesteps,
            )

            # --- dual-transformer routing ---
            boundary_timestep = model.boundary_ratio * num_train_timesteps
            t_mean = timestep.float().mean().item()
            active_transformer = model.transformer if t_mean >= boundary_timestep else model.transformer_2

            with model.transformer_autocast_context:
                predicted_flow = active_transformer(
                    hidden_states=noisy_latent.to(dtype=model.transformer_train_dtype.torch_dtype()),
                    timestep=timestep,
                    encoder_hidden_states=text_encoder_output.to(
                        dtype=model.transformer_train_dtype.torch_dtype()
                    ),
                    return_dict=False,
                )[0]

            flow_target = latent_noise - normalized_latent

            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow_target.to(predicted_flow.dtype),
            }
            if validation_expert_label is not None:
                model_output_data['validation_expert_label'] = validation_expert_label

            if config.debug_mode:
                with torch.no_grad():
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_latent", noisy_latent, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow_target", flow_target, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: WanModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=None,
        ).mean()

    def after_optimizer_step(
            self,
            model: WanModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass
