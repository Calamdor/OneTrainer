import copy
from collections.abc import Callable

import numpy as np
from modules.model.WanModel import WanModel
from diffusers import UniPCMultistepScheduler
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch
from tqdm import tqdm

from PIL import Image


def _find_shift_for_step_balance(
        scheduler_config: dict,
        num_steps: int,
        steps_high: int,
        boundary_ratio: float,
) -> float:
    """Binary-search for the flow_shift that places exactly steps_high timesteps
    at or above boundary_ratio * num_train_timesteps.

    Strategy: find the minimum shift where timesteps[steps_high] >= boundary_t,
    meaning the step JUST PAST the boundary is still in high-noise territory —
    i.e. at least steps_high steps are high-noise. This avoids the discrete
    count plateau problem.
    """
    boundary_t = boundary_ratio * scheduler_config['num_train_timesteps']

    def t_at_idx(shift: float) -> float:
        """Return the timestep value at index steps_high (0-based, descending)."""
        s = UniPCMultistepScheduler(**{**scheduler_config, 'flow_shift': shift})
        s.set_timesteps(num_steps)
        if steps_high >= len(s.timesteps):
            return float('inf')
        return s.timesteps[steps_high].item()

    # More shift → higher timesteps everywhere.
    # We want t_at_idx(shift) to cross boundary_t from below.
    # hi = minimum shift where t[steps_high] >= boundary_t  →  count_high >= steps_high+1
    # lo = maximum shift where t[steps_high] <  boundary_t  →  count_high == steps_high
    # We want lo (gives exactly steps_high high steps).
    lo, hi = 0.01, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if t_at_idx(mid) < boundary_t:
            lo = mid   # still below boundary, can go higher
        else:
            hi = mid   # above/at boundary, back off

    shift = round(lo, 3)

    # Verify
    s_check = UniPCMultistepScheduler(**{**scheduler_config, 'flow_shift': shift})
    s_check.set_timesteps(num_steps)
    actual = sum(1 for t in s_check.timesteps if t >= boundary_t)
    print(f"[WanSampler] auto shift={shift:.3f} → {actual} high / {num_steps - actual} low steps")
    return shift


class WanSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: WanModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            num_frames: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            cfg_scale_2: float | None = None,
            flow_shift: float | None = None,
            steps_high: int | None = None,
            steps_low: int | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # Ensure num_frames satisfies 4k+1 constraint
            num_frames = ((num_frames - 1) // 4) * 4 + 1

            # 1. Text encoding
            self.model.text_encoder_to(self.train_device)

            prompt_embeds, _ = self.model.encode_text(prompt, self.train_device)
            if cfg_scale != 1.0:
                negative_embeds, _ = self.model.encode_text(negative_prompt or "", self.train_device)
            else:
                negative_embeds = None

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # 2. Prepare latents (float32 for numerical stability during scheduler steps)
            vae_temporal_scale = 4
            vae_spatial_scale = 8
            num_latent_channels = 16
            num_latent_frames = (num_frames - 1) // vae_temporal_scale + 1

            latents = torch.randn(
                1, num_latent_channels, num_latent_frames,
                height // vae_spatial_scale,
                width // vae_spatial_scale,
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # 3. Scheduler — resolve flow_shift from step-balance spec or explicit override
            scheduler_cfg = dict(self.model.noise_scheduler.config)
            if steps_high is not None and steps_low is not None:
                # Auto-find shift that gives exactly steps_high steps above boundary
                num_steps = steps_high + steps_low
                resolved_shift = _find_shift_for_step_balance(
                    scheduler_cfg, num_steps, steps_high, self.model.boundary_ratio,
                )
                diffusion_steps = num_steps
            elif flow_shift is not None:
                resolved_shift = flow_shift
            else:
                resolved_shift = None

            noise_scheduler = self.model.noise_scheduler.__class__(
                **{**scheduler_cfg, **({"flow_shift": resolved_shift} if resolved_shift is not None else {})},
            )
            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            # 4. Compute boundary timestep for dual-transformer routing
            # transformer  = high-noise expert (t >= boundary_timestep)
            # transformer_2 = low-noise expert  (t <  boundary_timestep)
            if self.model.transformer_2 is not None:
                boundary_timestep = self.model.boundary_ratio * noise_scheduler.config.num_train_timesteps
                high_steps = sum(1 for t in timesteps if t >= boundary_timestep)
                print(
                    f"[WanSampler] step split: {high_steps} high-noise (transformer) "
                    f"/ {len(timesteps) - high_steps} low-noise (transformer_2) "
                    f"of {len(timesteps)} total  [boundary={boundary_timestep:.0f}]"
                )
            else:
                boundary_timestep = None

            # 5. Denoising loop — load each expert once, swap at the boundary

            # Debug: report companion LoRA patch status before sampling begins
            n_companion = len(self.model.companion_lora_handles)
            companion_expert = getattr(self.model, 'companion_lora_expert', None)
            if n_companion > 0:
                expert_label = "high-noise (transformer)" if companion_expert == 1 else "low-noise (transformer_2)"
                print(f"[WanSampler] Companion LoRA: {n_companion} forward patches on {expert_label}")
            else:
                print("[WanSampler] Companion LoRA: none")

            _companion_fired = {1: False, 2: False}  # track first-fire per expert
            current_expert = None  # 1 = high-noise, 2 = low-noise

            for i, t in enumerate(tqdm(timesteps, desc="sampling")):
                use_high_noise = (boundary_timestep is None or t >= boundary_timestep)
                desired_expert = 1 if use_high_noise else 2

                if desired_expert != current_expert:
                    # Offload the previous expert
                    if current_expert == 1:
                        self.model.transformer_1_to(self.temp_device)
                        torch_gc()
                    elif current_expert == 2:
                        self.model.transformer_2_to(self.temp_device)
                        torch_gc()

                    # Load the next expert
                    if desired_expert == 1:
                        self.model.transformer_1_to(self.train_device)
                    else:
                        self.model.transformer_2_to(self.train_device)
                    current_expert = desired_expert

                    # Debug: on first use of each expert, confirm companion hook presence
                    if not _companion_fired[desired_expert]:
                        _companion_fired[desired_expert] = True
                        label = "high-noise (transformer)" if desired_expert == 1 else "low-noise (transformer_2)"
                        has_companion = n_companion > 0 and companion_expert == desired_expert
                        companion_note = f"{n_companion} forward patches (companion LoRA active)" if has_companion else "no companion LoRA"
                        print(f"[WanSampler] step {i}: switching to {label} — {companion_note}")

                active_transformer = (
                    self.model.transformer if desired_expert == 1 else self.model.transformer_2
                )
                timestep = t.expand(1)

                with self.model.transformer_autocast_context:
                    noise_pred = active_transformer(
                        hidden_states=latents.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                        return_dict=False,
                    )[0]

                    if negative_embeds is not None:
                        noise_uncond = active_transformer(
                            hidden_states=latents.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                            timestep=timestep,
                            encoder_hidden_states=negative_embeds.to(dtype=self.model.transformer_train_dtype.torch_dtype()),
                            return_dict=False,
                        )[0]
                        active_cfg = (cfg_scale_2 if (cfg_scale_2 is not None and desired_expert == 2)
                                      else cfg_scale)
                        noise_pred = noise_uncond + active_cfg * (noise_pred - noise_uncond)

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                on_update_progress(i + 1, len(timesteps))

            # Offload final active expert
            if current_expert == 1:
                self.model.transformer_1_to(self.temp_device)
            elif current_expert == 2:
                self.model.transformer_2_to(self.temp_device)
            torch_gc()

            # 6. VAE decode
            self.model.vae_to(self.train_device)
            vae = self.model.vae
            vae.enable_tiling()

            # Denormalize latents before decoding
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                (1.0 / torch.tensor(vae.config.latents_std))
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std + latents_mean

            video = vae.decode(latents.to(dtype=vae.dtype), return_dict=False)[0]

            self.model.vae_to(self.temp_device)
            torch_gc()

            # 7. Post-process: (1, C, T, H, W) float → uint8
            # VAE output is in [-1, 1]; normalize to [0, 1]
            video = (video.float() + 1.0) / 2.0
            nan_count = torch.isnan(video).sum().item()
            if nan_count > 0:
                print(f"[WanSampler] WARNING: {nan_count} NaN values in decoded output — "
                      "clamping to 0. Check dtype settings (FP16 compute over BF16 weights "
                      "causes this).")
                video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
            video = video.clamp(0, 1).cpu()

            is_image = video.shape[2] == 1
            if is_image:
                frame = video[0, :, 0, :, :].permute(1, 2, 0).numpy()
                frame = (frame * 255).round().astype(np.uint8)
                return ModelSamplerOutput(
                    file_type=FileType.IMAGE,
                    data=Image.fromarray(frame),
                )
            else:
                # (T, H, W, C) uint8 tensor
                frames = video[0].permute(1, 2, 3, 0)
                frames = (frames * 255).round().to(torch.uint8)
                return ModelSamplerOutput(
                    file_type=FileType.VIDEO,
                    data=frames,
                    fps=16,
                )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        steps_high = getattr(sample_config, 'steps_high', None)
        steps_low = getattr(sample_config, 'steps_low', None)
        cfg_scale_2 = getattr(sample_config, 'cfg_scale_2', None)
        flow_shift = getattr(sample_config, 'flow_shift', None)

        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            num_frames=self.quantize_resolution(sample_config.frames - 1, 4) + 1,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            cfg_scale_2=cfg_scale_2,
            flow_shift=flow_shift,
            steps_high=steps_high,
            steps_low=steps_low,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, WanSampler, ModelType.WAN2_2_T2V)
