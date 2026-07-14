from collections.abc import Callable

from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.ltx2 import _vae_chunked_decode as _ccd
from modules.modelLoader.ltx2._ffn_chunk_patch import chunked_ffn
from modules.modelLoader.ltx2._sequential_cfg_patch import sequential_cfg
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LtxMultiScaleMode import LtxMultiScaleMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

import numpy as np
from PIL import Image

# LTX-2.3 inference extras -- intentionally empty. The pipeline defaults (stg_scale=0,
# modality_scale=1, guidance_rescale=0, use_cross_timestep=False) match what ComfyUI does:
# pure CFG with a single forward pass per step (two passes when CFG > 1 for cond + uncond).
_LTX_2_3_INFERENCE_EXTRAS: dict = {}

_BUCKET_DIVISIBILITY = 32          # LTX-2 patch / VAE constraint
_FRAME_QUANTIZATION_FACTOR = 8     # frames must satisfy (n - 1) % 8 == 0
_DEFAULT_FRAME_RATE = 24.0         # LTX-2 default; the model supports variable fps

# FFN chunking along the token dim -- not cosmetic; at typical sampling resolutions the FFN
# intermediate is the dominant per-block transient activation (see _ffn_chunk_patch.py).
# 4096 chunks the worst case (~60k tokens) into ~15 pieces (~250 MB each).
_SAMPLING_FFN_CHUNK = 4096

# Official Lightricks distilled-recipe sigma schedule for stage 2's refiner pass (verified
# from the reference ComfyUI workflow): a short partial-denoise starting from sigma=0.85 --
# fixed regardless of stage 1's mode, since it's architecturally a short refiner pass, not a
# from-scratch denoise. Stage 1 uses the user's own diffusion_steps/cfg_scale instead of a
# hardcoded schedule (see _sample_two_stage's docstring).
#
# The trailing 0.0 sigma in the reference workflow is dropped here: diffusers'
# FlowMatchEulerDiscreteScheduler.set_timesteps appends 0.0 automatically; including it
# would both mis-count the progress bar and divide-by-zero in the dynamic-shifting formula.
_DISTILLED_STAGE2_SIGMAS = [0.85, 0.725, 0.422]
_STAGE2_NOISE_T = 0.85  # noise blend ratio when re-noising upsampled latents


@factory.register(BaseModelSampler, ModelType.LTX_2_3)
class Ltx2Sampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: Ltx2Model,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    def _quantize_frames(self, num_frames: int) -> int:
        if num_frames <= 1:
            return 1
        return ((num_frames - 1) // _FRAME_QUANTIZATION_FACTOR) * _FRAME_QUANTIZATION_FACTOR + 1

    def _pad_embeds(self, embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LTX-2's connector requires 1024 prompt tokens; pad with zeros on the left side."""
        target_length = 1024
        current_length = embeds.shape[1]
        if current_length >= target_length:
            return embeds, mask

        pad_length = target_length - current_length
        pad_embed = torch.zeros(
            (embeds.shape[0], pad_length, embeds.shape[2]),
            device=embeds.device, dtype=embeds.dtype,
        )
        embeds = torch.cat([pad_embed, embeds], dim=1)

        if mask is not None:
            pad_mask = torch.zeros(
                (mask.shape[0], pad_length),
                device=mask.device, dtype=mask.dtype,
            )
            mask = torch.cat([pad_mask, mask], dim=1)

        return embeds, mask

    def _pick_upsampler(self, mode: LtxMultiScaleMode):
        """Return the model's upsampler matching the multi-scale mode, or None."""
        if mode == LtxMultiScaleMode.X1_5:
            return self.model.latent_upsampler_x1_5
        if mode == LtxMultiScaleMode.X2:
            return self.model.latent_upsampler_x2
        return None

    def _decode_video_latents(self, vae, latents_bf16: torch.Tensor) -> torch.Tensor:
        """Decode video latents, preferring ComfyUI-style chunked streaming decode.

        Diffusers' own tiled decode handles the *spatial* dimensions with real
        overlap-blend tiling (functional, verified against source), but its *temporal*
        tiling materializes the whole overlapping-chunk volume and is documented (see
        _vae_chunked_decode.py) as 20-200x slower / prone to shared-VRAM spillover at
        typical frame counts. The chunked streaming decode -- a genuine single-pass
        architecture with sliding-window conv caches, ported from ComfyUI's approach --
        is preferred whenever the loaded VAE matches its supported subset. Falls back to
        the plain (tiled, if enabled) vae.decode() on any mismatch, so it can never
        regress below the previous behavior.
        """
        if _ccd.is_enabled():
            try:
                return _ccd.chunked_decode(
                    vae, latents_bf16,
                    output_device=self.temp_device,
                    output_dtype=latents_bf16.dtype,
                )
            except Exception as e:
                print(f"[Ltx2 VAE] chunked decode unavailable ({type(e).__name__}: {e}); "
                      "falling back to tiled decode")
        return vae.decode(latents_bf16, return_dict=False)[0]

    @torch.no_grad()
    def _decode_video_and_audio(
            self,
            pipeline,
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor | None,
            is_video: bool,
    ):
        """Decode denormalized video (and, for video samples, audio) latents.

        Mirrors the non-"latent" path of LTX2Pipeline.__call__, called after moving the
        transformer + connectors off GPU so decode runs without 22+ GB of transformer
        weights also sitting on the device. The LTX-2.3 video VAE has
        timestep_conditioning=False, so decode is a straightforward VAE.decode() -- no
        noise injection at decode time. LTX-2.3 is a joint video+audio model -- the audio
        branch is always denoised alongside video (even for a T2V-trained LoRA, since the
        transformer forward always requires audio_hidden_states), so audio_latents is
        real output to decode and mux, not something to discard.
        """
        self.model.vae_to(self.train_device)
        self.model.audio_vae_to(self.temp_device)
        torch_gc()

        vae = pipeline.vae
        orig_vae_dtype = next(vae.parameters()).dtype
        if orig_vae_dtype != torch.bfloat16:
            vae.to(dtype=torch.bfloat16)
        try:
            video_pixels = self._decode_video_latents(vae, video_latents.to(torch.bfloat16))
        finally:
            if orig_vae_dtype != torch.bfloat16:
                vae.to(dtype=orig_vae_dtype)
        del video_latents

        self.model.vae_to(self.temp_device)
        torch_gc()

        video = pipeline.video_processor.postprocess_video(
            video_pixels, output_type="np" if is_video else "pil",
        )
        del video_pixels

        audio = None
        if audio_latents is not None and is_video:
            self.model.audio_vae_to(self.train_device)
            self.model.vocoder_to(self.train_device)
            torch_gc()
            audio_latents = audio_latents.to(pipeline.audio_vae.dtype)
            mel = pipeline.audio_vae.decode(audio_latents, return_dict=False)[0]
            del audio_latents
            # audio_vae is always loaded fp32 (matches the video vae's forced-fp32 loading,
            # see Ltx2ModelLoader), independent of the vocoder's own weight dtype -- cast the
            # mel spectrogram to whatever dtype the vocoder's weights actually are, same
            # pattern as the video vae's temporary dtype handling above.
            vocoder_dtype = next(pipeline.vocoder.parameters()).dtype
            audio = pipeline.vocoder(mel.to(dtype=vocoder_dtype))
            del mel
            self.model.audio_vae_to(self.temp_device)
            self.model.vocoder_to(self.temp_device)
            torch_gc()

        return video, audio

    def _sample_two_stage(
            self,
            pipeline,
            upsampler,
            multi_scale_mode: LtxMultiScaleMode,
            prompt_embeds: torch.Tensor,
            prompt_mask: torch.Tensor,
            neg_embeds: torch.Tensor | None,
            neg_mask: torch.Tensor | None,
            final_height: int,
            final_width: int,
            num_frames: int,
            frame_rate: float,
            diffusion_steps: int,
            cfg_scale: float,
            stage1_strength: float,
            stage2_strength: float,
            use_distilled_lora: bool,
            generator: torch.Generator,
            on_update_progress: Callable[[int, int], None],
    ):
        """Two-stage spatial-upsample sampling.

        ``final_height``/``final_width`` are the FINAL output resolution (the convention
        for this sampler regardless of mode -- see the SampleConfig field docs); stage 1's
        (lower) resolution is derived here by dividing by the mode's upscale factor, not
        supplied directly.

        Stage 2 (the upscale refiner) always uses the fixed 3-step partial-denoise
        schedule starting at sigma=0.85, CFG=1.0, and the distilled LoRA at
        ``stage2_strength`` if one is loaded -- this is architecturally fixed (a short
        refiner pass only makes sense fully distilled), not a per-run tunable.

        Stage 1 is the real tunable dial: it always runs a normal flow-matching denoise at
        the user's ``diffusion_steps``/``cfg_scale`` -- the distilled LoRA (if loaded and
        enabled) is applied at ``stage1_strength`` on top of that, not via a separate fixed
        schedule. This covers both ends of the tradeoff through ordinary parameters: "fast,
        heavily distilled" is few steps + low cfg + high strength (e.g. 8 steps, cfg=1,
        strength=0.8); "slower, more diverse" is many steps + normal cfg + low strength
        (e.g. 50 steps, cfg=3, strength=0.2) -- no hardcoded sigma schedule needed for
        stage 1, unlike stage 2's fixed refiner pass.
        """
        factor = multi_scale_mode.upscale_factor()
        stage1_h = self.quantize_resolution(int(round(final_height / factor)), _BUCKET_DIVISIBILITY)
        stage1_w = self.quantize_resolution(int(round(final_width / factor)), _BUCKET_DIVISIBILITY)

        has_distilled = use_distilled_lora and bool(self.model.distilled_lora_handles)
        stage1_steps = diffusion_steps
        stage1_kwargs = {"num_inference_steps": diffusion_steps}
        total_steps = stage1_steps + len(_DISTILLED_STAGE2_SIGMAS)

        print(
            f"[Ltx2 Sampler] two-stage {multi_scale_mode}: "
            f"stage 1 @ {stage1_w}x{stage1_h} ({stage1_steps} steps), "
            f"stage 2 @ {final_width}x{final_height} ({len(_DISTILLED_STAGE2_SIGMAS)} steps)"
        )

        if has_distilled:
            self.model.distilled_lora_strength = stage1_strength
            self.model.distilled_lora_to(self.train_device)
            self.model._resume_distilled_lora_hooks()

        with sequential_cfg(self.model.transformer), \
                chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK):
            stage1_latents, stage1_audio = pipeline(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                height=stage1_h,
                width=stage1_w,
                num_frames=num_frames,
                frame_rate=frame_rate,
                guidance_scale=cfg_scale,
                generator=generator,
                return_dict=False,
                output_type="latent",
                **stage1_kwargs,
                **_LTX_2_3_INFERENCE_EXTRAS,
            )
        on_update_progress(stage1_steps, total_steps)

        if has_distilled:
            self.model._pause_distilled_lora_hooks()

        # Upsample latents directly -- no VAE round-trip. The pipeline's output_type="latent"
        # path returns DENORMALIZED latents, and the upsampler operates on unnormalized
        # latents per its module docstring, so feeding the stage-1 output directly is correct.
        self.model.latent_upsampler_to(self.train_device, scale=factor)
        try:
            upsampler_dtype = next(upsampler.parameters()).dtype
            upsampled_latents = upsampler(stage1_latents.to(upsampler_dtype))
        finally:
            self.model.latent_upsampler_to(self.temp_device, scale=factor)
        del stage1_latents
        torch_gc()

        if has_distilled:
            self.model.distilled_lora_strength = stage2_strength
            self.model.distilled_lora_to(self.train_device)
            self.model._resume_distilled_lora_hooks()
        else:
            print("[Ltx2 Sampler] warning: two-stage refiner running WITHOUT a distilled "
                  "LoRA loaded -- the fixed 3-step partial-denoise schedule is designed for "
                  "a distilled model, quality may be poor")

        # Stage 2: pass DENORMALIZED upsampled video latents + stage-1 audio latents. Video:
        # the pipeline normalizes, adds noise_scale noise, then denoises. Audio: stage-1
        # audio is refined (not restarted from noise -- 3 steps from scratch would be
        # garbled) at the same noise_scale so both modalities partial-denoise consistently.
        with sequential_cfg(self.model.transformer), \
                chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK):
            video_latents, audio_latents = pipeline(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                height=final_height,
                width=final_width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                sigmas=_DISTILLED_STAGE2_SIGMAS,
                latents=upsampled_latents,
                audio_latents=stage1_audio,
                noise_scale=_STAGE2_NOISE_T,
                guidance_scale=1.0,
                generator=generator,
                return_dict=False,
                output_type="latent",
                **_LTX_2_3_INFERENCE_EXTRAS,
            )
        del upsampled_latents, stage1_audio
        on_update_progress(total_steps, total_steps)

        if has_distilled:
            self.model._pause_distilled_lora_hooks()
            self.model.distilled_lora_to(self.temp_device)

        return video_latents, audio_latents

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            vae_tiling: bool,
            use_distilled_lora: bool,
            multi_scale_mode: LtxMultiScaleMode,
            stage1_strength: float,
            stage2_strength: float,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            use_neg = cfg_scale != 1.0

            # 1. Encode prompts on the text encoder (Gemma3-12B), then offload it.
            self.model.text_encoder_to(self.train_device)
            prompt_embeds, prompt_mask = self.model.encode_text(prompt, self.train_device)
            prompt_embeds, prompt_mask = self._pad_embeds(prompt_embeds, prompt_mask)
            if use_neg:
                neg_embeds, neg_mask = self.model.encode_text(negative_prompt or "", self.train_device)
                neg_embeds, neg_mask = self._pad_embeds(neg_embeds, neg_mask)
            else:
                neg_embeds, neg_mask = None, None
            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # 2. Move diffusion components to GPU.
            self.model.connectors_to(self.train_device)
            self.model.transformer_to(self.train_device)
            torch_gc()

            pipeline = self.model.create_pipeline()
            # pipeline.device returns vae.device (vae is first in the __init__ signature).
            # With VAE on CPU, _execution_device = CPU -> prepare_latents creates CPU latents
            # -> CUDA generator type mismatch. Override on a throwaway subclass so the
            # pipeline creates latents on the correct device without keeping VAE on GPU.
            _td = self.train_device
            pipeline.__class__ = type(
                pipeline.__class__.__name__,
                (pipeline.__class__,),
                {"_execution_device": property(lambda self: _td)},
            )
            pipeline.set_progress_bar_config(disable=False)
            if vae_tiling and hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
            elif hasattr(pipeline.vae, "disable_tiling"):
                pipeline.vae.disable_tiling()

            is_video = num_frames > 1

            upsampler = self._pick_upsampler(multi_scale_mode) if multi_scale_mode.is_two_stage() else None
            if multi_scale_mode.is_two_stage() and upsampler is None:
                print(f"[Ltx2 Sampler] multi-scale mode {multi_scale_mode} requested but the "
                      "upsampler is not loaded -- falling back to FULL_SIZE")

            if multi_scale_mode.is_two_stage() and upsampler is not None:
                video_latents, audio_latents = self._sample_two_stage(
                    pipeline=pipeline,
                    upsampler=upsampler,
                    multi_scale_mode=multi_scale_mode,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    neg_embeds=neg_embeds,
                    neg_mask=neg_mask,
                    final_height=height,
                    final_width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    diffusion_steps=diffusion_steps,
                    cfg_scale=cfg_scale,
                    stage1_strength=stage1_strength,
                    stage2_strength=stage2_strength,
                    use_distilled_lora=use_distilled_lora,
                    generator=generator,
                    on_update_progress=on_update_progress,
                )
            else:
                # Single-stage (FULL_SIZE): denoise with output_type="latent", then offload
                # the transformer + connectors before VAE decode so the heavy decode step has
                # the GPU to itself.
                use_distilled = use_distilled_lora and bool(self.model.distilled_lora_handles)
                if use_distilled:
                    # Same "stage 1 strength" dial used by two-stage mode's first pass -- FULL_SIZE
                    # is just stage 1 with no stage 2 following it, not a differently-configured mode.
                    self.model.distilled_lora_strength = stage1_strength
                    # distilled_lora_to(GPU) pins the tensors in CPU memory (never moves them
                    # to VRAM -- see Ltx2Model.distilled_lora_to()'s docstring for why).
                    self.model.distilled_lora_to(self.train_device)
                    self.model._resume_distilled_lora_hooks()

                # sequential_cfg is a no-op when the batch isn't CFG-doubled (batch=1, i.e.
                # cfg_scale=1.0), so it's always safe to wrap -- matches how it's engaged
                # unconditionally in the reference implementation.
                with sequential_cfg(self.model.transformer), \
                        chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK):
                    video_latents, audio_latents = pipeline(
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_mask,
                        negative_prompt_embeds=neg_embeds,
                        negative_prompt_attention_mask=neg_mask,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=frame_rate,
                        num_inference_steps=diffusion_steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        return_dict=False,
                        output_type="latent",
                        **_LTX_2_3_INFERENCE_EXTRAS,
                    )
                on_update_progress(diffusion_steps, diffusion_steps)

                if use_distilled:
                    self.model._pause_distilled_lora_hooks()
                    self.model.distilled_lora_to(self.temp_device)

            self.model.transformer_to(self.temp_device)
            self.model.connectors_to(self.temp_device)
            torch_gc()

            video, audio = self._decode_video_and_audio(pipeline, video_latents, audio_latents, is_video)
            del video_latents, audio_latents

            del prompt_embeds, prompt_mask, neg_embeds, neg_mask
            torch_gc()

            # Capture audio for muxing into the output container.
            audio_waveform = None
            audio_sample_rate = None
            if audio is not None and is_video:
                try:
                    audio_waveform = audio[0].float().cpu()
                    audio_sample_rate = int(self.model.vocoder.config.output_sampling_rate)
                except Exception as e:
                    print(f"[Ltx2Sampler] could not capture audio for muxing: {e}")
            del audio

            if is_video:
                # `video` is a numpy array of shape (B, T, H, W, C) in [0, 1].
                frames_np = video[0]
                frames_np = (np.clip(frames_np, 0.0, 1.0) * 255).round().astype(np.uint8)
                frames_tensor = torch.from_numpy(frames_np)  # (T, H, W, C) uint8
                return ModelSamplerOutput(
                    file_type=FileType.VIDEO,
                    data=frames_tensor,
                    fps=int(round(frame_rate)),
                    audio=audio_waveform,
                    audio_sample_rate=audio_sample_rate,
                )
            else:
                frame = video[0][0] if isinstance(video[0], list) else video[0]
                if not isinstance(frame, Image.Image):
                    frame_np = np.asarray(frame)
                    if frame_np.dtype != np.uint8:
                        frame_np = (np.clip(frame_np, 0.0, 1.0) * 255).round().astype(np.uint8)
                    frame = Image.fromarray(frame_np)
                return ModelSamplerOutput(
                    file_type=FileType.IMAGE,
                    data=frame,
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
        multi_scale_mode = sample_config.ltx_multi_scale_mode or LtxMultiScaleMode.FULL_SIZE
        # Not user-configurable: the ComfyUI-style chunked streaming decode (_decode_video_latents)
        # is always attempted first regardless of this flag. This only controls diffusers' native
        # tiling on the rare fallback path (chunked decode unavailable/erroring), where tiling is
        # strictly safer than not, so it's always left on.
        vae_tiling = True

        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, _BUCKET_DIVISIBILITY),
            width=self.quantize_resolution(sample_config.width, _BUCKET_DIVISIBILITY),
            num_frames=self._quantize_frames(sample_config.frames),
            frame_rate=_DEFAULT_FRAME_RATE,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            vae_tiling=bool(vae_tiling),
            use_distilled_lora=bool(sample_config.ltx_use_distilled_lora),
            multi_scale_mode=multi_scale_mode,
            stage1_strength=float(sample_config.ltx_distilled_lora_stage1_strength or 0.8),
            stage2_strength=float(sample_config.ltx_distilled_lora_stage2_strength or 0.8),
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
            fps=int(round(_DEFAULT_FRAME_RATE)),
        )

        on_sample(sampler_output)
