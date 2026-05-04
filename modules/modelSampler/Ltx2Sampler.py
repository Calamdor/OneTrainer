import os
import time
from collections.abc import Callable
from contextlib import contextmanager

from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.ltx2._ffn_chunk_patch import attention_backend, chunked_ffn
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


# LTX-2.3 inference extras — intentionally empty.
#
# The pipeline defaults (stg_scale=0, modality_scale=1, guidance_rescale=0,
# use_cross_timestep=False) match what ComfyUI does: pure CFG with a single
# forward pass per step (two passes when CFG > 1 for cond + uncond).
#
# Previous extras that were removed:
#   stg_scale=1.0        → extra STG forward pass per step
#   modality_scale=3.0   → extra modality-isolation forward pass per step;
#                          delta scaled by (3.0-1)=2 applied to video latents —
#                          catastrophic for T2V with zero audio conditioning
#   guidance_rescale=0.7 → post-CFG rescale not used by ComfyUI
#   use_cross_timestep=True → ComfyUI model has no equivalent; False is correct
_LTX_2_3_INFERENCE_EXTRAS: dict = {}

# FFN chunking along the token dim. At ~60k tokens × hidden=16384 the FFN
# intermediate is ~3.7 GB and dominates per-block transient peak. ComfyUI uses
# 2 chunks by default. We use a fixed token-count chunk so behavior is
# stable across resolutions; 4096 chunks the worst case (~60k tokens) into 15
# pieces (~250 MB each).
_SAMPLING_FFN_CHUNK = 4096

# Diffusers attention backend for sampling. None = leave default ("native").
# "sage" requires sageattention >=2.1.1 installed; "flash" requires flash-attn.
_SAMPLING_ATTENTION_BACKEND: str | None = "sage"

_BUCKET_DIVISIBILITY = 32          # LTX-2 patch / VAE constraint
_FRAME_QUANTIZATION_FACTOR = 8     # frames must satisfy (n - 1) % 8 == 0
_DEFAULT_FRAME_RATE = 24.0         # LTX-2 default; the model supports variable fps

# Official Lightricks distilled-recipe sigma schedules (verified from the
# reference ComfyUI workflow JSON: LTX-2.3_T2V_I2V_Two_Stage_Distilled.json).
# Stage 1 at low resolution does the bulk of the denoising; stage 2 at the
# upsampled (= target) resolution does a short partial-denoise refiner pass
# starting from sigma=0.85.
#
# IMPORTANT: the trailing 0.0 sigma in the ComfyUI workflow is dropped here.
# diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps appends 0.0
# automatically (line 379 of scheduling_flow_match_euler_discrete.py).
# Including it in our list causes (a) the progress bar to show N+1 steps
# instead of N, and (b) a divide-by-zero in time_shift when sigma=0 hits
# the dynamic-shifting formula `(1/t - 1) ** sigma`.
_DISTILLED_STAGE1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
_DISTILLED_STAGE2_SIGMAS = [0.85, 0.725, 0.422]
_STAGE2_NOISE_T = 0.85  # noise blend ratio when re-noising upsampled latents

_LTX2_VRAM_DEBUG: bool = bool(os.environ.get("LTX2_VRAM_DEBUG"))


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

    def _reset_conductor_stats(self) -> None:
        """Reset per-stage conductor instrumentation counters."""
        if not _LTX2_VRAM_DEBUG:
            return
        conductor = getattr(self.model, "transformer_offload_conductor", None)
        if conductor is not None and hasattr(conductor, "reset_stats"):
            conductor.reset_stats()

    def _dump_conductor_stats(self, label: str) -> None:
        """Print accumulated conductor stats for the just-finished pipeline call."""
        if not _LTX2_VRAM_DEBUG:
            return
        conductor = getattr(self.model, "transformer_offload_conductor", None)
        if conductor is not None and hasattr(conductor, "dump_stats"):
            conductor.dump_stats(label)

    def _reset_lora_call_counter(self) -> None:
        if not _LTX2_VRAM_DEBUG:
            return
        from modules.model.Ltx2Model import _DistilledLoraCallStats
        _DistilledLoraCallStats.reset()

    def _dump_lora_stats(self, label: str) -> None:
        if not _LTX2_VRAM_DEBUG:
            return
        from modules.model.Ltx2Model import _DistilledLoraCallStats
        _DistilledLoraCallStats.dump(label)

    @contextmanager
    def _timed_phase(self, label: str):
        """Time a high-level sampling phase (TE encode, components→GPU, pipeline call, VAE decode).

        Brackets the block with ``torch.cuda.synchronize()`` at both ends so the
        measurement reflects actual GPU work — not just CPU dispatch latency.
        Gated by the ``LTX2_VRAM_DEBUG`` env var, identical to ``_vram_log``.
        """
        if not _LTX2_VRAM_DEBUG:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.train_device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.train_device)
            dt = time.perf_counter() - t0
            print(f"[Ltx2 Time] {label}: {dt*1000:.0f}ms ({dt:.2f}s)")

    def _vram_log(self, label: str, reset_peak: bool = True) -> None:
        """Diagnostic VRAM + RAM reporter — gates on env var so it's quiet in normal runs.

        VRAM (from cuda.memory_*):
        - ``alloc``: current bytes held by live tensors
        - ``peak``: max alloc since last reset (captures the spike during the
          phase that just ended; snapshot alone misses transient peaks during
          forward passes)
        - ``reserved``: caching-allocator pool size (alloc + idle slack);
          ``reserved - alloc`` is allocator overhead PyTorch hasn't returned
          to the OS
        - ``free/total``: the OS-side view from cuMemGetInfo — diverges from
          ``total - reserved`` if other processes are using the device

        RAM (from psutil, when available):
        - ``proc``: this process's resident set size (what shows up in Task
          Manager's "Memory" column)
        - ``sys used/total``: system-wide RAM in use vs installed; useful for
          spotting CPU-side spikes when components ride the bus to/from device

        With ``reset_peak=True`` (default), the peak counter resets after
        printing so the next call shows the peak of the upcoming phase only.
        """
        if not _LTX2_VRAM_DEBUG:
            return
        try:
            allocated = torch.cuda.memory_allocated(self.train_device) / 1e9
            peak = torch.cuda.max_memory_allocated(self.train_device) / 1e9
            reserved = torch.cuda.memory_reserved(self.train_device) / 1e9
            free_b, total_b = torch.cuda.mem_get_info(self.train_device)
            free, total = free_b / 1e9, total_b / 1e9
            ram_str = ""
            try:
                import psutil
                proc_rss = psutil.Process().memory_info().rss / 1e9
                vm = psutil.virtual_memory()
                ram_str = (
                    f" | RAM proc={proc_rss:.2f} "
                    f"sys={vm.used / 1e9:.2f}/{vm.total / 1e9:.2f} GB"
                )
            except Exception:
                pass
            print(
                f"[Ltx2 VRAM] {label}: "
                f"alloc={allocated:.2f} peak={peak:.2f} reserved={reserved:.2f} "
                f"free/total={free:.2f}/{total:.2f} GB{ram_str}"
            )
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(self.train_device)
        except Exception:
            pass

    @torch.no_grad()
    def _sample_two_stage(
            self,
            pipeline,
            upsampler,
            multi_scale_mode: LtxMultiScaleMode,
            prompt_embeds: torch.Tensor,
            prompt_mask: torch.Tensor,
            neg_embeds: torch.Tensor | None,
            neg_mask: torch.Tensor | None,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            diffusion_steps: int,
            cfg_scale: float,
            stage1_strength: float,
            stage2_strength: float,
            use_distilled_lora: bool,
            extras: dict,
            is_video: bool,
            generator: torch.Generator,
            on_update_progress: Callable[[int, int], None],
    ):
        """Two-stage spatial-upsample sampling.

        Stage 1: generate at the user's specified W×H using the configured
        diffusion_steps with a normal scheduler, output_type="latent" → no VAE decode.
        Upsample latents directly via the spatial upsampler model.
        Stage 2: refine at the upscaled (larger) resolution with the 3-step
        partial-denoise sigma schedule starting at sigma=0.85.

        The distilled LoRA (if loaded) is applied as a quality booster at the
        configured strength; it does not enforce a specific step count.
        """
        factor = multi_scale_mode.upscale_factor()
        up_h = self.quantize_resolution(int(round(height * factor)), _BUCKET_DIVISIBILITY)
        up_w = self.quantize_resolution(int(round(width * factor)), _BUCKET_DIVISIBILITY)
        # --- Stage 1: user-specified res, full denoise ---
        # When the distilled LoRA is active, follow the official Lightricks
        # two-stage workflow: stage 1 uses the 8-step distilled sigma schedule
        # (matches ComfyUI's `video_ltx2_3_t2v.json` template). Without distilled
        # LoRA, stage 1 honors the user's `diffusion_steps` with default
        # flow-match sigmas. Stage 2 is always the 3-step distilled refiner.
        if use_distilled_lora:
            stage1_steps = len(_DISTILLED_STAGE1_SIGMAS)
            stage1_kwargs = {"sigmas": _DISTILLED_STAGE1_SIGMAS}
        else:
            stage1_steps = diffusion_steps
            stage1_kwargs = {"num_inference_steps": diffusion_steps}
        total_steps = stage1_steps + len(_DISTILLED_STAGE2_SIGMAS)
        print(
            f"[Ltx2 Sampler] two-stage {multi_scale_mode}: "
            f"stage 1 @ {width}x{height} ({stage1_steps} steps), "
            f"stage 2 @ {up_w}x{up_h} ({len(_DISTILLED_STAGE2_SIGMAS)} steps)"
        )
        if use_distilled_lora:
            self.model.distilled_lora_strength = stage1_strength
            self.model._resume_distilled_lora_hooks()
        self._reset_conductor_stats()
        self._reset_lora_call_counter()
        with self._timed_phase(f"pipeline stage 1 ({stage1_steps} steps @ {width}x{height})"), \
                sequential_cfg(self.model.transformer), \
                chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK), \
                attention_backend(self.model.transformer, _SAMPLING_ATTENTION_BACKEND):
            stage1_latents, stage1_audio = pipeline(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                guidance_scale=cfg_scale,
                generator=generator,
                return_dict=False,
                output_type="latent",
                **stage1_kwargs,
                **extras,
            )
        self._dump_conductor_stats(f"stage 1 ({stage1_steps} steps)")
        self._dump_lora_stats(f"stage 1 ({stage1_steps} steps)")
        if use_distilled_lora:
            self.model._pause_distilled_lora_hooks()
        on_update_progress(stage1_steps, total_steps)
        print(f"[Ltx2 Sampler] stage 1 latents shape: {tuple(stage1_latents.shape)}, "
              f"stage 1 audio shape: {tuple(stage1_audio.shape) if stage1_audio is not None else None}")
        self._vram_log("after stage 1")

        # --- Upsample latents directly (no VAE round-trip) ---
        # The pipeline returned DENORMALIZED latents (output_type="latent" path
        # in pipeline_ltx2.py:1437-1441 calls _denormalize_latents). The latent
        # upsampler operates on UNNORMALIZED latents per its module docstring,
        # so feeding the denormalized stage-1 output directly is correct.
        with self._timed_phase("latent upsample"):
            self.model.latent_upsampler_to(self.train_device, scale=factor)
            try:
                upsampler_dtype = next(upsampler.parameters()).dtype
                upsampled_latents = upsampler(stage1_latents.to(upsampler_dtype))
            finally:
                self.model.latent_upsampler_to(self.temp_device, scale=factor)
            del stage1_latents
            torch_gc()
        print(f"[Ltx2 Sampler] upsampled latents shape: {tuple(upsampled_latents.shape)}")
        self._vram_log("after upsample")

        # --- Stage 2: high-res, partial denoise ---
        if use_distilled_lora:
            self.model.distilled_lora_strength = stage2_strength
            self.model._resume_distilled_lora_hooks()
        # CRITICAL: pass DENORMALIZED upsampled video latents + stage1 audio latents.
        # Video: pipeline normalizes → adds noise_scale noise → denoises.
        # Audio: pass stage1_audio so stage 2 refines the already-denoised audio
        #        rather than starting from random noise (3 steps from scratch = garbled).
        # Both use the same noise_scale so partial denoise is consistent.
        self._reset_conductor_stats()
        self._reset_lora_call_counter()
        with self._timed_phase(f"pipeline stage 2 ({len(_DISTILLED_STAGE2_SIGMAS)} steps @ {up_w}x{up_h})"), \
                sequential_cfg(self.model.transformer), \
                chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK), \
                attention_backend(self.model.transformer, _SAMPLING_ATTENTION_BACKEND):
            video_latents, audio_latents = pipeline(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                height=up_h,
                width=up_w,
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
                **extras,
            )
        self._dump_conductor_stats(f"stage 2 ({len(_DISTILLED_STAGE2_SIGMAS)} steps)")
        self._dump_lora_stats(f"stage 2 ({len(_DISTILLED_STAGE2_SIGMAS)} steps)")
        del upsampled_latents, stage1_audio
        self._vram_log("after stage 2 diffusion")

        # Offload transformer + connectors before VAE decode. distilled_lora_to
        # is a no-op for GPU (LoRA always lives in pinned CPU memory) — kept
        # here only so a hypothetical future "release pinning" can hook in.
        if use_distilled_lora:
            self.model._pause_distilled_lora_hooks()
        self.model.transformer_to(self.temp_device)
        self.model.connectors_to(self.temp_device)
        self.model.distilled_lora_to(self.temp_device)
        torch_gc()
        self._vram_log("after stage 2 transformer offload")

        video, audio = self._decode_video_and_audio(
            pipeline, video_latents, audio_latents, is_video,
        )
        del video_latents, audio_latents
        on_update_progress(total_steps, total_steps)
        self._vram_log("after stage 2 decode")
        return video, audio

    @torch.no_grad()
    def _decode_video_and_audio(
            self,
            pipeline,
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor | None,
            is_video: bool,
    ):
        """Manual decode of denormalized latents to (video frames, audio waveform).

        Mirrors the non-"latent" path of LTX2Pipeline.__call__ (lines 1467-1473)
        but is called by us AFTER moving the transformer + connectors off GPU,
        so VAE decode runs without 22+ GB of transformer + LoRA weights also
        sitting on the device. The LTX-2.3 video VAE has
        ``timestep_conditioning=False`` so the decode is straightforward —
        no noise injection at decode time.

        Inputs are denormalized (output_type="latent" from the pipeline already
        denormalizes via ``_denormalize_latents`` / ``_denormalize_audio_latents``).
        """
        # Video decode: bring video VAE to GPU, keep audio VAE + vocoder in RAM.
        self.model.vae_to(self.train_device)
        self.model.audio_vae_to(self.temp_device)
        torch_gc()
        self._vram_log("after VAE→GPU (pre video decode)")

        vae = pipeline.vae
        orig_vae_dtype = next(vae.parameters()).dtype
        if orig_vae_dtype != torch.bfloat16:
            vae.to(dtype=torch.bfloat16)
        try:
            video_pixels = vae.decode(video_latents.to(torch.bfloat16), return_dict=False)[0]
        finally:
            if orig_vae_dtype != torch.bfloat16:
                vae.to(dtype=orig_vae_dtype)
        del video_latents

        # Offload video VAE before audio decode.
        self.model.vae_to(self.temp_device)
        torch_gc()

        video = pipeline.video_processor.postprocess_video(
            video_pixels, output_type="np" if is_video else "pil",
        )
        del video_pixels

        # Audio decode — bring audio components to GPU only when needed.
        audio = None
        if audio_latents is not None and is_video:
            self.model.audio_vae_to(self.train_device)
            self.model.vocoder_to(self.train_device)
            torch_gc()
            audio_latents = audio_latents.to(pipeline.audio_vae.dtype)
            mel = pipeline.audio_vae.decode(audio_latents, return_dict=False)[0]
            del audio_latents
            audio = pipeline.vocoder(mel)
            del mel
            self.model.audio_vae_to(self.temp_device)
            self.model.vocoder_to(self.temp_device)
            torch_gc()

        return video, audio

    def _configure_vae_tiling(self, vae, tile_size: int = 256) -> None:
        """Spatial-only VAE tiling.

        tile_size=256 (default): each tile processes 256×256 pixels across all
        frames. Peak activation per tile ≈ 256²×121×128ch×2B ≈ 2 GB — safe on
        32 GB cards with the transformer already offloaded.

        tile_size=512 matches ComfyUI's defaults but peaks at ~8 GB per tile,
        which spills to shared memory on a 32 GB card.

        Overlap is always 64 px (stride = tile_size - 64).
        """
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        stride = tile_size - 64
        if hasattr(vae, "tile_sample_min_height"):
            vae.tile_sample_min_height = tile_size
        if hasattr(vae, "tile_sample_min_width"):
            vae.tile_sample_min_width = tile_size
        if hasattr(vae, "tile_sample_stride_height"):
            vae.tile_sample_stride_height = stride
        if hasattr(vae, "tile_sample_stride_width"):
            vae.tile_sample_stride_width = stride
        # No temporal tiling — process all frames per spatial tile
        if hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = False
        if hasattr(vae, "use_framewise_encoding"):
            vae.use_framewise_encoding = False

        print(f"[Ltx2 VAE] spatial tiling: tile={tile_size}px stride={stride}px (64px overlap), no temporal tiling")

    def _pick_upsampler(self, mode: LtxMultiScaleMode):
        """Return the model's upsampler matching the multi-scale mode, or None."""
        if mode == LtxMultiScaleMode.X1_5:
            return self.model.latent_upsampler_x1_5
        if mode == LtxMultiScaleMode.X2:
            return self.model.latent_upsampler_x2
        return None

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
            multi_scale_mode: LtxMultiScaleMode,
            vae_tiling: bool,
            vae_tile_size: int = 256,
            stage1_strength: float = 0.3,
            stage2_strength: float = 0.6,
            use_distilled_lora: bool = True,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
            on_update_preview: Callable[[int, int, torch.Tensor], None] | None = None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            self._vram_log("entry")
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # 1. Encode prompts on the text encoder (Gemma3-12B), then offload it.
            with self._timed_phase("TE→GPU"):
                self.model.text_encoder_to(self.train_device)
            self._vram_log("after TE→GPU")
            with self._timed_phase("encode_text (positive + optional negative)"):
                prompt_embeds, prompt_mask = self.model.encode_text(prompt, self.train_device)
                prompt_embeds, prompt_mask = self._pad_embeds(prompt_embeds, prompt_mask)
                if cfg_scale != 1.0:
                    neg_embeds, neg_mask = self.model.encode_text(
                        negative_prompt or "", self.train_device,
                    )
                    neg_embeds, neg_mask = self._pad_embeds(neg_embeds, neg_mask)
                else:
                    neg_embeds, neg_mask = None, None
            self._vram_log("after prompt encode")
            with self._timed_phase("TE→CPU + gc"):
                self.model.text_encoder_to(self.temp_device)
                torch_gc()
            self._vram_log("after TE→CPU + gc")

            # 2. Move diffusion components to GPU.
            # Connectors (~500 MB) stay on GPU during diffusion — pre-computing them
            # outside the pipeline caused quality regressions due to quantization
            # context differences; 500 MB is not worth the risk.
            with self._timed_phase("components→GPU (connectors + transformer + LoRA pin)"):
                self.model.connectors_to(self.train_device)
                self.model.transformer_to(self.train_device)
                if use_distilled_lora:
                    self.model.distilled_lora_to(self.train_device)
                torch_gc()
            self._vram_log("after diffusion components→GPU")

            pipeline = self.model.create_pipeline()
            # pipeline.device returns vae.device (vae is first in the __init__ signature).
            # With VAE on CPU, _execution_device = CPU → prepare_latents creates CPU latents
            # → CUDA generator type mismatch. Override on a throwaway subclass so the
            # pipeline creates latents on the correct device without keeping VAE on GPU.
            _td = self.train_device
            pipeline.__class__ = type(
                pipeline.__class__.__name__,
                (pipeline.__class__,),
                {"_execution_device": property(lambda self: _td)},
            )
            pipeline.set_progress_bar_config(disable=False)
            if vae_tiling:
                self._configure_vae_tiling(pipeline.vae, tile_size=vae_tile_size)
            else:
                if hasattr(pipeline.vae, "disable_tiling"):
                    pipeline.vae.disable_tiling()
                print("[Ltx2 VAE tiling] disabled (full-frame decode)")

            extras = dict(_LTX_2_3_INFERENCE_EXTRAS)
            is_video = num_frames > 1

            # 5. Two-stage flow if multi-scale mode is enabled and an upsampler
            #    is available; fall back to single-stage otherwise. We require
            #    the corresponding upsampler to be loaded — if not, the caller
            #    should have grouped the warning at setup time.
            upsampler = self._pick_upsampler(multi_scale_mode) if multi_scale_mode.is_two_stage() else None
            if multi_scale_mode.is_two_stage() and upsampler is None:
                print(f"[Ltx2 Sampler] multi-scale mode {multi_scale_mode} requested "
                      f"but the upsampler is not loaded — falling back to FULL_SIZE")

            if multi_scale_mode.is_two_stage() and upsampler is not None:
                video, audio = self._sample_two_stage(
                    pipeline=pipeline,
                    upsampler=upsampler,
                    multi_scale_mode=multi_scale_mode,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    neg_embeds=neg_embeds,
                    neg_mask=neg_mask,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    diffusion_steps=diffusion_steps,
                    cfg_scale=cfg_scale,
                    stage1_strength=stage1_strength,
                    stage2_strength=stage2_strength,
                    use_distilled_lora=use_distilled_lora,
                    extras=extras,
                    is_video=is_video,
                    generator=generator,
                    on_update_progress=on_update_progress,
                )
            else:
                # Single-stage: diffuse with output_type="latent", then offload
                # the transformer + connectors before VAE decode so the heavy
                # decode step has the GPU to itself. (LoRA already lives in
                # pinned CPU memory; nothing to offload there.)
                if use_distilled_lora:
                    self.model.distilled_lora_strength = stage1_strength
                    self.model._resume_distilled_lora_hooks()
                self._reset_conductor_stats()
                self._reset_lora_call_counter()
                with self._timed_phase(f"pipeline single-stage ({diffusion_steps} steps @ {width}x{height}, cfg={cfg_scale})"), \
                        sequential_cfg(self.model.transformer), \
                        chunked_ffn(self.model.transformer, _SAMPLING_FFN_CHUNK), \
                        attention_backend(self.model.transformer, _SAMPLING_ATTENTION_BACKEND):
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
                        **extras,
                    )
                self._dump_conductor_stats(f"single-stage ({diffusion_steps} steps)")
                self._dump_lora_stats(f"single-stage ({diffusion_steps} steps)")
                self._vram_log("after diffusion (latent output)")
                if use_distilled_lora:
                    self.model._pause_distilled_lora_hooks()
                with self._timed_phase("transformer+connectors→CPU + gc"):
                    self.model.transformer_to(self.temp_device)
                    self.model.connectors_to(self.temp_device)
                    self.model.distilled_lora_to(self.temp_device)
                    torch_gc()
                self._vram_log("after transformer+connectors offload")
                with self._timed_phase("VAE decode"):
                    video, audio = self._decode_video_and_audio(
                        pipeline, video_latents, audio_latents, is_video,
                    )
                del video_latents, audio_latents
                on_update_progress(diffusion_steps, diffusion_steps)

            # 5. Capture audio for muxing into the mp4. (Stage-2 audio when
            #    two-stage; otherwise the single-stage audio.)
            audio_waveform = None
            audio_sample_rate = None
            if audio is not None and is_video:
                try:
                    audio_waveform = audio[0].float().cpu()
                    audio_sample_rate = int(self.model.vocoder.config.output_sampling_rate)
                except Exception as e:
                    print(f"[Ltx2Sampler] could not capture audio for muxing: {e}")

            # 6. Free prompt embeds + send remaining components back to temp_device.
            #    transformer + connectors already on temp_device (offloaded before VAE decode).
            #    VAE/vocoder already on temp_device (managed in _decode_video_and_audio).
            del prompt_embeds, prompt_mask, neg_embeds, neg_mask, audio
            self.model.distilled_lora_to(self.temp_device)
            self.model.latent_upsampler_to(self.temp_device)
            torch_gc()

            if is_video:
                # `video` is a numpy array of shape (B, T, H, W, C) in [0, 1].
                frames_np = video[0]
                if frames_np.dtype != np.uint8:
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
                # Single-frame mode returns a list of PIL images.
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
            on_update_preview: Callable[[int, int, torch.Tensor], None] | None = None,
    ):
        frame_rate = getattr(sample_config, "frame_rate", None)
        if frame_rate is None or frame_rate <= 0:
            frame_rate = _DEFAULT_FRAME_RATE

        multi_scale_mode = getattr(sample_config, "ltx_multi_scale_mode", None) or LtxMultiScaleMode.FULL_SIZE
        vae_tiling = getattr(sample_config, "ltx_vae_tiling", None)
        if vae_tiling is None:
            vae_tiling = True
        vae_tile_size = int(getattr(sample_config, "ltx_vae_tile_size", None) or 256)

        use_distilled_lora = getattr(sample_config, "ltx_use_distilled_lora", None)
        if use_distilled_lora is None:
            use_distilled_lora = True
        stage1_strength = float(getattr(sample_config, "ltx_distilled_lora_stage1_strength", None) or 0.3) if use_distilled_lora else 0.0
        stage2_strength = float(getattr(sample_config, "ltx_distilled_lora_stage2_strength", None) or 0.6) if use_distilled_lora else 0.0

        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, _BUCKET_DIVISIBILITY),
            width=self.quantize_resolution(sample_config.width, _BUCKET_DIVISIBILITY),
            num_frames=self._quantize_frames(sample_config.frames),
            frame_rate=float(frame_rate),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            multi_scale_mode=multi_scale_mode,
            vae_tiling=bool(vae_tiling),
            vae_tile_size=vae_tile_size,
            stage1_strength=float(stage1_strength),
            stage2_strength=float(stage2_strength),
            use_distilled_lora=use_distilled_lora,
            on_update_progress=on_update_progress,
            on_update_preview=on_update_preview,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, Ltx2Sampler, ModelType.LTX_2_3)
