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
            initial_latents: torch.Tensor | None = None,
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
        # I2V stage 1 — pass the pre-prepared 5D latents (image at
        # frame 0, noise elsewhere) to the T2V pipeline as ``latents=``.
        # Per-step frame-0 re-clamp is handled sampler-side via
        # callback_on_step_end. Stage 2 starts from upsampled stage-1
        # latents (existing behavior) — frame 0 stays close to the
        # conditioning image without re-clamping.
        _stage1_i2v_kwargs = {}
        if initial_latents is not None:
            _stage1_i2v_kwargs["latents"] = initial_latents
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
                **_stage1_i2v_kwargs,
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
            video_pixels = self._decode_video_with_oom_fallback(
                vae, video_latents.to(torch.bfloat16))
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

    def _configure_vae_tiling(self, vae, tile_size: int = 512,
                              temporal_tile_size: int = 64,
                              temporal_overlap: int = 24) -> None:
        """Spatial + framewise (temporal-chunked) VAE tiling.

        Matches ComfyUI's "VAE Decode (Tiled)" gentle-on-GPU behaviour:
        spatial tiling is bounded by ``tile_size`` × ``tile_size`` pixels
        with 64 px overlap, AND framewise (temporal) tiling is enabled
        so each spatial tile processes only ``tile_sample_min_num_frames``
        pixel frames at a time instead of the full temporal volume.

        ``diffusers.AutoencoderKLLTX2Video`` defaults ``use_framewise_*``
        to ``False`` (verified at autoencoder_kl_ltx2.py:1173-1174).
        Without explicitly setting them to ``True``, ``_decode()`` falls
        through to ``tiled_decode`` which is spatial-only — each 256×256
        spatial tile then decodes ALL temporal frames in one shot, giving
        the 15-19 GB peaks seen on 361-frame LTX-2.3 samples.

        With framewise enabled, ``_decode()`` routes to
        ``_temporal_tiled_decode`` which iterates over temporal slices of
        ``tile_sample_min_num_frames`` pixel frames each, recursing into
        the spatial ``tiled_decode`` per slice.  Peak per spatial tile
        scales with the temporal slice depth (16 pixel frames by default)
        instead of the full video length.

        Spatial overlap is always 64 px (stride = tile_size - 64).
        Temporal slice/stride stay at the diffusers defaults
        (16-frame slice, 8-frame stride = 50% overlap).
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
        # Force framewise on — diffusers default is False, which makes
        # tiled_decode the spatial-only branch and explodes VRAM peak.
        if hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = True
        if hasattr(vae, "use_framewise_encoding"):
            vae.use_framewise_encoding = True
        # Temporal tile sizing (pixel frames). Matches ComfyUI LTX2_SM
        # TilingConfig.default(): 64-frame tile, 40-frame stride (24 overlap).
        temporal_stride = max(1, int(temporal_tile_size) - int(temporal_overlap))
        if hasattr(vae, "tile_sample_min_num_frames"):
            vae.tile_sample_min_num_frames = int(temporal_tile_size)
        if hasattr(vae, "tile_sample_stride_num_frames"):
            vae.tile_sample_stride_num_frames = temporal_stride

        print(f"[Ltx2 VAE] tiling: spatial tile={tile_size}px stride={stride}px (64px overlap), "
              f"temporal tile={temporal_tile_size}f stride={temporal_stride}f ({temporal_overlap}f overlap), framewise=ON")

    # --- VAE decode memory fitting ----------------------------------------
    # How ComfyUI stays gentle on consumer cards (comfy/sd.py VAE.decode +
    # model_management.get_free_memory): it does NOT rely on catching OOM. It
    # ESTIMATES the decode's peak from the latent shape, compares against the
    # REAL physical free VRAM from torch.cuda.mem_get_info(), and sizes the work
    # to fit before decoding. That matters on Windows, where the NVIDIA driver's
    # shared-memory fallback silently spills oversized allocations into host RAM
    # instead of raising OutOfMemoryError — so a try/except OOM ladder never
    # fires, it just crawls. We mirror ComfyUI: fit tiling to physical-free VRAM
    # up front (_fit_tiling_to_free_vram), reuse the shrink ladder, and keep the
    # OOM catch only as a backstop for platforms that do fault.

    _VAE_TILE_ATTRS = (
        "tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_num_frames",
        "tile_sample_stride_height", "tile_sample_stride_width", "tile_sample_stride_num_frames",
        "use_tiling", "use_framewise_decoding",
    )
    # ComfyUI's LTX VAE peak heuristic (comfy/sd.py:663, lightricks branch):
    #   bytes ~= 1200 * T_lat * H_lat * W_lat * (8*8*8) * dtype_size
    # _VAE_DECODE_FUDGE biases the estimate conservative (diffusers' framewise
    # path holds blend buffers + the full output accumulator on top of one tile's
    # activations, which the bare coefficient under-counts). Tune against the
    # logged "est vs free" line / the CUDA memory profiler on real hardware.
    _VAE_DECODE_BYTES_COEF = 1200 * (8 * 8 * 8)
    _VAE_DECODE_FUDGE = 2.0
    # Never let a single spatial tile approach the full frame (≈full-frame decode
    # is what spills to shared memory on Windows). 512px keeps several tiles even
    # at 1080p while staying well clear of the spill threshold.
    _VAE_SAFE_TILE_CEILING = 512

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        """True for CUDA out-of-memory errors across torch versions."""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _snapshot_vae_tiling(self, vae) -> dict:
        return {a: getattr(vae, a) for a in self._VAE_TILE_ATTRS if hasattr(vae, a)}

    def _restore_vae_tiling(self, vae, snapshot: dict) -> None:
        for a, v in snapshot.items():
            setattr(vae, a, v)

    def _shrink_vae_tiling(self, vae, spatial_floor: int = 128, temporal_floor: int = 16) -> bool:
        """Take one step down the OOM-fallback ladder. Returns False if nothing
        left to shrink (caller should then re-raise the OOM).

        Order, largest VRAM lever first (peak ~ tile_h * tile_w * tile_frames):
          0. enable tiling if it was off (full-frame decode that OOM'd),
          1. halve the spatial tile down to ``spatial_floor``,
          2. halve the temporal tile down to ``temporal_floor``.
        Spatial overlap follows _configure_vae_tiling's 64px convention
        (capped so stride stays positive at the floor); temporal overlap keeps
        its current fraction.
        """
        # Step 0: tiling disabled entirely -> turn it on (+ framewise) and retry.
        if hasattr(vae, "use_tiling") and not vae.use_tiling:
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            if hasattr(vae, "use_framewise_decoding"):
                vae.use_framewise_decoding = True
            print("[Ltx2 VAE] decode OOM — enabling tiling (was full-frame), retrying")
            return True

        # Step 1: halve spatial tile (height == width here) down to the floor.
        cur_spatial = getattr(vae, "tile_sample_min_height", None)
        if cur_spatial is not None and cur_spatial > spatial_floor:
            new_spatial = max(spatial_floor, cur_spatial // 2)
            overlap = min(64, new_spatial // 2)
            stride = new_spatial - overlap
            vae.tile_sample_min_height = new_spatial
            vae.tile_sample_min_width = new_spatial
            vae.tile_sample_stride_height = stride
            vae.tile_sample_stride_width = stride
            print(f"[Ltx2 VAE] decode OOM — shrinking spatial tile {cur_spatial}px -> "
                  f"{new_spatial}px (stride {stride}px), retrying")
            return True

        # Step 2: halve temporal tile down to the floor, preserving overlap.
        cur_frames = getattr(vae, "tile_sample_min_num_frames", None)
        if cur_frames is not None and cur_frames > temporal_floor:
            cur_stride = getattr(vae, "tile_sample_stride_num_frames", cur_frames)
            cur_overlap = max(0, cur_frames - cur_stride)
            new_frames = max(temporal_floor, cur_frames // 2)
            overlap = min(cur_overlap, new_frames // 2)
            stride = max(1, new_frames - overlap)
            vae.tile_sample_min_num_frames = new_frames
            vae.tile_sample_stride_num_frames = stride
            print(f"[Ltx2 VAE] decode OOM — shrinking temporal tile {cur_frames}f -> "
                  f"{new_frames}f (stride {stride}f), retrying")
            return True

        return False

    def _estimate_decode_bytes(self, t_lat: int, h_lat: int, w_lat: int) -> int:
        """ComfyUI's LTX-VAE peak-memory heuristic for a (latent) decode volume.
        ``2`` is the bf16 dtype size — decode always runs in bf16 here."""
        return int(self._VAE_DECODE_BYTES_COEF * t_lat * h_lat * w_lat * 2 * self._VAE_DECODE_FUDGE)

    def _fit_tiling_to_free_vram(self, vae, latents: torch.Tensor) -> None:
        """Size VAE tiling to fit *physical* free VRAM before decoding.

        Mirrors ComfyUI's proactive approach (estimate vs ``mem_get_info`` free)
        rather than reacting to an OOM that the Windows driver's shared-memory
        fallback would suppress. If the full volume fits real free VRAM, leave
        the configured tiling alone. Otherwise enable tiling, cap any oversized
        spatial tile to ``_VAE_SAFE_TILE_CEILING``, then shrink (reusing
        :meth:`_shrink_vae_tiling`) until the per-tile estimate fits.
        """
        dev = latents.device
        if dev.type != "cuda":
            return
        try:
            free, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            return  # can't measure -> rely on the OOM backstop

        GB = 1024 ** 3
        margin = int(1.5 * GB)
        sc = int(getattr(vae, "spatial_compression_ratio", 32) or 32)
        tc = int(getattr(vae, "temporal_compression_ratio", 8) or 8)
        t, h, w = int(latents.shape[-3]), int(latents.shape[-2]), int(latents.shape[-1])

        est_full = self._estimate_decode_bytes(t, h, w)
        print(f"[Ltx2 VAE] decode fit: est full-frame={est_full / GB:.1f} GB, "
              f"free(physical)={free / GB:.1f} GB (margin {margin / GB:.1f} GB)")
        if est_full <= free - margin:
            return  # whole volume fits real VRAM — no spill risk, keep config

        # Tiling required. Ensure it (and framewise) are on.
        if hasattr(vae, "use_tiling") and not vae.use_tiling:
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            if hasattr(vae, "use_framewise_decoding"):
                vae.use_framewise_decoding = True
            print("[Ltx2 VAE] est exceeds free VRAM — enabling tiling")

        # Cap an oversized spatial tile away from full-frame (the Windows spill
        # trigger), preserving the 64px-overlap convention.
        cur = int(getattr(vae, "tile_sample_min_height", self._VAE_SAFE_TILE_CEILING))
        if cur > self._VAE_SAFE_TILE_CEILING:
            ceil = self._VAE_SAFE_TILE_CEILING
            stride = ceil - min(64, ceil // 2)
            vae.tile_sample_min_height = ceil
            vae.tile_sample_min_width = ceil
            vae.tile_sample_stride_height = stride
            vae.tile_sample_stride_width = stride
            print(f"[Ltx2 VAE] capping spatial tile {cur}px -> {ceil}px (stride {stride}px)")

        # The full decoded video is held on-device regardless of tiling; subtract
        # it from the per-tile budget.
        out_bytes = int(((t - 1) * tc + 1) * (h * sc) * (w * sc) * 3 * 2)
        tile_budget = free - margin - out_bytes

        def per_tile_est() -> int:
            th = int(getattr(vae, "tile_sample_min_height", h * sc))
            tw = int(getattr(vae, "tile_sample_min_width", w * sc))
            tf = int(getattr(vae, "tile_sample_min_num_frames", (t - 1) * tc + 1))
            tl = min(t, max(1, tf // tc))
            hl = min(h, max(1, th // sc))
            wl = min(w, max(1, tw // sc))
            return self._estimate_decode_bytes(tl, hl, wl)

        guard = 0
        while per_tile_est() > tile_budget and guard < 16:
            if not self._shrink_vae_tiling(vae):
                print("[Ltx2 VAE] at minimum tile size; per-tile estimate still "
                      f"{per_tile_est() / GB:.2f} GB > budget {max(0, tile_budget) / GB:.2f} GB "
                      "— consider lowering resolution/frame count")
                break
            guard += 1
        print(f"[Ltx2 VAE] per-tile estimate {per_tile_est() / GB:.2f} GB vs tile budget "
              f"{max(0, tile_budget) / GB:.2f} GB")

    def _decode_video_with_oom_fallback(self, vae, latents_bf16: torch.Tensor) -> torch.Tensor:
        """``vae.decode`` fitted to physical free VRAM, with an OOM backstop.

        First sizes tiling to fit real free VRAM (:meth:`_fit_tiling_to_free_vram`)
        — the Windows-safe primary mechanism, since the driver's shared-memory
        fallback would otherwise suppress the OOM. If a CUDA OOM still fires
        (non-Windows, or an under-estimate), the tile config is shrunk one rung
        (:meth:`_shrink_vae_tiling`), the allocator cleared, and the decode
        retried. Non-OOM errors propagate. Tile attrs are restored afterwards.
        """
        # Experimental: OT-resident chunked streaming decode (ComfyUI-style single
        # low-VRAM pass). Off by default; opt in via LTX2_VAE_CHUNKED_DECODE=1. Any
        # error (unsupported VAE, residual-alignment/shape mismatch) falls through to
        # the tiled path below, so it can never regress the default decode.
        try:
            from modules.modelLoader.ltx2 import _vae_chunked_decode as _ccd
            if _ccd.is_enabled():
                try:
                    # CPU output buffer (temp_device) keeps the large decoded video off
                    # the GPU so cuDNN has room for conv workspaces. postprocess_video
                    # handles CPU tensors. Returned to GPU only if a downstream step needs it.
                    return _ccd.chunked_decode(
                        vae, latents_bf16,
                        output_device=self.temp_device,
                        output_dtype=latents_bf16.dtype,
                    )
                except Exception as _exc:
                    print(f"[Ltx2 VAE] chunked decode unavailable "
                          f"({type(_exc).__name__}: {_exc}); falling back to tiled decode")
        except Exception:
            pass

        snapshot = self._snapshot_vae_tiling(vae)
        # Tile-by-tile empty_cache hook: diffusers' tiled VAE decoder walks
        # (spatial × temporal) tiles in nested loops. PyTorch's CUDA caching
        # allocator keeps freed tile buffers around as cached slabs, so VRAM
        # climbs monotonically across tiles even though each tile's working
        # set is small. The hook fires after every decoder forward (one per
        # tile) and reclaims the slabs back to the driver — keeps VRAM flat
        # instead of climbing into shared-GPU-memory (page-to-RAM) territory.
        # Complements _fit_tiling_to_free_vram (which sizes the tiles); this
        # keeps the per-tile residency from accumulating during the walk.
        _empty_cache_handle = None
        if torch.cuda.is_available() and hasattr(vae, "decoder"):
            def _post_tile_empty_cache(_module, _inputs, _output):
                torch.cuda.empty_cache()
                return _output
            _empty_cache_handle = vae.decoder.register_forward_hook(_post_tile_empty_cache)
        try:
            self._fit_tiling_to_free_vram(vae, latents_bf16)
            while True:
                try:
                    return vae.decode(latents_bf16, return_dict=False)[0]
                except Exception as exc:
                    if not self._is_oom(exc):
                        raise
                    if not self._shrink_vae_tiling(vae):
                        print("[Ltx2 VAE] decode OOM at minimum tile size — cannot shrink "
                              "further, re-raising")
                        raise
                    torch_gc()
        finally:
            if _empty_cache_handle is not None:
                _empty_cache_handle.remove()
            self._restore_vae_tiling(vae, snapshot)

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
            vae_tile_size: int = 512,
            vae_temporal_tile_size: int = 64,
            vae_temporal_overlap: int = 24,
            stage1_strength: float = 0.3,
            stage2_strength: float = 0.6,
            use_distilled_lora: bool = True,
            initial_latents: torch.Tensor | None = None,
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
            # Sample-prompt text cache: sample prompts don't change across
            # intervals (or runs), so the TE (12B) round-trip — TE→GPU + encode +
            # TE→CPU+gc plus the GPU-residency dance — is wasted after the first
            # encode. Cache the (padded) embeds/masks keyed by the prompt pair AND
            # a fingerprint of the loaded TE weights, so changing/fixing the TE
            # (different file, norm fix, embedding fix) auto-invalidates the cache
            # — no stale-cache trap. In-memory (per run) + on-disk across runs at
            # <cache_dir>/sample_text_cache when the trainer wires the dir.
            import os as _os, hashlib as _hashlib
            _use_neg = cfg_scale != 1.0

            # TE fingerprint: cheap deterministic hash of a few loaded TE weights,
            # computed once per run. Changes if the TE weights change at all.
            _te_fp = getattr(self.model, "_te_fingerprint", None)
            if _te_fp is None:
                try:
                    _te = self.model.text_encoder
                    _tm = getattr(_te, "model", _te)
                    _ew = _tm.embed_tokens.weight
                    _fp_parts = (
                        tuple(_ew.shape),
                        round(float(_ew[:8].float().sum().item()), 4),
                        round(float(_ew[-8:].float().sum().item()), 4),
                        round(float(_tm.norm.weight.float().sum().item()), 4),
                        round(float(_tm.layers[0].input_layernorm.weight.float().sum().item()), 4),
                    )
                except Exception:
                    _fp_parts = ("te-fingerprint-unavailable",)
                _te_fp = self.model._te_fingerprint = _hashlib.sha1(repr(_fp_parts).encode()).hexdigest()[:16]

            _key = _hashlib.sha1(
                repr((prompt, negative_prompt or "", _use_neg, _te_fp)).encode()).hexdigest()[:24]
            _mem = getattr(self.model, "_sample_text_cache", None)
            if _mem is None:
                _mem = self.model._sample_text_cache = {}
            _disk_dir = getattr(self.model, "_sample_text_cache_dir", None)
            _disk_path = _os.path.join(_disk_dir, _key + ".pt") if _disk_dir else None

            _cached = _mem.get(_key)
            _hit_src = "memory"
            if _cached is None and _disk_path and _os.path.isfile(_disk_path):
                try:
                    _cached = torch.load(_disk_path, map_location="cpu")
                    _mem[_key] = _cached
                    _hit_src = "disk"
                except Exception as _e:
                    print(f"[Ltx2 Sampler] text cache: disk load failed ({_e}); re-encoding", flush=True)
                    _cached = None

            if _cached is not None:
                _pe, _pm, _ne, _nm = _cached
                prompt_embeds = _pe.to(self.train_device)
                prompt_mask = _pm.to(self.train_device)
                neg_embeds = _ne.to(self.train_device) if _ne is not None else None
                neg_mask = _nm.to(self.train_device) if _nm is not None else None
                print(f"[Ltx2 Sampler] text cache HIT ({_hit_src}, te={_te_fp}) — TE skipped", flush=True)
            else:
                with self._timed_phase("TE→GPU"):
                    self.model.text_encoder_to(self.train_device)
                self._vram_log("after TE→GPU")
                with self._timed_phase("encode_text (positive + optional negative)"):
                    prompt_embeds, prompt_mask = self.model.encode_text(prompt, self.train_device)
                    prompt_embeds, prompt_mask = self._pad_embeds(prompt_embeds, prompt_mask)
                    if _use_neg:
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
                _payload = (
                    prompt_embeds.detach().to("cpu"),
                    prompt_mask.detach().to("cpu"),
                    neg_embeds.detach().to("cpu") if neg_embeds is not None else None,
                    neg_mask.detach().to("cpu") if neg_mask is not None else None,
                )
                _mem[_key] = _payload
                if _disk_path:
                    try:
                        _os.makedirs(_disk_dir, exist_ok=True)
                        torch.save(_payload, _disk_path)
                        print(f"[Ltx2 Sampler] text cache MISS — encoded + saved to disk (te={_te_fp})", flush=True)
                    except Exception as _e:
                        print(f"[Ltx2 Sampler] text cache MISS — encoded (disk save failed: {_e})", flush=True)
                else:
                    print("[Ltx2 Sampler] text cache MISS — encoded (in-memory only; no cache_dir wired)", flush=True)

            # 2. Move diffusion components to GPU.
            # Connectors (~500 MB) stay on GPU during diffusion — pre-computing them
            # outside the pipeline caused quality regressions due to quantization
            # context differences; 500 MB is not worth the risk.
            # NOTE: split into per-component phases to localize the slow part.
            # Suspected SATA-resident diffusers (mmap'd connectors paged in on
            # first .to(GPU)) vs NVMe GGUF transformer.
            with self._timed_phase("connectors→GPU"):
                self.model.connectors_to(self.train_device)
            with self._timed_phase("transformer→GPU"):
                self.model.transformer_to(self.train_device)
            if use_distilled_lora:
                with self._timed_phase("distilled_lora pin→GPU"):
                    self.model.distilled_lora_to(self.train_device)
            with self._timed_phase("components→GPU gc"):
                torch_gc()
            self._vram_log("after diffusion components→GPU")

            # I2V via the regular T2V pipeline + sampler-side frame-0
            # clamp callback. Diffusers' LTX2ImageToVideoPipeline uses a
            # per-token timestep that allocates ~2 GB of adaln modulation
            # per transformer block — fine on 80+ GB datacenter cards,
            # but ~2× the VRAM of T2V at 1024p which doesn't fit on 16-32
            # GB cards. ComfyUI/kijai's I2V pre-encodes the image, splices
            # it into frame 0 of the latent grid, and re-clamps frame 0
            # after every scheduler step using a scalar timestep. Same
            # VRAM as T2V. We pass ``initial_latents`` (a 5D pre-noised
            # latent prepared sampler-side: image at frame 0, noise
            # elsewhere) and let the sampler-side wrapper inject a
            # callback_on_step_end that does the per-step re-clamp.
            pipeline = self.model.create_pipeline()
            _initial_latents = initial_latents
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
                self._configure_vae_tiling(
                    pipeline.vae,
                    tile_size=vae_tile_size,
                    temporal_tile_size=vae_temporal_tile_size,
                    temporal_overlap=vae_temporal_overlap,
                )
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
                    initial_latents=_initial_latents,
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
                # If I2V (initial_latents provided), pass them as ``latents=``
                # to T2V. T2V's prepare_latents 5D branch normalizes + packs.
                # Per-step frame-0 re-clamp is handled sampler-side via
                # callback_on_step_end (see ltx/backend.py).
                _single_stage_kwargs = {}
                if _initial_latents is not None:
                    _single_stage_kwargs["latents"] = _initial_latents
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
                        **_single_stage_kwargs,
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
                    # Per-frame conversion into a pre-allocated uint8 buffer.
                    # The naive ``(np.clip(x,0,1)*255).round().astype(uint8)``
                    # chain on a 241×H×W×3 fp32 video spawns three transient
                    # ~4.4 GB fp32 arrays before the ~1.1 GB uint8 lands —
                    # peaks past 13 GB and OOMs tight systems. ComfyUI's
                    # pattern: pre-alloc output, loop frames, scale + cast in
                    # tight per-frame steps so the working set is one frame.
                    T = frames_np.shape[0]
                    out = np.empty(frames_np.shape, dtype=np.uint8)
                    for t in range(T):
                        scratch = np.clip(frames_np[t], 0.0, 1.0)
                        scratch *= 255.0
                        np.round(scratch, out=scratch)
                        out[t] = scratch.astype(np.uint8)
                        del scratch
                    # Drop the fp32 source ASAP — ``video`` holds the buffer
                    # ``frames_np`` views into; both refs must go to free the
                    # ~4.4 GB before torch.from_numpy doubles nothing (it views).
                    del frames_np
                    video = None
                    frames_np = out
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
        vae_tile_size = int(getattr(sample_config, "ltx_vae_tile_size", None) or 512)
        vae_temporal_tile_size = int(getattr(sample_config, "ltx_vae_temporal_tile_size", None) or 64)
        vae_temporal_overlap = int(getattr(sample_config, "ltx_vae_temporal_overlap", None) or 24)

        use_distilled_lora = getattr(sample_config, "ltx_use_distilled_lora", None)
        if use_distilled_lora is None:
            use_distilled_lora = True
        stage1_strength = float(getattr(sample_config, "ltx_distilled_lora_stage1_strength", None) or 0.3) if use_distilled_lora else 0.0
        stage2_strength = float(getattr(sample_config, "ltx_distilled_lora_stage2_strength", None) or 0.6) if use_distilled_lora else 0.0

        # I2V: caller pre-prepares 5D ``initial_latents`` (image at frame 0,
        # noise elsewhere) sampler-side and stashes them on
        # ``sample_config.ltx_initial_latents``. The sampler-side wrapper
        # also injects a callback_on_step_end that re-clamps frame 0 to
        # the encoded image latent at each step's sigma. ``None``/missing
        # → regular T2V flow.
        initial_latents = getattr(sample_config, "ltx_initial_latents", None)

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
            vae_temporal_tile_size=vae_temporal_tile_size,
            vae_temporal_overlap=vae_temporal_overlap,
            stage1_strength=float(stage1_strength),
            stage2_strength=float(stage2_strength),
            use_distilled_lora=use_distilled_lora,
            initial_latents=initial_latents,
            on_update_progress=on_update_progress,
            on_update_preview=on_update_preview,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, Ltx2Sampler, ModelType.LTX_2_3)
