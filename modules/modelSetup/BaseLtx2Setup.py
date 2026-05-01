import os
from abc import ABCMeta

from modules.model.Ltx2Model import Ltx2Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import enable_checkpointing_for_ltx_transformer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert.lora.convert_ltx2_lora import (
    convert_ltx2_lora_original_to_diffusers,
    normalize_lora_ab_to_down_up,
    pair_lora_down_up,
)
from modules.util.dtype_util import create_autocast_context
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers import LTX2Pipeline
from modules.util.torch_util import torch_gc

import huggingface_hub
from safetensors.torch import load_file


class BaseLtx2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta,
):
    """Base setup for LTX-2.3 (video flow-matching, T2V).

    Implements predict() and calculate_loss() for flow-matching training.
    Audio branch receives zero latents (T2V preset; no audio loss computed).
    Distilled LoRA is loaded at setup time and paused during training;
    re-applied at sample time via _resume_distilled_lora_hooks().
    """

    # Substring-matched layer filter presets for LTX-2.3.
    #
    # Architecture note: each LTX2VideoTransformerBlock contains parallel video
    # and audio paths. Layer names follow these patterns:
    #   Video-only:      .attn1.  .attn2.  .ff.
    #   Audio-only:      .audio_attn1.  .audio_attn2.  .audio_ff.
    #   Cross-modal:     .audio_to_video_attn.  .video_to_audio_attn.
    #
    # Dot-bounded patterns (".attn1." etc.) avoid matching the audio variants,
    # since "audio_attn1" does not contain the substring ".attn1." with a
    # leading dot. This removes the need for regex in the common case.
    #
    # Default "blocks" targets all transformer_blocks — ComfyUI-compatible scope.
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ff"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks"],
        "full": [],
        # Targets only the video self-attn, video cross-attn, and video FF layers.
        # Excludes audio_attn1/2, audio_ff, and the a2v/v2a cross-modal attentions.
        # Use this for T2V LoRA when you want minimal audio-branch disruption.
        "video": [".attn1.", ".attn2.", ".ff."],
    }

    # Cached per-run constants — populated lazily on first predict() call
    # so model.vae and noise_scheduler are guaranteed to be loaded.
    _training_timesteps: Tensor | None
    _linear_sigmas: Tensor | None
    _vae_mean: Tensor | None    # (1, C, 1, 1, 1) float32
    _vae_std: Tensor | None     # (1, C, 1, 1, 1) float32
    _vae_scale: float
    _connector_padding_side: str | None
    _patch_size: int
    _patch_size_t: int
    _audio_in_channels: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_timesteps = None
        self._linear_sigmas = None
        self._vae_mean = None
        self._vae_std = None
        self._vae_scale = 1.0
        self._connector_padding_side = None
        self._patch_size = 1
        self._patch_size_t = 1
        self._audio_in_channels = 128

    def create_parameters(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        return NamedParameterGroupCollection()

    def setup_optimizations(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # Wrap transformer blocks in OffloadCheckpointLayer — this is the
        # single OT entry point that wires torch.compile, layer-async-offload,
        # and (during training) gradient checkpointing on the per-block level.
        # Mirror BaseWanSetup's invariant: trigger when ANY of those features
        # is requested by the user's config.
        if (
            config.gradient_checkpointing.enabled()
            or config.compile
            or float(getattr(config, "layer_offload_fraction", 0.0) or 0.0) > 0.0
        ):
            model.transformer_offload_conductor = \
                enable_checkpointing_for_ltx_transformer(model.transformer, config)

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
            ],
            config.enable_autocast_cache,
        )

        # Load spatial upsamplers if their paths are configured. Done before
        # quantize_layers so they get the same compute_dtype + quant treatment
        # as the rest of the components. Only used when sample-time multi-scale
        # mode is X1_5 / X2 — but we load both up front since the user can
        # toggle modes per sample without reloading the model.
        x1_5_path = (getattr(config, "ltx_spatial_upsampler_x1_5_path", "") or "").strip()
        x2_path = (getattr(config, "ltx_spatial_upsampler_x2_path", "") or "").strip()
        if x1_5_path and model.latent_upsampler_x1_5 is None:
            try:
                model.latent_upsampler_x1_5 = _load_ltx_upsampler(x1_5_path, scale=1.5)
                print(f"[Ltx2 Upsampler] x1.5 loaded from {os.path.basename(x1_5_path)}")
            except Exception as e:
                print(f"[Ltx2 Upsampler] Failed to load x1.5 from '{x1_5_path}': {e}")
        if x2_path and model.latent_upsampler_x2 is None:
            try:
                model.latent_upsampler_x2 = _load_ltx_upsampler(x2_path, scale=2.0)
                print(f"[Ltx2 Upsampler] x2 loaded from {os.path.basename(x2_path)}")
            except Exception as e:
                print(f"[Ltx2 Upsampler] Failed to load x2 from '{x2_path}': {e}")

        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.audio_vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.connectors, self.train_device, model.train_dtype, config)
        quantize_layers(model.vocoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        if model.latent_upsampler_x1_5 is not None:
            quantize_layers(model.latent_upsampler_x1_5, self.train_device, model.train_dtype, config)
        if model.latent_upsampler_x2 is not None:
            quantize_layers(model.latent_upsampler_x2, self.train_device, model.train_dtype, config)

    def setup_model(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # Apply the distilled LoRA via forward-method patching — the same
        # pattern Wan uses for companion LoRA. Patches sit on top of whatever
        # quantization the transformer's Linears use (FP8, W8A8 int/float,
        # NF4, GGUF-A8) — orig_fwd handles the quantized matmul, the LoRA
        # delta is added at full BF16 precision at the output.
        # No precision loss, fast load, removable for sample-during-training.
        path = (config.ltx_distilled_lora_path or "").strip()
        if path:
            try:
                _apply_distilled_lora(model, path)
            except Exception as e:
                print(f"[BaseLtx2Setup] Failed to load distilled LoRA: {e}")

    def prepare_text_caching(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        # Move everything to CPU first, then bring only the text encoder to GPU.
        # The text encoder (Gemma3) is never trained for LTX-2.3 LoRA — it is
        # always offloaded after caching is complete.
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()

    def setup_train_device(
            self,
            model: Ltx2Model,
            config: TrainConfig,
    ):
        model.eval()
        model.to(self.temp_device)

    def predict(
            self,
            model: Ltx2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step + 1
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)

            # --- text embeddings (cached; never re-encoded during training) ---
            text_encoder_output = batch['text_encoder_1_hidden_state']
            tokens_mask = batch.get('tokens_mask_1')

            # --- lazily cache VAE normalization constants and scheduler tensors ---
            if self._vae_mean is None:
                vae = model.vae
                self._vae_mean = vae.latents_mean.to(dtype=torch.float32, device=self.train_device).view(1, -1, 1, 1, 1)
                self._vae_std = vae.latents_std.to(dtype=torch.float32, device=self.train_device).view(1, -1, 1, 1, 1)
                self._vae_scale = float(vae.config.scaling_factor)
                num_t = model.noise_scheduler.config.num_train_timesteps
                self._training_timesteps = torch.arange(1, num_t + 1, dtype=torch.long, device=self.train_device)
                self._linear_sigmas = self._training_timesteps.float() / num_t
                self._connector_padding_side = (
                    getattr(model.tokenizer, "padding_side", "left") if model.tokenizer is not None else "left"
                )
                self._patch_size = getattr(model.transformer.config, 'patch_size', 1)
                self._patch_size_t = getattr(model.transformer.config, 'patch_size_t', 1)
                self._audio_in_channels = getattr(model.transformer.config, 'audio_in_channels', 128)

            latent_image = batch['latent_image']
            if latent_image.ndim == 4:
                latent_image = latent_image.unsqueeze(2)  # (B,C,H,W) → (B,C,1,H,W)

            normalized_latent = (latent_image.float() - self._vae_mean) * self._vae_scale / self._vae_std

            # --- noise & timestep (spatial domain before packing) ---
            latent_noise = self._create_noise(normalized_latent, config, generator)

            num_train_timesteps = self._training_timesteps.shape[0]
            timestep = self._get_timestep_discrete(
                num_train_timesteps,
                deterministic,
                generator,
                normalized_latent.shape[0],
                config,
            )

            noisy_latent, _ = self._add_noise_discrete(
                normalized_latent,
                latent_noise,
                timestep,
                self._training_timesteps,
            )

            # Flow-matching target: velocity = noise − x0 (spatial domain)
            flow_target = latent_noise - normalized_latent

            # --- run connectors (frozen; transforms Gemma3 embeddings for transformer) ---
            video_emb, audio_emb, attn_mask = model.connectors(
                text_encoder_output.to(device=self.train_device),
                tokens_mask.to(device=self.train_device) if tokens_mask is not None else None,
                padding_side=self._connector_padding_side,
            )

            # --- pack video latents: (B,C,T,H,W) → (B, T*H*W, C) for transformer ---
            ps = self._patch_size
            pt = self._patch_size_t
            packed_noisy = LTX2Pipeline._pack_latents(
                noisy_latent.to(dtype=model.train_dtype.torch_dtype()), ps, pt,
            )
            packed_target = LTX2Pipeline._pack_latents(flow_target.float(), ps, pt)

            B, C, T, H, W = noisy_latent.shape
            latent_num_frames = T // pt
            latent_height = H // ps
            latent_width = W // ps

            # Reshape latent_mask to (B, seq_len, 1) to broadcast with packed
            # losses of shape (B, seq_len, C). The data loader may deliver the
            # mask as (B, H, W), (B, 1, H, W), (B, T, H, W), or (B, 1, T, H, W).
            # We normalise to (B, T_lat, H_lat, W_lat) then flatten to seq_len.
            if 'latent_mask' in batch:
                T_lat = T // pt
                m = batch['latent_mask'].float()
                if m.ndim == 5:
                    # (B, C, T_lat, H_lat, W_lat) — drop channel dim
                    m = m.mean(1)
                if m.ndim == 4:
                    if m.shape[1] != T_lat:
                        # (B, 1, H_lat, W_lat) — expand across frames
                        m = m.mean(1, keepdim=True).expand(-1, T_lat, -1, -1)
                elif m.ndim == 3:
                    # (B, H_lat, W_lat) — no temporal dim, expand across frames
                    m = m.unsqueeze(1).expand(-1, T_lat, -1, -1)
                batch['latent_mask'] = m.reshape(B, -1, 1)

            # --- zero audio branch (T2V: no audio data in batch) ---
            # The joint transformer requires audio_hidden_states. Pass a single dummy
            # token of zeros; no audio loss is computed.
            audio_hidden = torch.zeros(
                B, 1, self._audio_in_channels,
                device=self.train_device,
                dtype=model.train_dtype.torch_dtype(),
            )

            # --- transformer forward ---
            video_pred, _ = model.transformer(
                hidden_states=packed_noisy,
                audio_hidden_states=audio_hidden,
                encoder_hidden_states=video_emb.to(dtype=model.train_dtype.torch_dtype()),
                audio_encoder_hidden_states=audio_emb.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                sigma=timestep,
                encoder_attention_mask=attn_mask,
                audio_encoder_attention_mask=attn_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                audio_num_frames=1,
                use_cross_timestep=True,
                return_dict=False,
            )

        # NaN guard — runs outside autocast but still inside no_grad.
        # Check every 10 steps to avoid a blocking GPU sync on the hot path;
        # nan_to_num is applied unconditionally only when a NaN is detected.
        if not deterministic and train_progress.global_step % 10 == 0:
            if video_pred.isnan().any():
                print(
                    f"[Ltx2 NaN] step={train_progress.global_step} "
                    f"timestep={timestep.tolist()} "
                    f"noisy_nan={packed_noisy.isnan().any().item()} "
                    f"text_nan={text_encoder_output.isnan().any().item()} "
                    f"— skipping step to prevent optimizer corruption"
                )
                video_pred = video_pred.nan_to_num(nan=0.0)

        return {
            'loss_type': 'target',
            'timestep': timestep,
            'predicted': video_pred,
            'target': packed_target.to(video_pred.dtype),
        }

    def calculate_loss(
            self,
            model: Ltx2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=self._linear_sigmas,  # cached; None on step 0, mixin sets __sigmas on first call
        ).mean()

    def after_optimizer_step(
            self,
            model: Ltx2Model,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass


def _resolve_distilled_lora_path(path: str) -> str:
    """Resolve a configured distilled-LoRA path to a local file.

    Accepted forms:
    - ``/abs/path/to/file.safetensors`` — used directly
    - ``relative/path/file.safetensors`` (exists locally) — used directly
    - ``Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors``
      (HF spec: ``<owner>/<repo>/<filename>``) — downloaded via hf_hub_download
    """
    return _resolve_hf_or_local_path(path, label="distilled LoRA")


def _resolve_hf_or_local_path(path: str, label: str = "file") -> str:
    """Generic resolver for either a local file or an HF hub spec.

    Returns a local filesystem path.

    HF spec form: ``<owner>/<repo>[/<subfolder>]/<filename>``
    - Two parts  → invalid
    - Three parts → ``<owner>/<repo>/<filename>`` (no subfolder)
    - Four+ parts → ``<owner>/<repo>/<subfolder...>/<filename>`` (subfolder support)

    Examples:
      Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
      Kijai/LTX2.3_comfy/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors
    """
    if os.path.isfile(path):
        return path
    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"{label} path '{path}' is not a local file and is not in HF "
            f"format <owner>/<repo>/<filename>"
        )
    repo_id = "/".join(parts[:2])
    remainder = "/".join(parts[2:])
    if "/" in remainder:
        subfolder, filename = remainder.rsplit("/", 1)
    else:
        subfolder, filename = None, remainder
    print(f"[Ltx2] Downloading {filename} from HF repo {repo_id}"
          + (f" subfolder={subfolder}" if subfolder else "") + "…")
    return huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )


def _load_ltx_upsampler(path: str, scale: float):
    """Load a Lightricks single-file LTX2 spatial upsampler safetensors.

    The single-file format ships from ``Lightricks/LTX-2.3`` as flat tensors
    matching the diffusers ``LTX2LatentUpsamplerModel`` parameter names —
    BUT the rational-resampler flag must match the scale type:

    - ``scale == 1.5`` ships with a ``SpatialRationalResampler`` upsampler
      (``upsampler.blur_down.kernel`` / ``upsampler.conv.*``); requires
      ``use_rational_resampler=True``.
    - ``scale == 2.0`` (integer) ships with a Sequential ConvTranspose3d
      upsampler (``upsampler.0.weight`` / ``upsampler.0.bias``); requires
      ``use_rational_resampler=False``.

    Verified by inspecting the safetensors headers of both files.
    """
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    local_path = _resolve_hf_or_local_path(path, label="LTX upsampler")
    state_dict = load_file(local_path)
    use_rational = not float(scale).is_integer()
    model = LTX2LatentUpsamplerModel(
        in_channels=128,
        mid_channels=1024,
        num_blocks_per_stage=4,
        rational_spatial_scale=float(scale),
        use_rational_resampler=use_rational,
    )
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    return model


def _apply_distilled_lora(model: Ltx2Model, path: str) -> None:
    """Load the distilled LoRA weights into the handle list.

    Builds ``model.distilled_lora_handles`` — one ``(module, orig_fwd,
    {"down": d, "up": u})`` 3-tuple per matched Linear — but does NOT apply
    any forward patches.  Patches are applied (with the correct per-stage
    strength) only at sample time via ``_resume_distilled_lora_hooks()``.
    """
    model._clear_distilled_lora_hooks()

    local_path = _resolve_distilled_lora_path(path)

    raw_state_dict = load_file(local_path)
    sd = convert_ltx2_lora_original_to_diffusers(raw_state_dict)
    sd = normalize_lora_ab_to_down_up(sd)
    pairs = pair_lora_down_up(sd, prefix_to_strip="diffusion_model.")
    del raw_state_dict, sd

    transformer = model.transformer
    if transformer is None:
        raise RuntimeError("transformer not loaded; cannot load distilled LoRA")

    handles: list[tuple] = []
    misses: list[str] = []

    while pairs:
        module_path, down, up = pairs.pop()

        # OffloadCheckpointLayer-aware traversal — wrapper blocks expose their
        # real submodules under ``.checkpoint``.
        try:
            target = transformer
            for part in module_path.split("."):
                target = getattr(target, part)
                if hasattr(target, "checkpoint") and isinstance(target.checkpoint, torch.nn.Module):
                    target = target.checkpoint
        except AttributeError:
            misses.append(module_path)
            del down, up
            continue

        if not isinstance(target, torch.nn.Linear):
            misses.append(module_path)
            del down, up
            continue

        d = down.detach()
        u = up.detach()
        del down, up

        handles.append((target, target.forward, {"down": d, "up": u}))

    del pairs

    if misses:
        print(
            f"[Ltx2 LoRA] {len(misses)} LoRA modules had no matching transformer "
            f"submodule (e.g. {misses[:5]}). The base model + LoRA versions may differ."
        )
    print(f"[Ltx2 LoRA] {len(handles)} pairs loaded from {os.path.basename(local_path)} — patches applied per-stage at sample time")
    model.distilled_lora_handles = handles
    model.distilled_lora_path = path
