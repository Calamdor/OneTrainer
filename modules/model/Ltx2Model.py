import contextlib
import math

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2ImageToVideoPipeline,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2Vocoder, LTX2VocoderWithBWE
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast


class Ltx2Model(BaseModel):
    # base model data
    tokenizer: GemmaTokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Gemma3ForConditionalGeneration | None
    vae: AutoencoderKLLTX2Video | None
    audio_vae: AutoencoderKLLTX2Audio | None
    connectors: LTX2TextConnectors | None
    vocoder: LTX2Vocoder | LTX2VocoderWithBWE | None
    transformer: LTX2VideoTransformer3DModel | None

    # original copy of tokenizer (preserved before any in-place mutations)
    orig_tokenizer: GemmaTokenizerFast | None

    # offload conductor
    transformer_offload_conductor: LayerOffloadConductor | None

    # lora
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    # Distilled LoRA -- frozen, applied via forward-method patches at sample time only.
    # Stored as 3-tuples (module, original_forward, {"down": d, "up": u}) so cleanup
    # restores the original forward without rebuilding the module. Tensors live in
    # PINNED CPU MEMORY, never resident in dedicated VRAM -- see distilled_lora_to()'s
    # docstring. This is the "ComfyUI-style low-VRAM streaming" mechanism: the community
    # distilled LoRA is 2.5GB, the "Sulphur-2" variant is 13.5GB, neither of which should
    # sit resident in VROM alongside the transformer.
    distilled_lora_handles: list[tuple]
    distilled_lora_path: str | None
    distilled_lora_strength: float

    # Spatial latent upsamplers (two-stage multi-scale sampling). Both optional; loaded
    # from Lightricks single-safetensors files at setup time. Move with the active
    # sampling stage via latent_upsampler_to(device, scale).
    latent_upsampler_x1_5: LTX2LatentUpsamplerModel | None
    latent_upsampler_x2: LTX2LatentUpsamplerModel | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.audio_vae = None
        self.connectors = None
        self.vocoder = None
        self.transformer = None

        self.orig_tokenizer = None

        self.transformer_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

        self.distilled_lora_handles = []
        self.distilled_lora_path = None
        self.distilled_lora_strength = 1.0

        self.latent_upsampler_x1_5 = None
        self.latent_upsampler_x2 = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    # --- LoRA/checkpoint key-conversion hooks (BaseModel contract) ---
    #
    # Verified against the actual diffusers LTX2VideoTransformer3DModel module tree
    # (meta-device instantiation, this venv's pinned diffusers commit, matching the
    # test suite's cross_attn_mod=True config that enables LTX-2.3's prompt-modulation
    # path) rather than assumed from the old branch's now-superseded convert_ltx2_lora.py.
    #
    # NOT yet covered here, deliberately left unresolved rather than guessed:
    # - The old branch's rename table also had "scale_shift_table_a2v_ca_video" /
    #   "scale_shift_table_a2v_ca_audio" entries. No submodule with that name exists in
    #   named_modules() -- self.scale_shift_table / self.audio_scale_shift_table are
    #   top-level nn.Parameters, not submodules, so this pair needs a named_parameters()
    #   level check (not named_modules()) before it can be confirmed or ported.
    # - Per-block audio/cross-modal attention modules (audio_attn1, audio_attn2,
    #   audio_to_video_attn, video_to_audio_attn, audio_ff, audio_norm*) need NO renames
    #   at all -- diffusers' native names already match Lightricks' checkpoint format
    #   for these (the old branch's rename table never mentions them, which is
    #   consistent with 1:1 naming, not an omission).

    def fusion_groups(self) -> list | None:
        # LTX-2.3 never fuses qkv -- attn1/attn2/audio_attn1/audio_attn2/audio_to_video_attn/
        # video_to_audio_attn all keep separate to_q/to_k/to_v projections in both diffusers
        # and the native Lightricks checkpoint (confirmed: no fused qkv module in the meta
        # model dump, and the community-trained kohya LoRA inspected separately trains
        # separate to_q/to_k/to_v, never a fused name).
        return None

    def diffusers_to_original(self) -> list | None:
        # Every attention submodule in a block (attn1/attn2/audio_attn1/audio_attn2/
        # audio_to_video_attn/video_to_audio_attn) shares the same internal layout: to_q/to_k/
        # to_v/to_out.0 keep their diffusers names 1:1, and only norm_q/norm_k rename to the
        # native q_norm/k_norm. strict=True conversion requires every key under transformer_blocks
        # to be covered -- an attention submodule listed with no explicit to_q/to_k/to_v/to_out.0
        # passthrough would leave those keys unmatched, not merely un-renamed.
        attn_body = [
            ("to_q", "to_q"),
            ("to_k", "to_k"),
            ("to_v", "to_v"),
            ("to_out.0", "to_out.0"),
            ("norm_q", "q_norm"),
            ("norm_k", "k_norm"),
            # Optional per-head gated-attention projection (Ltx2Attention.to_gate_logits,
            # transformer_ltx2.py apply_gated_attention=True) -- not present on every attention
            # submodule, but this identity rule only fires for keys that actually exist.
            ("to_gate_logits", "to_gate_logits"),
        ]
        return [
            ("audio_proj_in", "audio_patchify_proj"),
            ("proj_in", "patchify_proj"),
            ("audio_time_embed", "audio_adaln_single"),
            ("time_embed", "adaln_single"),
            ("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
            ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
            ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
            ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
            ("audio_prompt_adaln", "audio_prompt_adaln_single"),
            ("prompt_adaln", "prompt_adaln_single"),
            ("audio_caption_projection", "audio_caption_projection"),
            ("caption_projection", "caption_projection"),
            ("audio_proj_out", "audio_proj_out"),
            ("proj_out", "proj_out"),
            ("audio_scale_shift_table", "audio_scale_shift_table"),
            ("scale_shift_table", "scale_shift_table"),
            ("transformer_blocks.{i}", "transformer_blocks.{i}", [
                ("attn1", "attn1", attn_body),
                ("attn2", "attn2", attn_body),
                ("audio_attn1", "audio_attn1", attn_body),
                ("audio_attn2", "audio_attn2", attn_body),
                ("audio_to_video_attn", "audio_to_video_attn", attn_body),
                ("video_to_audio_attn", "video_to_audio_attn", attn_body),
                ("ff", "ff"),
                ("audio_ff", "audio_ff"),
                ("scale_shift_table", "scale_shift_table"),
                ("prompt_scale_shift_table", "prompt_scale_shift_table"),
                ("audio_scale_shift_table", "audio_scale_shift_table"),
                ("audio_prompt_scale_shift_table", "audio_prompt_scale_shift_table"),
                ("video_a2v_cross_attn_scale_shift_table", "video_a2v_cross_attn_scale_shift_table"),
                ("audio_a2v_cross_attn_scale_shift_table", "audio_a2v_cross_attn_scale_shift_table"),
            ]),
        ]

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        # T2V LoRA trains the transformer only; no text-encoder LoRA declared (the common,
        # empty-list case per BaseModel's docstring). Revisit if/when TE LoRA training is added.
        return []

    def _clear_distilled_lora_hooks(self) -> None:
        """Restore original forwards and release all distilled-LoRA tensors.

        Use this only when permanently unloading -- it frees all LoRA weight tensors
        (up to ~13.5 GB for the Sulphur-2 variant). For training, prefer _pause/_resume so
        the tensors survive in CPU RAM (pinned) between sample windows.
        """
        for handle in self.distilled_lora_handles:
            module, orig_forward = handle[0], handle[1]
            module.forward = orig_forward
        self.distilled_lora_handles = []

    def _pause_distilled_lora_hooks(self) -> None:
        """Remove distilled-LoRA forward patches without freeing the tensors.

        The weight tensors remain in distilled_lora_handles on pinned CPU RAM so
        _resume_distilled_lora_hooks() can re-apply them instantly without a disk reload.
        Call this before every training forward pass; call _resume before every sampling
        run.
        """
        for handle in self.distilled_lora_handles:
            module, orig_forward, payload = handle[0], handle[1], handle[2]
            if not isinstance(payload, dict):
                continue
            pre_resume = payload.pop("_pre_resume_fwd", None)
            # If _resume was called at least once, pre_resume holds whatever forward was
            # active before the patch (bare original or LoRA-wrapped). If pause is called
            # before the first resume (e.g. during setup_model), fall back to orig_forward
            # so LoRA hooks onto a clean, unpatched forward.
            module.forward = pre_resume if pre_resume is not None else orig_forward

    def _resume_distilled_lora_hooks(self) -> None:
        """Re-apply distilled-LoRA forward patches on top of the current forward.

        Wraps module.forward AS IT STANDS NOW -- so if LoRA is already hooked the
        distilled delta stacks on top of it correctly. No-op if distilled_lora_handles is
        empty.
        """
        if not self.distilled_lora_handles:
            return
        strength = self.distilled_lora_strength
        print(f"[Ltx2 LoRA] applying {len(self.distilled_lora_handles)} distilled LoRA patches (strength={strength})")
        for handle in self.distilled_lora_handles:
            module, payload = handle[0], handle[2]
            if not isinstance(payload, dict):
                continue
            d = payload["down"]
            u = payload["up"]
            # Capture whatever forward is active right now (bare original, or
            # LoRA-wrapped) so pause can restore exactly this state.
            current_fwd = module.forward
            payload["_pre_resume_fwd"] = current_fwd

            def _make_patched(base, _d, _u, _s):
                def patched(x):
                    # _d/_u live in pinned CPU memory (see distilled_lora_to). non_blocking=True
                    # lets the DMA copy overlap with `base(x)` compute on the same stream; the
                    # matmul that follows serializes on the transfer automatically.
                    d_gpu = _d.to(x.device, x.dtype, non_blocking=True)
                    u_gpu = _u.to(x.device, x.dtype, non_blocking=True)
                    return base(x) + F.linear(F.linear(x, d_gpu), u_gpu) * _s
                return patched

            module.forward = _make_patched(current_fwd, d, u, strength)

    def distilled_lora_to(self, device: torch.device) -> None:
        """Stage distilled LoRA tensors for the upcoming device usage.

        LoRA d/u tensors live in **pinned CPU memory** during sampling, never in dedicated
        VRAM. The patched forward (``_d.to(x.device, x.dtype, non_blocking=True)``)
        performs a fast async DMA copy per matmul. This keeps the LoRA's GPU footprint to a
        few transient MB per matmul instead of ~2.5-13.5 GB resident -- the dominant VRAM
        saver at sample time, and the mechanism that makes the community distilled LoRAs
        (2.5GB original, 13.5GB "Sulphur-2") usable at all on consumer VRAM.

        - ``device`` is GPU (``cuda``): pin tensors in CPU memory if not already pinned.
          Don't actually move to GPU.
        - ``device`` is CPU: ensure tensors are CPU-resident (no-op if already CPU; unpins
          implicitly via copy).
        """
        target_is_gpu = torch.device(device).type == "cuda"

        for handle in self.distilled_lora_handles:
            if len(handle) > 2 and handle[2] is not None:
                payload = handle[2]
                if isinstance(payload, dict):
                    for k in ("down", "up"):
                        t = payload.get(k)
                        if t is None or not hasattr(t, "data"):
                            continue
                        data = t.data
                        if target_is_gpu:
                            if data.device.type != "cpu":
                                # Came back from GPU somewhere; bring to CPU first.
                                data = data.to("cpu")
                            try:
                                already_pinned = data.is_pinned()
                            except Exception:
                                already_pinned = False
                            if not already_pinned:
                                # Fall back to pageable on failure; transfers will be slower.
                                with contextlib.suppress(Exception):
                                    data = data.pin_memory()
                            t.data = data
                        else:
                            if data.device.type != "cpu":
                                t.data = data.to(device)
                elif hasattr(payload, "to"):
                    payload.to(device)

    def latent_upsampler_to(self, device: torch.device, scale: float | None = None) -> None:
        """Move the active spatial upsampler to the given device.

        ``scale`` selects which one: 1.5 -> x1.5 model, 2.0 -> x2 model. If ``scale`` is
        None, both are moved (used when shutting down).
        """
        if scale is None or scale == 1.5:
            if self.latent_upsampler_x1_5 is not None:
                self.latent_upsampler_x1_5.to(device=device)
        if scale is None or scale == 2.0:
            if self.latent_upsampler_x2 is not None:
                self.latent_upsampler_x2.to(device=device)

    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)
        if self.audio_vae is not None:
            self.audio_vae.to(device=device)

    def audio_vae_to(self, device: torch.device):
        if self.audio_vae is not None:
            self.audio_vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            self.text_encoder.to(device=device)
        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.offload_activated():
            self.transformer_offload_conductor.to(device)
        elif self.transformer is not None:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def connectors_to(self, device: torch.device):
        if self.connectors is not None:
            self.connectors.to(device=device)

    def vocoder_to(self, device: torch.device):
        if self.vocoder is not None:
            self.vocoder.to(device=device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
        self.connectors_to(device)
        self.vocoder_to(device)
        self.latent_upsampler_to(device)

    def eval(self):
        if self.vae is not None:
            self.vae.eval()
        if self.audio_vae is not None:
            self.audio_vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        if self.connectors is not None:
            self.connectors.eval()
        if self.vocoder is not None:
            self.vocoder.eval()
        if self.latent_upsampler_x1_5 is not None:
            self.latent_upsampler_x1_5.eval()
        if self.latent_upsampler_x2 is not None:
            self.latent_upsampler_x2.eval()

    def calculate_timestep_shift(
        self,
        latent_height: int,
        latent_width: int,
        latent_num_frames: int = 1,
    ) -> float:
        # Resolution-aware shift constant for the SD3/linear shift formula. Mirrors what
        # diffusers' LTX-2 pipeline does at inference: derive mu from a linear interpolation
        # between (base_image_seq_len, base_shift) and (max_image_seq_len, max_shift), then
        # convert to a shift constant via exp(mu). The scheduler is configured
        # time_shift_type="exponential", so exp(mu) matches its behavior under OT's
        # linear-shift training math. LTX-2.3's transformer uses patch_size=1 and
        # patch_size_t=1 (no spatial/temporal patching), so the effective sequence length
        # is T_lat * H_lat * W_lat; clamp to [base_image_seq_len, max_image_seq_len] to stay
        # inside the scheduler's calibrated range.
        sched_cfg = self.noise_scheduler.config
        base_seq_len = int(sched_cfg.get("base_image_seq_len", 1024))
        max_seq_len = int(sched_cfg.get("max_image_seq_len", 4096))
        base_shift = float(sched_cfg.get("base_shift", 0.95))
        max_shift = float(sched_cfg.get("max_shift", 2.05))

        image_seq_len = max(1, int(latent_num_frames) * int(latent_height) * int(latent_width))
        image_seq_len = max(base_seq_len, min(image_seq_len, max_seq_len))

        m = (max_shift - base_shift) / max(1, (max_seq_len - base_seq_len))
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)

    def create_pipeline(self) -> LTX2Pipeline:
        return LTX2Pipeline(
            scheduler=self.noise_scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=self.vocoder,
        )

    def create_i2v_pipeline(self) -> LTX2ImageToVideoPipeline:
        # Same components as create_pipeline; only the prepare_latents + __call__ paths
        # differ to accept image= and lock frame 0.
        return LTX2ImageToVideoPipeline(
            scheduler=self.noise_scheduler,
            vae=self.vae,
            audio_vae=self.audio_vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=self.vocoder,
        )

    def encode_text(
            self,
            text: str | list[str],
            device: torch.device,
    ) -> tuple:
        # Gemma3 expects left padding for chat-style prompts
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(text, str):
            text = [text]
        text = [t.strip() for t in text]

        text_inputs = self.tokenizer(
            text,
            padding="longest",
            max_length=1024,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # LTX-2 uses every layer's hidden state, stacked along a new dim and flattened
        # into a single 3D tensor [batch, seq_len, hidden_size * num_layers].
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        embeddings = hidden_states.flatten(2, 3)

        return embeddings, attention_mask
