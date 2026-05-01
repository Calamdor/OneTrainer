
from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
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

    # lora (declared for training-phase compatibility; unused for sampling-only)
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    # Distilled LoRA — frozen, applied via forward-method patches at sample time
    # only. Stored as 3-tuples (module, original_forward, scale) so cleanup
    # restores the original forward without rebuilding the module.
    distilled_lora_handles: list[tuple]
    distilled_lora_path: str | None
    distilled_lora_strength: float

    # Spatial latent upsamplers (two-stage sampling). Both optional; loaded
    # from Lightricks single-safetensors files at model load time. Move with
    # the active sampling stage via ``latent_upsampler_to(device, scale)``.
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

    def _clear_distilled_lora_hooks(self) -> None:
        """Restore original forwards and release all distilled-LoRA tensors.

        Use this only when permanently unloading — it frees the ~7.6 GB of
        LoRA weight tensors.  For training, prefer _pause / _resume so the
        tensors survive on CPU RAM between sample windows.
        """
        for handle in self.distilled_lora_handles:
            module, orig_forward = handle[0], handle[1]
            module.forward = orig_forward
        self.distilled_lora_handles = []

    def _pause_distilled_lora_hooks(self) -> None:
        """Remove distilled-LoRA forward patches without freeing the tensors.

        The weight tensors (~7.6 GB BF16) remain in distilled_lora_handles on
        CPU RAM so _resume_distilled_lora_hooks() can re-apply them instantly
        without a disk reload.  Call this before every training forward pass;
        call _resume before every sampling run.
        """
        for handle in self.distilled_lora_handles:
            module, orig_forward, payload = handle[0], handle[1], handle[2]
            if not isinstance(payload, dict):
                continue
            pre_resume = payload.pop("_pre_resume_fwd", None)
            # If _resume was called at least once, pre_resume holds whatever forward
            # was active before the patch (bare original or LoRA-wrapped).  If pause
            # is called before the first resume (e.g. during setup_model), fall back
            # to orig_forward so LoRA hooks onto a clean, unpatched forward.
            module.forward = pre_resume if pre_resume is not None else orig_forward

    def _resume_distilled_lora_hooks(self) -> None:
        """Re-apply distilled-LoRA forward patches on top of the current forward.

        Wraps module.forward AS IT STANDS NOW — so if LoRA is already hooked
        the distilled delta stacks on top of it correctly.  No-op if
        distilled_lora_handles is empty.
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
                    return base(x) + F.linear(
                        F.linear(x, _d.to(x.device, x.dtype)),
                        _u.to(x.device, x.dtype),
                    ) * _s
                return patched

            module.forward = _make_patched(current_fwd, d, u, strength)

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
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        elif self.transformer is not None:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def distilled_lora_to(self, device: torch.device) -> None:
        """Move distilled LoRA weight tensors to device.

        Called explicitly by the sampler: to train_device before
        _resume_distilled_lora_hooks(), and back to temp_device after
        _pause_distilled_lora_hooks(). Kept separate from transformer_to()
        so training never pays the ~7.6 GB cost for these unused tensors.
        """
        for handle in self.distilled_lora_handles:
            if len(handle) > 2 and handle[2] is not None:
                payload = handle[2]
                if isinstance(payload, dict):
                    for k in ("down", "up"):
                        t = payload.get(k)
                        if t is not None and hasattr(t, "data"):
                            t.data = t.data.to(device)
                elif hasattr(payload, "to"):
                    payload.to(device)

    def connectors_to(self, device: torch.device):
        if self.connectors is not None:
            self.connectors.to(device=device)

    def vocoder_to(self, device: torch.device):
        if self.vocoder is not None:
            self.vocoder.to(device=device)

    def latent_upsampler_to(self, device: torch.device, scale: float | None = None) -> None:
        """Move the active spatial upsampler to the given device.

        ``scale`` selects which one: 1.5 → x1.5 model, 2.0 → x2 model. If
        ``scale`` is None, both are moved (used when shutting down).
        """
        if scale is None or scale == 1.5:
            if self.latent_upsampler_x1_5 is not None:
                self.latent_upsampler_x1_5.to(device=device)
        if scale is None or scale == 2.0:
            if self.latent_upsampler_x2 is not None:
                self.latent_upsampler_x2.to(device=device)

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
