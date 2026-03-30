
from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from transformers import T5TokenizerFast, UMT5EncoderModel


class WanModel(BaseModel):
    # base model data
    tokenizer: T5TokenizerFast | None
    noise_scheduler: UniPCMultistepScheduler | None
    text_encoder: UMT5EncoderModel | None
    vae: AutoencoderKLWan | None
    transformer: WanTransformer3DModel | None    # high-noise expert
    transformer_2: WanTransformer3DModel | None  # low-noise expert
    boundary_ratio: float

    # original copy of tokenizer
    orig_tokenizer: T5TokenizerFast | None

    # autocast context (transformer uses the shared model.autocast_context / model.train_dtype)

    # offload conductors
    transformer_offload_conductor: LayerOffloadConductor | None
    transformer_2_offload_conductor: LayerOffloadConductor | None

    # lora
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    transformer_2_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.transformer_2 = None
        self.boundary_ratio = 0.875

        self.orig_tokenizer = None

        self.transformer_offload_conductor = None
        self.transformer_2_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.transformer_2_lora = None
        self.lora_state_dict = None
        self.companion_lora_handles = []  # list of (module, orig_forward, rot_mod_or_None)
        self.companion_lora_expert = None  # 1 = high-noise (transformer), 2 = low-noise (transformer_2)
        self.companion_lora_path: str | None = None

    def _clear_companion_lora_hooks(self):
        """Restore original forwards from companion LoRA handles and reset companion state."""
        for handle in self.companion_lora_handles:
            m, orig_fwd = handle[0], handle[1]
            m.forward = orig_fwd
        self.companion_lora_handles = []
        self.companion_lora_expert = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
            self.transformer_2_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            self.text_encoder.to(device=device)
        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_1_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        elif self.transformer is not None:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)
        if self.companion_lora_expert == 1:
            for handle in self.companion_lora_handles:
                if len(handle) > 2 and handle[2] is not None:
                    handle[2].to(device)

    def transformer_2_to(self, device: torch.device):
        if self.transformer_2_offload_conductor is not None and \
                self.transformer_2_offload_conductor.layer_offload_activated():
            self.transformer_2_offload_conductor.to(device)
        elif self.transformer_2 is not None:
            self.transformer_2.to(device=device)
        if self.transformer_2_lora is not None:
            self.transformer_2_lora.to(device)
        if self.companion_lora_expert == 2:
            for handle in self.companion_lora_handles:
                if len(handle) > 2 and handle[2] is not None:
                    handle[2].to(device)

    def transformer_to(self, device: torch.device):
        self.transformer_1_to(device)
        self.transformer_2_to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        if self.transformer_2 is not None:
            self.transformer_2.eval()

    def create_pipeline(self) -> WanPipeline:
        return WanPipeline(
            tokenizer=self.orig_tokenizer if self.orig_tokenizer is not None else self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            transformer_2=self.transformer_2,
            vae=self.vae,
            scheduler=self.noise_scheduler,
            boundary_ratio=self.boundary_ratio,
        )

    def encode_text(
            self,
            text: str,
            device: torch.device,
    ) -> tuple:
        tokenizer_output = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenizer_output.input_ids.to(device)
        attention_mask = tokenizer_output.attention_mask.to(device)

        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        embeddings = encoder_output.last_hidden_state

        # Zero out padding positions — UMT5 pad-token embeddings are non-zero,
        # which corrupts cross-attention in the transformer.  The official
        # WanPipeline trims to actual sequence length then re-pads with zeros;
        # masking achieves the same result in one step.
        embeddings = embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)

        return embeddings, attention_mask
