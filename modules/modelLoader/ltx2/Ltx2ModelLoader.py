import os
import traceback

import torch

from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.ltx2 import _diffusers_patch  # noqa: F401  # patches diffusers LTX2 single-file converter
from modules.modelLoader.ltx2 import _sequential_cfg_patch  # noqa: F401  # wraps transformer.forward to support sequential CFG
from modules.modelLoader.ltx2.Gemma3GGUFLoader import load_gemma3_from_gguf
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import replace_linear_with_quantized_layers

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2VocoderWithBWE
from torch import nn
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast


class Ltx2ModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def __load_diffusers(
            self,
            model: Ltx2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            quantization: QuantizationConfig,
            transformer_path: str = "",
            text_encoder_path: str = "",
    ):
        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=["transformer", "vae", "audio_vae", "connectors", "vocoder"],
            transformers_modules=["text_encoder"],
        )

        tokenizer = GemmaTokenizerFast.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        _te_dtype = weight_dtypes.train_dtype.torch_dtype() or torch.bfloat16

        if text_encoder_path and weight_dtypes.text_encoder.is_gguf():
            # NOTE: HF transformers' from_pretrained(gguf_file=...) dequantizes
            # GGUF tensors to bf16 at load time, defeating the point of GGUF.
            # Use our custom loader instead, which keeps weights packed and
            # installs diffusers' GGUFLinear modules — to be upgraded to
            # LinearGGUFA8 below by replace_linear_with_quantized_layers.
            is_causal = os.path.isfile(text_encoder_path)
            gguf_path = (
                text_encoder_path
                if is_causal
                else os.path.join(base_model_name, "text_encoder", text_encoder_path)
            )
            text_encoder = load_gemma3_from_gguf(
                gguf_path=gguf_path,
                base_model_name=base_model_name,
                dtype=_te_dtype,
                is_causal_lm=is_causal,
            )
        else:
            text_encoder = self._load_transformers_sub_module(
                Gemma3ForConditionalGeneration,
                weight_dtypes.text_encoder,
                weight_dtypes.train_dtype,
                base_model_name,
                "text_encoder",
            )
        # Drop the SigLIP vision tower — LTX2 doesn't use it and it costs ~4GB.
        if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "vision_tower"):
            text_encoder.model.vision_tower = None
        # Replace lm_head with Identity. Gemma3 ties lm_head.weight to
        # embed_tokens.weight; the state-dict only contains the embedding copy, so
        # lm_head.weight stays as a meta tensor after load. LTX2 never uses lm_head
        # (we only read outputs.hidden_states), so swapping it for Identity sidesteps
        # the meta-tensor crash during quantization and avoids a redundant ~1.5GB.
        if hasattr(text_encoder, "lm_head"):
            text_encoder.lm_head = nn.Identity()

        # Upgrade diffusers' GGUFLinear modules (installed by load_gemma3_from_gguf)
        # to OneTrainer's LinearGGUFA8, mirroring the LTX2 transformer GGUF path.
        if text_encoder_path and weight_dtypes.text_encoder.is_gguf():
            replace_linear_with_quantized_layers(
                parent_module=text_encoder,
                dtype=weight_dtypes.text_encoder,
                keep_in_fp32_modules=[],
                quantization=quantization,
            )

        vae = self._load_diffusers_sub_module(
            AutoencoderKLLTX2Video,
            DataType.FLOAT_32,
            DataType.FLOAT_32,
            base_model_name,
            "vae",
        )

        audio_vae = self._load_diffusers_sub_module(
            AutoencoderKLLTX2Audio,
            DataType.FLOAT_32,
            DataType.FLOAT_32,
            base_model_name,
            "audio_vae",
        )

        connectors = self._load_diffusers_sub_module(
            LTX2TextConnectors,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "connectors",
            quantization,
        )

        vocoder = self._load_diffusers_sub_module(
            LTX2VocoderWithBWE,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "vocoder",
            quantization,
        )

        _gguf_compute_dtype = weight_dtypes.train_dtype.torch_dtype() or torch.bfloat16

        if transformer_path and weight_dtypes.transformer.is_gguf():
            transformer = LTX2VideoTransformer3DModel.from_single_file(
                transformer_path,
                config=base_model_name,
                subfolder="transformer",
                dtype=_gguf_compute_dtype,
                quantization_config=GGUFQuantizationConfig(compute_dtype=_gguf_compute_dtype),
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                LTX2VideoTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.audio_vae = audio_vae
        model.connectors = connectors
        model.vocoder = vocoder
        model.transformer = transformer

    def __load_internal(
            self,
            model: Ltx2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            quantization: QuantizationConfig,
            transformer_path: str = "",
            text_encoder_path: str = "",
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, quantization, transformer_path, text_encoder_path)
        else:
            raise Exception("not an internal model")

    def __after_load(self, model: Ltx2Model):
        pass

    def load(
            self,
            model: Ltx2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []
        transformer_path = getattr(model_names, 'transformer_model', '') or ''
        text_encoder_path = getattr(model_names, 'text_encoder_model', '') or ''

        try:
            self.__load_internal(model, model_type, weight_dtypes, model_names.base_model, quantization, transformer_path, text_encoder_path)
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, model_names.base_model, quantization, transformer_path, text_encoder_path)
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
