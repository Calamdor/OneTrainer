import copy
import json
import os
import traceback

from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler, WanTransformer3DModel
from transformers import T5TokenizerFast, UMT5EncoderModel


class WanModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def __load_diffusers(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            quantization: QuantizationConfig,
            transformer_1_path: str = "",
            transformer_2_path: str = "",
    ):
        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=["transformer", "transformer_2", "vae"],
            transformers_modules=["text_encoder"],
        )

        tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = UniPCMultistepScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            UMT5EncoderModel,
            weight_dtypes.text_encoder,
            weight_dtypes.train_dtype,
            base_model_name,
            "text_encoder",
        )

        vae = self._load_diffusers_sub_module(
            AutoencoderKLWan,
            DataType.FLOAT_32,
            DataType.FLOAT_32,
            base_model_name,
            "vae",
        )

        if transformer_1_path and weight_dtypes.transformer.is_gguf():
            transformer = WanTransformer3DModel.from_single_file(
                transformer_1_path,
                config=base_model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self.__load_with_dtype_cache(
                WanTransformer3DModel, weight_dtypes, base_model_name, "transformer", quantization,
            )

        if transformer_2_path and weight_dtypes.transformer.is_gguf():
            transformer_2 = WanTransformer3DModel.from_single_file(
                transformer_2_path,
                config=base_model_name,
                subfolder="transformer_2",
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            )
            transformer_2 = self._convert_diffusers_sub_module_to_dtype(
                transformer_2, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer_2 = self.__load_with_dtype_cache(
                WanTransformer3DModel, weight_dtypes, base_model_name, "transformer_2", quantization,
            )

        # Read boundary_ratio from model_index.json if present
        boundary_ratio = 0.875
        if os.path.isdir(base_model_name):
            model_index_path = os.path.join(base_model_name, "model_index.json")
            if os.path.isfile(model_index_path):
                with open(model_index_path, "r") as f:
                    model_index = json.loads(f.read())
                boundary_ratio = model_index.get("boundary_ratio", 0.875)

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.transformer_2 = transformer_2
        model.boundary_ratio = boundary_ratio

    def __load_internal(
            self,
            model: WanModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            quantization: QuantizationConfig,
            transformer_1_path: str = "",
            transformer_2_path: str = "",
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, quantization,
                                  transformer_1_path, transformer_2_path)
        else:
            raise Exception("not an internal model")

    def __load_with_dtype_cache(
            self,
            model_class,
            weight_dtypes,
            base_model_name: str,
            subfolder: str,
            quantization,
    ):
        """Load a transformer sub-module, using a dtype-converted cache if configured.

        On cache miss: load normally from base_model_name, then save the converted weights
        to {dtype_cache_dir}/{subfolder}_{dtype}/ so future runs skip the fp32→dtype pass.

        On cache hit: load directly from the cache dir (same OT loading path, but the source
        data is already at the target dtype so the conversion inside _load_diffusers_sub_module
        is a no-op).  All quantization wrapper setup still runs normally.

        Only applies to non-GGUF pure dtype conversions.  GGUF and quantized dtypes (INT8, FP8,
        etc.) use their own loading paths and cannot be serialized by save_pretrained.
        """
        transformer_dtype = weight_dtypes.transformer
        dtype_cache_dir = getattr(quantization, 'dtype_cache_dir', None) if quantization else None
        if dtype_cache_dir and transformer_dtype.is_quantized():
            dtype_cache_dir = None

        if dtype_cache_dir:
            dtype_name = weight_dtypes.transformer.name.lower()
            cache_subdir = os.path.join(dtype_cache_dir, f"{subfolder}_{dtype_name}")
            cache_config = os.path.join(cache_subdir, "config.json")

            if os.path.isfile(cache_config):
                print(
                    f"[WanModelLoader] {subfolder}: loading from dtype cache "
                    f"({dtype_name}) — {cache_subdir}"
                )
                return self._load_diffusers_sub_module(
                    model_class,
                    weight_dtypes.transformer,
                    weight_dtypes.train_dtype,
                    cache_subdir,
                    "",
                    quantization,
                )

        # Normal load from original source
        module = self._load_diffusers_sub_module(
            model_class,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            subfolder,
            quantization,
        )

        if dtype_cache_dir:
            dtype_name = weight_dtypes.transformer.name.lower()
            cache_subdir = os.path.join(dtype_cache_dir, f"{subfolder}_{dtype_name}")
            print(
                f"[WanModelLoader] {subfolder}: saving dtype cache "
                f"({dtype_name}) → {cache_subdir}"
            )
            os.makedirs(cache_subdir, exist_ok=True)
            module.save_pretrained(cache_subdir, safe_serialization=True)
            print(f"[WanModelLoader] {subfolder}: dtype cache saved")

        return module

    def __after_load(self, model: WanModel):
        model.orig_tokenizer = copy.deepcopy(model.tokenizer)

    def load(
            self,
            model: WanModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []
        t1 = getattr(model_names, 'transformer_model', '')
        t2 = getattr(model_names, 'transformer_2_model', '')

        try:
            self.__load_internal(model, model_type, weight_dtypes, model_names.base_model, quantization, t1, t2)
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(model, model_type, weight_dtypes, model_names.base_model, quantization, t1, t2)
            self.__after_load(model)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
