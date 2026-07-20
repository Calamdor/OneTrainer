"""Diagnostic: does the transformer actually load as FP8, or silently stay bf16?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import modules.util.create as create  # noqa: F401
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.quantization_util import quantize_layers

from diffusers import LTX2VideoTransformer3DModel


class _Loader(HFModelLoaderMixin):
    pass


def main():
    config = TrainConfig.default_values()
    config.transformer.weight_dtype = DataType.FLOAT_8
    config.train_dtype = DataType.BFLOAT_16
    weight_dtypes = config.weight_dtypes()
    quantization = config.quantization

    loader = _Loader()
    transformer = loader._load_diffusers_sub_module(
        LTX2VideoTransformer3DModel,
        weight_dtypes.transformer,
        weight_dtypes.train_dtype,
        "dg845/LTX-2.3-Diffusers",
        "transformer",
        quantization,
    )

    print("--- BEFORE quantize_layers() ---")
    total_before = sum(p.numel() * p.element_size() for p in transformer.parameters())
    print(f"Total parameter memory: {total_before / 1e9:.2f} GB")

    quantize_layers(transformer, torch.device("cuda"), config.train_dtype, config)

    print("--- AFTER quantize_layers() ---")
    dtypes = {}
    total_bytes = 0
    for name, p in transformer.named_parameters():
        dt = str(p.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
        total_bytes += p.numel() * p.element_size()

    print("Parameter dtype histogram:", dtypes)
    print(f"Total parameter memory: {total_bytes / 1e9:.2f} GB")

    # sample a linear layer's actual class + weight dtype
    for name, m in transformer.named_modules():
        if "to_q" in name and hasattr(m, "weight"):
            print(f"Sample module {name}: class={type(m).__name__}, weight dtype={m.weight.dtype}, weight shape={tuple(m.weight.shape)}")
            break


if __name__ == "__main__":
    main()
