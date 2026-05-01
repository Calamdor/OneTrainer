from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.LtxMultiScaleMode import LtxMultiScaleMode
from modules.util.enum.NoiseScheduler import NoiseScheduler


def _get_model_defaults(model_type) -> dict:
    """
    Returns model-specific default values for sampling parameters.
    Returns dict with keys: width, height, diffusion_steps, cfg_scale, noise_scheduler
    """

    defaults = {
        "width": 512,
        "height": 512,
        "frames": 1,
        "diffusion_steps": 30,
        "cfg_scale": 7.0,
        "noise_scheduler": NoiseScheduler.DDIM,
        "negative_prompt": "",
    }

    if model_type is None:
        return defaults

    if model_type.is_sd_v1():
        defaults.update({
            "width": 512,
            "height": 512,
            "diffusion_steps": 30,
            "cfg_scale": 7.5,
            "noise_scheduler": NoiseScheduler.EULER_A,
        })
    elif model_type.is_sd_v2():
        defaults.update({
            "width": 768,
            "height": 768,
            "diffusion_steps": 30,
            "cfg_scale": 7.5,
            "noise_scheduler": NoiseScheduler.DDIM,
        })
    elif model_type.is_stable_diffusion_xl():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 30,
            "cfg_scale": 7.5,
            "noise_scheduler": NoiseScheduler.EULER_A,
        })
    elif model_type.is_stable_diffusion_3():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 28,
            "cfg_scale": 7.0,
        })
    elif model_type.is_flux_1():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 30,
            "cfg_scale": 3.5,
        })
    elif model_type.is_chroma():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 30,
            "cfg_scale": 3.5,
            "negative_prompt": (
                "This low-quality, greyscale, unfinished sketch is inaccurate and flawed. "
                "The image is very blurred and lacks detail, with excessive chromatic "
                "aberrations and artifacts. The image is overly saturated with excessive bloom."
            ),
        })
    elif model_type.is_flux_2():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 30,
            "cfg_scale": 4.0,
        })
    elif model_type.is_qwen():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 25,
            "cfg_scale": 3.5,
        })
    elif model_type.is_z_image():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 28,
            "cfg_scale": 4.0,
        })
    elif model_type.is_hunyuan_video():
        defaults.update({
            "width": 848,
            "height": 480,
            "diffusion_steps": 30,
            "cfg_scale": 6.0,
        })
    elif model_type.is_wan_video():
        defaults.update({
            "width": 832,
            "height": 480,
            "frames": 81,
            "diffusion_steps": 20,
            "cfg_scale": 4.0,
            "cfg_scale_2": 3.0,
            "steps_high": 10,
            "steps_low": 10,
        })
    elif model_type.is_ltx_video():
        defaults.update({
            "width": 1024,
            "height": 576,
            "frames": 121,
            "diffusion_steps": 20,
            "cfg_scale": 3.0,
            "negative_prompt": (
                "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, "
                "compression artifacts, jpeg artifacts, glitches, watermark, text, logo, "
                "signature, copyright, subtitles, distorted sound, saturated sound, loud"
            ),
            "frame_rate": 24.0,
            "ltx_multi_scale_mode": LtxMultiScaleMode.X2,
            "ltx_vae_tiling": True,
            "ltx_vae_tile_size": 256,
            "ltx_distilled_lora_stage1_strength": 0.3,
            "ltx_distilled_lora_stage2_strength": 0.6,
        })
    elif model_type.is_sana():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 20,
            "cfg_scale": 4.5,
        })
    elif model_type.is_hi_dream():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 28,
            "cfg_scale": 5.0,
        })
    elif model_type.is_pixart():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 25,
            "cfg_scale": 4.5,
        })
    elif model_type.is_wuerstchen():
        defaults.update({
            "width": 1024,
            "height": 1024,
            "diffusion_steps": 25,
            "cfg_scale": 4.0,
        })

    return defaults


class SampleConfig(BaseConfig):
    enabled: bool
    prompt: str
    negative_prompt: str
    height: int
    width: int
    frames: int
    length: float
    seed: int
    random_seed: bool
    diffusion_steps: int
    cfg_scale: float
    noise_scheduler: NoiseScheduler
    flow_shift: float | None
    steps_high: int | None
    steps_low: int | None
    cfg_scale_2: float | None
    frame_rate: float | None
    ltx_multi_scale_mode: LtxMultiScaleMode | None
    ltx_vae_tiling: bool | None
    ltx_vae_tile_size: int | None
    ltx_distilled_lora_stage1_strength: float | None
    ltx_distilled_lora_stage2_strength: float | None

    text_encoder_1_layer_skip: int
    text_encoder_1_sequence_length: int | None
    text_encoder_2_layer_skip: int
    text_encoder_2_sequence_length: int | None
    text_encoder_3_layer_skip: int
    text_encoder_4_layer_skip: int
    transformer_attention_mask: bool
    force_last_timestep: bool

    sample_inpainting: bool
    base_image_path: str
    mask_image_path: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def from_train_config(self, train_config):
        self.text_encoder_1_layer_skip = train_config.text_encoder_layer_skip
        self.text_encoder_1_sequence_length = train_config.text_encoder_sequence_length
        self.text_encoder_2_layer_skip = train_config.text_encoder_2_layer_skip
        self.text_encoder_2_sequence_length = train_config.text_encoder_2_sequence_length
        self.text_encoder_3_layer_skip = train_config.text_encoder_3_layer_skip
        self.text_encoder_4_layer_skip = train_config.text_encoder_4_layer_skip
        self.transformer_attention_mask = train_config.transformer.attention_mask
        self.force_last_timestep = train_config.rescale_noise_scheduler_to_zero_terminal_snr

    @staticmethod
    def default_values(model_type=None):
        defaults = _get_model_defaults(model_type)
        data = []

        data.append(("enabled", True, bool, False))
        data.append(("prompt", "", str, False))
        data.append(("negative_prompt", defaults["negative_prompt"], str, False))
        data.append(("height", defaults["height"], int, False))
        data.append(("width", defaults["width"], int, False))
        data.append(("frames", defaults["frames"], int, False))
        data.append(("length", 10.0, float, False))
        data.append(("seed", 42, int, False))
        data.append(("random_seed", False, bool, False))
        data.append(("diffusion_steps", defaults["diffusion_steps"], int, False))
        data.append(("cfg_scale", defaults["cfg_scale"], float, False))
        data.append(("noise_scheduler", defaults["noise_scheduler"], NoiseScheduler, False))
        data.append(("flow_shift", defaults.get("flow_shift", None), float, True))
        data.append(("steps_high", defaults.get("steps_high", None), int, True))
        data.append(("steps_low", defaults.get("steps_low", None), int, True))
        data.append(("cfg_scale_2", defaults.get("cfg_scale_2", None), float, True))
        data.append(("frame_rate", defaults.get("frame_rate", None), float, True))
        data.append(("ltx_multi_scale_mode", defaults.get("ltx_multi_scale_mode", None), LtxMultiScaleMode, True))
        data.append(("ltx_vae_tiling", defaults.get("ltx_vae_tiling", None), bool, True))
        data.append(("ltx_vae_tile_size", defaults.get("ltx_vae_tile_size", None), int, True))
        data.append(("ltx_distilled_lora_stage1_strength", defaults.get("ltx_distilled_lora_stage1_strength", None), float, True))
        data.append(("ltx_distilled_lora_stage2_strength", defaults.get("ltx_distilled_lora_stage2_strength", None), float, True))

        data.append(("text_encoder_1_layer_skip", 0, int, False))
        data.append(("text_encoder_1_sequence_length", None, int, True))
        data.append(("text_encoder_2_layer_skip", 0, int, False))
        data.append(("text_encoder_2_sequence_length", None, int, True))
        data.append(("text_encoder_3_layer_skip", 0, int, False))
        data.append(("text_encoder_4_layer_skip", 0, int, False))
        data.append(("transformer_attention_mask", False, bool, False))
        data.append(("force_last_timestep", False, bool, False))

        data.append(("sample_inpainting", False, bool, False))
        data.append(("base_image_path", "", str, False))
        data.append(("mask_image_path", "", str, False))

        return SampleConfig(data)
