"""One-off script: real end-to-end Ltx2Sampler verification against the cached checkpoint.

Loads the real dg845/LTX-2.3-Diffusers weights (transformer @ FP8, everything else bf16 to
fit comfortably in 32GB VRAM) and runs Ltx2Sampler.sample() for real, saving an MP4.

Usage:
    python scripts/run_ltx2_sample_test.py [--cfg 3.0] [--steps 25] [--width 480] [--height 480]
        [--distilled-lora PATH]
        [--multi-scale-mode FULL_SIZE|X1_5|X2] [--upsampler-x1-5 PATH] [--upsampler-x2 PATH]
        [--stage1-strength 0.8] [--stage2-strength 0.8] [--out NAME]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch

import modules.util.create as create  # noqa: F401  # populates the factory registry
from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.ltx2.Ltx2ModelLoader import Ltx2ModelLoader
from modules.modelSampler.Ltx2Sampler import Ltx2Sampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.LtxMultiScaleMode import LtxMultiScaleMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--distilled-lora", type=str, default="")
    p.add_argument("--multi-scale-mode", type=str, default="FULL_SIZE", choices=["FULL_SIZE", "X1_5", "X2"])
    p.add_argument("--upsampler-x1-5", type=str, default="")
    p.add_argument("--upsampler-x2", type=str, default="")
    p.add_argument("--stage1-strength", type=float, default=0.8)
    p.add_argument("--stage2-strength", type=float, default=0.8)
    p.add_argument("--out", type=str, default="ltx23_test_sample")
    p.add_argument("--transformer-dtype", type=str, default="FLOAT_8",
                    help="e.g. FLOAT_8, INT_W8A8, FLOAT_W8A8, GGUF_A8_INT, GGUF_A8_FLOAT")
    p.add_argument("--transformer-path", type=str, default="",
                    help="local .gguf (or single-file safetensors) path for the transformer")
    p.add_argument("--compile", action="store_true", help="enable torch.compile on transformer blocks")
    args = p.parse_args()

    device = torch.device("cuda")

    config = TrainConfig.default_values()
    config.model_type = ModelType.LTX_2_3
    config.base_model_name = "dg845/LTX-2.3-Diffusers"
    config.transformer.weight_dtype = DataType[args.transformer_dtype]
    config.transformer.model_name = args.transformer_path
    config.text_encoder.weight_dtype = DataType.BFLOAT_16
    config.train_dtype = DataType.BFLOAT_16
    config.ltx_distilled_lora_path = args.distilled_lora
    config.ltx_spatial_upsampler_x1_5_path = args.upsampler_x1_5
    config.ltx_spatial_upsampler_x2_path = args.upsampler_x2
    config.compile = args.compile

    weight_dtypes = config.weight_dtypes()
    model_names = config.model_names()
    quantization = config.quantization

    def mem(label: str) -> None:
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[mem] {label}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB", flush=True)

    model = Ltx2Model(model_type=ModelType.LTX_2_3)
    loader = Ltx2ModelLoader()

    print(f"[run] loading model (transformer dtype={args.transformer_dtype}, "
          f"path={args.transformer_path or '(hf repo)'})...", flush=True)
    t0 = time.time()
    loader.load(model, ModelType.LTX_2_3, model_names, weight_dtypes, quantization)
    print(f"[run] model loaded in {time.time() - t0:.1f}s", flush=True)
    mem("after model load")

    # Call the REAL production setup path instead of hand-rolling a subset of it. This is
    # what BaseLtx2Setup.setup_optimizations() does: load spatial upsamplers, set up
    # checkpointing/offload conductors, build model.autocast_context (defaults to a no-op
    # nullcontext() otherwise -- BaseModel.py -- which silently hides dtype-mismatch bugs
    # that only surface when autocast isn't actually active), then quantize_layers() on
    # every component. Reimplementing pieces of this by hand previously produced a false
    # crash (missing autocast) that looked like a LinearGGUFA8 bug but wasn't one.
    from modules.modelSetup.Ltx2LoRASetup import Ltx2LoRASetup
    setup = Ltx2LoRASetup(train_device=device, temp_device=torch.device("cpu"), debug_mode=False)

    print(f"[run] setup_optimizations (transformer={args.transformer_dtype})...", flush=True)
    t0q = time.time()
    setup.setup_optimizations(model, config)
    print(f"[run] setup_optimizations done in {time.time() - t0q:.1f}s", flush=True)
    mem("after setup_optimizations")

    if args.distilled_lora:
        # Mirrors BaseLtx2Setup.setup_model()'s call -- must happen after quantization, same
        # as the real training/setup path, since the forward patches wrap whatever Linear
        # class the transformer's projections use post-quantization.
        print("[run] loading distilled LoRA...", flush=True)
        t0d = time.time()
        setup._setup_distilled_lora(model, config)
        print(f"[run] distilled LoRA loaded in {time.time() - t0d:.1f}s "
              f"({len(model.distilled_lora_handles)} handles)", flush=True)

    model.eval()
    for part in (model.transformer, model.text_encoder, model.vae, model.audio_vae, model.connectors, model.vocoder):
        if part is not None:
            part.requires_grad_(False)

    sample_config = SampleConfig.default_values()
    sample_config.prompt = (
        "A beautiful woman in a bikini walking towards the camera on a sunny beach, "
        "ocean waves in the background, cinematic lighting, photorealistic"
    )
    sample_config.negative_prompt = (
        "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, "
        "compression artifacts, jpeg artifacts, glitches, watermark, text, logo, signature, "
        "copyright, subtitles, distorted sound, saturated sound, loud"
    )
    sample_config.width = args.width
    sample_config.height = args.height
    sample_config.frames = 121
    sample_config.seed = 42
    sample_config.random_seed = False
    sample_config.diffusion_steps = args.steps
    sample_config.cfg_scale = args.cfg
    sample_config.ltx_use_distilled_lora = bool(args.distilled_lora)
    sample_config.ltx_multi_scale_mode = LtxMultiScaleMode(args.multi_scale_mode)
    sample_config.ltx_distilled_lora_stage1_strength = args.stage1_strength
    sample_config.ltx_distilled_lora_stage2_strength = args.stage2_strength

    sampler = Ltx2Sampler(
        train_device=device,
        temp_device=torch.device("cpu"),
        model=model,
        model_type=ModelType.LTX_2_3,
    )

    dest_dir = Path(__file__).resolve().parent.parent / "workspace_out"
    dest_dir.mkdir(exist_ok=True)
    destination = str(dest_dir / args.out)

    step_t = {"last": None}

    def on_progress(step, total):
        now = time.time()
        dt = now - step_t["last"] if step_t["last"] is not None else 0.0
        step_t["last"] = now
        print(f"[run] sampling step {step}/{total} ({dt:.2f}s/it)", flush=True)
        mem(f"step {step}/{total}")

    print(f"[run] sampling (cfg={args.cfg}, steps={args.steps}, mode={args.multi_scale_mode}, "
          f"{args.width}x{args.height}, "
          f"distilled_lora={'on strength=' + str(args.stage1_strength) if args.distilled_lora else 'off'})...",
          flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    mem("before sampling (peak reset)")
    t1 = time.time()
    step_t["last"] = time.time()
    sampler.sample(
        sample_config=sample_config,
        destination=destination,
        video_format=VideoFormat.MP4,
        on_update_progress=on_progress,
    )
    print(f"[run] sampled in {time.time() - t1:.1f}s", flush=True)
    mem("after sampling")
    print(f"[run] DONE. Output: {destination}{VideoFormat.MP4.extension()}", flush=True)


if __name__ == "__main__":
    main()
