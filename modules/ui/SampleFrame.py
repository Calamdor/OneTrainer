from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.LtxMultiScaleMode import LtxMultiScaleMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class SampleFrame(ctk.CTkFrame):
    def __init__(
            self,
            parent,
            sample: SampleConfig,
            ui_state: UIState,
            model_type: ModelType,
            include_prompt: bool = True,
            include_settings: bool = True,
    ):
        ctk.CTkFrame.__init__(self, parent, fg_color="transparent")

        self.sample = sample
        self.ui_state = ui_state
        self.model_type = model_type

        is_flow_matching = model_type.is_flow_matching()
        is_inpainting_model = model_type.has_conditioning_image_input()
        is_video_model = model_type.is_video_model()

        if include_prompt and include_prompt:
            self.grid_rowconfigure(0, weight=0)
            self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        if include_prompt:
            top_frame = ctk.CTkFrame(self, fg_color="transparent")
            top_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

            top_frame.grid_columnconfigure(0, weight=0)
            top_frame.grid_columnconfigure(1, weight=1)

        if include_settings:
            bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
            bottom_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

            bottom_frame.grid_columnconfigure(0, weight=0)
            bottom_frame.grid_columnconfigure(1, weight=1)
            bottom_frame.grid_columnconfigure(2, weight=0)
            bottom_frame.grid_columnconfigure(3, weight=1)

        if include_prompt:
            # prompt
            components.label(top_frame, 0, 0, "prompt:")
            components.entry(top_frame, 0, 1, self.ui_state, "prompt")

            # negative prompt
            components.label(top_frame, 1, 0, "negative prompt:")
            components.entry(top_frame, 1, 1, self.ui_state, "negative_prompt")

        if include_settings:
            # width
            components.label(bottom_frame, 0, 0, "width:")
            components.entry(bottom_frame, 0, 1, self.ui_state, "width")

            # height
            components.label(bottom_frame, 0, 2, "height:")
            components.entry(bottom_frame, 0, 3, self.ui_state, "height")

            if is_video_model:
                # frames
                components.label(bottom_frame, 1, 0, "frames:",
                                tooltip="Number of frames to generate. Only used when generating videos.")
                components.entry(bottom_frame, 1, 1, self.ui_state, "frames")

                # frame rate (LTX-2.3: variable fps; Wan/Hunyuan: fixed)
                if model_type and model_type.is_ltx_video():
                    components.label(bottom_frame, 1, 2, "fps:",
                                    tooltip="Output frame rate. LTX-2.3 supports variable fps; the model conditions on this value during generation. Default 24.")
                    components.entry(bottom_frame, 1, 3, self.ui_state, "frame_rate")
                else:
                    # length
                    components.label(bottom_frame, 1, 2, "length:",
                                    tooltip="Length in seconds of audio output.")
                    components.entry(bottom_frame, 1, 3, self.ui_state, "length")

            # seed
            components.label(bottom_frame, 2, 0, "seed:")
            components.entry(bottom_frame, 2, 1, self.ui_state, "seed")

            # random seed
            components.label(bottom_frame, 2, 2, "random seed:")
            components.switch(bottom_frame, 2, 3, self.ui_state, "random_seed")

            # cfg scale
            components.label(bottom_frame, 3, 0, "cfg scale:")
            components.entry(bottom_frame, 3, 1, self.ui_state, "cfg_scale")

            # cfg scale 2 (transformer_2 / low-noise expert) — Wan2.2 only
            if model_type and model_type.is_wan_video():
                components.label(bottom_frame, 3, 2, "cfg scale 2:",
                                 tooltip="CFG scale for the low-noise expert (transformer_2). Default 3.0 per Wan2.2 T2V standard.")
                components.entry(bottom_frame, 3, 3, self.ui_state, "cfg_scale_2")

            # sampler
            if not is_flow_matching:
                components.label(bottom_frame, 4, 2, "sampler:")
                components.options_kv(bottom_frame, 4, 3, [
                    ("DDIM", NoiseScheduler.DDIM),
                    ("Euler", NoiseScheduler.EULER),
                    ("Euler A", NoiseScheduler.EULER_A),
                    # ("DPM++", NoiseScheduler.DPMPP), # TODO: produces noisy samples
                    # ("DPM++ SDE", NoiseScheduler.DPMPP_SDE), # TODO: produces noisy samples
                    ("UniPC", NoiseScheduler.UNIPC),
                    ("Euler Karras", NoiseScheduler.EULER_KARRAS),
                    ("DPM++ Karras", NoiseScheduler.DPMPP_KARRAS),
                    ("DPM++ SDE Karras", NoiseScheduler.DPMPP_SDE_KARRAS),
                    ("UniPC Karras", NoiseScheduler.UNIPC_KARRAS)
                ], self.ui_state, "noise_scheduler")

            # steps
            components.label(bottom_frame, 4, 0, "steps:")
            components.entry(bottom_frame, 4, 1, self.ui_state, "diffusion_steps")

            # multi-scale mode (LTX-2.3 only) — slots into the sampler dropdown
            # column on row 4 since LTX is flow-matching and skips that block.
            if model_type and model_type.is_ltx_video():
                components.label(bottom_frame, 4, 2, "multi-scale:",
                                tooltip="LTX-2.3 two-stage sampling. 'Full size' = "
                                        "single-pass at the requested W×H. '1.5x upscale' "
                                        "and '2x upscale' generate at the requested W×H first, "
                                        "then upsample latents and run a 3-step refiner "
                                        "at the larger output resolution. Output will be "
                                        "1.5× or 2× larger than the specified W×H.",
                                wide_tooltip=True)
                _ltx_multi_scale_widget = components.options_kv(bottom_frame, 4, 3, [
                    ("Full size",     LtxMultiScaleMode.FULL_SIZE),
                    ("1.5x upscale",  LtxMultiScaleMode.X1_5),
                    ("2x upscale",    LtxMultiScaleMode.X2),
                ], self.ui_state, "ltx_multi_scale_mode")

                # vae tiling toggle (LTX-2.3 only) — slots into row 5 col 2/3
                components.label(bottom_frame, 5, 2, "vae tiling:",
                                tooltip="Tile the VAE decode spatially. "
                                        "Default ON. Tile size controls peak VRAM: "
                                        "256px ≈ 2 GB/tile (safe on 32 GB), "
                                        "512px ≈ 8 GB/tile (may spill to shared memory).",
                                wide_tooltip=True)
                components.switch(bottom_frame, 5, 3, self.ui_state, "ltx_vae_tiling")
                components.label(bottom_frame, 5, 4, "tile size:",
                                tooltip="Spatial tile size in pixels for tiled VAE decode. "
                                        "256 = safe on 32 GB cards (~2 GB peak per tile). "
                                        "512 = ComfyUI default but peaks at ~8 GB per tile.")
                components.entry(bottom_frame, 5, 5, self.ui_state, "ltx_vae_tile_size")

                # Distilled LoRA toggle + per-stage weights (row 6).
                # Disabling the toggle also disables multi-scale mode and strength entries
                # since two-stage sampling depends on the distilled LoRA.
                components.label(bottom_frame, 6, 0, "distill LoRA:",
                                tooltip="Enable the distilled LoRA during sampling. "
                                        "Disable to save ~7.6 GB VRAM without clearing the path. "
                                        "Two-stage multi-scale and strength controls are disabled when off.")
                _distill_dependent_widgets = []

                def _update_distill_dependent():
                    enabled = self.ui_state.get_var("ltx_use_distilled_lora").get() in (True, "True", "1", 1)
                    state = "normal" if enabled else "disabled"
                    for w in _distill_dependent_widgets:
                        try:
                            w.configure(state=state)
                        except Exception:
                            pass

                components.switch(bottom_frame, 6, 1, self.ui_state, "ltx_use_distilled_lora",
                                 command=_update_distill_dependent)
                components.label(bottom_frame, 6, 2, "stage 1 str:",
                                tooltip="Distilled LoRA strength for stage 1. ComfyUI community default: 0.30.")
                _distill_dependent_widgets.append(
                    components.entry(bottom_frame, 6, 3, self.ui_state, "ltx_distilled_lora_stage1_strength")
                )
                components.label(bottom_frame, 6, 4, "stage 2 str:",
                                tooltip="Distilled LoRA strength for stage 2 (high-res refinement pass). ComfyUI community default: 0.60.")
                _distill_dependent_widgets.append(
                    components.entry(bottom_frame, 6, 5, self.ui_state, "ltx_distilled_lora_stage2_strength")
                )
                _distill_dependent_widgets.append(_ltx_multi_scale_widget)
                _update_distill_dependent()

            # inpainting
            if is_inpainting_model:
                components.label(bottom_frame, 5, 0, "inpainting:",
                                tooltip="Enables inpainting sampling. Only available when sampling from an inpainting model.")
                components.switch(bottom_frame, 5, 1, self.ui_state, "sample_inpainting")

                # base image path
                components.label(bottom_frame, 6, 0, "base image path:",
                                tooltip="The base image used when inpainting.")
                components.file_entry(bottom_frame, 6, 1, self.ui_state, "base_image_path",
                                    mode="file",
                                    allow_model_files=False,
                                    allow_image_files=True,
                                    )

                # mask image path
                components.label(bottom_frame, 6, 2, "mask image path:",
                                tooltip="The mask used when inpainting.")
                components.file_entry(bottom_frame, 6, 3, self.ui_state, "mask_image_path",
                                    mode="file",
                                    allow_model_files=False,
                                    allow_image_files=True,
                                    )
