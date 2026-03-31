
from modules.modelSetup.mixin.ModelSetupNoiseMixin import (
    ModelSetupNoiseMixin,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.WanExpertMode import WanExpertMode
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import torch
from torch import Tensor

import customtkinter as ctk
from customtkinter import AppearanceModeTracker, ThemeManager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TimestepGenerator(ModelSetupNoiseMixin):

    def __init__(
            self,
            timestep_distribution: TimestepDistribution,
            min_noising_strength: float,
            max_noising_strength: float,
            noising_weight: float,
            noising_bias: float,
            timestep_shift: float,
    ):
        super().__init__()

        self.timestep_distribution = timestep_distribution
        self.min_noising_strength = min_noising_strength
        self.max_noising_strength = max_noising_strength
        self.noising_weight = noising_weight
        self.noising_bias = noising_bias
        self.timestep_shift = timestep_shift

    def generate(self) -> Tensor:
        generator = torch.Generator()
        generator.seed()

        config = TrainConfig.default_values()
        config.timestep_distribution = self.timestep_distribution
        config.min_noising_strength = self.min_noising_strength
        config.max_noising_strength = self.max_noising_strength
        config.noising_weight = self.noising_weight
        config.noising_bias = self.noising_bias
        config.timestep_shift = self.timestep_shift


        return self._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=1000000,
            config=config,
        )


class TimestepDistributionWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            config: TrainConfig,
            ui_state: UIState,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.title("Timestep Distribution")
        self.resizable(True, True)

        self.config = config
        self.ui_state = ui_state
        self.image_preview_file_index = 0
        self.ax = None
        self.canvas = None

        is_wan_both = (config.model_type.is_wan_video()
                       and config.wan_expert_mode == WanExpertMode.BOTH)
        self.geometry("900x600" if is_wan_both else "900x450")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = self.__content_frame(self)
        frame.grid(row=0, column=0, sticky='nsew')
        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.after(200, lambda: set_window_icon(self))
        self.grab_set()
        self.focus_set()

    def __add_expert_fields(self, frame, row, prefix):
        """Add the 5 per-expert distribution fields starting at `row`.
        `prefix` is e.g. 'wan_high_noise_' or 'wan_low_noise_'.
        Returns the next available row.
        """
        components.label(frame, row, 0, "Min Noising Strength",
                         tooltip="Lower bound of the timestep range sampled for this expert.")
        components.entry(frame, row, 1, self.ui_state, f"{prefix}min_strength")
        row += 1

        components.label(frame, row, 0, "Max Noising Strength",
                         tooltip="Upper bound of the timestep range sampled for this expert.")
        components.entry(frame, row, 1, self.ui_state, f"{prefix}max_strength")
        row += 1

        components.label(frame, row, 0, "Noising Weight",
                         tooltip="Controls the spread/scale of the timestep distribution for this expert. "
                                 "For logit-normal: scale = weight + 1.0. "
                                 "Negative values tighten the distribution around the bias center; "
                                 "positive values widen it.")
        components.entry(frame, row, 1, self.ui_state, f"{prefix}noising_weight")
        row += 1

        components.label(
            frame, row, 0, "Noising Bias",
            tooltip="Shifts the center of the logit-normal within this expert's [min, max] window. "
                    "0.0 = center at window midpoint. "
                    "Positive values move the center toward max; negative toward min.",
        )
        components.entry(frame, row, 1, self.ui_state, f"{prefix}noising_bias")
        row += 1

        components.label(frame, row, 0, "Timestep Shift",
                         tooltip="Flow schedule shift applied after sampling. "
                                 "1.0 = identity (recommended). "
                                 "Values > 1 push samples toward higher noise and can bleed past max.")
        components.entry(frame, row, 1, self.ui_state, f"{prefix}timestep_shift")
        row += 1

        return row

    def __content_frame(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        is_wan = self.config.model_type.is_wan_video()
        is_wan_both = is_wan and self.config.wan_expert_mode == WanExpertMode.BOTH
        row = 0

        # timestep distribution — always shown
        components.label(frame, row, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        components.options(frame, row, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state,
                           "timestep_distribution")
        row += 1

        if is_wan_both:
            # --- Wan BOTH mode: two expert panels ---

            # BOTH low-noise fraction
            components.label(
                frame, row, 0, "BOTH Low-Noise Fraction",
                tooltip=(
                    "In BOTH mode, the fraction of training batches assigned to the low-noise expert "
                    "(transformer_2). Default 0.5 gives a 50/50 split between experts. "
                    "Set to 0.55 to give the low-noise expert ~10% more batches — recommended "
                    "because the low-noise expert covers a wider sigma range and "
                    "typically needs more gradient updates to converge. "
                    "Only affects BOTH mode; ignored in HIGH_NOISE and LOW_NOISE modes."
                ),
                wide_tooltip=True,
            )
            components.entry(frame, row, 1, self.ui_state, "wan_low_noise_fraction")
            row += 1

            # High-Noise Expert section
            ctk.CTkLabel(
                frame, text="High-Noise Expert",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(14, 2))
            row += 1

            row = self.__add_expert_fields(frame, row, "wan_high_noise_")

            # Low-Noise Expert section
            ctk.CTkLabel(
                frame, text="Low-Noise Expert",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 2))
            row += 1

            row = self.__add_expert_fields(frame, row, "wan_low_noise_")

            # dynamic timestep shifting
            components.label(frame, row, 0, "Dynamic Timestep Shifting",
                             tooltip="Dynamically shift the timestep distribution based on resolution. "
                                     "If enabled, the shifting parameters are taken from the model's scheduler "
                                     "configuration and Timestep Shift is ignored.",
                             wide_tooltip=True)
            components.switch(frame, row, 1, self.ui_state, "dynamic_timestep_shifting")
            row += 1

            button_row = row + 1  # +1 spacer row
            frame.grid_rowconfigure(row, weight=1)
            canvas_rowspan = button_row

        elif is_wan:
            # --- Wan single-expert mode: standard layout bound to active expert ---

            if self.config.wan_expert_mode == WanExpertMode.HIGH_NOISE:
                prefix = "wan_high_noise_"
            else:
                prefix = "wan_low_noise_"

            row = self.__add_expert_fields(frame, row, prefix)

            # dynamic timestep shifting
            components.label(frame, row, 0, "Dynamic Timestep Shifting",
                             tooltip="Dynamically shift the timestep distribution based on resolution. "
                                     "If enabled, the shifting parameters are taken from the model's scheduler "
                                     "configuration and Timestep Shift is ignored.",
                             wide_tooltip=True)
            components.switch(frame, row, 1, self.ui_state, "dynamic_timestep_shifting")
            row += 1

            # spacer so the canvas has room to breathe below the content
            frame.grid_rowconfigure(row + 1, weight=1)
            button_row = row + 3
            canvas_rowspan = button_row

        else:
            # --- Non-Wan: standard layout ---

            # min noising strength
            components.label(frame, row, 0, "Min Noising Strength",
                             tooltip="Specifies the minimum noising strength used during training. "
                                     "This can help to improve composition, but prevents finer details "
                                     "from being trained")
            components.entry(frame, row, 1, self.ui_state, "min_noising_strength")
            row += 1

            # max noising strength
            components.label(frame, row, 0, "Max Noising Strength",
                             tooltip="Specifies the maximum noising strength used during training. "
                                     "This can be useful to reduce overfitting, but also reduces the "
                                     "impact of training samples on the overall image composition")
            components.entry(frame, row, 1, self.ui_state, "max_noising_strength")
            row += 1

            # noising weight
            components.label(frame, row, 0, "Noising Weight",
                             tooltip="Controls the weight parameter of the timestep distribution function. "
                                     "Use the preview to see more details.")
            components.entry(frame, row, 1, self.ui_state, "noising_weight")
            row += 1

            # noising bias
            components.label(frame, row, 0, "Noising Bias",
                             tooltip="Controls the bias parameter of the timestep distribution function. "
                                     "Use the preview to see more details.")
            components.entry(frame, row, 1, self.ui_state, "noising_bias")
            row += 1

            # timestep shift
            components.label(frame, row, 0, "Timestep Shift",
                             tooltip="Shift the timestep distribution. Use the preview to see more details.")
            components.entry(frame, row, 1, self.ui_state, "timestep_shift")
            row += 1

            # dynamic timestep shifting
            components.label(frame, row, 0, "Dynamic Timestep Shifting",
                             tooltip="Dynamically shift the timestep distribution based on resolution. "
                                     "If enabled, the shifting parameters are taken from the model's scheduler "
                                     "configuration and Timestep Shift is ignored. Dynamic Timestep Shifting is "
                                     "not shown in the preview. Note: For Z-Image and Flux2, the dynamic shifting "
                                     "parameters are likely wrong and unknown. Use with care or set your own, "
                                     "fixed shift.",
                             wide_tooltip=True)
            components.switch(frame, row, 1, self.ui_state, "dynamic_timestep_shifting")
            row += 1

            # spacer so the canvas has room to breathe below the content
            frame.grid_rowconfigure(row + 1, weight=1)
            button_row = row + 3
            canvas_rowspan = button_row

        # plot
        appearance_mode = AppearanceModeTracker.get_mode()
        background_color = self.winfo_rgb(ThemeManager.theme["CTkToplevel"]["fg_color"][appearance_mode])
        text_color = self.winfo_rgb(ThemeManager.theme["CTkLabel"]["text_color"][appearance_mode])
        background_color = f"#{int(background_color[0]/256):x}{int(background_color[1]/256):x}{int(background_color[2]/256):x}"
        text_color = f"#{int(text_color[0]/256):x}{int(text_color[1]/256):x}{int(text_color[2]/256):x}"

        fig, ax = plt.subplots()
        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=canvas_rowspan)

        fig.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        ax.tick_params(axis='x', colors=text_color, which="both")
        ax.tick_params(axis='y', colors=text_color, which="both")
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

        self.__update_preview()

        # update button
        components.button(frame, button_row, 3, "Update Preview", command=self.__update_preview)

        frame.pack(fill="both", expand=1)
        return frame

    def __update_preview(self):
        self.ax.cla()

        if self.config.model_type.is_wan_video():
            expert_mode = self.config.wan_expert_mode
            if expert_mode == WanExpertMode.BOTH:
                # Overlay both expert distributions
                for min_ns, max_ns, weight, bias, shift, color, label in [
                    (
                        self.config.wan_high_noise_min_strength,
                        self.config.wan_high_noise_max_strength,
                        self.config.wan_high_noise_noising_weight,
                        self.config.wan_high_noise_noising_bias,
                        self.config.wan_high_noise_timestep_shift,
                        'steelblue', 'High-Noise',
                    ),
                    (
                        self.config.wan_low_noise_min_strength,
                        self.config.wan_low_noise_max_strength,
                        self.config.wan_low_noise_noising_weight,
                        self.config.wan_low_noise_noising_bias,
                        self.config.wan_low_noise_timestep_shift,
                        'mediumseagreen', 'Low-Noise',
                    ),
                ]:
                    gen = TimestepGenerator(
                        timestep_distribution=self.config.timestep_distribution,
                        min_noising_strength=min_ns,
                        max_noising_strength=max_ns,
                        noising_weight=weight,
                        noising_bias=bias,
                        timestep_shift=shift,
                    )
                    self.ax.hist(gen.generate(), bins=1000, range=(0, 999),
                                 color=color, alpha=0.6, label=label)
                self.ax.legend()
            elif expert_mode == WanExpertMode.HIGH_NOISE:
                gen = TimestepGenerator(
                    timestep_distribution=self.config.timestep_distribution,
                    min_noising_strength=self.config.wan_high_noise_min_strength,
                    max_noising_strength=self.config.wan_high_noise_max_strength,
                    noising_weight=self.config.wan_high_noise_noising_weight,
                    noising_bias=self.config.wan_high_noise_noising_bias,
                    timestep_shift=self.config.wan_high_noise_timestep_shift,
                )
                self.ax.hist(gen.generate(), bins=1000, range=(0, 999), color='steelblue')
            else:  # LOW_NOISE
                gen = TimestepGenerator(
                    timestep_distribution=self.config.timestep_distribution,
                    min_noising_strength=self.config.wan_low_noise_min_strength,
                    max_noising_strength=self.config.wan_low_noise_max_strength,
                    noising_weight=self.config.wan_low_noise_noising_weight,
                    noising_bias=self.config.wan_low_noise_noising_bias,
                    timestep_shift=self.config.wan_low_noise_timestep_shift,
                )
                self.ax.hist(gen.generate(), bins=1000, range=(0, 999), color='mediumseagreen')
        else:
            generator = TimestepGenerator(
                timestep_distribution=self.config.timestep_distribution,
                min_noising_strength=self.config.min_noising_strength,
                max_noising_strength=self.config.max_noising_strength,
                noising_weight=self.config.noising_weight,
                noising_bias=self.config.noising_bias,
                timestep_shift=self.config.timestep_shift,
            )
            self.ax.hist(generator.generate(), bins=1000, range=(0, 999))

        self.canvas.draw()

    def __ok(self):
        self.destroy()
