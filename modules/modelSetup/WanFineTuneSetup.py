from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseWanSetup import BaseWanSetup
from modules.util import factory
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch


class WanFineTuneSetup(BaseWanSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )


factory.register(BaseModelSetup, WanFineTuneSetup, ModelType.WAN2_2_T2V, TrainingMethod.FINE_TUNE)
