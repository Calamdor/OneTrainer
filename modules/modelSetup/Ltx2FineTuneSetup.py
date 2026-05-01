from modules.modelSetup.BaseLtx2Setup import BaseLtx2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch


class Ltx2FineTuneSetup(BaseLtx2Setup):
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


factory.register(BaseModelSetup, Ltx2FineTuneSetup, ModelType.LTX_2_3, TrainingMethod.FINE_TUNE)
