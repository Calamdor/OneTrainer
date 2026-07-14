from modules.model.Ltx2Model import Ltx2Model
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.ltx2.Ltx2LoRASaver import Ltx2LoRASaver
from modules.util.enum.ModelType import ModelType

Ltx2LoRAModelSaver = make_lora_model_saver(
    ModelType.LTX_2_3,
    model_class=Ltx2Model,
    lora_saver_class=Ltx2LoRASaver,
    embedding_saver_class=None,
)
