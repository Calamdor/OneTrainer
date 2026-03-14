from modules.model.WanModel import WanModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.wan.WanLoRASaver import WanLoRASaver
from modules.util.enum.ModelType import ModelType

WanLoRAModelSaver = make_lora_model_saver(
    ModelType.WAN2_2_T2V,
    model_class=WanModel,
    lora_saver_class=WanLoRASaver,
    embedding_saver_class=None,
)
