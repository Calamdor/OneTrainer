from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.ltx2.Ltx2LoRALoader import Ltx2LoRALoader
from modules.modelLoader.ltx2.Ltx2ModelLoader import Ltx2ModelLoader
from modules.util.enum.ModelType import ModelType

Ltx2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.LTX_2_3: "resources/sd_model_spec/ltx2_3_t2v-lora.json"},
    model_class=Ltx2Model,
    model_loader_class=Ltx2ModelLoader,
    embedding_loader_class=None,
    lora_loader_class=Ltx2LoRALoader,
)
