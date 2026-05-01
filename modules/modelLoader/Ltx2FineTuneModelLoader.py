from modules.model.Ltx2Model import Ltx2Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.ltx2.Ltx2ModelLoader import Ltx2ModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

Ltx2FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.LTX_2_3: "resources/sd_model_spec/ltx2_3_t2v.json"},
    model_class=Ltx2Model,
    model_loader_class=Ltx2ModelLoader,
    embedding_loader_class=None,
    training_methods=[TrainingMethod.FINE_TUNE],
)
