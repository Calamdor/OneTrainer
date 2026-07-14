from modules.model.Ltx2Model import Ltx2Model
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.ltx2.Ltx2ModelSaver import Ltx2ModelSaver
from modules.util.enum.ModelType import ModelType

Ltx2FineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.LTX_2_3,
    model_class=Ltx2Model,
    model_saver_class=Ltx2ModelSaver,
    embedding_saver_class=None,
)
