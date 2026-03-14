from modules.model.WanModel import WanModel
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class WanEmbeddingLoader(EmbeddingLoaderMixin):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: WanModel,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
