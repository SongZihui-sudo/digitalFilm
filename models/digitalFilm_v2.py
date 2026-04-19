import torch

from .base_model import model_base
from .disc import patchDiscriminator
from .digitalFilm_g import digitalFilm_G
from utils.utils import compile_model


class digitalFilmv2(model_base):
    def __init__(self, rank, global_config: dict, config: dict) -> None:
        super().__init__(rank, global_config)

        self.__config = config

        self.g: digitalFilm_G = digitalFilm_G(config)
        self.d: patchDiscriminator = patchDiscriminator(3, config.ndf, config.disc_n_layers, torch.nn.InstanceNorm2d, config.disc_no_antialias,
                                                        config.disc_patch_size)
        self.g = compile_model(self.g, global_config.mfast)
        self.d = compile_model(self.d, global_config.mfast)

        self._model = [["g", self.g], ["d", self.d]]
