from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from overrides import EnforceOverrides

from .model_desc import ModelDesc, CellDesc
from ..common.trainer import Trainer
from ..common.config import Config

class MicroBuilder(ABC, EnforceOverrides):
    def register_ops(self)->None:
        pass
    def build(self, model_desc:ModelDesc)->None:
        pass

