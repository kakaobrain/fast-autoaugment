from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from overrides import EnforceOverrides

from .model_desc import ModelDesc, CellDesc
from ..common.trainer import Trainer
from ..common.config import Config

class DagMutator(ABC, EnforceOverrides):
    @abstractmethod
    def mutate(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)


