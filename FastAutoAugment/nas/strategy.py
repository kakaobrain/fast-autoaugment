from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from overrides import EnforceOverrides

from .model_desc import ModelDesc, CellDesc

class Strategy(ABC, EnforceOverrides):
    def apply(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._add_cell_nodes(cell_desc)

    @abstractmethod
    def _add_cell_nodes(self, cell_desc:CellDesc)->None:
        pass
