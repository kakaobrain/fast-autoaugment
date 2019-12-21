from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .model import Model

from ..common.config import Config
from .model_desc import ModelDesc
from ..common.metrics import Metrics

class ArchTrainer(ABC, EnforceOverrides):
    """Find architecture for given dataset and model with arch parameters"""

    @abstractmethod
    def fit(self, conf_search:Config, model:Model, device,
            train_dl:DataLoader, val_dl:Optional[DataLoader],
            epochs:int, plotsdir:str, report_freq:int)\
                ->Tuple[ModelDesc, Metrics, Optional[Metrics]]:
        pass