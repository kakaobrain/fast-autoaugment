from overrides import overrides

from .strategy import Strategy
from .model_desc import ModelDesc

class DartsStrategy(Strategy):
    @overrides
    def get_model_desc(self, conf_ds:dict, conf_model_desc:dict)->ModelDesc:
        pass

