import yaml
import argparse
from collections import UserDict

from argparse import ArgumentError

_config:'Config' = None

class Config(UserDict):
    def __init__(self, config_filepath=None, app_desc=None, use_args=True,
        defaults_filepath=None)->None:

        super(Config, self).__init__()

        self.args, self.extra_args = None, []
        if use_args:
            parser = argparse.ArgumentParser(description=app_desc)
            parser.add_argument('--config-filepath', '-c', type=str, default=None,
                help='config filepath, defaults are loaded from defaults.yaml')
            self.args, self.extra_args = parser.parse_known_args()
            config_filepath = self.args.config_filepath or config_filepath

        # first load defaults
        if defaults_filepath:
            with open(defaults_filepath, 'r') as f:
                self.update(yaml.safe_load(f))
                print('defaults config loaded from: ', config_filepath)
        if config_filepath:
            with open(config_filepath, 'r') as f:
                self.update(yaml.safe_load(f))
                print('config loaded from: ', config_filepath)

        # merge from command line
        for i, arg in enumerate(self.extra_args):
            if i % 2 != 0:
                continue
            if i == len(self.extra_args)-1:
                raise ArgumentError('Value is expected after argument {key}')
            if arg.startswith(("--")):
                arg = arg[len("--"):]
                path = arg.split('.')
                c = self
                for key in path[:-1]:
                    if not key in c:
                        raise ArgumentError('{key} argument not recognized')
                    c = c[key]
                key = path[-1]
                if not key in c:
                    raise ArgumentError('{key} argument not recognized')
                if c[key] is None:
                    raise ArgumentError('{key} argument type cannot be determined as its value in yaml is None')
                c[key] = type(c[key])(self.extra_args[i+1])


        # without below Python would let static method override instance method
        self.get = super(Config, self).get

    @staticmethod
    def set(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get()->'Config':
        global _config
        return _config
