import yaml
import argparse
from collections import UserDict
from typing import List
from argparse import ArgumentError

_config:'Config' = None

class Config(UserDict):
    def __init__(self, config_filepath=None, app_desc=None, use_args=True,
        defaults_filepath=None, param_args:List[str]=[])->None:

        super(Config, self).__init__()

        self.args, self.extra_args = None, []
        if use_args:
            parser = argparse.ArgumentParser(description=app_desc)
            parser.add_argument('--config-filepath', '-c', type=str, default=None,
                help='config filepath, this overrides defaults loaded froj default config')
            parser.add_argument('--defaults-filepath', '-dc', type=str, default=None,
                help='if config does not have key then key from this config is used')

            self.args, self.extra_args = parser.parse_known_args()

            config_filepath = self.args.config_filepath or config_filepath
            defaults_filepath = self.args.defaults_filepath or defaults_filepath

        # get defaults
        self.default_yaml = {}
        if defaults_filepath:
            with open(defaults_filepath, 'r') as f:
                self.default_yaml = yaml.safe_load(f)
                print('defaults config loaded from: ', config_filepath)

        # get main config that would override defaults
        self.main_yaml = {}
        if config_filepath:
            with open(config_filepath, 'r') as f:
                self.main_yaml = yaml.safe_load(f)
                print('config loaded from: ', config_filepath)

        # merge from params
        Config._update_config_from_args(self.main_yaml, param_args)
        # merge from command line
        Config._update_config_from_args(self.main_yaml, self.extra_args)

        # load defaults
        self.update(self.default_yaml)
        # override defaults with main
        self.update(self.main_yaml)

        # without below Python would let static method override instance method
        self.get = super(Config, self).get

    @staticmethod
    def _update_config_from_args(conf:dict, extra_args:List[str])->None:
        for i, arg in enumerate(extra_args):
            if i % 2 != 0:
                continue
            if i == len(extra_args)-1:
                raise ArgumentError('Value is expected after argument {key}')
            if arg.startswith(("--")):
                arg = arg[len("--"):]
                path = arg.split('.')
                c = conf
                for key in path[:-1]:
                    if not key in c:
                        raise ArgumentError('{key} argument not recognized')
                    c = c[key]
                key = path[-1]
                if not key in c:
                    raise ArgumentError('{key} argument not recognized')
                if c[key] is None:
                    raise ArgumentError('{key} argument type cannot be determined as its value in yaml is None')
                c[key] = type(c[key])(extra_args[i+1])

    @staticmethod
    def set(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get()->'Config':
        global _config
        return _config
