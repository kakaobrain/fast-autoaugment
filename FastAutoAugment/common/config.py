import yaml
import argparse
from collections import UserDict
from typing import List
from argparse import ArgumentError
from collections.abc import Mapping, MutableMapping

def deep_update(d:MutableMapping, u:Mapping)->Mapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# global config instance
_config:'Config' = None

class Config(UserDict):
    def __init__(self, config_filepath=None, app_desc=None, use_args=True,
        defaults_filepath=None, param_args:List[str]=[])->None:

        super(Config, self).__init__()
        # without below Python would let static method override instance method
        self.get = super(Config, self).get

        self.args, self.extra_args = None, []

        # should we lok for program args for config file locations?
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
        default_yaml = {}
        if defaults_filepath:
            with open(defaults_filepath, 'r') as f:
                default_yaml = yaml.safe_load(f)
                print('defaults config loaded from: ', config_filepath)
        self.update(default_yaml)

        # get main config that would override defaults
        main_yaml = {}
        if config_filepath:
            with open(config_filepath, 'r') as f:
                main_yaml = yaml.safe_load(f)
                print('config loaded from: ', config_filepath)

        # merge from params
        Config._update_config_from_args(main_yaml, param_args)
        # merge from command line
        Config._update_config_from_args(main_yaml, self.extra_args)

        # override defaults with main
        deep_update(self, main_yaml)


    @staticmethod
    def _update_config_from_args(conf:dict, extra_args:List[str])->None:
        for i, arg in enumerate(extra_args):
            if i % 2 != 0:
                continue
            if i == len(extra_args)-1:
                raise ArgumentError('Value is expected after argument {key}')
            if arg.startswith(("--")):
                arg = arg[len("--"):]
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

