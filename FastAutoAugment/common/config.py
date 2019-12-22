import argparse
from typing import Type, Optional
from collections import UserDict
from typing import Sequence
from argparse import ArgumentError
from collections.abc import Mapping, MutableMapping
import yaml

# global config instance
_config:'Config' = None

def deep_update(d:MutableMapping, u:Mapping, map_type:Type[MutableMapping]=dict)\
        ->MutableMapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, map_type()), v)
        else:
            d[k] = v
    return d

class Config(UserDict):
    def __init__(self, config_filepath:str=None, app_desc:str=None, use_args=False,
                 defaults_filepath: str = None, param_args: Sequence[Optional[str]] = []) -> None:
        """Create config from specified files and args

        Config is simply a dictionary of key, value map. The value can itself be dictionary so config can be hierarchical. This class allows to load config from yaml. You can specify two yaml files: defaults_filepath which is loaded first and config_filepath which overrides defaults. The idea is that config_filepath provides some good defaults and config_filepath provides values specific to
        some environment or algorithm. On the top of that you can use param_args to override parameters for a given run.

        Keyword Arguments:
            config_filepath {[str]} -- [Yaml file to load config from] (default: {None})
            app_desc {[str]} -- [app description that will show up in --help] (default: {None})
            use_args {bool} -- [if true then command line parameters will override parameters from config files] (default: {False})
            defaults_filepath {[str]} -- [this file is loaded first and provides defaults. The config_filepath will override parameters specified in defaults_filepath] (default: {None})
            param_args {Sequence[str]} -- [parameters specified as ['--key1',val1,'--key2',val2,...] which will override parameters from config file.] (default: {[]})
        """
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
        if defaults_filepath:
            default_yaml = {}
            with open(defaults_filepath, 'r') as f:
                default_yaml = yaml.load(f, Loader=yaml.Loader)
                print('defaults config loaded from: ', config_filepath)
            deep_update(self, default_yaml, map_type=Config)

        # get main config that would override defaults
        main_yaml = {}
        if config_filepath:
            with open(config_filepath, 'r') as f:
                main_yaml = yaml.load(f)
                print('config loaded from: ', config_filepath)

        # merge from params
        Config._update_config_from_args(main_yaml, param_args)
        # merge from command line
        Config._update_config_from_args(main_yaml, self.extra_args)

        # override defaults with main
        deep_update(self, main_yaml, map_type=Config)


    @staticmethod
    def _update_config_from_args(conf:dict, args:Sequence[str])->None:
        for i, arg in enumerate(args):
            if i % 2 != 0:
                continue
            if i == len(args)-1:
                raise ArgumentError('Value is expected after argument {key}')
            if arg.startswith(("--")):
                key = arg[len("--"):]
                if key not in conf:
                    raise ArgumentError('{key} argument not recognized')
                if conf[key] is None:
                    raise ArgumentError('{key} argument type cannot be determined as its value in yaml is None')
                conf[key] = type(conf[key])(args[i+1])

    @staticmethod
    def set(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get()->'Config':
        global _config
        return _config

