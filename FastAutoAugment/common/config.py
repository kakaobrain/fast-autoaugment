import argparse
from typing import Type, Optional
from collections import UserDict
from typing import Sequence
from argparse import ArgumentError
from collections.abc import Mapping, MutableMapping
import os
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
    def __init__(self, config_filepath:Optional[str]=None,
                 config_defaults_filepath:Optional[str]=None,
                 app_desc:Optional[str]=None,
                 use_args=False, param_args: Sequence = []) -> None:
        """Create config from specified files and args

        Config is simply a dictionary of key, value map. The value can itself be
        a dictionary so config can be hierarchical. This class allows to load
        config from yaml. A special key 'include' can specify another yaml relative
        file path which will be loaded first and the key-value pairs in the main file
        will override the ones in include file. You can think of include file as
        defaults provider. This allows to create one base config and then several
        environment/experiment specific configs. On the top of that you can use
        param_args to override parameters for a given run.

        Keyword Arguments:
            config_filepath {[str]} -- [Yaml file to load config from] (default: {None})
            config_defaults_filepath {[str]} -- [Yaml file to load defaults from] (default: {None})
            app_desc {[str]} -- [app description that will show up in --help] (default: {None})
            use_args {bool} -- [if true then command line parameters will override parameters from config files] (default: {False})
            param_args {Sequence} -- [parameters specified as ['--key1',val1,'--key2',val2,...] which will override parameters from config file.] (default: {[]})
        """
        super(Config, self).__init__()
        # without below Python would let static method override instance method
        self.get = super(Config, self).get

        self.args, self.extra_args = None, []

        if use_args:
            # let command line args specify/override config file
            parser = argparse.ArgumentParser(description=app_desc)
            parser.add_argument('--config', type=str, default=None,
                help='config filepath in yaml format')
            parser.add_argument('--config-defaults', type=str, default=None,
                help='yaml file to supply defaults, file specified by --config will override any values')
            self.args, self.extra_args = parser.parse_known_args()
            config_filepath = self.args.config or config_filepath
            config_defaults_filepath = self.args.config_defaults or config_defaults_filepath

        self._load_from_file(config_defaults_filepath)
        self._load_from_file(config_filepath)
        self._update_from_args(param_args)      # merge from params
        self._update_from_args(self.extra_args) # merge from command line

    def _load_from_file(self, filepath:Optional[str])->None:
        if filepath:
            with open(os.path.expanduser(filepath), 'r') as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)
            if '__include__' in config_yaml:
                include_filepath = os.path.join(os.path.dirname(filepath),
                                                config_yaml['__include__'])
                self._load_from_file(include_filepath)
            deep_update(self, config_yaml, map_type=Config)
            print('config loaded from: ', filepath)

    def _update_from_args(self, args:Sequence)->None:
        for i, arg in enumerate(args):
            if i % 2 != 0:
                continue
            if i == len(args)-1:
                raise ArgumentError('Value is expected after argument {key}')
            if arg.startswith(("--")):
                path = arg[len("--"):].split('.')
                section = self
                for i in range(len(path)-1):
                    section = section[path[i]]
                key = path[-1]
                section[key] = type(section[key])(args[i+1])

    @staticmethod
    def set(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get()->'Config':
        global _config
        return _config

