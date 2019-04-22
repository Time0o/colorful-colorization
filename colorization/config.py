import json
import logging
import logging.config
import os
import sys
from collections import Mapping, Sequence
from importlib import import_module
from typing import Dict, Generator, Tuple, Union

from .cielab import ABGamut, CIELAB
from .model import Model


# recursive dictionary operations

def _recurse_dictionary(d: dict) -> Generator[Tuple[list, object], None, None]:
    # Return all values plus corresponding keys in a nested dictionary in the
    # form `[keys, val]` where `keys` is the list of successive keys necessary
    # to obtain `val`, I.e. if `keys = [key_1, ..., key_n]` we have
    # `d[key_1]...[key_n]` = val`.

    for k, v in d.items():
        if isinstance(v, Mapping):
            yield from [
                ([k] + g[0], g[1]) for g in _recurse_dictionary(v)
            ]
        else:
            yield [k], v


def _get_nested_dictionary(d: dict, path: list) -> dict:
    # Obtain a directory nested inside another directory based on a list of
    # keys (`path`). Keys are taken from the front of this list and used to
    # recursively index `d` until all keys are exhausted or a non-dictionary is
    # encountered (in which case the last encountered dictionary is returned).

    d = d
    for next_dict in path:
        if isinstance(d[next_dict], Mapping):
            d = d[next_dict]
        else:
            break

    return d


def _update_recursive(d: dict, u: dict) -> dict:
    # Udpate `d` based on `u` similar to `dict.update` except recurse into
    # nested dictionaries instead of overwriting them.

    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _update_recursive(d.get(k, {}), v)
        else:
            d[k] = v

    return d


# file path operations

def _is_path(path: object) -> bool:
    # Check whether a configuration dictionary value constitutes a file "path",
    # represented by lists of length two whose first element is the string
    # `'path'` and whose second element is an absolute or relative file path.

    return isinstance(path, Sequence) and path[0] == 'path'


def _resolve_path(path: Tuple[str, str], root: str) -> str:
    # Transform a file path of the form checked by `_is_path` into an absolute
    # file path, prepending `root` to the path if it is not already absolute
    # (`root` should be an absolute file path itself).

    path = path[1]

    if os.path.isabs(path):
        return path
    else:
        return os.path.join(root, path)


def _resolve_paths(config: dict, root: str) -> None:
    # Recursively apply `_resolve_path` to all paths in a configuration
    # dictionary.

    for val_path, val in _recurse_dictionary(config):
        if _is_path(val):
            parent_dict = _get_nested_dictionary(config, val_path)
            parent_dict[val_path[-1]] = _resolve_path(val, root)


# configuration post-processing

def _apply_hooks(config: Dict[str, dict]) -> None:
    # Transform values for specific keys in a configuration dictionary according
    # to predefined rules. This way the configuration file can be a simple JSON
    # file instead of a Python snippet.

    HOOKS = {
        # TODO
    }

    for val_path, val in _recurse_dictionary(config):
        if tuple(val_path) in HOOKS:
            parent_dict = _get_nested_dictionary(config, val_path)

            new_key, new_val = HOOKS[tuple(val_path)](val)

            del parent_dict[val_path[-1]]
            parent_dict[new_key] = new_val


# config loading and merging

def _load_config(path: str) -> Dict[str, dict]:
    # Load a configuratio dictionary from a JSON file.

    with open(path, 'r') as f:
        try:
            config = json.load(f)
        except json.decoder.JSONDecodeError:
            fmt = "not a valid JSON file: '{}'"
            raise ValueError(fmt.format(path))

    return config


def _merge_configs(config: Dict[str, dict],
                   default: Dict[str, dict]) -> Dict[str, dict]:
    # Merge `config` into `default` recursively, i.e. fall back on configuration
    # settings in `default` if they are not specified in `config`.

    default_copy = default.copy()

    return _update_recursive(default_copy, config)



# string to object conversion operations

def _get_class(name: str) -> object:
    # Obtain a class object from its string representation, e.g. if `name` is
    # `'torch.optim.Adam'`, `torch.optim.Adam` will be returned.

    module, classname = name.rsplit('.', 1)

    return getattr(import_module(module), classname)


def _construct_class(config: Dict[str, Union[str, dict]],
                     *extra_args,
                     **extra_kwargs) -> object:
    # Construct an object based on a configuration dictionary which contains
    # the objects type (in string representation) under the key `'type'` and
    # constructor keyword parameters under the key `'params'`.

    constructor = _get_class(config['type'])
    params = config.get('params', {})

    return constructor(*extra_args, **extra_kwargs, **params)


# public functions

def get_config(path: str, default_path=None) -> Dict[str, dict]:
    config = _load_config(path)

    if default_path is not None:
        default = _load_config(default_path)
        config = _merge_configs(config, default)

    _resolve_paths(config, os.path.dirname(os.path.abspath(path)))

    _apply_hooks(config)

    return config


def dataloader_from_config(config):
    # create dataset
    dataset = _construct_class(config['dataset_args'])

    # create dataloader
    dataloader = _construct_class(config['dataloader_args'], dataset)

    return dataloader


def model_from_config(config, trainable=True):
    # create network
    network = _construct_class(config['network_args'])

    if trainable:
        # create loss function
        loss = _construct_class(config['loss_args'])

        # create optimizer
        optimizer = _construct_class(config['optimizer_args'],
                                     network.parameters())
    else:
        loss = None
        optimizer = None

    # construct model
    return Model(network=network,
                 loss=loss,
                 optimizer=optimizer,
                 **config['model_args'])


def logger_from_config(config):
    # configure logger
    if 'log_args' in config:
        logging.config.dictConfig(config['log_args'])
