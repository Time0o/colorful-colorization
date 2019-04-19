import json
import os
import sys
from collections import Mapping, Sequence
from importlib import import_module

from .cielab import ABGamut, CIELAB
from .model import Model


# recursive dictionary operations

def _recurse_dictionary(d):
    for k, v in d.items():
        if isinstance(v, Mapping):
            yield from [
                [[k] + path[0], path[1]] for path in _recurse_dictionary(v)
            ]
        else:
            yield [[k], v]


def _get_nested_dictionary(d, path):
    d = d
    for next_dict in path:
        if isinstance(d[next_dict], Mapping):
            d = d[next_dict]
        else:
            break

    return d


def _update_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _update_recursive(d.get(k, {}), v)
        else:
            d[k] = v

    return d


# file path operations

def _is_path(path):
    return isinstance(path, Sequence) and path[0] == 'path'


def _adapt_path(path, root):
    path = path[1]

    if os.path.isabs(path):
        return path
    else:
        return os.path.join(root, path)


def _adapt_paths(config, root):
    for val_path, val in _recurse_dictionary(config):
        if _is_path(val):
            parent_dict = _get_nested_dictionary(config, val_path)
            parent_dict[val_path[-1]] = _adapt_path(val, root)


# config loading and merging

def _load_config(path):
    with open(path, 'r') as f:
        try:
            config = json.load(f)
        except json.decoder.JSONDecodeError:
            fmt = "not a valid JSON file: '{}'"
            raise ValueError(fmt.format(path))

    return config


def _merge_configs(config, default):
    default_copy = default.copy()

    return _update_recursive(default_copy, config)


# configuration post-processing

def _apply_hooks(config):
    def construct_cielab(path):
        if path is None:
            cielab = CIELAB()
        else:
            cielab = CIELAB(ABGamut.from_file(path))

        return 'cielab', cielab

    HOOKS = {
        ('network_args', 'params', 'ab_gamut_file'): construct_cielab
    }

    for val_path, val in _recurse_dictionary(config):
        if tuple(val_path) in HOOKS:
            parent_dict = _get_nested_dictionary(config, val_path)

            new_key, new_val = HOOKS[tuple(val_path)](val)

            del parent_dict[val_path[-1]]
            parent_dict[new_key] = new_val


def _get_config(path, default_path):
    config = _load_config(path)

    if default_path is not None:
        default = _load_config(default_path)
        config = _merge_configs(config, default)

    _adapt_paths(config, os.path.dirname(os.path.abspath(path)))

    _apply_hooks(config)

    return config


# string to object conversion operations

def _get_class(name):
    module, classname = name.rsplit('.', 1)

    return getattr(import_module(module), classname)


def _construct_class(config, *extra_args, **extra_kwargs):
    constructor = _get_class(config['type'])
    params = config.get('params', {})

    return constructor(*extra_args, **extra_kwargs, **params)


# public functions

def dataloader_from_config(path, default_path=None):
    config = _get_config(path, default_path)

    # create dataset
    dataset = _construct_class(config['dataset_args'])

    # create dataloader
    dataloader = _construct_class(config['dataloader_args'], dataset)

    return dataloader


def model_from_config(path, default_path=None):
    config = _get_config(path, default_path)

    # create network
    network = _construct_class(config['network_args'])

    # create loss function
    loss = _construct_class(config['loss_args'])

    # create optimizer
    optimizer = _construct_class(config['optimizer_args'], network.parameters())

    # construct model
    return Model(network=network, loss=loss, optimizer=optimizer)
