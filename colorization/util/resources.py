import os
from importlib import import_module


_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, '../../resources')


def get_resource_path(path):
   return os.path.join(_RESOURCE_DIR, path)


def get_class(name: str) -> object:
    # Obtain a class object from its string representation, e.g. if `name` is
    # `'torch.optim.Adam'`, `torch.optim.Adam` will be returned.

    module, classname = name.rsplit('.', 1)

    return getattr(import_module(module), classname)
