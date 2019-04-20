import os


_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, '../../resources')


def resource_path(path):
    return os.path.join(_RESOURCE_DIR, path)
