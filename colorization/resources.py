import os


_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, '../resources')


def get_resource_path(path):
    return os.path.join(_RESOURCE_DIR, path)
