import os
import sys

source_dir = os.path.dirname(__file__)
parent_dir = os.path.join(source_dir, '..')

sys.path.append(os.path.abspath(parent_dir))
