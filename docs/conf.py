import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Colorful Image Colorization'
copyright = '2019, Carolina Bianchi, Álvaro Orgaz Expósito, Timo Nicolai'
author = 'Carolina Bianchi, Álvaro Orgaz Expósito, Timo Nicolai'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False
}

html_static_path = ['_static']
