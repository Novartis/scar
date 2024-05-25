# -*- coding: utf-8 -*-
from importlib.metadata import version
__version__ = version("scar")

from .main._scar import model
from .main._setup import setup_anndata
from .main import _data_generater as data_generator
