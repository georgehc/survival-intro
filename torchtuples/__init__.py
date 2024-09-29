# -*- coding: utf-8 -*-

"""Top-level package for torchtuples."""

__author__ = """Haavard Kvamme"""
__email__ = 'haavard.kvamme@gmail.com'
__version__ = '0.2.0'

try:
    import torch
    if torch.__version__ < '1.1.0':
        raise ImportError("""Need a torch version that is at least '1.1.0'""")
except ModuleNotFoundError:
    raise ModuleNotFoundError("""You need to install pytorch! See https://pytorch.org/get-started/locally/""")


import torchtuples.tupletree
from torchtuples.tupletree import TupleTree, tuplefy, to_device, numpy_to_tensor, tensor_to_numpy, make_dataloader
import torchtuples.data
import torchtuples.base
import torchtuples.callbacks
import torchtuples.practical
import torchtuples.optim

from torchtuples.base import Model

cb = torchtuples.callbacks
