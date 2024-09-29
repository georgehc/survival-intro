import numpy as np
import torch 
from torchtuples.tupletree import TupleTree, tuplefy


def assert_tupletree_equal(a, b, check_dtypes=True):
    assert type(a) == type(b) == TupleTree, 'Not TupleTree'
    assert a.numerate() == b.numerate(), 'Not same structure'
    assert a.types() == b.types(), 'Not same types'
    if check_dtypes:
        ad, bd = (tuplefy(a, b)
                  .apply(lambda x: x.dtype if hasattr(x, 'dtype') else 'not_tensor'))
        assert ad == bd, 'Not same dtype'

    for aa, bb in zip(a.flatten(), b.flatten()):
        if hasattr(aa, 'dtype'):
            assert (aa == bb).all(), 'Not equal values'
        else:
            assert aa == bb, 'Not equal values'