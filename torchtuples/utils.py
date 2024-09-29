import time
import random
import numpy as np
import torch
from torchtuples import tuplefy, TupleTree

def make_name_hash(name='', file_ending='.pt'):
    year, month, day, hour, minute, second = time.localtime()[:6]
    ascii_letters_digits = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    random_hash = ''.join(random.choices(ascii_letters_digits, k=20))
    path = f"{name}_{year}-{month}-{day}_{hour}-{minute}-{second}_{random_hash}{file_ending}"
    return path

class TimeLogger:
    def __init__(self, start=None):
        self.start = self.time() if start is None else start
        self.prev = self.start

    @staticmethod
    def time():
        return time.time()

    def diff(self):
        prev, self.prev = (self.prev, self.time())
        return self.prev - self.start, self.prev - prev

    @staticmethod
    def _hms_from_sec(sec):
        """Hours, minutes, seconds."""
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s

    @staticmethod
    def _hms_str(h, m, s, shorten=True):
        """Hours, minutes, seconds."""
        hs = f"{int(h)}h:"
        ms = f"{int(m)}m:"
        ss = f"{int(s)}s"
        if shorten:
            if h == 0:
                hs = ''
                if m == 0:
                    ms = ''
        return f"{hs}{ms}{ss}"
        # return f"{int(h)}h:{int(m)}m:{int(s)}s"

    def hms_diff(self, shorten=True):
        diff_start, diff_prev = self.diff()
        hms_start = self._hms_from_sec(diff_start)
        hms_prev = self._hms_from_sec(diff_prev)
        return self._hms_str(*hms_start, shorten), self._hms_str(*hms_prev, shorten)


def array_or_tensor(tensor, numpy, input):
    """Returs a tensor if numpy is False or input is tensor.
    Else it returns numpy array, even if input is a DataLoader.
    """
    is_tensor = None
    if numpy is False:
        is_tensor = True
    elif (numpy is True) or is_dl(input):
        is_tensor = False
    elif not (is_data(input) or is_dl(input)):
        raise ValueError(f"Do not understand type of `input`: {type(input)}")
    elif tuplefy(input).type() is torch.Tensor:
        is_tensor = True
    elif tuplefy(input).type() is np.ndarray:
        is_tensor = False
    else:
        raise ValueError("Something wrong")
    
    if is_tensor:
        tensor = tuplefy(tensor).to_tensor().val_if_single()
    else:
        tensor = tuplefy(tensor).to_numpy().val_if_single()
    return tensor

def is_data(input):
    """Returns True if `input` is data of type tuple, list, TupleTree, np.array, torch.Tensor."""
    datatypes = [np.ndarray, torch.Tensor, tuple, list, TupleTree]
    return any([isinstance(input, ct) for ct in datatypes])

def is_dl(input):
    """Returns True if `input` is a DataLoader (inherit from DataLoader)."""
    return isinstance(input, torch.utils.data.DataLoader)
