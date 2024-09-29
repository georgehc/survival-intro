from __future__ import absolute_import, division, print_function

import os
from datetime import datetime

import numpy as np
import pandas as pd

SEP = "__"  # Avoid common signs like "_".


def to_np(x):
    return x.data.cpu().numpy()


class MyPrinter(object):
    def __init__(self, verbose, exp_name=None, log_path=None, debug=False):
        self.verbose = verbose
        self.debug = debug
        if log_path is not None:
            self.log_file = os.path.join(log_path, "%s.log" % exp_name)
        else:
            self.log_file = None

    def print(self, content, level=0, print_time=False):
        if self.verbose > level:
            if print_time or self.debug:
                print(" ".join(
                    ["\033[90m---",
                     str(datetime.now()), "---\033[0m"]))
            print(content)
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                if print_time or self.debug:
                    f.write("--- %s ---\n" % str(datetime.now()))
                f.write(content + "\n")
