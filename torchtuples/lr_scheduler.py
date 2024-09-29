import math
import numpy as np
from torch.optim import Optimizer
try:
    import pandas as pd
except:
    pass


class LRSchedulerBatch(object):
    def __init__(self, optimizer, batch_iter=-1):
        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        if batch_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"""param 'initial_lr' is not specified
                                   in param_groups[{i}] when resuming an optimizer""")
        self.batch_iter = batch_iter + (batch_iter == -1)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        raise NotImplementedError

    def step(self, score=None, batch_iter=None):
        if batch_iter is not None:
            self.batch_iter = batch_iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self.batch_iter += 1


class LRFinderScheduler(LRSchedulerBatch):
    def __init__(self, optimizer, lr_lower=1e-7, lr_upper=10., n_steps=100):
        super().__init__(optimizer, batch_iter=-1)
        ratio = np.power(lr_upper/lr_lower, 1/(n_steps-1))
        self.lrs = lr_lower * np.power(ratio, np.arange(n_steps))
    
    def get_lr(self):
        return [self.lrs[self.batch_iter]] * len(self.optimizer.param_groups)
        

class LRBatchCosineAnnealing(LRSchedulerBatch):
    """Cosine anealing of learning rate (per batch update and not per epoch as in original paper.)
    
    Arguments:
        optimizer {torch.optim.Optimizer} -- Torch optimizer.
    
    Keyword Arguments:
        cycle_len {int} -- Length of cycle (T_i in paper) (default: {1})
        cycle_multiplier {int} -- Multiplier of cycle_len after each finished cycle (default: {2})
        cycle_eta_multiplier {float} -- Multiply eta with this number after each finished cycle.
            Reaonalble values should be between 1 and 0. (default: {1.})
        eta_min {int} -- Minimul learing rate muliplier (will not actually be zero) (default: {0})
        last_batch {int} -- Index of last batch (default: {-1})
    """
    def __init__(self, optimizer, cycle_len=1, cycle_multiplier=2, cycle_eta_multiplier=1.,
                 eta_min=0, batch_iter=-1, keep_etas=True):
        super().__init__(optimizer, batch_iter)
        self.cycle_len = cycle_len
        self.cycle_multiplier = cycle_multiplier
        if (cycle_eta_multiplier > 1) or (cycle_eta_multiplier < 0):
            raise ValueError("cycle_eta_multiplier should be in (0, 1].")
        self.cycle_eta_multiplier = cycle_eta_multiplier
        self.eta_max = 1.
        self.eta_min = eta_min
        self.keep_etas = keep_etas
        if not keep_etas:
            raise ValueError("For now, we require that 'keep_etas' is True")
        self.etas = []
        self.cycle_iter = self.batch_iter % self.cycle_len
    
    @property
    def cycle_len(self):
        return self._cycle_len
    
    @cycle_len.setter
    def cycle_len(self, cycle_len):
        if cycle_len < 1:
            raise ValueError(f"Need cycle_len >= 1.")
        assert type(cycle_len) is int
        self._cycle_len = cycle_len

    def get_lr(self):
        # eta = (self.eta_min + 0.5 * (1. - self.eta_min)
        #        * (1 + math.cos(math.pi * self.cycle_iter / self.cycle_len)))
        eta = (self.eta_min + 0.5 * (self.eta_max - self.eta_min)
               * (1 + math.cos(math.pi * self.cycle_iter / self.cycle_len)))
        if self.keep_etas:
            self.etas.append(eta)
        return [eta * group['initial_lr'] for group in self.optimizer.param_groups]

    def step(self, score=None, batch_iter=None):
        super().step(score, batch_iter)
        self.cycle_iter += 1
        if self.cycle_iter == self.cycle_len:
            self.cycle_iter = 0
            self.cycle_len *= self.cycle_multiplier
            self.eta_max *= self.cycle_eta_multiplier

    def to_pandas(self):
        etas = pd.Series(self.etas).rename('eta')
        etas.index.name = 'batch'
        return etas

    def plot(self, **kwargs):
        return self.to_pandas().to_frame().plot(**kwargs)
