from torch import optim
import torchtuples.callbacks as cb

class OptimWrap(cb.CallbackHandler):
    """Wraps a torch.optim.Optimizer object so we can call some extra methods on it.
    The torch.optim.Optimizer can be obtained through the property 'optimizer'. 
    """
    def __init__(self, optimizer, callbacks=None):
        self.optimizer = optimizer
        if callbacks is None:
            callbacks = {}
        super().__init__(callbacks)
    
    def step(self, closure=None):
        return self.optimizer.step(closure)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def set(self, key, val):
        for param_group in self.optimizer.param_groups:
            param_group[key] = val
        # return self

    # def set_decoupled_weight_decay(self, wd):
    #     """Set decoupled weight decay"""
    #     return self.set('decoupled_weight_decay', wd)

    def set_lr(self, lr):
        """Sets 'initial_lr' and 'lr' to value 'lr'"""
        self.set('initial_lr', lr)
        self.set('lr', lr)
        # return self

    def set_momentum(self, momentum):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (momentum, param_group['betas'][1])
        elif 'momentum' in first_gr:
            self.set('momentum', momentum)
        else:
            raise ValueError("No momentum found")
        # return self

    def set_beta(self, beta):
        first_gr = self.optimizer.parameter_groups[0]
        if 'betas' in first_gr:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = (param_group['betas'][0], beta)
        elif 'alpha' in first_gr:
            self.set('alpha', beta)
        else:
            raise ValueError("No beta found")
        # return self
    
    def drop_scheduler(self):
        pass

    def reinitialize(self, params=None, **kwargs):
        print('Optimizer is not reinitialized. Need to do this manually')
        return NotImplemented

class OptimWrapReinit(OptimWrap):
    optim_func = NotImplemented
    init_args = NotImplemented

    def __init__(self, callbacks=None, params=None):
        super().__init__(None, callbacks)
        if params is not None:
            self.init_optimizer(params)

    def __call__(self, params):
        self.init_optimizer(params)
        return self

    def init_optimizer(self, params):
        call_args = set(self.optim_func.__init__.__code__.co_varnames[1:])
        init_args = {name: self.init_args[name] for name in (call_args & self.init_args.keys())}
        self.optimizer = self.optim_func(params, **init_args)

    def reinitialize(self, params=None, **kwargs):
        if params is None:
            if hasattr(self, 'model'):
                params = self.model.net.parameters()
        init_args = self.init_args.copy()
        init_args.update(kwargs)
        return self._constructor(**init_args, params=params)

    @property
    def _constructor(self):
        raise NotImplementedError


class SGD(OptimWrapReinit):
    r"""Wrapper to torch.optim.SGD where 'params' are not needed.

    Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups. If not specified, it will get parameters from model.

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    optim_func = optim.SGD
    def __init__(self, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, params=None):
        self.init_args = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                              nesterov=nesterov)
        super().__init__(None, params)

    @property
    def _constructor(self):
        return SGD


class RMSprop(OptimWrapReinit):
    """Wrapper to torch.optim.RMSprop where 'params' are not 'needed'.

    Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """
    optim_func = optim.RMSprop
    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 params=None):
        self.init_args = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                              centered=centered)
        super().__init__(None, params)

    @property
    def _constructor(self):
        return RMSprop

class Adam(OptimWrapReinit):
    r"""Wrapper to torch.optim.Adam where 'params' are not 'needed'.

    Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups. If not specified, it will get parameters from model.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    optim_func = optim.Adam
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 params=None):
        self.init_args = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(None, params)

    @property
    def _constructor(self):
        return Adam


class OptimizerDecoupledWeightDecay(OptimWrapReinit):
    def __init__(self, decoupled_weight_decay=0, dwd_normalize=False,
                 nb_epochs=None, params=None, callbacks=None):
        callbacks = callbacks if callbacks else {}
        if 'decoupled_weight_decay' in callbacks.keys():
            raise ValueError("decoupled_weight_decay allreday exists")
        self.decoupled_weight_decay = cb.DecoupledWeightDecay(decoupled_weight_decay, dwd_normalize,
                                                              nb_epochs)
        callbacks['decoupled_weight_decay'] = self.decoupled_weight_decay
        super().__init__(callbacks, params)

    def init_optimizer(self, params):
        return super().init_optimizer(params)
        # self.set_decoupled_weight_decay(self.init_args['decoupled_weight_decay'])

    def step(self, closure=None):
        if not hasattr(self, 'model'):
            raise RuntimeError("Optimizer with decoupled weight decay needs assignent of a 'Model' to function properly")
        return super().step(closure)


class AdamW(OptimizerDecoupledWeightDecay):
    optim_func = optim.Adam
    def __init__(self, lr=1e-3, betas=(0.9, 0.99), decoupled_weight_decay=0., dwd_normalize=False,
                 nb_epochs=None, eps=1e-8, params=None):
        self.init_args = dict(lr=lr, betas=betas, eps=eps, decoupled_weight_decay=decoupled_weight_decay,
                              dwd_normalize=dwd_normalize, nb_epochs=nb_epochs)
        super().__init__(decoupled_weight_decay, dwd_normalize, nb_epochs, params)

    @property
    def _constructor(self):
        return AdamW


class AdamWR(OptimizerDecoupledWeightDecay):
    """Adam with decoupled weight decay and warm restarts
    Eta is multiplied with this learning rate.
    
    Keyword Arguments:
        lr {float} -- Learning rate. (default: {1e-3})
        betas {tuple} -- Betas in Adam. (default: {(0.9, 0.99)})
        wd {float} -- Decoupled weight decay (default: {0.})
        wd_normalize {bool} -- Normalized weight decay. (default: {True})
        cycle_len {int} -- Number of epochs in each cycle. (default: {1})
        cycle_multiplier {int} -- After each cycle multiply cycle len with this (default: {2})
        cycle_eta_multiplier {[type]} -- After each cycle multiply eta_max with this (default: {1.})
        eta_min {int} -- Min eta  (default: {0})
        params {[type]} -- torch net parameters (default: {None})
        eps {[type]} --  (default: {1e-8})
    """
    optim_func = optim.Adam
    def __init__(self, lr=1e-3, betas=(0.9, 0.99), decoupled_weight_decay=0., dwd_normalize=True, 
                 cycle_len=1, cycle_multiplier=2, cycle_eta_multiplier=1.,
                 eta_min=0, params=None, eps=1e-8):

        self.init_args = dict(lr=lr, betas=betas, eps=eps, decoupled_weight_decay=decoupled_weight_decay,
                              dwd_normalize=dwd_normalize, cycle_len=cycle_len, cycle_multiplier=cycle_multiplier,
                              cycle_eta_multiplier=cycle_eta_multiplier, eta_min=eta_min)
        self.lr_scheduler = cb.LRCosineAnnealing(cycle_len, cycle_multiplier,
                                                 cycle_eta_multiplier,eta_min)
        callbacks = {'lr_scheduler': self.lr_scheduler}
        nb_epochs = self.lr_scheduler.get_cycle_len if dwd_normalize else None
        super().__init__(decoupled_weight_decay, dwd_normalize, nb_epochs, params, callbacks)
    
    def drop_scheduler(self):
        self.callbacks.pop('lr_scheduler')
        self.decoupled_weight_decay.nb_epochs = 1

    @property
    def _constructor(self):
        return AdamWR

