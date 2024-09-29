'''
Callbacks.
'''
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
import math
import numpy as np
try:
    import pandas as pd
except:
    pass
import torch
import torchtuples
from . import lr_scheduler 
from torchtuples.utils import make_name_hash, TimeLogger


class Callback:
    '''Temple for how to write callbacks.
    '''
    def give_model(self, model):
        self.model = model

    def on_fit_start(self):
        pass

    def on_epoch_start(self):
        pass

    def on_batch_start(self):
        pass

    def before_step(self):
        """Called after loss.backward(), but before optim.step()."""
        pass

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_fit_end(self):
        pass


class CallbackHandler:
    def __init__(self, callbacks):
        if type(callbacks) in (list, tuple, torchtuples.TupleTree):
            self.callbacks = OrderedDict()
            for c in callbacks:
                self[self._make_name(c)] = c
        else:
            self.callbacks = OrderedDict(callbacks)

    def _make_name(self, obj):
        name = type(obj).__name__
        i = 0
        new_name = name
        while new_name in self.keys():
            new_name = f"{name}_{i}"
            i += 1
            if i >= 100:
                raise RuntimeError("Stopped while loop. Too many callbacks with same name")
        return new_name
    
    def __getitem__(self, name):
        return self.callbacks[name]
    
    def __setitem__(self, name, callback):
        self.append(callback, name)

    def append(self, callback, name=None):
        if hasattr(self, 'model'):
            callback.give_model(self.model)
        if name is None:
            name = self._make_name(callback)
        assert name not in self.callbacks.keys(), f"Duplicate name: {name}"
        self.callbacks[name] = callback
    
    def items(self):
        return self.callbacks.items()
    
    def keys(self):
        return self.callbacks.keys()
    
    def values(self):
        return self.callbacks.values()
    
    def __len__(self):
        return len(self.callbacks)

    def apply_callbacks(self, func):
        stop_signal = False
        for c in self.values():
            stop = func(c)
            stop = stop if stop else False
            stop_signal = stop_signal or stop
        return stop_signal

    def give_model(self, model):
        self.model = model
        stop_signal = self.apply_callbacks(lambda x: x.give_model(model))
        return stop_signal

    def on_fit_start(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_fit_start())
        return stop_signal

    def on_epoch_start(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_epoch_start())
        return stop_signal

    def on_batch_start(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_batch_start())
        return stop_signal

    def before_step(self):
        stop_signal = self.apply_callbacks(lambda x: x.before_step())
        return stop_signal

    def on_batch_end(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_batch_end())
        return stop_signal

    def on_epoch_end(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_epoch_end())
        return stop_signal

    def on_fit_end(self):
        stop_signal = self.apply_callbacks(lambda x: x.on_fit_end())
        return stop_signal


class TrainingCallbackHandler(CallbackHandler):
    '''Object for holding all callbacks.
    '''
    def __init__(self, optimizer, train_metrics, log, val_metrics=None, callbacks=None):
        super().__init__(dict(log=log, optimizer=optimizer, train_metrics=train_metrics))
        if val_metrics:
            self['val_metrics'] = val_metrics
        if callbacks is not None:
            if type(callbacks) in (list, tuple, torchtuples.TupleTree):
                for c in callbacks:
                    self.append(c)
            else:
                for name, c in callbacks.items():
                    self[name] = c

    def append(self, callback, name=None):
        super().append(callback, name)
        self.callbacks.move_to_end('log')


class TrainingLogger(Callback):
    def __init__(self, verbose=1):
        self.epoch = 0
        self.epochs = []
        self.loss = []
        self._verbose = verbose
    
    @property
    def monitors(self):
        return self._monitors

    @monitors.setter
    def monitors(self, monitor_dict):
        self._monitors = monitor_dict

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def on_fit_start(self):
        self.time_logger = TimeLogger()

    def on_epoch_end(self):
        self.epochs.append(self.epoch)
        if self.verbose:
            self.print_on_epoch_end()
        self.epoch += 1
        return False
    
    def print_on_epoch_end(self):
        tot, prev = self.time_logger.hms_diff()
        string = f"{self.epoch}:\t[{prev} / {tot}],\t"
        print(string + self.get_measures(), flush=True)

    def get_measures(self):
        measures = self.monitors
        if self.verbose.__class__ in [dict, OrderedDict]:
            measures = OrderedDict(measures, **self.verbose)
        string = ''
        for prefix, mm in measures.items():
            for name, score in mm.to_pandas().iloc[-1].items(): # slow but might not matter
                if (score is not None) and (np.isnan(score) == False):
                    string += '\t%s:' % (prefix + name)
                    string += ' %.4f,' % score
        return string[:-1]

    def to_pandas(self, colnames=None):
        '''Get data in dataframe.
        '''
        assert colnames is None, 'Not implemented'
        dfs = []
        for prefix, mm in self.monitors.items():
            df = mm.to_pandas()
            df.columns = prefix + df.columns
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df

    def plot(self, colnames=None, **kwargs):
        return self.to_pandas(colnames).plot(**kwargs)



class MonitorMetrics(Callback):
    def __init__(self, per_epoch=1):
        self.scores = dict()
        self.epoch = -1
        self.per_epoch = per_epoch

    def on_epoch_end(self):
        self.epoch += 1

    def append_score_if_epoch(self, name, val):
        if self.epoch % self.per_epoch == 0:
            self.append_score(name, val)
    
    def append_score(self, name, val):
        scores = self.scores.get(name, {'epoch': [], 'score': []})
        scores['epoch'].append(self.epoch)
        scores['score'].append(val)
        self.scores[name] = scores

    def to_pandas(self):
        '''Return scores as a pandas dataframe'''
        scores = [pd.Series(score['score'], index=score['epoch']).rename(name)
                  for name, score in self.scores.items()]
        scores = pd.concat(scores, axis=1)
        if type(scores) is pd.Series:
            scores = scores.to_frame()
        return scores


class _MonitorFitMetricsTrainData(MonitorMetrics):
    # def __init__(self, per_epoch=1):
    #     super().__init__(per_epo)
        # self.per_epoch = per_epoch

    def on_epoch_start(self):
        self.batch_metrics = defaultdict(list)
        return super().on_epoch_start()
    
    def on_batch_end(self):
        for name, score in self.model.batch_metrics.items():
            self.batch_metrics[name].append(score.item())
        return super().on_batch_end()

    def on_epoch_end(self):
        super().on_epoch_end()
        for name, vals in self.batch_metrics.items():
            self.append_score_if_epoch(name, np.mean(vals))
        # if self.epoch % self.per_epoch != 0:
        #     return False
        # for name, vals in self.batch_metrics.items():
        #     self.append_score(name, np.mean(vals))
        # return False


class MonitorFitMetrics(MonitorMetrics):
    def __init__(self, dataloader=None, per_epoch=1):
        super().__init__(per_epoch)
        self.dataloader = dataloader

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch != 0:
            return None
        if self.dataloader is None:
            scores = {name: np.nan for name in self.model.metrics.keys()}
        else:
            scores = self.model.score_in_batches_dataloader(self.dataloader)
        for name, val in scores.items():
            self.append_score(name, val)


class MonitorTrainMetrics(Callback):
    '''Monitor metrics for training loss.

    Parameters:
        per_epoch: How often to calculate.
    '''
    @property
    def scores(self):
        return self.model.train_metrics.scores
    
    def to_pandas(self):
        '''Return scores as a pandas dataframe'''
        scores = [pd.Series(score[name], index=score['epoch']).rename(name)
                  for name, score in self.scores.items()]
        scores = pd.concat(scores, axis=1)
        if type(scores) is pd.Series:
            scores = scores.to_frame()
        return scores


# class MonitorXy(MonitorBase):
#     '''Monitor metrics for classification and regression.
#     Same as MonitorBase but we input a pair, X, y instead of data.

#     For survival methods, see e.g. MonitorCox.

#     Parameters:
#         X: Numpy array with features.
#         y: Numpy array with labels.
#         monitor_funcs: Function, list, or dict of functions giving quiatities that should
#             be monitored.
#             The function takes argumess (df, preds) and should return a score.
#         batch_size: Batch size used for calculating the scores.
#         **kwargs: Can be passed to predict method.
#     '''
#     def __init__(self, X, y, monitor_funcs, per_epoch=1, batch_size=512, **kwargs):
#         self.X, self.y = X, y
#         self.kwargs = kwargs
#         super().__init__(monitor_funcs, per_epoch, batch_size)

#     def get_score_inputs(self):
#         '''This function should create arguments to the monitor function.
#         Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
#         '''
#         preds = self.model.predict(self.X, self.batch_size, **self.kwargs)
#         return self.y, preds


# class MonitorSklearn(MonitorXy):
#     '''Class for monitoring metrics of from sklearn metrics

#     Parameters:
#         X: Numpy array with features.
#         y: Numpy array with labels.
#         monitor: Name for method in sklearn.metrics.
#             E.g. 'log_loss', or pass sklearn.metrics.log_loss.
#             For additional parameter, specify the function with a lambda statemetn.
#                 e.g. {'accuracy': lambda y, p: metrics.accuracy_score(y, p > 0.5)}
#         batch_size: Batch size used for calculating the scores.
#         **kwargs: Can be passed to predict method.
#     '''
#     def __init__(self, X, y, monitor, per_epoch=1, batch_size=512, **kwargs):
#         if monitor.__class__ is str:
#             monitor = {monitor: monitor}
#         elif monitor.__class__ is list:
#             monitor = {mon if mon.__class__ is str else str(i): mon
#                        for i, mon in enumerate(monitor)}

#         monitor = {name: getattr(metrics, mon) if mon.__class__ == str else mon
#                    for name, mon in monitor.items()}
#         super().__init__(X, y, monitor, per_epoch, batch_size)


class ClipGradNorm(Callback):
    '''Callback for clipping gradients.
    
    See torch.nn.utils.clip_grad_norm_.

    Parameters:
        net: Network wtih parameters() function.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    '''
    def __init__(self, net, max_norm, norm_type=2):
        self.net = net
        self.max_norm = max_norm
        self.norm_type = norm_type

    def before_step(self):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm, self.norm_type)
        stop_signal = False
        return stop_signal


class LRScheduler(Callback):
    '''Wrapper for pytorch.optim.lr_scheduler objects.

    Parameters:
        scheduler: A pytorch.optim.lr_scheduler object.
        mm_obj: Monitor object, where first metric is used for early stopping.
            E.g. MonitorSurvival(df_val, 'cindex').
    '''
    def __init__(self, scheduler, mm_obj):
        self.scheduler = scheduler
        self.mm_obj = mm_obj

    def on_epoch_end(self):
        score = self.mm_obj.scores[0][-1]
        self.scheduler.step(score)
        stop_signal = False
        return stop_signal


class LRSchedulerBatch(Callback):
    '''Wrapper for schedulers

    Parameters:
        scheduler: A scheduler, e.g. BatchCosineAnnealingLR()
    '''
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_fit_start(self):
        self.scheduler.step()  
    
    def on_batch_end(self):
        self.scheduler.step()
        return False


class LRCosineAnnealing(LRSchedulerBatch):
    def __init__(self, cycle_len=1, cycle_multiplier=2, cycle_eta_multiplier=1., eta_min=0):
        self.first_cycle_len = cycle_len
        self.cycle_multiplier = cycle_multiplier
        self.cycle_eta_multiplier = cycle_eta_multiplier
        self.eta_min = eta_min
        scheduler = None
        super().__init__(scheduler)
    
    def on_fit_start(self):
        # if self.first_cycle_len == "epoch":
        if self.scheduler is None:
            cycle_len = self.first_cycle_len * self.model.fit_info['batches_per_epoch']
            scheduler = lr_scheduler.LRBatchCosineAnnealing(self.model.optimizer, int(cycle_len),
                                                            self.cycle_multiplier,
                                                            self.cycle_eta_multiplier,
                                                            self.eta_min,
                                                            keep_etas=True)
            self.scheduler = scheduler
        elif self.model.optimizer is not self.scheduler.optimizer:
            raise RuntimeError(
                "Changed optimizer, and we have not implemented cosine annealing for this")
        super().on_fit_start()
    
    def get_cycle_len(self):
        return self.scheduler.cycle_len
    
    def get_epochs_per_cycle(self):
        return self.get_cycle_len() / self.model.fit_info['batches_per_epoch']

    def get_etas(self):
        return self.scheduler.etas

    def to_pandas(self):
        return self.scheduler.to_pandas()

    def plot(self, **kwargs):
        return self.scheduler.plot(**kwargs)


class LRFinder(Callback):
    def __init__(self, lr_lower=1e-7, lr_upper=10., n_steps=100, tolerance=np.inf):
        self.lr_lower = lr_lower
        self.lr_upper = lr_upper
        self.n_steps = n_steps
        self.lowest_loss = np.inf
        self.tolerance = tolerance

    def on_fit_start(self):
        self.batch_loss = []
        self.scheduler = lr_scheduler.LRFinderScheduler(self.model.optimizer, self.lr_lower,
                                                        self.lr_upper, self.n_steps)
        # self.scheduler.step()

    def on_batch_start(self):
        self.scheduler.step()
        for group in self.model.optimizer.param_groups:
            group['initial_lr'] = group['lr']  # Needed when using weight decay.
    
    def on_batch_end(self):
        # self.scheduler.step()
        batch_loss = self.model.batch_loss.item()
        self.batch_loss.append(batch_loss)
        if batch_loss > self.tolerance:
            return True
        # if (batch_loss / self.lowest_loss) > self.tolerance:
        #     return True
        self.lowest_loss = min(self.lowest_loss, batch_loss)
        if self.scheduler.batch_iter == self.n_steps:
            return True
        return False
    
    def to_pandas(self, smoothed=0):
        res = pd.DataFrame(dict(train_loss=self.batch_loss),
                           index=self.lrs[:len(self.batch_loss)])
        res.index.name = 'lr'
        if smoothed:
            res = res.apply(_smooth_curve, beta=smoothed)
        return res
    
    def plot(self, logx=True, smoothed=0.98, **kwargs):
        res = self.to_pandas(smoothed)
        ylabel = 'bach_loss'
        if smoothed:
            ylabel = ylabel + ' (smoothed)'
        ax = res.plot(logx=logx, **kwargs)
        ax.set_xlabel('lr')
        ax.set_ylabel(ylabel)
        return ax

    @property
    def lrs(self):
        return self.scheduler.lrs
    
    def get_best_lr(self, lr_min=1e-4, lr_max=1., _multiplier=10.):
        """Get suggestion for bets learning rate.
        It is beter to investigate the plot, but this might work too.
        
        Keyword Arguments:
            lower {float} -- Lower accepable learning rate (default: {1e-4})
            upper {float} -- Upper acceptable learning rate (default: {1.})
            _multiplier {float} -- See sorce code (default according to fast.ai) (default: {10})
        
        Returns:
            float -- Suggested best learning rate.
        """
        smoothed = _smooth_curve(self.batch_loss)
        idx_min = np.argmin(smoothed)
        best_lr = self.lrs[idx_min] / _multiplier
        best_lr = np.clip(best_lr, lr_min, lr_max)
        return best_lr

        # idx_lower = np.searchsorted(self.lrs, lr_min * _multiplier)
        # idx_upper = np.searchsorted(self.lrs, lr_max * _multiplier, 'right')
        # smoothed = _smooth_curve(self.batch_loss)[idx_lower:idx_upper]
        # idx_min = np.argmin(smoothed)
        # best_lr = self.lrs[idx_lower:idx_upper][idx_min] / _multiplier
        # return best_lr

def _smooth_curve(vals, beta=0.98):
    """From fastai"""
    avg_val = 0
    smoothed = []
    for (i,v) in enumerate(vals):
        avg_val = beta * avg_val + (1-beta) * v
        smoothed.append(avg_val/(1-beta**(i+1)))
    return smoothed


class DecoupledWeightDecay(Callback):
    """Same weight decay for all groups in the optimizer."""
    def __init__(self, weight_decay, normalized=False, nb_epochs=None):
        if (weight_decay >= 1) or (weight_decay < 0):
            warnings.warn(f"Weigth decay should be in in [0, 1), got {weight_decay}")
        self.weight_decay = weight_decay
        self.normalized = normalized
        # if self.normalized:
        #     if not ((type(nb_epochs) is int) or callable(nb_epochs)):
        #         raise ValueError(f"Need nb_epochs to be callable or int, not {nb_epochs}")
        self.nb_epochs = nb_epochs
    
    def on_fit_start(self):
        self._batches_per_epoch = self.model.fit_info['batches_per_epoch']

    @property
    def nb_epochs(self):
        if type(self._nb_epochs) is int:
            return self._nb_epochs
        elif callable(self._nb_epochs):
            return self._nb_epochs()
        raise RuntimeError('nb_epochs needs to be callable or int')

    @nb_epochs.setter
    def nb_epochs(self, nb_epochs):
        if self.normalized:
            if not ((type(nb_epochs) is int) or callable(nb_epochs)):
                raise ValueError(f"Need nb_epochs to be callable or int, not {type(nb_epochs)}")
        self._nb_epochs = nb_epochs
    
    def _normalized_weight_decay(self):
        # if type(self.nb_epochs) is int:
        #     nb_epochs = self.nb_epochs
        # elif callable(self.nb_epochs):
        #     nb_epochs = self.nb_epochs()
        # else:
        #     raise RuntimeError('nb_epochs needs to be callable or int')
        norm_const = math.sqrt(1 / (self._batches_per_epoch * self.nb_epochs))
        return self.weight_decay * norm_const
    
    def before_step(self):
        # Weight decay out of the loss. After the gradient computation but before the step.
        if self.normalized:
            weight_decay = self._normalized_weight_decay()
        else:
            weight_decay = self.weight_decay
        for group in self.model.optimizer.param_groups:
            lr = group['lr']
            # alpha = group.get('initial_lr', 1.)
            alpha = group.get('initial_lr', lr)
            eta = lr / alpha
            for p in group['params']:
                if p.grad is not None:
                    p.data = p.data.add(-weight_decay * eta, p.data)
                    # p.data.mul_(1 - weight_decay * eta)
        return False


# class EarlyStoppingCycle_old(Callback):
#     '''
#     TODO: Should rewrite with _ActionOnBestMetric.

#     Stop training when monitored quantity has not improved the last cycle.
#     Takes a Monitor object that is also a callback.
#     Use first metric in mm_obj to determine early stopping.

#     Parameters:
#         mm_obj: Monitor object, where first metric is used for early stopping.
#             E.g. MonitorSurvival(df_val, 'cindex').
#         get_score: Function for obtaining current scores. If string, we use validation metric.
#             Default is 'loss' which gives validation loss.
#         minimize: If we are to minimize or maximize monitor.
#         min_delta: Minimum change in the monitored quantity to qualify as an improvement,
#             i.e. an absolute change of less than min_delta, will count as no improvement.
#         patience: Number of cycles patience.
#         model_file_path: If spesified, the model weights will be stored whever a better score
#             is achieved.
#     '''
#     def __init__(self, lr_scheduler='optimizer', get_score='loss', minimize=True, min_delta=0,
#                  patience=1, min_cycles=4, model_file_path=None, load_best=False):
#         self.get_score = get_score
#         self._get_score = self.get_score
#         self.minimize = minimize
#         self.min_delta = min_delta
#         self.patience = patience
#         self.min_cycles = min_cycles
#         self.model_file_path = model_file_path
#         self.load_best = load_best
#         if self.load_best and (self.model_file_path is None):
#             raise ValueError("To use 'load_best' you need to provide a model_file_path")
#         self.lr_scheduler = lr_scheduler
#         self.cur_best = np.inf if self.minimize else -np.inf
#         self.cur_best_cycle_nb = None

#     def on_fit_start(self):
#         if type(self.get_score) is str:
#             self.get_score = lambda: self.model.val_metrics.scores[self._get_score]['score'][-1]
#         if self.lr_scheduler == 'optimizer':
#             self.lr_scheduler = self.model.optimizer.lr_scheduler
        
#     def on_epoch_end(self):
#         etas = self.lr_scheduler.get_etas()[:-1]  # Drop last because allready updated by optimizer
#         cycle_nb = (np.diff(etas) > 0).sum()
#         score = self.get_score()

#         if self.minimize:
#             if score < (self.cur_best - self.min_delta):
#                 self.cur_best = score
#                 self.cur_best_cycle_nb = cycle_nb
#         else:
#             if score > (self.cur_best + self.min_delta):
#                 self.cur_best = score
#                 self.cur_best_cycle_nb = cycle_nb

#         if (score == self.cur_best) and (self.model_file_path is not None):
#             self.model.save_model_weights(self.model_file_path)

#         stop_signal = ((cycle_nb > (self.cur_best_cycle_nb + self.patience)) and 
#                        (cycle_nb >= self.min_cycles))
#         return stop_signal

#     def on_fit_end(self):
#         if self.load_best:
#             self.model.load_model_weights(self.model_file_path)
#         return super().on_fit_end()


class StopIfExplodeOrNan(Callback):
    """Stop trainig if training or validation loss becomes larger than a threshold or becomes nan.
    Checks both train and val data.

    Keyword Arguments:
        threshold {float} -- Stop if train or val loss is 'nan' or larger than threshold (default: {np.inf})
        metric {str} -- Whick metric in model.log.monitors should be used.(default: {'loss'})
    """
    def __init__(self, threshold=np.inf, metric='loss'):
        self.threshold = threshold
        self.metric = metric

    def _update_cur_best(self, key):
        score = self.model.log.monitors[key].scores[self.metric]['score'][-1]
        if np.isnan(score):
            return True
        if score >= self.threshold:
            return True
        return False

    def on_epoch_end(self):
        return self._update_cur_best('train_') or self._update_cur_best('val_')


class _ActionOnBestMetric(Callback):
    """Abstract class used for e.g. EarlyStopping.
    """
    def __init__(self, metric='loss', dataset='val', get_score=None, minimize=True, min_delta=0.,
                 checkpoint_model=True, file_path=None, load_best=True, rm_file=True):
        self.metric = metric
        self.dataset = dataset
        self.get_score = get_score
        self.minimize = minimize
        self.min_delta = min_delta
        self._checkpoint_model = checkpoint_model
        if not self._checkpoint_model:
            assert load_best == False, "Need load best to be False when '_checkpoint_model' is False"
        if (not load_best and rm_file) and self._checkpoint_model:
            raise ValueError("If you really want not not load best but remove file you can instead remove this callback.")
        self.load_best = load_best
        self.rm_file = rm_file
        self.file_path = file_path if file_path else make_name_hash('weight_checkpoint')
        self.file_path = Path(self.file_path)
        self.cur_best = np.inf if self.minimize else -np.inf
        self._iter_since_best = 0

    def on_fit_start(self):
        if not self.file_path.exists() and self._checkpoint_model:
            self.model.save_model_weights(self.file_path)
        if self.get_score is None:
            if self.dataset == 'val':
                metrics = self.model.val_metrics
            elif self.dataset == 'train':
                metrics = self.model.train_metrics
            else:
                raise ValueError("Need dataset to be 'val' or 'train'.")
            self.get_score = lambda: metrics.scores[self.metric]['score'][-1]

    def on_epoch_end(self):
        score = self.get_score()
        if self.minimize:
            if score < (self.cur_best - self.min_delta):
                self.cur_best = score
                self._iter_since_best = -1
        else:
            if score > (self.cur_best + self.min_delta):
                self.cur_best = score
                self._iter_since_best = -1
        
        if (score == self.cur_best) and (self._checkpoint_model):
            self.model.save_model_weights(self.file_path)
        self._iter_since_best += 1

    def on_fit_end(self):
        if self._checkpoint_model:
            if self.load_best:
                self.load_weights()
            if self.rm_file:
                self.rm_weight_file()

    def load_weights(self):
        if not self._checkpoint_model:
            raise RuntimeError("This model has no stored weights")
        self.model.load_model_weights(self.file_path)

    def rm_weight_file(self):
        if self.file_path.exists():
            self.file_path.unlink()


class BestWeights(_ActionOnBestMetric):
    """
    Keep trac of the weight of the best performin model.
    If you also want early stopping, you can use EarlyStopping or EarlyStoppingCycle instead.
    
    Keyword Arguments:
        metric {str} -- Metric in model.train_metrics or model.val_metrics (default: {'loss'})
        dataset {str} -- Data set which is moitored train/val (default: {'val'})
        get_score {[type]} -- Alternative to metric, where you can give a function that returns the
            scores. (default: {None})
        minimize {bool} -- If we are minimizing or maximizing the score (default: {True})
        file_path {[type]} -- Alternative file path for model weight. If 'None' we generate one.
            (default: {None})
        load_best {bool} -- Load best weight into model object after training.
            If 'False' this needs to be done by calling the method 'load_weights' (default: {True})
        rm_file {bool} -- If we should delete the checkpoint weight file after finishing training.
            (default: {True})
    """
    def __init__(self, metric='loss', dataset='val', get_score=None, minimize=True, file_path=None,
                 load_best=True, rm_file=True):
        min_delta = 0.
        checkpoint_model = True
        super().__init__(metric, dataset, get_score, minimize, min_delta, checkpoint_model,
                         file_path, load_best, rm_file)


class EarlyStopping(_ActionOnBestMetric):
    """
    Stop training when monitored quantity has not improved the last epochs.
    
    Keyword Arguments:
        metric {str} -- Metric in model.train_metrics or model.val_metrics (default: {'loss'})
        dataset {str} -- Data set which is moitored train/val (default: {'val'})
        get_score {[type]} -- Alternative to metric, where you can give a function that returns the
            scores. (default: {None})
        minimize {bool} -- If we are minimizing or maximizing the score (default: {True})
        min_delta {[type]} -- Improvement required to consider the new score better than the
            previous best. (default: {0.})
        patience {int} -- Number of epochs to wait since the best score before stopping. (default: {10})
        checkpoint_model {bool} -- If we should keep track of the best model weights. (default: {True})
        file_path {[type]} -- Alternative file path for model weight. If 'None' we generate one.
            (default: {None})
        load_best {bool} -- Load best weight into model object after training.
            If 'False' this needs to be done by calling the method 'load_weights' (default: {True})
        rm_file {bool} -- If we should delete the checkpoint weight file after finishing training.
            (default: {True})
    """
    def __init__(self, metric='loss', dataset='val', get_score=None, minimize=True, min_delta=0.,
                 patience=10, checkpoint_model=True, file_path=None, load_best=True, rm_file=True):
        self.patience = patience
        super().__init__(metric, dataset, get_score, minimize, min_delta, checkpoint_model,
                         file_path, load_best, rm_file)

    def on_epoch_end(self):
        super().on_epoch_end()
        return self._iter_since_best >= self.patience


class EarlyStoppingCycle(_ActionOnBestMetric):
    """
    Stop training when monitored quantity has not improved the last cycles.
    
    Keyword Arguments:
        metric {str} -- Metric in model.train_metrics or model.val_metrics (default: {'loss'})
        dataset {str} -- Data set which is moitored train/val (default: {'val'})
        lr_scheduler {str} -- lr_scheduler object. If 'optimizer' use model.optimizer.lr_scheduler.
            (default: {'optimizer'})
        get_score {[type]} -- Alternative to metric, where you can give a function that returns the
            scores. (default: {None})
        minimize {bool} -- If we are minimizing or maximizing the score (default: {True})
        min_delta {[type]} -- Improvement required to consider the new score better than the
            previous best. (default: {0.})
        patience {int} -- Number of cycles to wait since the best score before stopping. (default: {1})
        min_cycles {int} -- Minimum number of cycles required before stopping. (default: {4})
        checkpoint_model {bool} -- If we should keep track of the best model weights. (default: {True})
        file_path {[type]} -- Alternative file path for model weight. If 'None' we generate one.
            (default: {None})
        load_best {bool} -- Load best weight into model object after training.
            If 'False' this needs to be done by calling the method 'load_weights' (default: {True})
        rm_file {bool} -- If we should delete the checkpoint weight file after finishing training.
            (default: {True})
    """
    def __init__(self, metric='loss', dataset='val', lr_scheduler='optimizer', get_score=None,
                 minimize=True, min_delta=0., patience=1, min_cycles=4, checkpoint_model=True,
                 file_path=None, load_best=True, rm_file=True):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.min_cycles = min_cycles
        self.cur_best_cycle_nb = 0
        super().__init__(metric, dataset, get_score, minimize, min_delta, checkpoint_model,
                         file_path, load_best, rm_file)

    def on_fit_start(self):
        if self.lr_scheduler == 'optimizer':
            self.lr_scheduler = self.model.optimizer.lr_scheduler
        return super().on_fit_start()

    def get_current_cycle_nb(self):
        """Get current cycle number"""
        etas = self.lr_scheduler.get_etas()[:-1]  # Drop last because allready updated by optimizer
        return (np.diff(etas) > 0).sum()

    def on_epoch_end(self):
        cycle_nb = self.get_current_cycle_nb()
        score = self.get_score()

        if self.minimize:
            if score < (self.cur_best - self.min_delta):
                self.cur_best = score
                self._iter_since_best = -1
                self.cur_best_cycle_nb = cycle_nb
        else:
            if score > (self.cur_best + self.min_delta):
                self.cur_best = score
                self._iter_since_best = -1
                self.cur_best_cycle_nb = cycle_nb
        self._iter_since_best += 1
        
        if (score == self.cur_best) and (self._checkpoint_model):
            self.model.save_model_weights(self.file_path)

        stop_signal = ((cycle_nb > (self.cur_best_cycle_nb + self.patience)) and 
                       (cycle_nb >= self.min_cycles))
        return stop_signal

