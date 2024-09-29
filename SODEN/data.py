from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from six.moves import cPickle as pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (BatchSampler, DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.dataloader import default_collate

from lifelines import KaplanMeierFitter

NUM_WORKERS = 0


# DictDataset.
class DictDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        sample_features = {}
        for key in self.features:
            sample_features[key] = self.features[key][idx]
        return sample_features, self.labels[idx]


# Batch random sampler that maintains order within each batch.
class OrderedBatchRandomSampler(object):
    def __init__(self, n, batch_size, seed=13, drop_last=False):
        super(OrderedBatchRandomSampler, self).__init__()
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.random_state = np.random.RandomState(seed)

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        else:
            return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for idx in self.random_state.permutation(self.n):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield sorted(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield sorted(batch)


def my_collate_fn(batch):
    LIST_KEYS = ["eval_t_seq"]
    if isinstance(batch[0][0], dict):
        # `batch` is a list of (`features`, `labels`) pair and `features` is a
        # dict. `batch[0][0]` is the `features` of the first data sample.
        # `LIST_KEYS` provides a list of fields that have variable-length list
        # for different samples so we need to pad them and deal with them
        # separately in collate_fn.
        collated_features = {}
        for list_key in LIST_KEYS:
            if list_key in batch[0][0]:
                batch_list = [d[0][list_key] for d in batch]
                batch_list = pad_sequence(batch_list, batch_first=True)
                collated_features[list_key] = batch_list
        for key in batch[0][0]:
            if key in LIST_KEYS:
                continue
            collated_features[key] = default_collate(
                [d[0][key] for d in batch])
        collated_labels = default_collate([d[1] for d in batch])
        collated_results = (collated_features, collated_labels)
        return collated_results
    return default_collate(batch)


def get_dataloader(t,
                   delta,
                   x=None,
                   batch_size=128,
                   random_state=None,
                   is_eval=False):
    """
    Arguments:
      t: A (N,) numpy array for time-to-event or censoring time.
      delta: A (N,) numpy array for censoring status (1 for observed events).
      x: A (N, d) numpy array for features.
    """
    # Sort
    N = len(t)
    idx = np.argsort(t)
    t = t[idx]
    delta = delta[idx]
    x = x[idx]
    init_cond = np.zeros_like(t)

    labels = torch.tensor(delta, dtype=torch.float)
    features = {}
    features["t"] = torch.tensor(t, dtype=torch.float)
    features["init_cond"] = torch.tensor(init_cond, dtype=torch.float)
    features["features"] = torch.tensor(x, dtype=torch.float)
    features["index"] = torch.arange(N, dtype=torch.long)

    _collate_fn = None
    if is_eval:
        constant_dict = {}  # constant values shared by all samples
        # Eval time steps for time-dependent C-index.
        constant_dict["eval_t"] = torch.unique(features["t"])

        # Eval time steps for quantile C-index
        ones = torch.ones_like(features["t"])
        features["t_q25"] = ones * t[int(0.25 * len(t))]
        features["t_q50"] = ones * t[int(0.5 * len(t))]
        features["t_q75"] = ones * t[int(0.75 * len(t))]

        # Eval min and max time steps for Brier Score
        constant_dict["t_min"] = torch.tensor(t[0], dtype=torch.float32)
        constant_dict["t_max"] = torch.tensor(t[-1], dtype=torch.float32)

        kmf = KaplanMeierFitter()
        kmf.fit(t, event_observed=(1 - delta))
        G_T = kmf.predict(t, interpolate=True).to_numpy()
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            constant_dict["t_max_{}".format(eps)] = torch.tensor(
                max(t[G_T > eps]), dtype=torch.float32)

        def _collate_fn(batch):
            if isinstance(batch[0][0], dict):
                collated_features = constant_dict  # add the constant fields
                for key in batch[0][0]:
                    collated_features[key] = default_collate(
                        [d[0][key] for d in batch])
                collated_labels = default_collate([d[1] for d in batch])
                collated_results = (collated_features, collated_labels)
                return collated_results
            return default_collate(batch)

    dataset = DictDataset(features, labels)

    if is_eval:
        sampler = BatchSampler(
            SequentialSampler(range(N)), batch_size=batch_size, drop_last=False)
    else:
        sampler = OrderedBatchRandomSampler(N, batch_size, drop_last=True)

    if _collate_fn is None:
        _collate_fn = default_collate
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=_collate_fn, pin_memory=True,
        num_workers=NUM_WORKERS)
    return dataloader


def get_mimic_dataloader(input_file, batch_size, random_state, is_eval=False):
        dt = np.load(input_file)
        std_x = dt["arr_0"]
        y = dt["arr_1"]

        delta = y[:, 1]
        t = y[:, 0] + 0.001
        feature_size = std_x.shape[1]
        dataloader = get_dataloader(
            t,
            delta,
            std_x,
            batch_size=batch_size,
            random_state=random_state,
            is_eval=is_eval)

        return dataloader, feature_size


def rnn_collate_fn(batch):
    if isinstance(batch[0][0], dict) and "seq_feat" in batch[0][0]:
        # `batch` is a list of (`features`, `labels`) pair and `features` is a
        # dict. `batch[0][0]` is the `features` of the first data sample.
        sorted_batch = sorted(batch, key=lambda x: x[0]["seq_feat"].size(0),
                              reverse=True)
        batch_seq_feat_list = [x[0]["seq_feat"] for x in sorted_batch]
        batch_seq_feat_tensor = pad_sequence(batch_seq_feat_list,
                                             batch_first=True)
        collated_features = {"seq_feat": batch_seq_feat_tensor}
        for key in sorted_batch[0][0]:
            if key == "seq_feat":
                continue
            collated_features[key] = default_collate(
                [d[0][key] for d in sorted_batch])
        collated_labels = default_collate([d[1] for d in sorted_batch])
        collated_results = (collated_features, collated_labels)
        return collated_results
    return default_collate(batch)


def get_mimic_seq_dataloader(input_file, batch_size, random_state,
                             is_eval=False):
    data = pickle.load(open(input_file, "rb"))
    fix_feat = data["fix_feat"]
    seq_feat = data["seq_feat"]
    t = data["label"][:, 0]
    delta = data["label"][:, 1]

    # Sort
    idx = np.argsort(t)
    t = t[idx]
    delta = delta[idx]
    fix_feat = fix_feat[idx]
    seq_feat = seq_feat[idx]

    init_cond = np.zeros_like(t)
    seq_feat_length = [s.shape[0] for s in seq_feat]

    feature_size = {}
    feature_size["fix_feat"] = fix_feat.shape[-1]
    feature_size["seq_feat"] = seq_feat[0].shape[-1]

    labels = torch.tensor(delta, dtype=torch.float)
    features = {}
    features["t"] = torch.tensor(t, dtype=torch.float)
    features["init_cond"] = torch.tensor(init_cond, dtype=torch.float)
    features["fix_feat"] = torch.tensor(fix_feat, dtype=torch.float)
    features["seq_feat"] = [torch.tensor(t,
                                         dtype=torch.float) for t in seq_feat]
    features["seq_feat_length"] = torch.tensor(seq_feat_length,
                                               dtype=torch.long)

    N = len(t)
    features["index"] = torch.arange(N, dtype=torch.long)

    _collate_fn = rnn_collate_fn
    if is_eval:
        constant_dict = {}  # constant values shared by all samples
        # Eval time steps for time-dependent C-index.
        constant_dict["eval_t"] = torch.unique(features["t"]).sort()[0]

        # Eval time steps for quantile C-index
        ones = torch.ones_like(features["t"])
        features["t_q25"] = ones * t[int(0.25 * len(t))]
        features["t_q50"] = ones * t[int(0.5 * len(t))]
        features["t_q75"] = ones * t[int(0.75 * len(t))]

        # Eval min and max time steps for Brier Score
        constant_dict["t_min"] = torch.tensor(t[0], dtype=torch.float32)
        constant_dict["t_max"] = torch.tensor(t[-1], dtype=torch.float32)

        kmf = KaplanMeierFitter()
        kmf.fit(t, event_observed=(1 - delta))
        G_T = kmf.predict(t, interpolate=True).to_numpy()
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            constant_dict["t_max_{}".format(eps)] = torch.tensor(
                max(t[G_T > eps]), dtype=torch.float32)

        def _collate_fn(batch):
            if isinstance(batch[0][0], dict) and "seq_feat" in batch[0][0]:
                # `batch` is a list of (`features`, `labels`) pair and `features` is a
                # dict. `batch[0][0]` is the `features` of the first data sample.
                sorted_batch = sorted(batch, key=lambda x: x[0]["seq_feat"].size(0),
                                      reverse=True)
                batch_seq_feat_list = [x[0]["seq_feat"] for x in sorted_batch]
                batch_seq_feat_tensor = pad_sequence(batch_seq_feat_list,
                                                     batch_first=True)
                collated_features = constant_dict
                collated_features["seq_feat"] = batch_seq_feat_tensor
                for key in sorted_batch[0][0]:
                    if key == "seq_feat":
                        continue
                    collated_features[key] = default_collate(
                        [d[0][key] for d in sorted_batch])
                collated_labels = default_collate([d[1] for d in sorted_batch])
                collated_results = (collated_features, collated_labels)
                return collated_results
            return default_collate(batch)

    dataset = DictDataset(features, labels)
    N = len(t)
    if is_eval:
        sampler = BatchSampler(
            SequentialSampler(range(N)), batch_size=batch_size, drop_last=False)
    else:
        sampler = BatchSampler(
            RandomSampler(range(N)), batch_size=batch_size, drop_last=True)

    dataloader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=_collate_fn, pin_memory=True,
        num_workers=NUM_WORKERS)
    return dataloader, feature_size
