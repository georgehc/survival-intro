"""
Various models

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import sys
sys.path.insert(0, '../')
import hnswlib
import numpy as np
import os
import pandas as pd
import pycox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
import xgboost as xgb
from pycox.models.data import pair_rank_mat
from pycox.models.interpolation import InterpolateLogisticHazard
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import MDS
from torch import Tensor
from torch.nn.modules.loss import _Loss, _WeightedLoss
from typing import Optional


class Model(tt.Model):
    """
    Although torchtuples already supports fitting a single epoch at a time,
    there's a bit of an overhead. This version aims to be a little faster
    albeit with less features (e.g., no callbacks, no verbose output).
    """
    def fit_dataloader(self, dataloader, epochs=1, callbacks=None,
                       verbose=False, metrics=None, val_dataloader=None):
        if 'fit_info' not in self.__dict__:
            self._setup_train_info(dataloader)
            self.metrics = self._setup_metrics(metrics)
            self.log.verbose = False
            self.val_metrics.dataloader = val_dataloader

        for _ in range(epochs):
            for data in dataloader:
                self.optimizer.zero_grad()
                if data[0].size(0) > 1:
                    self.batch_metrics = self.compute_metrics(data,
                                                              self.metrics)
                    self.batch_loss = self.batch_metrics['loss']
                    if self.batch_loss.grad_fn is not None:
                        self.batch_loss.backward()
                        self.optimizer.step()


class CoxPH(pycox.models.CoxPH):
    """
    We modify CoxPH in the same way we modify Model
    """
    def fit_dataloader(self, dataloader, epochs=1, callbacks=None,
                       verbose=False, metrics=None, val_dataloader=None):
        if 'fit_info' not in self.__dict__:
            self._setup_train_info(dataloader)
            self.metrics = self._setup_metrics(metrics)
            self.log.verbose = False
            self.val_metrics.dataloader = val_dataloader

        for _ in range(epochs):
            for data in dataloader:
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics['loss']
                if self.batch_loss.grad_fn is not None:
                    self.batch_loss.backward()
                    self.optimizer.step()


class DeepHitSingle(pycox.models.DeepHitSingle):
    """
    We modify DeepHit in the same way we modify Model
    """
    def fit_dataloader(self, dataloader, epochs=1, callbacks=None,
                       verbose=False, metrics=None, val_dataloader=None):
        if 'fit_info' not in self.__dict__:
            self._setup_train_info(dataloader)
            self.metrics = self._setup_metrics(metrics)
            self.log.verbose = False
            self.val_metrics.dataloader = val_dataloader

        for _ in range(epochs):
            for data in dataloader:
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics['loss']
                if self.batch_loss.grad_fn is not None:
                    self.batch_loss.backward()
                    self.optimizer.step()


class Hypersphere(nn.Module):
    def __init__(self, squared_radius: Optional[float] = 1.) -> None:
        super(Hypersphere, self).__init__()
        self.squared_radius = squared_radius

    def forward(self, input: Tensor) -> Tensor:
        return F.normalize(input, dim=1) / np.sqrt(self.squared_radius)


# symmetric squared Euclidean distance calculation from:
# https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
def symmetric_squared_pairwise_distances(x):
    r = torch.mm(x, x.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    return diag + diag.t() - 2*r


class NLLKernelHazardLoss(torch.nn.Module):
    def __init__(self, alpha, sigma, gamma):
        super(NLLKernelHazardLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma

    def forward(self, phi: Tensor, idx_durations: Tensor,
                events: Tensor) -> Tensor:
        return nll_kernel_hazard(phi, idx_durations, events, self.alpha,
                                 self.sigma, self.gamma)


def nll_kernel_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                      alpha: float, sigma: float, gamma: float,
                      reduction: str = 'mean') -> Tensor:
    """
    Computes the kernel hazard function loss in the paper [1] (including a
    ranking loss) and adds a maximum grad determinant regularizer from [2]
    (encourages hyperspherical uniformity if the base neural net maps to a
    hypersphere).

    [1] George H. Chen. Deep Kernel Survival Analysis and Subject-Specific
        Survival Time Prediction Intervals. MLHC 2020.
    [2] Weiyang Liu, Rongmei Lin, Zhen Liu, Li Xiong, Bernhard Schölkopf,
        Adrian Weller. Learning with Hyperspherical Uniformity. AISTATS 2021.

    The inputs are the same as for pycox's nll_logistic_hazard function, where
    phi is the output of the base neural net. Time is assumed to have already
    been discretized.
    """
    if events.dtype is not torch.float:
        events = events.float()

    batch_size = phi.size(0)
    num_durations = idx_durations.max().item() + 1
    if alpha > 0:
        rank_mat_np = pair_rank_mat(idx_durations.detach().cpu().numpy(),
                                    events.detach().cpu().numpy())
        # rank_normalization_constant = rank_mat_np.sum()
        rank_mat = torch.tensor(rank_mat_np, device=phi.device)
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1, 1)
    y_bce = torch.zeros((batch_size, num_durations), dtype=torch.float,
                        device=phi.device).scatter(1, idx_durations, events)

    # compute kernel matrix
    kernel_matrix = (-symmetric_squared_pairwise_distances(phi)).exp()
    weights_loo = kernel_matrix - torch.eye(batch_size, device=phi.device)

    # bin weights in the same time index together (only for columns)
    weights_loo_discretized = \
        torch.matmul(weights_loo,
                     torch.zeros((batch_size, num_durations),
                                 dtype=torch.float,
                                 device=phi.device).scatter(1, idx_durations,
                                                            1))

    # kernel hazard function calculation
    num_at_risk = ((weights_loo_discretized.flip(1)).cumsum(1)).flip(1) + 1e-12
    num_events = torch.matmul(weights_loo, y_bce)
    hazards = torch.clamp(num_events / num_at_risk, 1e-12, 1. - 1e-12)

    if not torch.all((hazards >= 0) & (hazards <= 1)):
        # weird corner case
        return torch.tensor(np.inf, dtype=torch.float32)

    bce = F.binary_cross_entropy(hazards, y_bce, reduction='none')
    nll_loss = bce.cumsum(1).gather(1, idx_durations).view(-1)

    if gamma > 0:
        uniformity_loss = gamma * \
            -torch.logdet(kernel_matrix
                          + 1e-6 * torch.eye(batch_size, device=phi.device))
    else:
        uniformity_loss = 0.

    if alpha > 0:
        surv = (1 - hazards).add(1e-12).log().cumsum(1).exp()
        ones = torch.ones((batch_size, 1), device=phi.device)
        A = surv.matmul(y_bce.transpose(0, 1))
        diag_A = A.diag().view(1, -1)
        differences = (ones.matmul(diag_A) - A).transpose(0, 1)
        rank_loss = \
            (rank_mat * torch.exp(differences / sigma)).mean(1, keepdim=True)
            # / \
            # rank_normalization_constant

        return alpha * pycox.models.loss._reduction(nll_loss, reduction) + \
            (1 - alpha) * pycox.models.loss._reduction(rank_loss, reduction) \
            + uniformity_loss
    else:
        return pycox.models.loss._reduction(nll_loss, reduction) \
            + uniformity_loss


class KernelModel(Model):
    def __init__(self, net, loss=None, optimizer=None, device=None,
                 beta=0., tau=np.sqrt(-np.log(1e-4)), brute_force=False,
                 max_n_neighbors=256, dkn_max_n_neighbors=256,
                 ann_random_seed=2591188910,
                 ann_deterministic_construction=True):
        super(KernelModel, self).__init__(net, loss, optimizer, device)

        self.beta = beta
        self.tau = tau
        self.brute_force = brute_force
        self.n_neighbors = max_n_neighbors
        self.dkn_n_neighbors = dkn_max_n_neighbors
        self.ann_random_seed = ann_random_seed
        self.ann_deterministic_construction = ann_deterministic_construction

        self.training_data = None
        self.train_embeddings = None
        self.ann_index = None

    def build_ANN_index(self):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        train_embeddings = self.train_embeddings
        if type(train_embeddings) == torch.Tensor:
            train_embeddings = train_embeddings.detach().cpu().numpy()
        n_train = train_embeddings.shape[0]

        self.average_label = self.training_data[1].mean(axis=0)

        if self.beta == 0:
            ann_index = hnswlib.Index(space='l2',
                                      dim=train_embeddings.shape[1])
            if self.ann_deterministic_construction:
                ann_index.set_num_threads(1)
            else:
                ann_index.set_num_threads(os.cpu_count())
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items(train_embeddings, np.arange(n_train))
            ann_index.set_ef(min(2*self.n_neighbors, n_train))

            self.ann_index = ann_index

        else:
            eps = self.beta * self.tau
            eps_squared = eps * eps

            # build epsilon-net
            ann_index = \
                hnswlib.Index(space='l2', dim=train_embeddings.shape[1])
            ann_index.set_num_threads(1)
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.dkn_n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items([train_embeddings[0]], [0])
            assignments = [[0]]
            for idx in range(1, n_train):
                pt = train_embeddings[idx]
                labels, sq_dists = ann_index.knn_query([pt], k=1)
                nearest_exemplar_idx = labels[0, 0]
                sq_dist = sq_dists[0, 0]
                if sq_dist > eps_squared:
                    ann_index.add_items([pt], [len(assignments)])
                    assignments.append([idx])
                else:
                    assignments[nearest_exemplar_idx].append(idx)
            n_exemplars = len(assignments)
            if n_exemplars < self.dkn_n_neighbors:
                self.dkn_n_neighbors = n_exemplars
            ann_index.set_ef(min(2*self.dkn_n_neighbors, n_exemplars))

            self.ann_index = ann_index

            # compute summary statistics
            self.exemplar_assignments = [np.array(_) for _ in assignments]
            self.exemplar_sizes = np.array([len(_) for _ in assignments])
            self.exemplar_labels = \
                np.array([self.training_data[1][_].sum(axis=0)
                          for _ in assignments])

    def predict_classification(self, input, batch_size=256, numpy=None,
                               eval_=True, grads=False, to_cpu=False,
                               num_workers=0, is_dataloader=None, func=None,
                               **kwargs):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        test_embeddings = self.predict(input, batch_size, numpy, eval_,
                                       grads, to_cpu, num_workers,
                                       is_dataloader, func, **kwargs)
        if type(test_embeddings) == torch.Tensor:
            with torch.no_grad():
                test_embeddings = test_embeddings.detach().cpu().numpy()

        tau_squared = self.tau ** 2
        if self.brute_force:
            if self.beta == 0:
                sq_dists = cdist(test_embeddings, self.train_embeddings,
                                 'sqeuclidean')
            else:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings,
                                 self.train_embeddings[exemplars],
                                 'sqeuclidean')

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

        else:
            if self.beta == 0:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))

            if num_workers <= 0:
                n_threads = os.cpu_count()
            else:
                n_threads = num_workers

            if self.beta == 0:
                k = self.n_neighbors
            else:
                k = self.dkn_n_neighbors
            labels, sq_dists = self.ann_index.knn_query(test_embeddings, k=k,
                                                        num_threads=n_threads)

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            row_col_pairs = \
                np.array([(row, col)
                          for row, cols in enumerate(labels) for col in cols],
                         dtype=np.int64)
            rows = row_col_pairs[:, 0]
            cols = row_col_pairs[:, 1]
            sq_dists_flat = sq_dists.flatten()
            weights = csr_matrix((np.exp(-sq_dists_flat)
                                  * (sq_dists_flat <= tau_squared),
                                 (rows, cols)), shape=weights_shape)

        # what the paper calls gamma * n
        gamma_n = np.exp(-tau_squared) / self.train_embeddings.shape[0]

        if self.beta == 0:
            unnormalized = weights.dot(self.training_data[1]) \
                + gamma_n * self.average_label
        else:
            unnormalized = weights.dot(self.exemplar_labels) \
                + gamma_n * self.average_label
        return unnormalized

    def predict_regression(self, input, batch_size=256, numpy=None,
                           eval_=True, grads=False, to_cpu=False,
                           num_workers=0, is_dataloader=None, func=None,
                           **kwargs):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        test_embeddings = self.predict(input, batch_size, numpy, eval_,
                                       grads, to_cpu, num_workers,
                                       is_dataloader, func, **kwargs)
        if type(test_embeddings) == torch.Tensor:
            with torch.no_grad():
                test_embeddings = test_embeddings.detach().cpu().numpy()

        tau_squared = self.tau ** 2
        if self.brute_force:
            if self.beta == 0:
                sq_dists = cdist(test_embeddings, self.train_embeddings,
                                 'sqeuclidean')
            else:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings,
                                 self.train_embeddings[exemplars],
                                 'sqeuclidean')

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

        else:
            if self.beta == 0:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))

            if num_workers <= 0:
                n_threads = os.cpu_count()
            else:
                n_threads = num_workers

            if self.beta == 0:
                k = self.n_neighbors
            else:
                k = self.dkn_n_neighbors
            labels, sq_dists = self.ann_index.knn_query(test_embeddings, k=k,
                                                        num_threads=n_threads)

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            row_col_pairs = \
                np.array([(row, col)
                          for row, cols in enumerate(labels) for col in cols],
                         dtype=np.int64)
            rows = row_col_pairs[:, 0]
            cols = row_col_pairs[:, 1]
            sq_dists_flat = sq_dists.flatten()
            weights = csr_matrix((np.exp(-sq_dists_flat)
                                  * (sq_dists_flat <= tau_squared),
                                 (rows, cols)), shape=weights_shape)

        # what the paper calls gamma * n
        gamma_n = np.exp(-tau_squared) / self.train_embeddings.shape[0]

        if self.beta == 0:
            unnormalized = weights.dot(self.training_data[1]) \
                + gamma_n * self.average_label
            denom = weights.sum(axis=1) + gamma_n
        else:
            unnormalized = weights.dot(self.exemplar_labels) \
                + gamma_n * self.average_label
            denom = weights.dot(self.exemplar_sizes) + gamma_n
        if type(denom) != np.ndarray:
            denom = np.asarray(denom).ravel()
        return unnormalized / denom


class NKS(KernelModel):
    """
    Neural kernel survival analysis estimator; paired with a deep net as the
    base neural net/encoder, then one obtains a deep kernel survival analysis
    model
    """
    def __init__(self, net, optimizer=None, device=None, loss=None,
                 alpha=0.0, sigma=1.0, gamma=0.0, beta=0.0,
                 tau=np.sqrt(-np.log(1e-4)), brute_force=False,
                 max_n_neighbors=256, dkn_max_n_neighbors=256,
                 ann_random_seed=2591188910,
                 ann_deterministic_construction=True):
        ann_det_con = ann_deterministic_construction
        if loss is None:
            loss = NLLKernelHazardLoss(alpha, sigma, gamma)
        super(NKS, self).__init__(net, loss=loss, optimizer=optimizer,
                                  device=device, beta=beta, tau=tau,
                                  brute_force=brute_force,
                                  max_n_neighbors=max_n_neighbors,
                                  dkn_max_n_neighbors=dkn_max_n_neighbors,
                                  ann_random_seed=ann_random_seed,
                                  ann_deterministic_construction=ann_det_con)

    def build_ANN_index(self):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        train_embeddings = self.train_embeddings
        if type(train_embeddings) == torch.Tensor:
            train_embeddings = train_embeddings.detach().cpu().numpy()
        n_train = train_embeddings.shape[0]
        num_durations = len(self.duration_index)
        idx_durations, events = self.training_data[1]

        self.overall_event_at_risk_counts = \
            survival_summarize(idx_durations, events, num_durations)

        self.overall_hazard = \
            np.clip(self.overall_event_at_risk_counts[0] /
                    self.overall_event_at_risk_counts[1],
                    1e-12, 1. - 1e-12)
        self.overall_KM = \
            np.exp(np.cumsum(np.log(1 - self.overall_hazard + 1e-7)))

        if self.beta == 0:
            ann_index = hnswlib.Index(space='l2',
                                      dim=train_embeddings.shape[1])
            if self.ann_deterministic_construction:
                ann_index.set_num_threads(1)
            else:
                ann_index.set_num_threads(os.cpu_count())
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items(train_embeddings, np.arange(n_train))
            ann_index.set_ef(min(2*self.n_neighbors, n_train))

            self.ann_index = ann_index

            # store some helpful information for prediction
            train_event_matrix = np.zeros((n_train, num_durations))
            train_obs_matrix = np.zeros((n_train, num_durations))
            for row, (idx, event) in enumerate(zip(idx_durations, events)):
                train_event_matrix[row, idx] = event
                train_obs_matrix[row, idx] = 1
            self.train_event_matrix = train_event_matrix
            self.train_obs_matrix = train_obs_matrix

        else:
            eps = self.beta * self.tau
            eps_squared = eps * eps

            # build epsilon-net
            ann_index = \
                hnswlib.Index(space='l2', dim=train_embeddings.shape[1])
            ann_index.set_num_threads(1)
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.dkn_n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items([train_embeddings[0]], [0])
            assignments = [[0]]
            for idx in range(1, n_train):
                pt = train_embeddings[idx]
                labels, sq_dists = ann_index.knn_query([pt], k=1)
                nearest_exemplar_idx = labels[0, 0]
                sq_dist = sq_dists[0, 0]
                if sq_dist > eps_squared:
                    ann_index.add_items([pt], [len(assignments)])
                    assignments.append([idx])
                else:
                    assignments[nearest_exemplar_idx].append(idx)
            n_exemplars = len(assignments)
            if n_exemplars < self.dkn_n_neighbors:
                self.dkn_n_neighbors = n_exemplars
                if self.n_neighbors > self.dkn_n_neighbors:
                    self.n_neighbors = self.dkn_n_neighbors
            ann_index.set_ef(min(2*self.dkn_n_neighbors, n_exemplars))

            self.ann_index = ann_index

            # compute summary statistics
            self.exemplar_assignments = [np.array(_) for _ in assignments]
            self.exemplar_sizes = np.array([len(_) for _ in assignments])
            self.exemplar_labels = \
                np.array([survival_summarize(idx_durations[_], events[_],
                                             num_durations)
                          for _ in assignments])

        self.baseline_event_counts = None
        self.baseline_at_risk_counts = None
        self.baseline_hazard = None

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateNKS(self, scheme, duration_index, sub)

    def predict_surv_df(self, input, batch_size=256, eval_=True,
                        num_workers=0, mode='hazard', **kwargs):
        surv = self.predict_surv(input, batch_size, True, eval_, True,
                                 num_workers, mode=mode, **kwargs)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=256, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, epsilon=1e-7, mode='hazard',
                     **kwargs):
        if mode != 'hazard':
            raise NotImplementedError
        hazard = self.predict_hazard(input, batch_size, False, eval_,
                                     to_cpu, num_workers, **kwargs)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(self, input, batch_size=256, numpy=None, eval_=True,
                       to_cpu=False, num_workers=0, **kwargs):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       True, num_workers)
        train_embeddings = self.train_embeddings
        tau_squared = self.tau ** 2

        if self.beta == 0:
            # compute kernel matrix
            if self.brute_force:
                sq_dists = cdist(test_embeddings, train_embeddings,
                                 'sqeuclidean')
                weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

            else:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
                if num_workers <= 0:
                    n_threads = os.cpu_count()
                else:
                    n_threads = num_workers

                k = self.n_neighbors
                labels, sq_dists = \
                    self.ann_index.knn_query(test_embeddings, k=k,
                                             num_threads=n_threads)
                row_col_pairs = \
                    np.array([(row, col)
                              for row, cols in enumerate(labels)
                              for col in cols],
                             dtype=np.int64)
                rows = row_col_pairs[:, 0]
                cols = row_col_pairs[:, 1]
                sq_dists_flat = sq_dists.flatten()
                weights = csr_matrix((np.exp(-sq_dists_flat)
                                      * (sq_dists_flat <= tau_squared),
                                     (rows, cols)), shape=weights_shape)

            # convert observed time matrix to be for test data
            weights_discretized = weights.dot(self.train_obs_matrix)

            # kernel hazard function calculation
            num_at_risk = \
                np.flip(np.cumsum(np.flip(weights_discretized,
                                          axis=1), axis=1), axis=1) + 1e-12
            num_events = weights.dot(self.train_event_matrix)

            hazards = np.zeros(num_at_risk.shape)
            row_sums = num_events.sum(axis=1)
            row_zero_mask = (row_sums == 0)
            if np.any(row_zero_mask):
                hazards[row_zero_mask, :] = self.overall_hazard
            row_nonzero_mask = ~row_zero_mask
            hazards[row_nonzero_mask] = \
                np.clip(num_events[row_nonzero_mask]
                        / num_at_risk[row_nonzero_mask], 1e-12, 1. - 1e-12)

            # hazards = np.clip(num_events / num_at_risk, 1e-12, 1. - 1e-12)
        else:
            if self.brute_force:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings, train_embeddings[exemplars],
                                 'sqeuclidean')
                weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))
                if num_workers <= 0:
                    n_threads = os.cpu_count()
                else:
                    n_threads = num_workers

                k = self.dkn_n_neighbors
                labels, sq_dists = \
                    self.ann_index.knn_query(test_embeddings, k=k,
                                             num_threads=n_threads)
                row_col_pairs = \
                    np.array([(row, col)
                              for row, cols in enumerate(labels)
                              for col in cols],
                             dtype=np.int64)
                rows = row_col_pairs[:, 0]
                cols = row_col_pairs[:, 1]
                sq_dists_flat = sq_dists.flatten()
                weights = csr_matrix((np.exp(-sq_dists_flat)
                                      * (sq_dists_flat <= tau_squared),
                                     (rows, cols)), shape=weights_shape)

            numer = weights.dot(self.exemplar_labels[:, 0, :])
            denom = weights.dot(self.exemplar_labels[:, 1, :])
            hazards = np.zeros(numer.shape)

            if self.baseline_event_counts is not None:
                numer = numer + self.baseline_event_counts[np.newaxis, :]
                denom = denom + self.baseline_at_risk_counts[np.newaxis, :]

            denom_nonzero_mask = (denom > 0)
            if np.any(denom_nonzero_mask):
                hazards[denom_nonzero_mask] = numer[denom_nonzero_mask] / \
                    denom[denom_nonzero_mask]
            row_sums = numer.sum(axis=1)
            row_zero_mask = (row_sums == 0)
            if np.any(row_zero_mask):
                hazards[row_zero_mask, :] = self.overall_hazard

            hazards = np.clip(hazards, 1e-12, 1. - 1e-12)

        return tt.utils.array_or_tensor(hazards, numpy, input)

    def fit(self, input, target=None, batch_size=256, epochs=1, callbacks=None,
            verbose=True, num_workers=0, shuffle=True, metrics=None,
            val_data=None, val_batch_size=256, **kwargs):
        raise NotImplementedError

    def save_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().save_model_weights(path + extension, **kwargs)

    def load_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().load_model_weights(path + extension, **kwargs)

    def get_summary_functions(self):
        return self.exemplar_labels.copy()

    def load_summary_functions(self, exemplar_summary_functions,
                               baseline_event_counts=None,
                               baseline_at_risk_counts=None):
        self.exemplar_labels = exemplar_summary_functions
        self.baseline_event_counts = baseline_event_counts
        self.baseline_at_risk_counts = baseline_at_risk_counts


def one_hot(x):
    x_one_hot = np.zeros((x.size, int(x.max()) + 1))
    x_one_hot[np.arange(x.size), x.astype(np.int64)] = 1
    return x_one_hot.astype(np.float32)


def convert_times_to_indices(times):
    unique_times = np.unique(times)
    time_to_idx = {time: idx for idx, time in enumerate(unique_times)}
    indices = np.array([time_to_idx[time] for time in times])
    return indices, unique_times, time_to_idx


def survival_summarize(idx_durations, events, num_durations, weights=None):
    unnormalized_Z_q_bar = np.zeros(num_durations)
    unnormalized_Z_q_plus = np.zeros(num_durations)
    if weights is None:
        for idx, event in zip(idx_durations, events):
            unnormalized_Z_q_bar[idx] += event
            unnormalized_Z_q_plus[idx] += 1
    else:
        for idx, event, weight in zip(idx_durations, events, weights):
            unnormalized_Z_q_bar[idx] += event * weight
            unnormalized_Z_q_plus[idx] += weight
    unnormalized_Z_q_plus = np.flip(np.cumsum(np.flip(unnormalized_Z_q_plus)))
    return np.array([unnormalized_Z_q_bar, unnormalized_Z_q_plus])


def build_epsilon_nets(features, epsilons, ef, M, random_seed,
                       presorted=False, verbose=False):
    if not presorted:
        epsilons = np.sort(epsilons)
    if not type(epsilons) == np.ndarray:
        epsilons = np.array(epsilons)
    epsilons_squared = epsilons ** 2

    n, d = features.shape
    ann_indices = []
    n_eps = len(epsilons)
    n_exemplars = np.ones(n_eps, dtype=np.int64)
    exemplars = []
    for idx in range(n_eps):
        ann_index = hnswlib.Index(space='l2', dim=d)
        ann_index.set_num_threads(1)
        ann_index.init_index(max_elements=n, ef_construction=ef, M=M,
                             random_seed=random_seed)
        ann_index.add_items([features[0]], [0])
        ann_indices.append(ann_index)
        exemplars.append([0])

    if verbose:
        print('Constructing epsilon-nets with epsilons:', epsilons, flush=True)
        current_progress = 0
    for idx in range(1, n):
        pt = features[idx]
        labels, sq_dists = ann_index.knn_query([pt], k=1)
        sq_dist = sq_dists[0, 0]
        for ann_idx, eps_squared in enumerate(epsilons_squared):
            if sq_dist > eps_squared:
                ann_indices[ann_idx].add_items([pt], [n_exemplars[ann_idx]])
                n_exemplars[ann_idx] += 1
                exemplars[ann_idx].append(idx)
            else:
                break

        if verbose:
            progress = int((idx + 1) / n * 20)
            if progress > current_progress:
                print('Progress: %d%%' % (progress * 5), flush=True)
                current_progress = progress

    for ann_idx in range(n_eps):
        ann_indices[ann_idx].set_ef(min(ef, n_exemplars[ann_idx]))

    return ann_indices, [np.array(_, dtype=np.int64) for _ in exemplars], \
        n_exemplars


class KernelPretrainMSELoss(_Loss):
    """
    Code nearly identical to the standard (non-kernel) version from the
    official PyTorch repo; the forward() function is changed to accommodate
    leave-one-out kernel prediction
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super(KernelPretrainMSELoss, self).__init__(size_average, reduce,
                                                    reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target_kernel = 1. - torch.cdist(target, target, 0) / target.size(1)
        input_kernel = (-symmetric_squared_pairwise_distances(input)).exp() \
            - torch.eye(input.size(0), device=input.device)

        return F.mse_loss(input_kernel.view(-1), target_kernel.view(-1),
                          reduction=self.reduction)


def compute_mds_embeddings(X, xgb_model, n_output_features, mds_n_init=4,
                           mds_random_seed=2306758833):
    leaves = xgb_model.predict(xgb.DMatrix(X), pred_leaf=True)

    forest_kernel = 1. - squareform(pdist(leaves, 'hamming'))
    forest_est_dist = \
        np.sqrt(np.clip(np.log(1. / (forest_kernel + 1e-12)), 0., None))

    mds = MDS(n_components=n_output_features, n_init=mds_n_init,
              dissimilarity='precomputed', random_state=mds_random_seed)
    mds_embeddings = \
        mds.fit_transform(forest_est_dist).astype(np.float32)

    return mds_embeddings


class NKSSummaryLoss(nn.Module):
    def __init__(self, alpha, sigma):
        super(NKSSummaryLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, hazards: Tensor, idx_durations: Tensor, events: Tensor,
                reduction: str = 'mean') -> Tensor:
        if not torch.all((hazards >= 0) & (hazards <= 1)):
            # weird corner case
            return torch.tensor(np.inf, dtype=torch.float32)

        if events.dtype is not torch.float:
            events = events.float()

        alpha = self.alpha
        sigma = self.sigma

        batch_size = hazards.size(0)
        num_durations = hazards.size(1)
        if alpha > 0:
            rank_mat_np = pair_rank_mat(idx_durations.detach().cpu().numpy(),
                                        events.detach().cpu().numpy())
            # rank_normalization_constant = rank_mat_np.sum()
            rank_mat = torch.tensor(rank_mat_np, device=hazards.device)
        idx_durations = idx_durations.view(-1, 1)
        events = events.view(-1, 1)
        y_bce = torch.zeros((batch_size, num_durations), dtype=torch.float,
                            device=hazards.device).scatter(1, idx_durations,
                                                           events)

        bce = F.binary_cross_entropy(hazards, y_bce, reduction='none')
        nll_loss = bce.cumsum(1).gather(1, idx_durations).view(-1)

        if alpha > 0:
            surv = (1 - hazards).add(1e-12).log().cumsum(1).exp()
            ones = torch.ones((batch_size, 1), device=hazards.device)
            A = surv.matmul(y_bce.transpose(0, 1))
            diag_A = A.diag().view(1, -1)
            differences = (ones.matmul(diag_A) - A).transpose(0, 1)
            rank_loss = \
                (rank_mat
                 * torch.exp(differences / sigma)).mean(1, keepdim=True)

            return alpha * pycox.models.loss._reduction(nll_loss, reduction) \
                + (1 - alpha) \
                * pycox.models.loss._reduction(rank_loss, reduction)
        else:
            return pycox.models.loss._reduction(nll_loss, reduction)


class InterpolateNKS(InterpolateLogisticHazard):
    def predict_surv(self, input, batch_size=256, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, mode='hazard', **kwargs):
        if self.scheme in ['const_hazard', 'exp_surv']:
            surv = self._surv_const_haz(input, batch_size, numpy, eval_,
                                        to_cpu, num_workers, **kwargs)
        elif self.scheme in ['const_pdf', 'lin_surv']:
            surv = self._surv_const_pdf(input, batch_size, numpy, eval_,
                                        to_cpu, num_workers, mode=mode,
                                        **kwargs)
        else:
            raise NotImplementedError
        return surv

    def predict_surv_df(self, input, batch_size=256, eval_=True, to_cpu=False,
                        num_workers=0, mode='hazard', **kwargs):
        surv = self.predict_surv(input, batch_size, True, eval_, to_cpu,
                                 num_workers, mode=mode, **kwargs)
        index = None
        if self.duration_index is not None:
            index = pycox.models.utils.make_subgrid(self.duration_index,
                                                    self.sub)
        return pd.DataFrame(surv.transpose(), index)

    def _surv_const_pdf(self, input, batch_size=256, numpy=None, eval_=True,
                        to_cpu=False, num_workers=0, mode='hazard', **kwargs):
        s = self.model.predict_surv(input, batch_size, False, eval_, to_cpu,
                                    num_workers, mode=mode, **kwargs).float()
        n, m = s.shape
        device = s.device
        diff = \
            (s[:, 1:] -
             s[:, :-1]).contiguous().view(-1, 1).repeat(1,
                                                        self.sub).view(n, -1)
        rho = torch.linspace(0, 1, self.sub+1,
                             device=device)[:-1].contiguous().repeat(n, m-1)
        s_prev = \
            s[:, :-1].contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        surv = torch.zeros(n, int((m-1)*self.sub + 1))
        surv[:, :-1] = diff * rho + s_prev
        surv[:, -1] = s[:, -1]
        return tt.utils.array_or_tensor(surv, numpy, input)


class KernelModelSummary(nn.Module):
    def __init__(self, kernel_model, exemplar_labels, average_label,
                 log_output=False):
        super(KernelModelSummary, self).__init__()
        self.ann_index = kernel_model.ann_index
        self.dkn_n_neighbors = kernel_model.dkn_n_neighbors
        self.tau_squared = kernel_model.tau**2
        device = kernel_model.device
        exemplar_indices = \
            np.array([_[0] for _ in kernel_model.exemplar_assignments],
                     dtype=np.int64)
        exemplar_embeddings = kernel_model.train_embeddings[exemplar_indices]
        exemplar_sizes = kernel_model.exemplar_sizes
        n_train = kernel_model.train_embeddings.shape[0]
        if type(exemplar_embeddings) is not torch.Tensor:
            exemplar_embeddings = torch.tensor(exemplar_embeddings,
                                               dtype=torch.float32,
                                               device=device)
        else:
            exemplar_embeddings = torch.clone(exemplar_embeddings.detach())
        self.exemplar_embeddings = exemplar_embeddings
        self.exemplar_sizes = exemplar_sizes
        assert type(exemplar_labels) == np.ndarray
        self.exemplar_labels = nn.Parameter(torch.tensor(exemplar_labels,
                                                         dtype=torch.float32,
                                                         device=device))
        self.average_label = nn.Parameter(torch.tensor(average_label,
                                                       dtype=torch.float32,
                                                       device=device))
        self.exemplar_sizes = torch.tensor(exemplar_sizes,
                                           dtype=torch.float32,
                                           device=device)
        self.distance_threshold = torch.tensor(kernel_model.tau,
                                               dtype=torch.float32,
                                               device=device)
        self.gamma_n = (-self.distance_threshold**2).exp() / \
            torch.tensor(n_train, dtype=torch.float32, device=device)
        self.log_output = log_output

    def forward(self, input, num_workers: int = 0) -> Tensor:
        # --------------------------------------------------------------------
        # Get kernel weights (done on CPU; at this point the embedding space
        # and thus also the kernel weights are treated as fixed)
        #
        if type(input) == torch.Tensor:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input

        weights_shape = (input_np.shape[0], self.exemplar_embeddings.size(0))
        if num_workers <= 0:
            n_threads = os.cpu_count()
        else:
            n_threads = num_workers
        labels, sq_dists = \
            self.ann_index.knn_query(input_np,
                                     k=self.dkn_n_neighbors,
                                     num_threads=n_threads)
        row_col_pairs = \
            np.array([(row, col)
                      for row, cols in enumerate(labels)
                      for col in cols],
                     dtype=np.int64)
        rows = row_col_pairs[:, 0]
        cols = row_col_pairs[:, 1]
        sq_dists_flat = sq_dists.flatten()
        weights = csr_matrix((np.exp(-sq_dists_flat)
                              * (sq_dists_flat <= self.tau_squared),
                             (rows, cols)), shape=weights_shape)
        kernel_weights = torch.tensor(weights.toarray(), dtype=torch.float32,
                                      device=self.exemplar_labels.device)

        # --------------------------------------------------------------------
        # Compute regression predictions
        #
        if len(self.exemplar_labels.size()) == 1:
            numer = torch.matmul(kernel_weights,
                                 self.exemplar_labels.view(-1, 1)) \
                + self.gamma_n * self.average_label
        else:
            numer = torch.matmul(kernel_weights, self.exemplar_labels) \
                + self.gamma_n * self.average_label
        denom = torch.matmul(kernel_weights, self.exemplar_sizes.view(-1, 1)) \
            + self.gamma_n

        if len(self.exemplar_labels.size()) == 1:
            output = (numer / denom).view(-1)
        else:
            output = numer / denom

        if self.log_output:
            return (output + 1e-12).log()
        return output


class NKSSummary(nn.Module):
    def __init__(self, kernel_model, exemplar_labels):
        super(NKSSummary, self).__init__()
        self.ann_index = kernel_model.ann_index
        self.dkn_n_neighbors = kernel_model.dkn_n_neighbors
        self.tau_squared = kernel_model.tau**2
        device = kernel_model.device
        exemplar_indices = \
            np.array([_[0] for _ in kernel_model.exemplar_assignments],
                     dtype=np.int64)
        exemplar_embeddings = kernel_model.train_embeddings[exemplar_indices]
        if type(exemplar_embeddings) is not torch.Tensor:
            exemplar_embeddings = torch.tensor(exemplar_embeddings,
                                               dtype=torch.float32,
                                               device=device)
        else:
            exemplar_embeddings = torch.clone(exemplar_embeddings.detach())
        self.exemplar_embeddings = exemplar_embeddings
        assert type(exemplar_labels) == np.ndarray
        exemplar_event_counts = exemplar_labels[:, 0, :]
        exemplar_at_risk_counts = \
            np.hstack((exemplar_labels[:, 1, :],
                       np.zeros((exemplar_labels.shape[0], 1))))
        exemplar_censor_counts = \
            exemplar_at_risk_counts[:, :-1] - exemplar_at_risk_counts[:, 1:] \
            - exemplar_event_counts
        self.log_exemplar_event_counts = \
            nn.Parameter(torch.tensor(exemplar_event_counts,
                                      dtype=torch.float32,
                                      device=device).clamp(
                                          min=1e-12).log())
        self.log_exemplar_censor_counts = \
            nn.Parameter(torch.tensor(exemplar_censor_counts,
                                      dtype=torch.float32,
                                      device=device).clamp(
                                          min=1e-12).log())
        n_durations = exemplar_event_counts.shape[1]
        self.log_baseline_event_counts = \
            nn.Parameter(-27.6310211159 * torch.ones(n_durations,
                                                     dtype=torch.float32,
                                                     device=device))
        self.log_baseline_censor_counts = \
            nn.Parameter(-27.6310211159 * torch.ones(n_durations,
                                                     dtype=torch.float32,
                                                     device=device))

    def forward(self, input, num_workers: int = 0) -> Tensor:
        # --------------------------------------------------------------------
        # Get kernel weights (done on CPU; at this point the embedding space
        # and thus also the kernel weights are treated as fixed)
        #
        if type(input) == torch.Tensor:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input

        weights_shape = (input_np.shape[0], self.exemplar_embeddings.size(0))
        if num_workers <= 0:
            n_threads = os.cpu_count()
        else:
            n_threads = num_workers
        labels, sq_dists = \
            self.ann_index.knn_query(input_np,
                                     k=self.dkn_n_neighbors,
                                     num_threads=n_threads)
        row_col_pairs = \
            np.array([(row, col)
                      for row, cols in enumerate(labels)
                      for col in cols],
                     dtype=np.int64)
        rows = row_col_pairs[:, 0]
        cols = row_col_pairs[:, 1]
        sq_dists_flat = sq_dists.flatten()
        weights = csr_matrix((np.exp(-sq_dists_flat)
                              * (sq_dists_flat <= self.tau_squared),
                             (rows, cols)), shape=weights_shape)
        kernel_weights = \
            torch.tensor(weights.toarray(), dtype=torch.float32,
                         device=self.log_baseline_event_counts.device)

        # --------------------------------------------------------------------
        # Compute hazard functions
        #
        baseline_event_counts = self.log_baseline_event_counts.exp()
        baseline_censor_counts = self.log_baseline_censor_counts.exp()
        baseline_at_risk_counts = \
            (baseline_event_counts
             + baseline_censor_counts).flip(0).cumsum(0).flip(0)

        exemplar_event_counts = self.log_exemplar_event_counts.exp()
        exemplar_censor_counts = self.log_exemplar_censor_counts.exp()
        exemplar_at_risk_counts = \
            (exemplar_event_counts
             + exemplar_censor_counts).flip(1).cumsum(1).flip(1)

        numer = torch.matmul(kernel_weights, exemplar_event_counts) \
            + baseline_event_counts.view(1, -1)
        denom = torch.matmul(kernel_weights, exemplar_at_risk_counts) \
            + baseline_at_risk_counts.view(1, -1) + 1e-12

        return torch.clamp(numer / denom, 1e-12, 1. - 1e-12)  # hazards

    def get_exemplar_summary_functions_baseline_event_at_risk_counts(self):
        log_exemplar_event_counts = self.log_exemplar_event_counts
        log_exemplar_censor_counts = self.log_exemplar_censor_counts
        log_baseline_event_counts = self.log_baseline_event_counts
        log_baseline_censor_counts = self.log_baseline_censor_counts

        exemplar_event_counts = \
            np.exp(log_exemplar_event_counts.detach(
                ).cpu().numpy())
        exemplar_censor_counts = \
            np.exp(log_exemplar_censor_counts.detach(
                ).cpu().numpy())
        exemplar_at_risk_counts = \
            np.flip(
                np.cumsum(
                    np.flip(exemplar_event_counts
                            + exemplar_censor_counts,
                            axis=1),
                    axis=1),
                axis=1)
        baseline_event_counts = \
            np.exp(log_baseline_event_counts.detach(
                ).cpu().numpy())
        baseline_censor_counts = \
            np.exp(log_baseline_censor_counts.detach(
                ).cpu().numpy())
        baseline_at_risk_counts = \
            np.flip(
                np.cumsum(
                    np.flip(baseline_event_counts
                            + baseline_censor_counts)))

        n_exemplars = exemplar_event_counts.shape[0]
        n_time = exemplar_event_counts.shape[1]
        exemplar_labels = np.zeros((n_exemplars, 2, n_time))
        exemplar_labels[:, 0, :] = exemplar_event_counts
        exemplar_labels[:, 1, :] = exemplar_at_risk_counts
        return exemplar_labels, baseline_event_counts, baseline_at_risk_counts
