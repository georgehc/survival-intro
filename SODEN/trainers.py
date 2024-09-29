from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from six.moves import cPickle as pickle
from torch.autograd import Variable
from utils import SEP, MyPrinter, to_np


def detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, dict):
        detached_data = {}
        for key in data:
            detached_data[key] = detach(data[key])
    elif type(data) == list:
        detached_data = []
        for x in data:
            detached_data.append(detach(x))
    else:
        raise NotImplementedError("Type {} not supported.".format(type(data)))
    return detached_data


class SODENTrainer(object):
    """SODENTrainer."""

    def __init__(self,
                 model=None,
                 device="cuda",
                 criterions=None,
                 optimizer=None,
                 dataloaders=None,
                 metrics=None,
                 earlystop_metric_name=None,
                 batch_size=4,
                 num_epochs=50,
                 patience=15,
                 grad_clip=None,
                 result_path=None,
                 model_path=None,
                 log_path=None,
                 log_step=200,
                 exp_name=None,
                 verbose=1,
                 fine_tune=False,
                 debug=False):
        """Initializes a SODENTrainer.

        Arguments:
          model: An instantiation of model.
          device: A string of "cuda" or "cpu".
          criterions: A dict with values being instantiations of training
            objectives.
          optimizer: An instantiation of an optimizer. The instantiation should
            have already take model.parameters() in instantiation.
          dataloaders: A dict of dataloaders where keys are "train", "valid",
            and "test" and each values is a PyTorch Dataloader.
          metrics: A dict of evaluation metrics of interest. Each key is the
            metric name and each value is a metric object that has `add` and
            `value` functions. `add` takes model_outputs and targets as input.
          earlystop_metric_name: A string indicating the name of metrics to be
            used for early stopping.
          batch_size: A number indicating batch_size. This should align with
            the dataloaders.
          num_epochs: Maximum number of epochs to run.
          patience: Initial patience for early stopping.
          grad_clip: A number for grad_clip.
          result_path: A string of path to store the results.
          model_path: A string of path to store the models.
          log_path: A string of path to store the logs.
          log_step: The logging interavl in terms of training steps.
          exp_name: The name of this experiment.
          verbose: Verbose mark.
          debug: Debug flag.
        """
        self.model = model
        self.device = device
        self.criterions = criterions
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.earlystop_metric_name = earlystop_metric_name

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        # Logging config.
        self.result_path = result_path
        self.model_path = model_path
        self.log_path = log_path
        self.log_step = log_step
        self.exp_name = exp_name
        self.verbose = verbose
        self.fine_tune = fine_tune
        self.debug = debug

        # Initialization.
        self.model.to(self.device)

        # State recording.
        self.curr_epoch = 0
        self.curr_patience = self.patience
        self.curr_step = 0
        self.global_step = 0

        self.running_loss_dict = {}
        self.clear_running_loss()

        self.best_earlystop_metric = self.initialize_metric()
        # The metric objects with methods `add` and `value`
        self.curr_metrics = {}
        self.curr_metrics["valid"] = deepcopy(self.metrics)
        self.curr_metrics["test"] = deepcopy(self.metrics)
        # The metric values at each eval step
        self.metric_value_trajectories = {}
        self.metric_value_trajectories["valid"] = []
        self.metric_value_trajectories["test"] = []

        if fine_tune:
            model_file = "%s.pt" % SEP.join(["best_ckpt", self.exp_name])
            model_file = os.path.join(model_path, model_file)
            ckpt = torch.load(model_file, map_location=torch.device(self.device))
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            self.exp_name = SEP.join([self.exp_name, "fine_tune"])
            if self.log_path:
                self.log_path = SEP.join([self.log_path, "fine_tune"])

        self.printer = MyPrinter(self.verbose, exp_name=exp_name,
                                 log_path=log_path, debug=self.debug)

    def initialize_metric(self):
        # TODO: more general implementation.
        return np.inf

    def metric_imporved(self, current, previous):
        """Check if the metric is improved."""
        # TODO: more general implementation.
        return current < previous

    def wrap_batch_data(self, batch_data):
        features, labels = batch_data
        if isinstance(features, dict):
            for name in features:
                # TODO: refactor it to a preprocessing function.
                if name == "init_cond" and "features" not in features and "seq_feat" not in features:
                    features[name] = Variable(
                        torch.tensor([0], dtype=torch.float)).to(self.device)
                else:
                    features[name] = Variable(features[name]).to(self.device)
        else:
            features = Variable(features).to(self.device)
        if isinstance(labels, dict):
            for name in labels:
                labels[name] = Variable(labels[name]).to(self.device)
        else:
            labels = Variable(labels).to(self.device)
        return features, labels

    def calculate_loss(self, outputs, labels):
        loss_dict = {}
        loss_dict = {}
        for name in self.criterions:
            if name == "":
                # TODO: Check if there are some parts of the training objective
                # that do not take `outputs` and `labels` as input; or the
                # `outputs` and `labels` are dicts. Multiple if-branches could
                # be added here.
                raise NotImplementedError("Need to check the implementation of"
                                          " `calculate_loss` method.")
            else:
                loss_dict[name] = self.criterions[name](outputs, labels)
        return loss_dict

    def train_one_step(self, features, labels):
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss_dict = self.calculate_loss(outputs, labels)
        loss = sum(loss_dict.values())
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss_dict

    def clear_running_loss(self):
        for name in self.criterions:
            self.running_loss_dict[name] = 0.0

    def maybe_log(self, loss_dict):
        """"""
        # Update running losses.
        for name in loss_dict:
            self.running_loss_dict[name] += loss_dict[name].item()
        # Check if global_step reaches log step.
        if self.global_step % self.log_step == 0:
            for name in self.running_loss_dict:
                self.running_loss_dict[name] /= self.log_step
            total_loss = sum(self.running_loss_dict.values())
            # Printing.
            self.printer.print(
                "step %5d total loss: %.6f" % (self.curr_step, total_loss),
                level=2)
            self.clear_running_loss()

    def train_one_epoch(self, train_loader):
        self.model.train()
        self.curr_step = 0
        for batch_data in train_loader:
            features, labels = self.wrap_batch_data(batch_data)
            loss_dict = self.train_one_step(features, labels)
            self.curr_step += 1
            self.global_step += 1
            self.maybe_log(loss_dict)

    def clear_curr_metrics(self, phase):
        self.curr_metrics[phase] = deepcopy(self.metrics)

    def get_metric_value(self, metric_name, phase):
        value = self.curr_metrics[phase][metric_name].value()
        return value

    def get_metric_value_dict(self, phase):
        assert phase in ["valid", "test"]
        metric_value_dict = {}
        for metric_name in self.curr_metrics[phase]:
            metric_value_dict[metric_name] = self.get_metric_value(metric_name,
                                                                   phase)
        return metric_value_dict

    def eval_update_one_step(self, features, labels, phase="valid"):
        outputs = self.model(features)
        for metric_name in self.curr_metrics[phase]:
            self.curr_metrics[phase][metric_name].add(detach(outputs),
                                                      detach(labels))

    def eval(self, phase="valid"):
        assert phase in ["valid", "test"]
        self.model.eval()

        self.clear_curr_metrics(phase)
        for batch_data in self.dataloaders[phase]:
            features, labels = self.wrap_batch_data(batch_data)
            self.eval_update_one_step(features, labels, phase)

    def maybe_save_ckpt(self, ckpt_name=None):
        if self.model_path is not None:
            if ckpt_name is None:
                ckpt_name = "best_ckpt"
            model_file = "%s.pt" % SEP.join([ckpt_name, self.exp_name])
            model_file = os.path.join(self.model_path, model_file)
            torch.save({
                "global_step": self.global_step,
                "metric": self.best_earlystop_metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, model_file)

    def is_earlystop_metric(self, metric_name):
        """Determine if `metric_name` should be counted in earlystop."""
        if self.earlystop_metric_name == metric_name:
            return True
        return False

    def check_earlystop_metric(self, metric_value_dict):
        """Update best_earlystop_metric and curr_patience."""
        earlystop_metrics = []
        for metric_name in metric_value_dict:
            if self.is_earlystop_metric(metric_name):
                earlystop_metrics.append(
                    metric_value_dict[metric_name][0])
        earlystop_metric = np.mean(earlystop_metrics)
        if self.metric_imporved(earlystop_metric, self.best_earlystop_metric):
            self.best_earlystop_metric = earlystop_metric
            self.curr_patience = self.patience
            self.maybe_save_ckpt()
        self.printer.print(
            "early stop metric: %.6f" % earlystop_metric, level=1)

    def maybe_save_result(self):
        if self.result_path is not None:
            result_file = "%s.pkl" % SEP.join(
                ["%.6f" % self.best_earlystop_metric, self.exp_name])
            result_file = os.path.join(self.result_path, result_file)
            with open(result_file, "wb") as f:
                pickle.dump(self.metric_value_trajectories, f)

    def train(self):
        while self.curr_epoch < self.num_epochs and self.curr_patience > 0:
            self.curr_patience -= 1
            self.printer.print(
                "epoch %d" % self.curr_epoch, level=1, print_time=True)
            self.train_one_epoch(self.dataloaders["train"])

            self.eval(phase="valid")
            metric_value_dict = self.get_metric_value_dict(phase="valid")
            self.metric_value_trajectories["valid"].append(metric_value_dict)

            self.eval(phase="test")
            metric_value_dict = self.get_metric_value_dict(phase="test")
            self.metric_value_trajectories["test"].append(metric_value_dict)

            # Compare the current earlystop metric to the best one. Possibly
            # renew the self.curr_patience. Maybe save ckpt.
            metric_value_dict = self.metric_value_trajectories["valid"][-1]
            self.check_earlystop_metric(metric_value_dict)

            for phase in ["valid", "test"]:
                metric_value_dict = self.metric_value_trajectories[phase][-1]
                for name in metric_value_dict:
                    self.printer.print(
                        "Phase: %s, metric: %s: %.6f" % (
                            phase, name, metric_value_dict[name][0]), level=1)

            if self.debug:
                self.maybe_save_ckpt("ckpt_%d" % self.curr_epoch)

            self.curr_epoch += 1

        self.maybe_save_result()
