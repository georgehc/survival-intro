import numpy as np
import torch
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

NUM_INT_STEPS = 1000


class CIndexMeter(object):  # TODO: handle tied time
    def __init__(self, save_raw=False):
        super(CIndexMeter, self).__init__()
        self.reset()
        self.save_raw = save_raw

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        if self.eval_t is None:
            self.eval_t = self.to_numpy(outputs["eval_t"])
        if self.cum_hazard_seqs is None:
            self.cum_hazard_seqs = self.to_numpy(outputs["cum_hazard_seqs"])
        else:
            self.cum_hazard_seqs = np.concatenate(
                [self.cum_hazard_seqs, self.to_numpy(outputs["cum_hazard_seqs"])],
                axis=1)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        num = 0
        s = 0
        for i in np.argwhere(self.labels > 0).reshape(-1):
            for j in range(i + 1, self.cum_hazard_seqs.shape[1]):
                k = np.argwhere(self.eval_t == self.t[i])[0, 0]
                num += 1
                if self.cum_hazard_seqs[k, i] > self.cum_hazard_seqs[k, j]:
                    s += 1

        if self.save_raw:
            raw = (self.t, self.labels, self.eval_t, self.cum_hazard_seqs)
            return [float(s) / num, raw]
        else:
            return [float(s) / num]

    def reset(self):
        self.t = np.array([])
        self.eval_t = None
        self.cum_hazard_seqs = None
        self.labels = np.array([])


class IPWCIndexMeter(object):
    def __init__(self, eps=1e-8):
        super(IPWCIndexMeter, self).__init__()
        self.reset()
        self.eps = eps

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        if self.eval_t is None:
            self.eval_t = self.to_numpy(outputs["eval_t"])
        if self.cum_hazard_seqs is None:
            self.cum_hazard_seqs = self.to_numpy(outputs["cum_hazard_seqs"])
        else:
            self.cum_hazard_seqs = np.concatenate(
                [self.cum_hazard_seqs, self.to_numpy(outputs["cum_hazard_seqs"])],
                axis=1)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        kmf = KaplanMeierFitter()
        kmf.fit(self.t, event_observed=(1 - self.labels))
        G_T = kmf.predict(self.t, interpolate=True).to_numpy()
        G_T[G_T == 0] = self.eps / 2  # still smaller than eps
        inv_G_T_square = 1. / G_T**2

        num = 0
        s = 0
        for i in np.argwhere(self.labels > 0).reshape(-1):
            if G_T[i] < self.eps:  # remove unstable estimates
                continue
            k = np.argwhere(self.eval_t == self.t[i])[0, 0]
            idx = np.logical_or(
                self.t > self.t[i],
                np.logical_and(self.t == self.t[i], self.labels == 0))
            s += sum(self.cum_hazard_seqs[k, idx] < self.cum_hazard_seqs[k, i]) * inv_G_T_square[i]
            num += sum(idx) * inv_G_T_square[i]
        return [float(s) / num]

    def reset(self):
        self.t = np.array([])
        self.eval_t = None
        self.cum_hazard_seqs = None
        self.labels = np.array([])


class BrierScoreMeter(object):
    def __init__(self, eps=0):
        super(BrierScoreMeter, self).__init__()
        assert eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.eps = eps
        self.reset()

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        if self.eps == 0:
            surv_key = "survival_seqs"
        else:
            surv_key = "survival_seqs_{}".format(self.eps)
        if self.survival_seqs is None:
            self.survival_seqs = self.to_numpy(outputs[surv_key])
        else:
            self.survival_seqs = np.concatenate(
                [self.survival_seqs, self.to_numpy(outputs[surv_key])],
                axis=1)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        S = self.survival_seqs
        kmf = KaplanMeierFitter()
        kmf.fit(self.t, event_observed=(1 - self.labels))
        G_T = kmf.predict(self.t, interpolate=True).to_numpy()

        t_span = np.linspace(self.t.min(), max(self.t[G_T > self.eps]),
                             S.shape[0])
        G_t = kmf.predict(t_span, interpolate=True).to_numpy()

        ind = ((self.t.reshape(1, -1) <= t_span.reshape(-1, 1))).astype(float)
        labels = self.labels

        # Remove indices where G_t are zero
        S = S[G_t > 1e-8]
        ind = ind[G_t > 1e-8]
        G_t = G_t[G_t > 1e-8]

        # Remove indices where G_T are zero
        S = S[:, G_T > 1e-8]
        ind = ind[:, G_T > 1e-8]
        labels = labels[G_T > 1e-8]
        G_T = G_T[G_T > 1e-8]

        labels = labels.reshape(1, -1)
        G_t = G_t.reshape(-1, 1)
        G_T = G_T.reshape(1, -1)

        brier = S**2 * labels * ind / G_T
        brier += (1 - S)**2 * (1 - ind) / G_t
        return brier.mean()

    def reset(self):
        self.t = np.array([])
        self.survival_seqs = None
        self.labels = np.array([])


class ConcordanceMeter(object):
    def __init__(self, output_key="prod", save_raw=False):
        super(ConcordanceMeter, self).__init__()
        self.output_key = output_key
        self.reset()
        self.save_raw = save_raw

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        self.prod = np.concatenate(
            [self.prod, self.to_numpy(outputs[self.output_key])], axis=0)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        if self.save_raw:
            raw = (self.t, self.labels, self.prod)
            return [concordance_index(
                event_times=self.t, predicted_scores=-self.prod,
                event_observed=self.labels), raw]
        return [concordance_index(
            event_times=self.t, predicted_scores=-self.prod,
            event_observed=self.labels)]

    def reset(self):
        self.t = np.array([])
        self.prod = np.array([])
        self.labels = np.array([])


class IPWConcordanceMeter(object):
    def __init__(self, output_key="prod", eps=1e-8):
        super(IPWConcordanceMeter, self).__init__()
        self.output_key = output_key
        self.eps = eps
        self.reset()

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        self.prod = np.concatenate(
            [self.prod, self.to_numpy(outputs[self.output_key])], axis=0)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        kmf = KaplanMeierFitter()
        kmf.fit(self.t, event_observed=(1 - self.labels))
        G_T = kmf.predict(self.t, interpolate=True).to_numpy()
        G_T[G_T == 0] = self.eps / 2  # still smaller than eps
        inv_G_T_square = 1. / G_T**2

        num = 0
        s = 0
        for i in np.argwhere(self.labels > 0).reshape(-1):
            if G_T[i] < self.eps:
                continue
            idx = np.logical_or(
                self.t > self.t[i],
                np.logical_and(self.t == self.t[i], self.labels == 0))
            s += sum(self.prod[idx] < self.prod[i]) * inv_G_T_square[i]
            num += sum(idx) * inv_G_T_square[i]
        return [float(s) / num]

    def reset(self):
        self.t = np.array([])
        self.prod = np.array([])
        self.labels = np.array([])


class QuantileConcordanceMeter(ConcordanceMeter):
    def __init__(self, q=0.5):
        super(QuantileConcordanceMeter, self).__init__()
        self.q = q
        self.output_key = "Lambda_q%02d" % int(100 * self.q)
        self.reset()

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        self.prod = np.concatenate(
            [self.prod, self.to_numpy(outputs[self.output_key])], axis=0)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)


class BinomialLogLikelihoodMeter(object):
    def __init__(self, eps=0):
        super(BinomialLogLikelihoodMeter, self).__init__()
        assert eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.eps = eps
        self.reset()

    def to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("cpu").numpy()
        if isinstance(data, dict):
            np_data = {}
            for key in data:
                np_data[key] = self.to_numpy(data[key])
        elif type(data) == list:
            np_data = []
            for x in data:
                np_data.append(self.to_numpy(x))
        else:
            raise NotImplementedError("Type {} not supported.".format(
                type(data)))
        return np_data

    def add(self, outputs, labels):
        self.t = np.concatenate([self.t, self.to_numpy(outputs["t"])], axis=0)
        if self.eps == 0:
            surv_key = "survival_seqs"
        else:
            surv_key = "survival_seqs_{}".format(self.eps)
        if self.survival_seqs is None:
            self.survival_seqs = self.to_numpy(outputs[surv_key])
        else:
            self.survival_seqs = np.concatenate(
                [self.survival_seqs, self.to_numpy(outputs[surv_key])],
                axis=1)
        self.labels = np.concatenate(
            [self.labels, self.to_numpy(labels)], axis=0)

    def value(self):
        S = self.survival_seqs
        kmf = KaplanMeierFitter()
        kmf.fit(self.t, event_observed=(1 - self.labels))
        G_T = kmf.predict(self.t, interpolate=True).to_numpy()

        t_span = np.linspace(self.t.min(), max(self.t[G_T > self.eps]),
                             S.shape[0])
        G_t = kmf.predict(t_span, interpolate=True).to_numpy()

        ind = ((self.t.reshape(1, -1) <= t_span.reshape(-1, 1))).astype(float)
        labels = self.labels

        # Remove indices where G_t are zero
        S = S[G_t > 1e-8]
        ind = ind[G_t > 1e-8]
        G_t = G_t[G_t > 1e-8]

        # Remove indices where G_T are zero
        S = S[:, G_T > 1e-8]
        ind = ind[:, G_T > 1e-8]
        labels = labels[G_T > 1e-8]
        G_T = G_T[G_T > 1e-8]

        labels = labels.reshape(1, -1)
        G_t = G_t.reshape(-1, 1)
        G_T = G_T.reshape(1, -1)

        bll = np.log(1 - S + 1e-10) * labels * ind / G_T
        bll += np.log(S + 1e-10) * (1 - ind) / G_t
        return bll.mean()

    def reset(self):
        self.t = np.array([])
        self.survival_seqs = None
        self.labels = np.array([])
