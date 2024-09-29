import torch
from torch import nn
from torchtuples import tuplefy

def init_embedding(emb):
    """Weight initialization of embeddings (in place).
    Best practise from fastai
    
    Arguments:
        emb {torch.nn.Embedding} -- Embedding
    """
    w = emb.weight.data
    sc = 2 / (w.shape[1]+1)
    w.uniform_(-sc, sc)

def _accuracy(input, target):
    """Accuracy, i.e. mean(input == target)"""
    return input.eq(target.view_as(input)).float().mean()

def accuracy_binary(input, target):
    """Accuracy for binary models on input for logit models in (-inf, inf).
    Do not used for models with sigmoid output activation.
    """
    if len(input.shape) == 1:
        raise NotImplementedError(f"`accuracy_argmax` not implemented for shape {input.shape}")
    assert (target.min() == 0) and (target.max() == 1), 'We have binary classfication so we need 0/1'
    pred = torch.zeros_like(input).to(target.dtype)
    pred[input >= 0.] = 1
    return _accuracy(pred, target)

def accuracy_argmax(input, target):
    """Accuracy after argmax on input for logit models in (-inf, inf).
    Do not used for models with sigmoid/softmax output activation.

    Tupycally used as a metric passed to Model.fit()
    If input is one dimensional, we assume we have binary classification.
    """
    if len(input.shape) == 1:
        raise NotImplementedError(f"`accuracy_argmax` not implemented for shape {input.shape}")
    if input.shape[1] == 1:
        raise NotImplementedError("`accuracy_argmax` not for binary data. See `accuracy_binary` instead.")
    else:
        pred = input.argmax(dim=1, keepdim=True)
    return _accuracy(pred, target)


class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLPVanilla(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)


class EntityEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, dropout=0.):
        super().__init__()
        if not hasattr(num_embeddings, '__iter__'):
            num_embeddings = [num_embeddings]
        if not hasattr(embedding_dims, '__iter__'):
            embedding_dims = [embedding_dims]
        if len(num_embeddings) != len(embedding_dims):
            raise ValueError("Need 'num_embeddings' and 'embedding_dims' to have the same length")
        self.embeddings = nn.ModuleList()
        for n_emb, emb_dim in zip(num_embeddings, embedding_dims):
            emb = nn.Embedding(n_emb, emb_dim)
            init_embedding(emb)
            self.embeddings.append(emb)
        self.dropout = nn.Dropout(dropout) if dropout else None
    
    def forward(self, input):
        if input.shape[1] != len(self.embeddings):
            raise RuntimeError(f"Got input of shape '{input.shape}', but need dim 1 to be {len(self.embeddings)}.")
        input = [emb(input[:, i]) for i, emb in enumerate(self.embeddings)]
        input = torch.cat(input, 1)
        if self.dropout:
            input = self.dropout(input)
        return input


class MixedInputMLP(nn.Module):
    def __init__(self, in_features, num_embeddings, embedding_dims, num_nodes, out_features,
                 batch_norm=True, dropout=None, activation=nn.ReLU, dropout_embedding=0.,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.embeddings = EntityEmbeddings(num_embeddings, embedding_dims, dropout_embedding)
        input_mlp = in_features + sum(embedding_dims)
        self.mlp = MLPVanilla(input_mlp, num_nodes, out_features, batch_norm, dropout, activation,
                              output_activation, output_bias, w_init_)

    def forward(self, input_numeric, input_categoric):
        input = torch.cat([input_numeric, self.embeddings(input_categoric)], 1)
        return self.mlp(input)
