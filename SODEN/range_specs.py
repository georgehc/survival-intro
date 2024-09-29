import json
import os
from collections import OrderedDict

TRAIN_CATEGORY_TYPE_SPECS = {"optim": True, "batch_size": True}
TRAIN_RANGE_SPECS = {
    "batch_size": {
        "categories": [512, 1024]
    },
    # TODO: make learning rate sample range dependent on optimizer type.
    "learning_rate": {
        "low": -4.5,
        "high": -1.5,
        "scale": "log10",
    },
    "grad_clip": {
        "low": 1,
        "high": 100,
    },
    "weight_decay": {
        "low": -9,
        "high": -4,
        "scale": "log10",
    },
    "momentum": {
        "low": 0.85,
        "high": 0.99,
    },
    "optim": {
        "categories": ["RMSprop", "Adam"],
    },
}

LAYER_TYPES = [
    "conv2d", "fc", "pool2d", "bn1d", "bn2d", "relu", "lrelu", "drop",
    "surv_ode", "rnn", "nnet_surv", "nnet_surv_cox", "deephit", "deepsurv",
    "cox_time", "rdeephit"
]
LAYER_CATEGORY_TYPE_SPECS = {}
LAYER_RANGE_SPECS = {}

# Define LAYER_RANGE_SPECS and LAYER_CATEGORY_TYPE_SPECS for each layer type
for layer_type in LAYER_TYPES:
    if layer_type == "conv2d":
        is_category_types = {}
        sample_specs = {
            "out_channels": {
                "low": 3,
                "high": 8,
                "scale": "log2"
            },
            "kernel_size": {
                "low": 1,
                "high": 7
            },
            "stride": {
                "low": 0,
                "high": 4
            },
            "padding": {
                "low": 0,
                "high": 4
            },
        }
        for var_name in sample_specs:
            if var_name not in is_category_types:
                sample_specs[var_name]["is_int"] = True
    elif layer_type == "fc":
        is_category_types = {}
        sample_specs = {
            "out_features": {
                "low": 3,
                "high": 12,
                "scale": "log2",
                "is_int": True,
            },
        }
    elif layer_type == "pool2d":
        is_category_types = {}
        sample_specs = {
            "kernel_size": {
                "low": 2,
                "high": 3
            },
            "stride": {
                "low": 0,
                "high": 4
            },
            "padding": {
                "low": 0,
                "high": 4
            }
        }
        for var_name in sample_specs:
            sample_specs[var_name]["is_int"] = True
    elif layer_type == "bn1d" or layer_type == "bn2d":
        is_category_types = {"affine": True}
        sample_specs = {
            "momentum": {
                "low": 0.05,
                "high": 0.2,
            },
            "affine": {
                "categories": [True, False]
            },
        }
    elif layer_type == "drop":
        is_category_types = {}
        sample_specs = {
            "p": {
                "low": 0.01,
                "high": 0.5,
            },
        }
    elif layer_type == "surv_ode":
        is_category_types = {
            "num_layers": True,
            "batch_norm": True,
            "func_type": True,
            "has_feature": True
        }
        sample_specs = {
            "hidden_size": {
                "low": 2,
                "high": 7,
                "scale": "log2",
                "is_int": True,
            },
            "num_layers": {
                "categories": [1, 2, 4]
            },
            "batch_norm": {
                "categories": [True, False]
            },
            "func_type": {
                "categories": [
                    "mlp", "exponential", "weibull", "log_logistic",
                    "cox_mlp_exp", "cox_mlp_mlp"
                ]
            },
            "has_feature": {
                "categories": [True, False]
            },
        }
    elif layer_type == "rnn":
        is_category_types = {"rnn_type": True}
        sample_specs = {
            "hidden_size": {
                "low": 3,
                "high": 8,
                "scale": "log2",
                "is_int": True,
            },
            "num_layers": {
                "low": 1,
                "high": 3,
                "is_int": True,
            },
            "rnn_type": {
                "categories": ["LSTM", "GRU"]
            },
        }
    else:
        is_category_types = {}
        sample_specs = {}

    LAYER_CATEGORY_TYPE_SPECS[layer_type] = is_category_types
    LAYER_RANGE_SPECS[layer_type] = sample_specs


if __name__ == "__main__":
    pass
