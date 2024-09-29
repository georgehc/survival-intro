# Demo code for "An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes"

Author: George H. Chen (georgechen [at symbol] cmu.edu)

This repository provides accompanying code for the monograph:

```
George H. Chen. "An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes".
```

Throughout this repository, we regularly refer to specific sections, examples, equations, etc from the monograph. We also regularly cite papers listed in the bibliography at the end of the monograph.

## Prerequisites

- Anaconda Python 3 (tested with Anaconda version 2024.06 running Python 3.12.4)
- Additional packages: `lifelines`, `scikit-survival`, `torch` (tested with PyTorch version 2.3.1 with CUDA 12.1), `torchsurv`, `xgboost`, `hnswlib`, and `torchdiffeq`; for more precise details, see the [requirements.txt](requirements.txt) file (includes the packages we just mentioned along with various other packages that should already come with Anaconda Python 3 or that are dependencies for some packages we rely on) which can be installed via `pip install -r requirements.txt`

This repository comes with slightly modified versions of the following packages/files (which means that you do not need to manually obtain these yourself):
- Haavard Kvamme's [PyCox](https://github.com/havakv/pycox) and [torchtuples](https://github.com/havakv/torchtuples) packages (some bug fixes/print flushing) --- these are in the directories `./pycox/` and `./torchtuples/` respectively (note that Kvamme's code is under a BSD 2-clause license)
- One file copied from my earlier [survival-kernets](https://github.com/georgehc/survival-kernets) repository -- this is saved within the directory `./survival_kernets/` (my earlier code is under an MIT license)
- The [SODEN code by the original authors](https://github.com/jiaqima/SODEN) (Tang et al., 2022) --- this code is in the directory `./SODEN/` and I have modified `models.py` to make one of the imports relative (the SODEN code is under an MIT license)
- Two files copied (one of which has been slightly edited) from Vincent Jeanselme's [PyTorch Dynamic-DeepHit implementation](https://github.com/Jeanselme/DynamicDeepHit/) --- these are in the directory `./ddh/` (currently there is no license for these files from what I can tell)

## Datasets

- The SUPPORT dataset (Knaus et al., 1995) was obtained from: https://hbiostat.org/data/
- The PBC dataset (Fleming and Harrington, 1991) was obtained from: https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets/pbc2.csv
- The Rotterdam tumor bank (Foekens et al 2000) and GBSG (Schumacher et al 1994) datasets (train on Rotterdam and test on GBSG) are taken from the DeepSurv (Katzman et al 2018): https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/gbsg

The SUPPORT dataset is used for our Jupyter notebooks accompanying Sections 2-5; these particular notebooks can also be modified to work instead with the Rotterdam/GBSG datasets (we show what changes are needed only for the exponential time-to-event prediction model). The PBC dataset is used for our Jupyter notebooks accompanying Section 6.

## Outline of code notebooks

Below, we list the code notebooks that are provided. Note that these notebooks are *not* listed here in the same order as how their corresponding models are presented of the monograph. Instead, they are listed in an indented fashion, where the indentation is meant to indicate dependencies as to which notebook to view first.

- `S2.2.2_Exponential.ipynb`: exponential time-to-event prediction model (Examples 2.2.1 & 2.2.2 of the monograph) applied to the SUPPORT dataset; this notebook also includes more detailed descriptions of various code cells for loading and preprocessing the SUPPORT dataset, putting together the experimental setup, and using evaluation metrics which are all key ideas that show up in subsequent demos
    - `S2.2.2_Exponential_RotterdamGBSG.ipynb`: the same demo as `S2.2.2_Exponential.ipynb` except instead of using the SUPPORT dataset, we train on the Rotterdam tumor bank dataset and test on the GBSG dataset
    - `S2.2.2_Weibull.ipynb`: Weibull time-to-event prediction model (Example 2.2.3 of the monograph) applied to the SUPPORT dataset
    - `S2.3.3_DeepHit_single.ipynb`: DeepHit model *without* competing risks (Example 2.3.1 of the monograph) applied to the SUPPORT dataset; importantly, this notebook shows how to handle **time discretization**
        - `S2.3.3_Nnet-survival.ipynb`: Nnet-survival model (Example 2.3.2 of the monograph) applied to the SUPPORT dataset; this notebook uses the same time discretization seen in the DeepHit notebook
        - `S4.3_Survival_Kernets.ipynb`: Survival kernets model (Section 4.3 of the monograph); this notebook uses the same time discretization seen in the DeepHit notebook
        - `S6.1.4_DeepHit_competing.ipynb`: DeepHit model with competing events/risks (Section 6.1.4 of the monograph) applied to the PBC dataset (this is actually a time series dataset but per data point, we only consider the initial time step, so that we reduce it to a tabular dataset); the code is somewhat similar to DeepHit with a single critical event but is more involved
            - `S6.2.3_Dynamic-DeepHit.ipynb`: Dynamic-DeepHit model (Section 6.2.3 of the monograph) applied to the PBC dataset (where we now consider each data point as a time series)
    - `S2.3.3_Kaplan-Meier_Nelson-Aalen.ipynb`: Kaplan-Meier and Nelson-Aalen estimators (Example 2.3.3 of the monograph) applied to the SUPPORT dataset
    - `S3.3_DeepSurv.ipynb`: DeepSurv model (Section 3.3 of the monograph) applied to the SUPPORT dataset; note that the classical Cox model is a special case if the base neural net is just a linear layer that outputs a single number and has no bias (i.e., the base neural net just does an inner product)
        - `S3.4_Cox-Time.ipynb`: Cox-Time model (Section 3.4 of the monograph) applied to the SUPPORT dataset
    - `S5.2_SODEN.ipynb`: SODEN model (Sections 5.1 and 5.2, where the training and prediction procedures are specifically in Section 5.2) applied to the SUPPORT dataset
