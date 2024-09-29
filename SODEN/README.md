# SODEN

This repo provides a pytorch implementation for the proposed model, *SODEN*, in the following paper:

[SODEN: A Scalable Continuous-Time Survival Model through Ordinary Differential Equation Networks](https://arxiv.org/abs/2008.08637)

[Weijing Tang](https://sites.google.com/umich.edu/weijingtang/home)\*, [Jiaqi Ma](http://www.jiaqima.com)\*, [Qiaozhu Mei](http://www-personal.umich.edu/~qmei/), and [Ji Zhu](http://dept.stat.lsa.umich.edu/~jizhu/). JMLR, Vol. 23, pp.34-1.

(\* equal contribution.)

## Requirements

Most required libraries should be included in `environment.yml`. To prepare the environment, run the following commands:
```shell
conda env create -f environment.yml
conda activate soden
```

## Datasets
The 10 splits of SUPPORT and METABRIC datasets are available at `data/support` and `data/metabric` respectively.

The MIMIC and MIMIC-SEQ datasets are derived from the [MIMIC-III database](https://mimic.physionet.org/gettingstarted/access/), which requires individual licenses to access. We preprocess the data using the [code](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/tree/master/Codes/mimic3_mvcv) by [PuruShotham and Meng et al. (2018)](https://www.sciencedirect.com/science/article/pii/S1532046418300716). In particular, the MIMIC dataset in our paper is the result of [12_get_avg_first24hrs_17-features-processed(fromdb).ipynb](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/Codes/mimic3_mvcv/12_get_avg_first24hrs_17-features-processed(fromdb).ipynb). And the MIMIC-SEQ dataset in our paper is the result of [11_get_time_series_sample_17-features-processed_24hrs.ipynb](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/Codes/mimic3_mvcv/11_get_time_series_sample_17-features-processed_24hrs.ipynb).


## Usage
### Generate config files
1. The folder `configs/` includes model config templates for various models. `range_specs.py` defines tuning ranges of hyper-parameters.

2. Run `generate_config.py` to randomly generate complete hyper-parameter configurations for random search. For example, to generate 100 random configurations for the proposed SODEN model on SUPPORT data, run the following command:
```shell
python generate_config.py --basic_model_config_file configs/support__rec_mlp.json --num_trials 100 --starting_trial 0 --random_seed 0
```
The generated 100 complete hyper-parameter configuration files will be stored in the folder `data/hp_configs`.

(The proposed SODEN model corresponds to `rec_mlp` function type in the code; and the ablation baselines SODEN-PH and SODEN-Cox respectively correspond to `cox_mlp_mlp` and `cox_mlp_exp` in the code.)

### Train
Example command to run a single trial of training SODEN on the SUPPORT data with the split 1 (out of 1-10):
```shell
python main.py --save_model --save_log --dataset support --path data/support --split 1 --seed 1 --model_config_file data/hp_configs/support__rec_mlp__0__model.json --train_config_file data/hp_configs/support__rec_mlp__0__train.json
```
In particular, the model and training hyper-parameters are specified by the files `support__rec_mlp__0__model.json` and `support__rec_mlp__0__train.json` we generated before. 

A model config filename takes the form `<dataset>__<model_type>__<trial_id>__model.json` and a training config filename takes the form `<dataset>__<model_type>__<trial_id>__train.json`.

### Evaluation
A couple of evaluation metrics including the loss function will be calculated throughout the training procedure. But some evaluation metrics are time-consuming so their calculations are left to a dedicated evaluation mode. After training is completed, run the following command to evaluate.
```shell
python main.py --evaluate --dataset support --path data/support --split 1 --seed 1 --model_config_file data/hp_configs/support__rec_mlp__0__model.json --train_config_file data/hp_configs/support__rec_mlp__0__train.json
```
This command will load the best model checkpoint and make the evaluation (make sure the `--save_model` argument is added to the training command).


## Acknowledgement
[Yuanhao Liu](https://statistics.rutgers.edu/people-pages/faculty/people/135-graduate-students/568-yuanhao-liu) and [Chenkai Sun](https://chenkaisun.github.io) contributed to part of the baseline implementation and simulation study in this project.
