# Neural RMTPP baseline runner (EasyTPP)

The historical C++ code below is PtPack. For the neural RMTPP baseline, this
repository uses `Models/EasyTPP/easy_tpp/model/torch_model/torch_rmtpp.py`.
`run_experiment.py` and `run.sh` provide the complete pipeline for every
adapted benchmark: conversion, training, best-checkpoint selection by
validation log-likelihood, final test, train/validation likelihood, RMSE and
accuracy plots, and a compact `tar.gz` export containing only log/CSV/plots.

From the repository root, install the Python dependencies once:

```bash
python3 -m venv .venv-rmtpp
source .venv-rmtpp/bin/activate
python -m pip install --upgrade pip
python -m pip install -e Models/EasyTPP
python -m pip install matplotlib
```

Run from the RMTPP model directory:

```bash
cd /Volumes/shenzm/Shuang_RA/Baseline/Models/RMTPP
bash run.sh amazon
bash run.sh retweet
bash run.sh taxi
bash run.sh stackoverflow
bash run.sh taobao
bash run.sh covid_policy_tracker
bash run.sh dws 13
bash run.sh dws 15
bash run.sh dws 17
bash run.sh dws 20
```

Use CPU, select a GPU, or override hyperparameters:

```bash
GPU=-1 bash run.sh taxi
GPU=1 bash run.sh dws 13
EPOCHS=80 BATCH_SIZE=64 LEARNING_RATE=0.001 bash run.sh amazon
```

Following the FullyNN runner layout, each run writes to
`Result/<group>/<dataset>_training_results/` under `Models/RMTPP`, with the
matching `.tar.gz`. Checkpoint/config/runtime files use temporary scratch space
and are removed after the run; the result directory and archive contain only
`log/`, `csv/`, and `plot/`.

# PtPack: The C++ Multivariate Temporal Point Process Package
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

PtPack is a C++ software library of high-dimensional temporal point processes. It aims to provide flexible modeling, learning, and inference of general multivariate temporal point processes to capture the latent dynamics governing the sheer volume of various temporal events arising from social networks, online media, financial trading, modern health-care, recommender systems, etc. Please check out the [project site](https://dunan.github.io/ptpack/html/index.html) for more details and documents.

## Prerequisites

- PtPack can be built on OS X and Linux.
- [gnuplot](http://www.gnuplot.info)

## Features

- Learning sparse interdependency structure of terminating point processes with applications in continuous-time information diffusions.

- Scalable continuous-time influence estimation and maximization.

- Learning multivariate Hawkes processes with different structural constraints, like: sparse, low-rank, customized triggering kernels, etc.

- Learning low-rank Hawkes processes for time-sensitive recommendations.

- Efficient simulation of standard multivariate Hawkes processes.

- Learning multivariate self-correcting processes.

- Simulation of customized general temporal point processes.

- Basic residual analysis and model checking of customized temporal point processes.

- Visualization of triggering kernels, intensity functions, and simulated events.

## Build static library

- cd MultiVariatePointProcess
- make

The built library will be saved under the directory build/lib

## Build examples

- cd MultiVariatePointProcess/example
- make

All built examples will be saved under the directory example/build
