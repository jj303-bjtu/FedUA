# Source Code for FedUA: From Ill-Posed to Well-Posed: Federated Uncertainty Aware Learning via Constrained Inference

PyTorch implementation of FedUA: From Ill-Posed to Well-Posed: Federated Uncertainty Aware Learning via Constrained Inference. 

## Data Setup

Folder `data/` contains scripts for generating non-IID client partitions, and for generating corrupted versions of CIFAR datasets.

A simple shell script `data/partition.sh` is provided to generate the CIFAR10/100 partitions.

Instructions and code for setting up EMNIST, partitioned by writers, can be found in `data/utils/emnist_setup.py`.

Corrupted CIFAR datasets can be generated with `python generate_corruptions.py`.

## Launch Script

We provide an example launch script in `launch.sh`. 

The launch script can be modified to run other configurations.

## Environment Details

We run our code using PyTorch 2.1 with CUDA 12. We provide a reference conda environment in `environment.yml`.
