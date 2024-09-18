#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=resnet152.out

python fedavg_pretraining.py -c fedavg_resnet152_cifar100.yml