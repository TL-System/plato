#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=resnet18.out

./../../run -c fedavg_resnet18_cifar100.yml