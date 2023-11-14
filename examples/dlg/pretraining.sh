#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=pretrain.out

./../../run -c fedavg_vgg16_cifar100.yml