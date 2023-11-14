#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=eval.out

python dlg.py -c rec_resnet32_cifar100_eval.yml