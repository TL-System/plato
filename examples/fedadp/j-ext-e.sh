#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=o-ext-e.out

CUDA_LAUNCH_BLOCKING=1 python fedadp.py  -c configs/ext-e.yml
