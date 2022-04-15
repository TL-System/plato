#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=o-iid-c.out

CUDA_LAUNCH_BLOCKING=1 python scaffold.py -c configs/iid-c.yml
