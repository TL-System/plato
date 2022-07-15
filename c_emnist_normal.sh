#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --output=c_emnist_normal.out

python3 examples/dlg/dlg.py -c examples/dlg/convergence_emnist_normal.yml
