#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=oort_FEMNIST_lenet5_500_20eachRound.out

python oort.py -c oort_FEMNIST_lenet5_500_20eachRound.yml -b /data/ykang/plato