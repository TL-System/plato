#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=oort_EMNIST_lenet5_500_20eachRound_10least_no_test_stalenss_10.out

python oort.py -c oort_EMNIST_lenet5.yml -b /data/ykang/plato
