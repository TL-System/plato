#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=FedBuff_FEMNIST_lenet5_500_20eachRound_10least_10staleness.out

./run -c ./configs/FEMNIST/fedbuff_lenet5.yml -b /data/ykang/plato