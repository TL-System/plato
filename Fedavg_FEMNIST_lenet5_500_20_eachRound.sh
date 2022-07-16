#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=FedAvg_FEMNIST_lenet5_500_20eachRound_sync.out

./run -c ./configs/FEMNIST/fedavg_lenet5.yml -b /data/ykang/plato