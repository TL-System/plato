#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=FedAvg_EMNIST_lenet5_500_20eachRound_10least_no_test_stalenss_10_sleep10.out
filename='configs/EMNIST/fedavg_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
./run -c ./configs/EMNIST/fedavg_lenet5.yml -b /data/ykang/plato