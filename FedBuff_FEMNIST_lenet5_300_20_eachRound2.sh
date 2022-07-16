#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=FedBuff_FEMNIST_lenet5_500_20eachRound_10least_stale10_sleep10.out
filename='oort_FEMNIST_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
./run -c ./configs/FEMNIST/fedavg_lenet5.yml -b /data/ykang/plato