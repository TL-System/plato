#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=1G
#SBATCH --output=FedBuff_EMNIST_lenet5_500_20eachRound_10least_10stalenss.out
filename='configs/FEMNIST/fedavg_lenet5.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
./run -c ./configs/MNIST/fedavg_lenet5.yml

