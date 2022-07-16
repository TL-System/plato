#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=FedBuff_MNIST_lenet5_500_20eachRound_10least_no_test_stalenss_10.out
filename='./configs/MNIST/fedbuff_lenet5_noiid.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename
./run -c ./configs/MNIST/fedbuff_lenet5_noniid.yml -b /data/ykang/plato
